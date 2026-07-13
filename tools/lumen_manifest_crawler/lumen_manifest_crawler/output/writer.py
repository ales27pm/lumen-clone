from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from lumen_manifest_crawler.fleet_artifacts import FleetArtifacts
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ValidationReport
from lumen_manifest_crawler.dataset.adapter_export import (
    adapter_runtime_manifest,
    agent_adapter_export_plan,
    augment_unsloth_config_for_adapter_export,
)
from lumen_manifest_crawler.dataset.adapter_evaluation import build_evaluation_fingerprint_bundle
from lumen_manifest_crawler.dataset.fine_tuning import AgentFineTuningDataset
from lumen_manifest_crawler.dataset.public_adapter_eval_registry import (
    build_public_adapter_eval_fingerprint_bundle,
)
from lumen_manifest_crawler.output.hashing import sha256_file


EMBEDDING_DATASET_ALIASES = {
    "embedding_corpus": ("embedding", "corpus.jsonl"),
    "embedding_train_pairs": ("embedding", "train_pairs.jsonl"),
    "embedding_val_pairs": ("embedding", "val_pairs.jsonl"),
    "embedding_train_triplets": ("embedding", "train_triplets.jsonl"),
    "embedding_val_triplets": ("embedding", "val_triplets.jsonl"),
    "embedding_hard_negatives": ("embedding", "hard_negatives.jsonl"),
    "embedding_eval_retrieval": ("embedding", "eval_retrieval.jsonl"),
}

RERANKER_DATASET_ALIASES = {
    "reranker_train_pairs": ("reranker", "train_pairs.jsonl"),
    "reranker_val_pairs": ("reranker", "val_pairs.jsonl"),
    "reranker_hard_negative_pairs": ("reranker", "hard_negative_pairs.jsonl"),
    "reranker_eval_reranking": ("reranker", "eval_reranking.jsonl"),
}

CANONICAL_DATASET_ALIASES = {
    **EMBEDDING_DATASET_ALIASES,
    **RERANKER_DATASET_ALIASES,
}


def write_outputs(
    output_dir: Path,
    manifest: AgentBehaviorManifest,
    report: ValidationReport,
    datasets: dict[str, list[dict[str, Any]]],
    *,
    pretty: bool,
    fleet_artifacts: FleetArtifacts | None = None,
    manifest_markdown: str | None = None,
    cross_model_train_dir: Path | None = None,
    incremental_fingerprint: str | None = None,
    fine_tuning_datasets: dict[str, AgentFineTuningDataset] | None = None,
    fine_tuning_output_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # AgentBehaviorManifest.json is intentionally source-derived and deterministic.
    # Its artifactStatus field declares runtimeEvidence=false so synced app bundles
    # cannot be mistaken for live TestFlight proof.
    canonical_path = output_dir / "AgentBehaviorManifest.json"
    manifest.write_json(canonical_path, pretty=False)
    if pretty:
        manifest.write_json(output_dir / "AgentBehaviorManifest.pretty.json", pretty=True)
    (output_dir / "AgentBehaviorManifest.sha256").write_text(sha256_file(canonical_path) + "\n", encoding="utf-8")
    if incremental_fingerprint is not None:
        (output_dir / "AgentBehaviorManifest.incremental.sha256").write_text(incremental_fingerprint + "\n", encoding="utf-8")
    (output_dir / "manifest_validation_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _write_tool_registry_csv(output_dir / "tool_registry.csv", manifest)
    _write_routing_matrix_csv(output_dir / "routing_matrix.csv", manifest)

    if fleet_artifacts is not None:
        _write_fleet_artifacts(output_dir, fleet_artifacts, cross_model_train_dir)
    elif manifest_markdown is not None:
        (output_dir / "AgentBehaviorManifest.md").write_text(manifest_markdown, encoding="utf-8")

    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    legacy_dataset_manifest_jsonl = dataset_dir / "dataset_manifest.jsonl"
    if legacy_dataset_manifest_jsonl.exists():
        legacy_dataset_manifest_jsonl.unlink()

    dataset_manifest_records = datasets.get("dataset_manifest", [])
    if len(dataset_manifest_records) > 1:
        raise ValueError(
            f"Expected at most one dataset_manifest record while writing outputs to {output_dir}, "
            f"but got {len(dataset_manifest_records)}. Multiple dataset manifests would make lineage ambiguous."
        )
    if len(dataset_manifest_records) == 1:
        (output_dir / "dataset_manifest.json").write_text(
            json.dumps(dataset_manifest_records[0], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _write_dataset_index(output_dir / "dataset_index.csv", datasets)
    for name, records in datasets.items():
        if name == "dataset_manifest" or name in CANONICAL_DATASET_ALIASES:
            continue
        path = dataset_dir / f"{name}.jsonl"
        if not records:
            path.unlink(missing_ok=True)
            continue
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    self_model_cards = datasets.get("self_model_cards", [])
    if self_model_cards:
        _write_jsonl(output_dir / "self_model_cards.jsonl", self_model_cards)

    _write_embedding_outputs(output_dir / "embedding", datasets)
    _write_reranker_outputs(output_dir / "reranker", datasets)
    _write_dataset_aliases(dataset_dir, datasets)
    _write_runtime_grounding_outputs(output_dir, manifest, datasets)

    if fine_tuning_datasets is not None:
        _write_fine_tuning_outputs(fine_tuning_output_dir or (output_dir / "fine_tuning"), fine_tuning_datasets)


def _write_fleet_artifacts(output_dir: Path, artifacts: FleetArtifacts, cross_model_train_dir: Path | None) -> None:
    (output_dir / "fleet_system_prompts.json").write_text(
        json.dumps(artifacts.system_prompts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "AgentBehaviorManifest.md").write_text(artifacts.markdown, encoding="utf-8")
    target_dir = cross_model_train_dir or (output_dir / "cross_model_training")
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(target_dir / "cross_model_training.jsonl", artifacts.cross_model_training)
    _write_jsonl(target_dir / "orchestration_evals.jsonl", artifacts.orchestration_evals)
    split_records = _split_cross_model_records(artifacts.cross_model_training)
    for filename, records in split_records.items():
        _write_jsonl(target_dir / filename, records)
    _write_cross_model_index(target_dir / "cross_model_training_index.csv", artifacts.cross_model_training)


def _split_cross_model_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    sft = [record for record in records if record.get("recordType") == "sft"]
    dpo = [record for record in records if record.get("recordType") == "dpo"]
    sft_train, sft_val = _stable_split_records(sft)
    dpo_train, dpo_val = _stable_split_records(dpo)
    return {
        "train_sft_cross.jsonl": sft_train,
        "val_sft_cross.jsonl": sft_val,
        "dpo_train_cross.jsonl": dpo_train,
        "dpo_val_cross.jsonl": dpo_val,
    }


def _stable_split_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []
    ranked = sorted(records, key=lambda record: str(record.get("id") or ""))
    val_count = 1 if len(ranked) > 1 else 0
    if len(ranked) >= 10:
        val_count = max(1, round(len(ranked) * 0.15))
    val_ids = {str(record.get("id") or "") for record in ranked[:val_count]}
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for record in records:
        cloned = {**record}
        cloned["split"] = "validation" if str(record.get("id") or "") in val_ids else "train"
        if cloned["split"] == "validation":
            val.append(cloned)
        else:
            train.append(cloned)
    return train, val


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_dataset_aliases(dataset_dir: Path, datasets: dict[str, list[dict[str, Any]]]) -> None:
    for dataset_name, (canonical_dir, canonical_filename) in CANONICAL_DATASET_ALIASES.items():
        if dataset_name not in datasets:
            continue
        alias = dataset_dir / f"{dataset_name}.jsonl"
        target = Path("..") / canonical_dir / canonical_filename
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        alias.symlink_to(target)


def _write_cross_model_index(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["recordType", "agentRole", "taskType", "recordCount"])
        counts: dict[tuple[str, str, str], int] = {}
        for record in records:
            key = (
                str(record.get("recordType") or "unknown"),
                str(record.get("agentRole") or "unknown"),
                str(record.get("taskType") or "unknown"),
            )
            counts[key] = counts.get(key, 0) + 1
        for (record_type, agent_role, task_type), count in sorted(counts.items()):
            writer.writerow([record_type, agent_role, task_type, count])


def _write_dataset_index(path: Path, datasets: dict[str, list[dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["family", "recordCount", "splits", "roles", "taskTypes"])
        for name, records in sorted(datasets.items()):
            if name == "dataset_manifest" or not records:
                continue
            splits = sorted({str(record.get("split")) for record in records if record.get("split") is not None})
            roles = sorted({str(record.get("agentRole")) for record in records if record.get("agentRole") is not None})
            task_types = sorted({str(record.get("taskType")) for record in records if record.get("taskType") is not None})
            writer.writerow([name, len(records), ";".join(splits), ";".join(roles), ";".join(task_types)])


def _write_tool_registry_csv(path: Path, manifest: AgentBehaviorManifest) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "displayName", "requiresApproval", "permissionKey", "argumentCount", "source"])
        for tool in manifest.tools:
            writer.writerow([tool.id, tool.displayName or "", tool.requiresApproval, tool.permissionKey or "", len(tool.arguments), tool.source or ""])


def _write_routing_matrix_csv(path: Path, manifest: AgentBehaviorManifest) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["intent", "allowedTools", "forbiddenTools"])
        for entry in manifest.routingMatrix:
            writer.writerow([entry.intent, ";".join(entry.allowedTools), ";".join(entry.forbiddenTools)])


def _write_embedding_outputs(root: Path, datasets: dict[str, list[dict[str, Any]]]) -> None:
    embedding_files = {
        "corpus.jsonl": datasets.get("embedding_corpus", []),
        "train_pairs.jsonl": datasets.get("embedding_train_pairs", []),
        "val_pairs.jsonl": datasets.get("embedding_val_pairs", []),
        "train_triplets.jsonl": datasets.get("embedding_train_triplets", []),
        "val_triplets.jsonl": datasets.get("embedding_val_triplets", []),
        "hard_negatives.jsonl": datasets.get("embedding_hard_negatives", []),
        "eval_retrieval.jsonl": datasets.get("embedding_eval_retrieval", []),
    }
    if not any(embedding_files.values()) and not datasets.get("embedding_dataset_card"):
        return
    root.mkdir(parents=True, exist_ok=True)
    for filename, records in embedding_files.items():
        _write_jsonl(root / filename, records)
    card_records = datasets.get("embedding_dataset_card", [])
    card = card_records[0] if card_records else {
        "schemaVersion": "1.0.0",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "counts": {filename.removesuffix(".jsonl"): len(records) for filename, records in embedding_files.items()},
    }
    (root / "dataset_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_reranker_outputs(root: Path, datasets: dict[str, list[dict[str, Any]]]) -> None:
    reranker_files = {
        "train_pairs.jsonl": datasets.get("reranker_train_pairs", []),
        "val_pairs.jsonl": datasets.get("reranker_val_pairs", []),
        "hard_negative_pairs.jsonl": datasets.get("reranker_hard_negative_pairs", []),
        "eval_reranking.jsonl": datasets.get("reranker_eval_reranking", []),
    }
    if not any(reranker_files.values()) and not datasets.get("reranker_dataset_card"):
        return
    root.mkdir(parents=True, exist_ok=True)
    for filename, records in reranker_files.items():
        _write_jsonl(root / filename, records)
    card_records = datasets.get("reranker_dataset_card", [])
    card = card_records[0] if card_records else {
        "schemaVersion": "1.0.0",
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "counts": {filename.removesuffix(".jsonl"): len(records) for filename, records in reranker_files.items()},
    }
    (root / "dataset_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_runtime_grounding_outputs(
    output_dir: Path,
    manifest: AgentBehaviorManifest,
    datasets: dict[str, list[dict[str, Any]]],
) -> None:
    codebase_records = [
        record for record in datasets.get("codebase_home_corpus", [])
        if isinstance(record, dict)
    ]
    if not codebase_records:
        return

    module_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    for record in codebase_records:
        module = str(record.get("module") or "unknown")
        language = str(record.get("language") or "unknown")
        module_counts[module] = module_counts.get(module, 0) + 1
        language_counts[language] = language_counts.get(language, 0) + 1

    selected_records = _select_runtime_grounding_records(codebase_records)
    bundle = {
        "schemaVersion": "1.0.0",
        "artifactKind": "agent_grounding_runtime_bundle",
        "sourceFamilies": ["codebase_home_corpus", "codebase_home_sft", "codebase_home_chunks", "codebase_home_chunk_sft"],
        "manifestCommit": manifest.sourceIntegrity.commit,
        "manifestToolCount": len(manifest.tools),
        "manifestIntentCount": len(manifest.intents),
        "codebaseHome": {
            "recordCount": len(codebase_records),
            "moduleCounts": dict(sorted(module_counts.items())),
            "languageCounts": dict(sorted(language_counts.items())),
            "selectedFiles": selected_records,
        },
        "injectionPolicy": {
            "target": "AgentGroundingPromptComposer",
            "purpose": "Give every bundled fleet prompt a compact map of Lumen's actual app home: modules, files, responsibilities, and source hashes.",
            "privacy": "static repo source only; generated/build/model/private-local folders excluded",
            "maxPromptCharacters": 6000,
        },
    }
    (output_dir / "runtime_grounding_bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "runtime_grounding_prompt.md").write_text(
        _runtime_grounding_prompt(bundle),
        encoding="utf-8",
    )


def _select_runtime_grounding_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority_tokens = (
        "AgentGrounding/",
        "Assistant/",
        "Services/Agent",
        "Services/Tools/",
        "Services/Intent",
        "Memory/",
        "RAG/",
        "Tools/",
        "tools/lumen_manifest_crawler/",
        "docs/",
    )

    def score(record: dict[str, Any]) -> tuple[int, str]:
        path = str(record.get("path") or "")
        priority = min((idx for idx, token in enumerate(priority_tokens) if token in path), default=99)
        return priority, path

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in sorted(records, key=score):
        path = str(record.get("path") or "")
        if path == "." or path in seen:
            continue
        seen.add(path)
        selected.append({
            "path": path,
            "module": record.get("module"),
            "language": record.get("language"),
            "sha256": record.get("sha256"),
            "responsibility": record.get("responsibility"),
            "symbols": list(record.get("symbols") or [])[:16],
            "imports": list(record.get("imports") or [])[:16],
        })
        if len(selected) >= 80:
            break
    return selected


def _runtime_grounding_prompt(bundle: dict[str, Any]) -> str:
    home = bundle.get("codebaseHome") if isinstance(bundle.get("codebaseHome"), dict) else {}
    modules = home.get("moduleCounts") if isinstance(home.get("moduleCounts"), dict) else {}
    selected = home.get("selectedFiles") if isinstance(home.get("selectedFiles"), list) else []
    top_modules = sorted(modules.items(), key=lambda item: (-int(item[1]), str(item[0])))[:24]
    lines = [
        "# Lumen Runtime Grounding Bundle",
        "",
        "Use this compact codebase-home map as bundled source grounding. It is generated at build time from static repo files and should be treated as navigational context, not private user data.",
        "",
        f"- Manifest commit: `{bundle.get('manifestCommit') or 'unknown'}`",
        f"- Tools: `{bundle.get('manifestToolCount')}`",
        f"- Intents: `{bundle.get('manifestIntentCount')}`",
        f"- Codebase-home records: `{home.get('recordCount')}`",
        "",
        "## Top Modules",
    ]
    lines.extend(f"- `{name}`: {count} files" for name, count in top_modules)
    lines.extend(["", "## Key Files"])
    for record in selected[:60]:
        path = record.get("path") or ""
        module = record.get("module") or ""
        responsibility = " ".join(str(record.get("responsibility") or "").split())[:220]
        symbols = ", ".join(str(symbol) for symbol in (record.get("symbols") or [])[:8])
        if symbols:
            lines.append(f"- `{path}` ({module}): {responsibility} Symbols: {symbols}.")
        else:
            lines.append(f"- `{path}` ({module}): {responsibility}")
    lines.append("")
    return "\n".join(lines)


def _write_fine_tuning_outputs(root: Path, datasets: dict[str, AgentFineTuningDataset]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_runtime_manifest.json").write_text(
        json.dumps(adapter_runtime_manifest(datasets), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "public_evaluation_fingerprints.json").write_text(
        json.dumps(
            build_public_adapter_eval_fingerprint_bundle(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for agent, dataset in sorted(datasets.items()):
        d = root / agent
        d.mkdir(parents=True, exist_ok=True)
        _write_jsonl(d / "train_sft.jsonl", dataset.train_sft)
        _write_jsonl(d / "val_sft.jsonl", dataset.val_sft)
        _write_jsonl(d / "train_dpo.jsonl", dataset.train_dpo)
        _write_jsonl(d / "val_dpo.jsonl", dataset.val_dpo)
        _write_jsonl(d / "eval.jsonl", dataset.eval)
        (d / "dataset_card.json").write_text(
            json.dumps(dataset.dataset_card, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (d / "evaluation_fingerprints.json").write_text(
            json.dumps(
                build_evaluation_fingerprint_bundle(dataset.eval),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (d / "contamination_report.json").write_text(
            json.dumps(dataset.contamination_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (d / "experiment_manifest.json").write_text(
            json.dumps(dataset.experiment_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        experiments_root = d / "experiments"
        current_variants = set(dataset.experiment_variants)
        if experiments_root.exists():
            for existing in experiments_root.iterdir():
                if existing.name in current_variants:
                    continue
                if existing.is_dir() and not existing.is_symlink():
                    shutil.rmtree(existing)
                else:
                    existing.unlink()
        for variant, artifacts in sorted(dataset.experiment_variants.items()):
            variant_root = experiments_root / variant
            variant_root.mkdir(parents=True, exist_ok=True)
            _write_jsonl(variant_root / "train_sft.jsonl", artifacts["train_sft"])
            _write_jsonl(variant_root / "val_sft.jsonl", artifacts["val_sft"])
            _write_jsonl(variant_root / "train_dpo.jsonl", artifacts["train_dpo"])
            _write_jsonl(variant_root / "val_dpo.jsonl", artifacts["val_dpo"])
            (variant_root / "contamination_report.json").write_text(
                json.dumps(artifacts["contamination_report"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (variant_root / "variant_manifest.json").write_text(
                json.dumps(artifacts["variant_manifest"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        adapter_first_config = augment_unsloth_config_for_adapter_export(agent, dataset.unsloth_config)
        (d / "unsloth_config.json").write_text(
            json.dumps(adapter_first_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (d / "adapter_export_plan.json").write_text(
            json.dumps(
                agent_adapter_export_plan(agent, dataset.dataset_card, adapter_first_config),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
