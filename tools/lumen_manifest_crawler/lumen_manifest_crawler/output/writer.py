from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from lumen_manifest_crawler.fleet_artifacts import FleetArtifacts
from lumen_manifest_crawler.manifest import AgentBehaviorManifest, ValidationReport
from lumen_manifest_crawler.dataset.adapter_export import (
    adapter_runtime_manifest,
    agent_adapter_export_plan,
    augment_unsloth_config_for_adapter_export,
)
from lumen_manifest_crawler.dataset.fine_tuning import AgentFineTuningDataset
from lumen_manifest_crawler.output.hashing import sha256_file


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
        if name == "dataset_manifest":
            continue
        with (dataset_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    _write_embedding_outputs(output_dir / "embedding", datasets)

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


def _write_cross_model_index(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
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
        writer = csv.writer(handle)
        writer.writerow(["family", "recordCount", "splits", "roles", "taskTypes"])
        for name, records in sorted(datasets.items()):
            if name == "dataset_manifest":
                continue
            splits = sorted({str(record.get("split")) for record in records if record.get("split") is not None})
            roles = sorted({str(record.get("agentRole")) for record in records if record.get("agentRole") is not None})
            task_types = sorted({str(record.get("taskType")) for record in records if record.get("taskType") is not None})
            writer.writerow([name, len(records), ";".join(splits), ";".join(roles), ";".join(task_types)])


def _write_tool_registry_csv(path: Path, manifest: AgentBehaviorManifest) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "displayName", "requiresApproval", "permissionKey", "argumentCount", "source"])
        for tool in manifest.tools:
            writer.writerow([tool.id, tool.displayName or "", tool.requiresApproval, tool.permissionKey or "", len(tool.arguments), tool.source or ""])


def _write_routing_matrix_csv(path: Path, manifest: AgentBehaviorManifest) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
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


def _write_fine_tuning_outputs(root: Path, datasets: dict[str, AgentFineTuningDataset]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_runtime_manifest.json").write_text(
        json.dumps(adapter_runtime_manifest(datasets), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
