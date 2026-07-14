from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from .adapter_artifact import verify_adapter_artifact
    from .train_sft import _verify_base_model_lineage
except ImportError:
    module_dir = str(Path(__file__).resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    from adapter_artifact import verify_adapter_artifact
    from train_sft import _verify_base_model_lineage


AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")
GGUF_MARKERS = {"gguf", "merged", "release", "bake", "finetune", "finetuned"}
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _emit(message: str) -> None:
    sys.stdout.write(message.rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optionally bake per-agent LoRA adapters into merged GGUF artifacts. "
            "Adapter-first training is the default; pass --release-bake to merge/export."
        )
    )
    parser.add_argument(
        "--release-bake",
        action="store_true",
        help="Explicitly enable optional adapter merge/export. Without this flag, no GGUF merge is performed.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to a per-agent Unsloth config JSON. Can be repeated.",
    )
    parser.add_argument(
        "--config-dir",
        default="generated/fine_tuning",
        help=(
            "Directory containing generated per-agent configs (used when --config is omitted). "
            "Each config is read from <agent>/unsloth_config.json."
        ),
    )
    parser.add_argument(
        "--agents",
        default=",".join(AGENTS),
        help="Comma-separated agents to process.",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="Override GGUF quantization method (for example: q4_k_m, q8_0, f16).",
    )
    parser.add_argument(
        "--output-root",
        default="models/gguf_release_bake",
        help="Root directory for optional release-baked merged GGUF artifacts.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="Optional Hugging Face model repo id to upload GGUF files to.",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Create HF repo as private if it does not exist.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip Hugging Face upload even if repo id is available.",
    )
    parser.add_argument(
        "--max-memory-usage",
        type=float,
        default=None,
        help="Override Unsloth maximum_memory_usage for GGUF export.",
    )
    parser.add_argument(
        "--manifest-output",
        default="generated/fine_tuning/release_bake_gguf_manifest.json",
        help="Path to write optional release-baked GGUF artifact manifest.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing merged GGUF files when present instead of re-exporting.",
    )
    return parser.parse_args()


def _tokenize_path(value: str) -> set[str]:
    return set("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def _validate_path_tokens(*, path: str, required_token: str, markers: set[str], label: str) -> None:
    tokens = _tokenize_path(path)
    if required_token not in tokens:
        raise ValueError(f"{label} must include slot token '{required_token}'. Got: {path}")
    if not markers.intersection(tokens):
        options = ", ".join(sorted(markers))
        raise ValueError(f"{label} must include one marker token in [{options}]. Got: {path}")


def _adapter_dir(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("adapter_output_dir") or cfg["output_dir"])).resolve()


def _portable_manifest_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "agent",
        "base_model_name",
        "baseModelRevision",
        "baseModelIndexDigest",
        "baseModelIndexReferencedShardNames",
        "baseModelIndexShardBindingSHA256",
        "baseModelArtifactDigest",
        "baseModelWeightShards",
        "baseModelTokenizerDigest",
        "max_seq_length",
        "output_dir",
    }
    missing = [key for key in sorted(required) if key not in cfg]
    if missing:
        raise ValueError(f"{path} missing required keys: {', '.join(missing)}")
    agent = str(cfg["agent"]).strip().lower()
    if agent not in AGENTS:
        raise ValueError(f"{path} has unsupported agent '{agent}'")
    _validate_path_tokens(
        path=str(_adapter_dir(cfg)),
        required_token=agent,
        markers={"lora", "adapter", "sft", "dpo", "orpo", "finetune", "finetuned"},
        label="adapter_output_dir",
    )
    if cfg.get("merge_adapters_by_default") is not False:
        raise ValueError(f"{path} must set merge_adapters_by_default=false for adapter-first training")
    if cfg.get("release_bake_enabled_by_default") is not False:
        raise ValueError(f"{path} must set release_bake_enabled_by_default=false")
    return cfg


def gather_configs(config_paths: list[str], config_dir: str, selected_agents: list[str]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    if config_paths:
        for raw in config_paths:
            configs.append(load_config(Path(raw).resolve()))
    else:
        root = Path(config_dir).resolve()
        for agent in selected_agents:
            nested = root / agent / "unsloth_config.json"
            configs.append(load_config(nested if nested.is_file() else root / f"{agent}.json"))

    filtered = [cfg for cfg in configs if str(cfg["agent"]).strip().lower() in set(selected_agents)]
    filtered.sort(key=lambda item: selected_agents.index(str(item["agent"]).strip().lower()))
    return filtered


def ensure_hf_repo(repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for upload. Install it in your Python environment.") from exc
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)


def upload_file(repo_id: str, local_path: Path, remote_name: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for upload. Install it in your Python environment.") from exc
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload release-baked GGUF: {remote_name}",
    )


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_adapter_training_lineage(
    cfg: dict[str, Any], artifact: dict[str, Any]
) -> tuple[str, str | None]:
    phase = str(cfg.get("adapter_training_phase") or "sft")
    configured_parent = cfg.get("parent_sft_adapter_sha256")
    artifact_parent = artifact.get("parentSFTAdapterSHA256")
    if phase == "sft":
        if configured_parent is not None or artifact_parent is not None:
            raise ValueError("SFT GGUF export must not declare a parent SFT adapter")
        return phase, None
    if phase != "sft_dpo":
        raise ValueError("adapter_training_phase must be either sft or sft_dpo")
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(configured_parent or "")) is None
        or artifact_parent != configured_parent
    ):
        raise ValueError(
            "SFT-to-DPO GGUF export must bind the finalized adapter to the configured parent SFT digest"
        )
    return phase, str(configured_parent)


def _verified_release_bake_lineage(cfg: dict[str, Any]) -> dict[str, Any]:
    """Verify the adapter and finalized experiment lineage before any GGUF use."""

    _verify_base_model_lineage(cfg)
    agent = str(cfg["agent"]).strip().lower()
    adapter_dir = _adapter_dir(cfg)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found for {agent}: {adapter_dir}")

    finalized_path = Path(
        str(
            cfg.get("finalized_variant_manifest")
            or (Path(str(cfg["output_dir"])) / "finalized_variant_manifest.json")
        )
    ).resolve()
    if not finalized_path.is_file():
        raise FileNotFoundError(f"Finalized variant manifest not found for {agent}: {finalized_path}")
    finalized = json.loads(finalized_path.read_text(encoding="utf-8"))
    if not isinstance(finalized, dict):
        raise ValueError(f"Finalized variant manifest must be a JSON object: {finalized_path}")
    expected_manifest_sha = finalized.get("variantManifestSHA256")
    unsigned = dict(finalized)
    unsigned.pop("variantManifestSHA256", None)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(expected_manifest_sha or "")) is None
        or _canonical_sha256(unsigned) != expected_manifest_sha
    ):
        raise ValueError(f"Finalized variant manifest integrity check failed: {finalized_path}")

    variant = cfg.get("variant")
    source_manifest_sha = cfg.get("variantManifestSHA256")
    if (
        finalized.get("agent") != agent
        or not isinstance(variant, str)
        or finalized.get("variant") != variant
        or re.fullmatch(r"[0-9a-f]{64}", str(source_manifest_sha or "")) is None
        or finalized.get("sourceVariantManifestSHA256") != source_manifest_sha
    ):
        raise ValueError("Finalized variant manifest is not bound to the selected agent and source variant")

    for field in (
        "baseModelRevision",
        "baseModelIndexDigest",
        "baseModelIndexReferencedShardNames",
        "baseModelIndexShardBindingSHA256",
        "baseModelArtifactDigest",
        "baseModelWeightShards",
        "baseModelTokenizerDigest",
    ):
        if finalized.get(field) != cfg.get(field):
            raise ValueError(f"Finalized variant manifest {field} does not match the GGUF config")
    if finalized.get("baseModelID") != cfg.get("baseModelID", cfg["base_model_name"]):
        raise ValueError("Finalized variant manifest baseModelID does not match the GGUF config")
    training_environment_sha = cfg.get("trainingEnvironmentSHA256")
    training_environment = finalized.get("trainingEnvironment")
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(training_environment_sha or "")) is None
        or finalized.get("trainingEnvironmentSHA256") != training_environment_sha
        or not isinstance(training_environment, dict)
        or _canonical_sha256(training_environment) != training_environment_sha
    ):
        raise ValueError("Finalized variant manifest training environment does not match the GGUF config")
    attestation = cfg.get("variantAttestation")
    if (
        not isinstance(attestation, dict)
        or attestation.get("variant") != variant
        or attestation.get("variantManifestSHA256") != source_manifest_sha
        or attestation.get("trainingEnvironmentSHA256") != training_environment_sha
        or finalized.get("trainingCorpusSHA256") != attestation.get("trainingCorpusSHA256")
        or finalized.get("trainingConfigSHA256")
        != attestation.get("effectiveTrainingConfigSHA256")
        or {
            name: contract.get("sha256")
            for name, contract in sorted((finalized.get("datasets") or {}).items())
            if isinstance(contract, dict)
            and isinstance(contract.get("sha256"), str)
        }
        != attestation.get("laneHashes")
    ):
        raise ValueError("Finalized variant manifest does not match the prepared training attestation")

    artifact = finalized.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Finalized variant manifest lacks adapter artifact lineage")
    expected_adapter_sha = artifact.get("adapterSHA256")
    expected_phase, expected_parent_sft_sha = _expected_adapter_training_lineage(
        cfg, artifact
    )
    if (
        artifact.get("status") != "trained"
        or re.fullmatch(r"[0-9a-f]{64}", str(expected_adapter_sha or "")) is None
        or artifact.get("adapterManifestSHA256") != expected_adapter_sha
        or artifact.get("trainingPhase") != expected_phase
    ):
        raise ValueError("Finalized variant manifest has invalid adapter artifact lineage")
    adapter_manifest = verify_adapter_artifact(
        adapter_dir,
        expected_adapter_sha256=expected_adapter_sha,
        expected_training_phase=expected_phase,
        expected_parent_sft_adapter_sha256=(
            str(expected_parent_sft_sha) if expected_parent_sft_sha is not None else None
        ),
    )

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if (
        not isinstance(adapter_config, dict)
        or adapter_config.get("base_model_name_or_path") != cfg["base_model_name"]
    ):
        raise ValueError("Adapter base_model_name_or_path does not match the pinned GGUF base model")
    return {
        "adapterSHA256": adapter_manifest["adapterSHA256"],
        "adapterTrainingPhase": expected_phase,
        "finalizedVariantManifestSHA256": expected_manifest_sha,
        "sourceVariantManifestSHA256": source_manifest_sha,
        "baseModelRevision": cfg["baseModelRevision"],
        "baseModelIndexDigest": cfg["baseModelIndexDigest"],
        "baseModelIndexReferencedShardNames": cfg["baseModelIndexReferencedShardNames"],
        "baseModelIndexShardBindingSHA256": cfg["baseModelIndexShardBindingSHA256"],
        "baseModelArtifactDigest": cfg["baseModelArtifactDigest"],
        "baseModelTokenizerDigest": cfg["baseModelTokenizerDigest"],
        "trainingEnvironmentSHA256": training_environment_sha,
    }


def _release_bake_skipped_manifest(configs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": "adapter_first",
        "release_bake_requested": False,
        "skipped": True,
        "reason": "Adapter-first training keeps LoRA adapters separate by default. Pass --release-bake to explicitly merge/export GGUF artifacts.",
        "manifest_output": _portable_manifest_path(args.manifest_output),
        "agents": {
            str(cfg["agent"]).strip().lower(): {
                "agent": str(cfg["agent"]).strip().lower(),
                "adapter_dir": _portable_manifest_path(str(cfg.get("adapter_output_dir") or cfg["output_dir"])),
                "base_model_name": cfg["base_model_name"],
                "merge_adapters_by_default": False,
                "release_bake_enabled_by_default": False,
            }
            for cfg in configs
        },
    }


def export_agent_gguf(
    cfg: dict[str, Any],
    *,
    output_root: Path,
    quantization_override: str | None,
    max_memory_usage_override: float | None,
) -> dict[str, Any]:
    try:
        from unsloth import FastLanguageModel  # type: ignore
        from unsloth.save import patch_saving_functions  # type: ignore
        from peft import PeftModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing Unsloth/PEFT dependencies. Run with .venv-unsloth/bin/python or install unsloth and peft."
        ) from exc

    agent = str(cfg["agent"]).strip().lower()
    adapter_dir = _adapter_dir(cfg)
    lineage = _verified_release_bake_lineage(cfg)

    quantization = str(
        quantization_override
        or cfg.get("gguf_quantization")
        or cfg.get("quantization_method")
        or "q4_k_m"
    ).lower()

    agent_output_dir = Path(
        str(cfg.get("gguf_output_dir") or (output_root / f"{agent}_release_bake_gguf"))
    ).resolve()
    _validate_path_tokens(
        path=str(agent_output_dir),
        required_token=agent,
        markers=GGUF_MARKERS,
        label="gguf_output_dir",
    )
    agent_output_dir.mkdir(parents=True, exist_ok=True)

    maximum_memory_usage = float(
        max_memory_usage_override
        if max_memory_usage_override is not None
        else cfg.get("gguf_maximum_memory_usage", 0.75)
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model_name"],
        revision=cfg["baseModelRevision"],
        max_seq_length=int(cfg["max_seq_length"]),
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    if not hasattr(model, "save_pretrained_gguf"):
        model = patch_saving_functions(model)

    scratch_dir = agent_output_dir / "_unsloth_release_bake"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    result = model.save_pretrained_gguf(
        str(scratch_dir),
        tokenizer,
        quantization_method=quantization,
        maximum_memory_usage=maximum_memory_usage,
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected GGUF export result for {agent}: {type(result)}")
    gguf_files = [Path(p) for p in result.get("gguf_files") or []]
    if not gguf_files:
        raise RuntimeError(f"No GGUF files produced for {agent}")

    selected: Path | None = None
    for candidate in gguf_files:
        name = candidate.name.lower()
        if name.endswith(".gguf") and "mmproj" not in name:
            selected = candidate
            break
    if selected is None:
        selected = gguf_files[0]

    target_name = f"lumen-{agent}-release-bake-{quantization}.gguf"
    target_path = agent_output_dir / target_name
    shutil.copy2(selected, target_path)

    summary = {
        "agent": agent,
        "mode": "optional_release_bake",
        "quantization": quantization,
        "adapter_dir": str(adapter_dir),
        "gguf_output_dir": str(agent_output_dir),
        "gguf_file": target_name,
        "gguf_path": str(target_path),
        "size_bytes": target_path.stat().st_size,
        "sha256": sha256sum(target_path),
        "base_model_name": cfg["base_model_name"],
        **lineage,
    }
    (agent_output_dir / "gguf_release_bake_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def existing_summary_for_agent(
    cfg: dict[str, Any],
    *,
    output_root: Path,
    quantization_override: str | None,
) -> dict[str, Any] | None:
    agent = str(cfg["agent"]).strip().lower()
    quantization = str(
        quantization_override
        or cfg.get("gguf_quantization")
        or cfg.get("quantization_method")
        or "q4_k_m"
    ).lower()
    agent_output_dir = Path(
        str(cfg.get("gguf_output_dir") or (output_root / f"{agent}_release_bake_gguf"))
    ).resolve()
    _validate_path_tokens(
        path=str(agent_output_dir),
        required_token=agent,
        markers=GGUF_MARKERS,
        label="gguf_output_dir",
    )
    target_name = f"lumen-{agent}-release-bake-{quantization}.gguf"
    target_path = agent_output_dir / target_name
    if not target_path.exists():
        return None
    lineage = _verified_release_bake_lineage(cfg)
    report_path = agent_output_dir / "gguf_release_bake_report.json"
    if not report_path.is_file():
        raise ValueError(f"Cannot reuse GGUF without its lineage report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"GGUF lineage report must be a JSON object: {report_path}")
    expected = {
        "agent": agent,
        "mode": "optional_release_bake",
        "quantization": quantization,
        "adapter_dir": str(_adapter_dir(cfg)),
        "gguf_output_dir": str(agent_output_dir),
        "gguf_file": target_name,
        "gguf_path": str(target_path),
        "size_bytes": target_path.stat().st_size,
        "sha256": sha256sum(target_path),
        "base_model_name": cfg["base_model_name"],
        **lineage,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Existing GGUF lineage report does not match current {key}")
    return {**expected, "reused_existing": True}


def _write_manifest(path: str, manifest: dict[str, Any]) -> Path:
    manifest_path = Path(path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    args = parse_args()
    selected_agents = [item.strip().lower() for item in args.agents.split(",") if item.strip()]
    for agent in selected_agents:
        if agent not in AGENTS:
            raise ValueError(f"Unsupported agent in --agents: {agent}")

    configs = gather_configs(args.config, args.config_dir, selected_agents)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.release_bake:
        manifest_path = _write_manifest(args.manifest_output, _release_bake_skipped_manifest(configs, args))
        _emit(f"Skipped GGUF release bake by default. Wrote adapter-first manifest: {manifest_path}")
        _emit("Pass --release-bake to explicitly merge adapters into GGUF artifacts.")
        return

    if args.hf_repo_id and not args.skip_upload:
        ensure_hf_repo(args.hf_repo_id, args.hf_private)

    manifest: dict[str, Any] = {
        "mode": "optional_release_bake",
        "release_bake_requested": True,
        "repo_id": args.hf_repo_id,
        "quantization_override": args.quantization,
        "agents": {},
    }

    for cfg in configs:
        summary = None
        if args.skip_existing:
            summary = existing_summary_for_agent(
                cfg,
                output_root=output_root,
                quantization_override=args.quantization,
            )
        if summary is None:
            summary = export_agent_gguf(
                cfg,
                output_root=output_root,
                quantization_override=args.quantization,
                max_memory_usage_override=args.max_memory_usage,
            )
        agent = summary["agent"]
        if args.hf_repo_id and not args.skip_upload:
            upload_file(
                repo_id=args.hf_repo_id,
                local_path=Path(summary["gguf_path"]),
                remote_name=summary["gguf_file"],
            )
            summary["hf_repo_id"] = args.hf_repo_id
            summary["hf_file_name"] = summary["gguf_file"]
        manifest["agents"][agent] = summary

    manifest_path = _write_manifest(args.manifest_output, manifest)
    _emit(f"Wrote GGUF release-bake manifest: {manifest_path}")


if __name__ == "__main__":
    main()
