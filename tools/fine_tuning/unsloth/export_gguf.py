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
    from .training_lineage import TRAINING_VARIANT_ATTESTATION_SCHEMA
    from .train_sft import (
        ADAPTER_BASE_TOKENIZER_FILES,
        ADAPTER_DERIVED_TOKENIZER_FILES,
        _controlled_torch_dtype,
        _load_verified_runtime_tokenizer_source,
        _runtime_tokenizer_evidence,
        _verify_base_model_lineage,
        _verify_runtime_model_binding,
        _verify_runtime_tokenizer_binding,
        _verified_private_runtime_model_snapshot,
    )
except ImportError:
    module_dir = str(Path(__file__).resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    from adapter_artifact import verify_adapter_artifact
    from training_lineage import TRAINING_VARIANT_ATTESTATION_SCHEMA
    from train_sft import (
        ADAPTER_BASE_TOKENIZER_FILES,
        ADAPTER_DERIVED_TOKENIZER_FILES,
        _controlled_torch_dtype,
        _load_verified_runtime_tokenizer_source,
        _runtime_tokenizer_evidence,
        _verify_base_model_lineage,
        _verify_runtime_model_binding,
        _verify_runtime_tokenizer_binding,
        _verified_private_runtime_model_snapshot,
    )


AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")
GGUF_MARKERS = {"gguf", "merged", "release", "bake", "finetune", "finetuned"}
DEFAULT_ADAPTER_FIRST_CONFIG_DIR = "generated/fine_tuning"
PREPARED_RUN_ROOT_ENV = "LUMEN_AIO_RUN_ROOT"
CONFIG_SOURCE_PATH_KEY = "_lumenExportConfigPath"
BASE_CONFIG_REQUIRED_KEYS = {
    "adapter_output_dir",
    "agent",
    "artifact_mode",
    "base_model_name",
    "baseModelID",
    "baseModelRevision",
    "baseModelIndexDigest",
    "baseModelIndexReferencedShardNames",
    "baseModelIndexShardBindingSHA256",
    "baseModelArtifactDigest",
    "baseModelWeightShards",
    "baseModelTokenizerDigest",
    "baseModelTokenizerFiles",
    "baseModelTokenizerClosureSHA256",
    "default_export_artifact",
    "max_seq_length",
    "merge_adapters_by_default",
    "output_dir",
    "release_bake_enabled_by_default",
}
PREPARED_RELEASE_BAKE_REQUIRED_KEYS = {
    "adapter_training_phase",
    "baseModelGenerationConfigFile",
    "baseModelTokenizerSnapshotPath",
    "baseModelTokenizerSnapshotVerification",
    "baseModelRuntimeSnapshotPath",
    "baseModelRuntimeSnapshotVerification",
    "bf16",
    "finalized_variant_manifest",
    "fp16",
    "trainingEnvironmentSHA256",
    "variant",
    "variantAttestation",
    "variantManifestSHA256",
}
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
        default=None,
        help=(
            "Config directory used when --config is omitted. Adapter-first mode defaults to "
            "generated/fine_tuning/<agent>/unsloth_config.json. Release-bake mode defaults to "
            "$LUMEN_AIO_RUN_ROOT/configs/<agent>.final.json and never falls back to the SFT config."
        ),
    )
    parser.add_argument(
        "--agents",
        default=None,
        help=(
            "Comma-separated agents to process. Defaults to the agents declared by explicit "
            "--config files, or all six agents in directory mode."
        ),
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
        help=(
            "Optional publication target to record in the manifest. Direct upload is "
            "unsupported here and requires --skip-upload."
        ),
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Retained for CLI compatibility; repo creation belongs to the isolated uploader.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Required when --hf-repo-id is supplied; export never receives Hub credentials.",
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


def _validate_config(
    cfg: dict[str, Any],
    *,
    path: Path,
    require_release_bake_lineage: bool = False,
) -> dict[str, Any]:
    required = set(BASE_CONFIG_REQUIRED_KEYS)
    if require_release_bake_lineage:
        required.update(PREPARED_RELEASE_BAKE_REQUIRED_KEYS)
    missing = [key for key in sorted(required) if key not in cfg]
    if missing:
        detail = f"{path} missing required keys: {', '.join(missing)}"
        if require_release_bake_lineage:
            detail += (
                ". Release bake requires the prepared <agent>.final.json config from the "
                "Ubuntu run root; checked-in generated configs intentionally omit run-scoped "
                "snapshot and finalized-adapter evidence"
            )
        raise ValueError(detail)
    if cfg["baseModelID"] != cfg["base_model_name"]:
        raise ValueError(
            f"{path} baseModelID must exactly match base_model_name"
        )
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
    if cfg.get("artifact_mode") != "adapter_first":
        raise ValueError(f"{path} must set artifact_mode=adapter_first")
    if cfg.get("default_export_artifact") != "lora_adapter":
        raise ValueError(f"{path} must set default_export_artifact=lora_adapter")
    if require_release_bake_lineage:
        phase = cfg["adapter_training_phase"]
        if phase not in {"sft", "sft_dpo"}:
            raise ValueError(
                f"{path} adapter_training_phase must be explicitly sft or sft_dpo for release bake"
            )
        if phase == "sft_dpo" and re.fullmatch(
            r"[0-9a-f]{64}", str(cfg.get("parent_sft_adapter_sha256") or "")
        ) is None:
            raise ValueError(
                f"{path} sft_dpo release bake requires parent_sft_adapter_sha256"
            )
        if phase == "sft" and cfg.get("parent_sft_adapter_sha256") is not None:
            raise ValueError(
                f"{path} sft release bake must not declare parent_sft_adapter_sha256"
            )
    return cfg


def load_config(
    path: Path,
    *,
    require_release_bake_lineage: bool = False,
) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if CONFIG_SOURCE_PATH_KEY in cfg:
        raise ValueError(f"{path} must not declare reserved key {CONFIG_SOURCE_PATH_KEY}")
    validated = _validate_config(
        cfg,
        path=path,
        require_release_bake_lineage=require_release_bake_lineage,
    )
    validated[CONFIG_SOURCE_PATH_KEY] = str(path.resolve())
    return validated


def _resolve_config_dir(
    *,
    config_paths: list[str],
    config_dir: str | None,
    release_bake: bool,
) -> str | None:
    """Resolve a deterministic mode-appropriate config source.

    Prepared snapshot evidence is run-scoped, so release bake must never discover a
    recent run or silently reuse the checked-in adapter-first configs.
    """

    if config_paths and config_dir is not None:
        raise ValueError("Use either --config or --config-dir, not both")
    if config_paths or config_dir is not None:
        return config_dir
    if not release_bake:
        return DEFAULT_ADAPTER_FIRST_CONFIG_DIR
    run_root = os.environ.get(PREPARED_RUN_ROOT_ENV, "").strip()
    if not run_root:
        raise ValueError(
            "--release-bake requires prepared final configs. Pass --config "
            "<run-root>/configs/<agent>.final.json, pass --config-dir "
            "<run-root>/configs, or set LUMEN_AIO_RUN_ROOT; the exporter will not "
            "guess a recent run or use checked-in generated configs"
        )
    return str(Path(run_root) / "configs")


def _selected_agents(
    *,
    agents_arg: str | None,
    config_paths: list[str],
) -> list[str]:
    if agents_arg is not None:
        return [item.strip().lower() for item in agents_arg.split(",") if item.strip()]
    if not config_paths:
        return list(AGENTS)

    selected: list[str] = []
    for raw in config_paths:
        path = Path(raw).resolve()
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError(f"{path} must contain a JSON object")
        agent = str(cfg.get("agent") or "").strip().lower()
        if not agent:
            raise ValueError(f"{path} must declare an agent")
        selected.append(agent)
    return selected


def gather_configs(
    config_paths: list[str],
    config_dir: str | None,
    selected_agents: list[str],
    *,
    require_release_bake_lineage: bool = False,
) -> list[dict[str, Any]]:
    if not selected_agents:
        raise ValueError("At least one agent must be selected")
    if len(selected_agents) != len(set(selected_agents)):
        raise ValueError("Selected agents must be unique")
    configs: list[dict[str, Any]] = []
    if config_paths:
        for raw in config_paths:
            configs.append(
                load_config(
                    Path(raw).resolve(),
                    require_release_bake_lineage=require_release_bake_lineage,
                )
            )
    else:
        if config_dir is None:
            raise ValueError("Config directory was not resolved")
        root = Path(config_dir).resolve()
        for agent in selected_agents:
            if require_release_bake_lineage:
                config_path = root / f"{agent}.final.json"
                if not config_path.is_file():
                    raise FileNotFoundError(
                        "Prepared final config not found for release bake: "
                        f"{config_path}. The exporter will not fall back to {agent}.json"
                    )
            else:
                nested = root / agent / "unsloth_config.json"
                config_path = nested if nested.is_file() else root / f"{agent}.json"
            configs.append(
                load_config(
                    config_path,
                    require_release_bake_lineage=require_release_bake_lineage,
                )
            )

    by_agent: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        agent = str(cfg["agent"]).strip().lower()
        if agent in by_agent:
            raise ValueError(f"Duplicate config supplied for selected agent: {agent}")
        by_agent[agent] = cfg
    missing_agents = [agent for agent in selected_agents if agent not in by_agent]
    unexpected_agents = sorted(set(by_agent).difference(selected_agents))
    if missing_agents or unexpected_agents:
        details: list[str] = []
        if missing_agents:
            details.append("missing " + ", ".join(missing_agents))
        if unexpected_agents:
            details.append("unexpected " + ", ".join(unexpected_agents))
        raise ValueError("Config coverage does not match --agents: " + "; ".join(details))
    filtered = [by_agent[agent] for agent in selected_agents]
    return filtered


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
        "baseModelTokenizerFiles",
        "baseModelTokenizerClosureSHA256",
    ):
        if finalized.get(field) != cfg.get(field):
            raise ValueError(f"Finalized variant manifest {field} does not match the GGUF config")
    if finalized.get("baseModelID") != cfg["baseModelID"]:
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
        or attestation.get("schema")
        != TRAINING_VARIANT_ATTESTATION_SCHEMA
    ):
        raise ValueError("GGUF config lacks a variant training attestation")
    for field in (
        "effectiveTrainingConfigSHA256",
        "trainingConfigInvariantSHA256",
    ):
        if re.fullmatch(
            r"[0-9a-f]{64}",
            str(attestation.get(field) or ""),
        ) is None:
            raise ValueError(
                f"GGUF variant attestation lacks a valid {field}"
            )
    if (
        attestation.get("variant") != variant
        or attestation.get("variantManifestSHA256") != source_manifest_sha
        or attestation.get("trainingEnvironmentSHA256") != training_environment_sha
        or finalized.get("trainingCorpusSHA256") != attestation.get("trainingCorpusSHA256")
        or finalized.get("trainingConfigSHA256")
        != attestation.get("effectiveTrainingConfigSHA256")
        or finalized.get("trainingConfigInvariantSHA256")
        != attestation.get("trainingConfigInvariantSHA256")
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
        expected_base_model=cfg["baseModelID"],
        expected_base_revision=cfg["baseModelRevision"],
    )

    artifact_files = adapter_manifest.get("files")
    artifact_by_path = {
        item.get("path"): item
        for item in artifact_files
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    } if isinstance(artifact_files, list) else {}
    base_by_path = {
        item.get("path"): item
        for item in cfg["baseModelTokenizerFiles"]
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    unapproved_derived = sorted(
        set(artifact_by_path).intersection(ADAPTER_DERIVED_TOKENIZER_FILES)
    )
    for filename in ADAPTER_BASE_TOKENIZER_FILES:
        artifact_file = artifact_by_path.get(filename)
        base_file = base_by_path.get(filename)
        if (
            not isinstance(artifact_file, dict)
            or not isinstance(base_file, dict)
            or artifact_file.get("sizeBytes") != base_file.get("sizeBytes")
            or artifact_file.get("sha256") != base_file.get("sha256")
        ):
            raise ValueError(
                "Adapter tokenizer files are not exact bytes from the pinned base closure"
            )
    if unapproved_derived:
        raise ValueError(
            "Adapter contains unapproved derived tokenizer files: "
            + ", ".join(unapproved_derived)
        )

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if (
        not isinstance(adapter_config, dict)
        or adapter_config.get("base_model_name_or_path") != cfg["base_model_name"]
        or adapter_config.get("revision") != cfg["baseModelRevision"]
    ):
        raise ValueError("Adapter base_model_name_or_path does not match the pinned GGUF base model")
    return {
        "adapterSHA256": adapter_manifest["adapterSHA256"],
        "adapterTrainingPhase": expected_phase,
        "finalizedVariantManifestSHA256": expected_manifest_sha,
        "sourceVariantManifestSHA256": source_manifest_sha,
        "trainingConfigSHA256": attestation[
            "effectiveTrainingConfigSHA256"
        ],
        "trainingConfigInvariantSHA256": attestation[
            "trainingConfigInvariantSHA256"
        ],
        "baseModelRevision": cfg["baseModelRevision"],
        "baseModelIndexDigest": cfg["baseModelIndexDigest"],
        "baseModelIndexReferencedShardNames": cfg["baseModelIndexReferencedShardNames"],
        "baseModelIndexShardBindingSHA256": cfg["baseModelIndexShardBindingSHA256"],
        "baseModelArtifactDigest": cfg["baseModelArtifactDigest"],
        "baseModelGenerationConfigFile": cfg["baseModelGenerationConfigFile"],
        "baseModelTokenizerDigest": cfg["baseModelTokenizerDigest"],
        "baseModelTokenizerFiles": cfg["baseModelTokenizerFiles"],
        "baseModelTokenizerClosureSHA256": cfg[
            "baseModelTokenizerClosureSHA256"
        ],
        "trainingEnvironmentSHA256": training_environment_sha,
    }


def _prepared_run_root(cfg: dict[str, Any]) -> Path:
    agent = str(cfg["agent"]).strip().lower()
    source_value = cfg.get(CONFIG_SOURCE_PATH_KEY)
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("Release bake config lacks its verified source path")
    source_path = Path(source_value).resolve()
    if (
        source_path.name != f"{agent}.final.json"
        or source_path.parent.name != "configs"
    ):
        raise ValueError(
            "Release bake requires the canonical prepared config at "
            f"<run-root>/configs/{agent}.final.json"
        )
    run_root = source_path.parent.parent.resolve()
    if source_path != run_root / "configs" / f"{agent}.final.json":
        raise ValueError("Release bake config escaped its prepared run root")
    return run_root


def _verified_release_bake_qualification(
    configs: list[dict[str, Any]],
    lineages: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    run_roots = {_prepared_run_root(cfg) for cfg in configs}
    if len(run_roots) != 1:
        raise ValueError("A release bake cannot mix prepared configs from multiple runs")
    run_root = next(iter(run_roots))
    run_manifest_path = run_root / "aio_run_manifest.json"
    if not run_manifest_path.is_file() or run_manifest_path.is_symlink():
        raise FileNotFoundError(
            f"Prepared run manifest not found for release bake: {run_manifest_path}"
        )
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest_agents = run_manifest.get("agents") if isinstance(run_manifest, dict) else None
    if not isinstance(manifest_agents, list):
        raise ValueError("Prepared run manifest lacks its agent list")
    run_agents = [
        str(item.get("agent") or "").strip().lower()
        for item in manifest_agents
        if isinstance(item, dict)
    ]
    if len(run_agents) != len(manifest_agents) or any(agent not in AGENTS for agent in run_agents):
        raise ValueError("Prepared run manifest has an invalid agent list")

    try:
        from .ubuntu_pipeline import _verified_completed_summary
    except ImportError:
        from tools.fine_tuning.unsloth.ubuntu_pipeline import (
            _verified_completed_summary,
        )

    summary = _verified_completed_summary(run_root, run_agents)
    if (
        summary.get("status") not in {"complete", "complete_without_gguf"}
        or summary.get("evaluationStatus") != "quality_gate_passed"
        or summary.get("evaluationScope") != "full"
        or summary.get("qualification") != "quality_gate_passed"
        or summary.get("promotionEligible") is not True
        or summary.get("preferenceTraining") is not True
    ):
        raise ValueError(
            "Release bake requires a verified full evaluation with "
            "quality_gate_passed qualification"
        )
    summary_agents = summary.get("agents")
    if not isinstance(summary_agents, dict):
        raise ValueError("Completed run summary lacks agent qualification evidence")
    for cfg in configs:
        agent = str(cfg["agent"]).strip().lower()
        item = summary_agents.get(agent)
        lineage = lineages.get(agent)
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("finalPhase"), dict)
            or not isinstance(lineage, dict)
            or item["finalPhase"].get("adapterSHA256")
            != lineage.get("adapterSHA256")
        ):
            raise ValueError(
                f"Completed run qualification is not bound to the selected {agent} adapter"
            )
    return {
        "sourceRunRoot": str(run_root),
        "sourceRunSummarySHA256": summary["summarySHA256"],
        "sourceRunStatus": summary["status"],
        "sourceRunEvaluationStatus": summary["evaluationStatus"],
        "sourceRunEvaluationScope": summary["evaluationScope"],
        "sourceRunQualification": summary["qualification"],
        "sourceRunPromotionEligible": summary["promotionEligible"],
    }


def _agent_output_dir(output_root: Path, agent: str) -> Path:
    return (output_root / f"{agent}_release_bake_gguf").resolve()


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
    verified_lineage: dict[str, Any] | None = None,
    qualification_evidence: dict[str, Any] | None = None,
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
    lineage = verified_lineage or _verified_release_bake_lineage(cfg)

    quantization = str(
        quantization_override
        or cfg.get("gguf_quantization")
        or cfg.get("quantization_method")
        or "q4_k_m"
    ).lower()

    agent_output_dir = _agent_output_dir(output_root, agent)
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

    (
        expected_runtime_tokenizer,
        runtime_tokenizer_snapshot_path,
        runtime_tokenizer_snapshot_verification,
    ) = _load_verified_runtime_tokenizer_source(cfg)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(runtime_tokenizer_snapshot_path),
        revision=cfg["baseModelRevision"],
        tokenizer_name=str(runtime_tokenizer_snapshot_path),
        max_seq_length=int(cfg["max_seq_length"]),
        dtype=_controlled_torch_dtype(cfg),
        load_in_4bit=True,
        local_files_only=True,
        trust_remote_code=False,
        use_exact_model_name=True,
    )
    runtime_model_binding = _verify_runtime_model_binding(
        cfg,
        runtime_model=model,
        snapshot_path=runtime_tokenizer_snapshot_path,
        snapshot_verification=runtime_tokenizer_snapshot_verification,
    )
    runtime_tokenizer_binding = _verify_runtime_tokenizer_binding(
        cfg,
        expected_tokenizer=expected_runtime_tokenizer,
        runtime_tokenizer=tokenizer,
        snapshot_path=runtime_tokenizer_snapshot_path,
        snapshot_verification=runtime_tokenizer_snapshot_verification,
    )
    runtime_tokenizer_evidence = _runtime_tokenizer_evidence(
        cfg,
        snapshot_path=runtime_tokenizer_snapshot_path,
        snapshot_verification=runtime_tokenizer_snapshot_verification,
        runtime_model_binding=runtime_model_binding,
        runtime_binding=runtime_tokenizer_binding,
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
        **(qualification_evidence or {}),
        **runtime_tokenizer_evidence,
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
    verified_lineage: dict[str, Any] | None = None,
    qualification_evidence: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    agent = str(cfg["agent"]).strip().lower()
    quantization = str(
        quantization_override
        or cfg.get("gguf_quantization")
        or cfg.get("quantization_method")
        or "q4_k_m"
    ).lower()
    agent_output_dir = _agent_output_dir(output_root, agent)
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
    lineage = verified_lineage or _verified_release_bake_lineage(cfg)
    report_path = agent_output_dir / "gguf_release_bake_report.json"
    if not report_path.is_file():
        raise ValueError(f"Cannot reuse GGUF without its lineage report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"GGUF lineage report must be a JSON object: {report_path}")
    runtime_snapshot_path, runtime_snapshot_verification = (
        _verified_private_runtime_model_snapshot(cfg)
    )
    runtime_evidence = _runtime_tokenizer_evidence(
        cfg,
        snapshot_path=runtime_snapshot_path,
        snapshot_verification=runtime_snapshot_verification,
        runtime_model_binding=report.get("runtimeModelBinding") or {},
        runtime_binding=report.get("runtimeTokenizerBinding") or {},
    )
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
        **(qualification_evidence or {}),
        **runtime_evidence,
        **lineage,
    }
    if report != expected:
        drifted = sorted(
            key
            for key in set(report) | set(expected)
            if report.get(key) != expected.get(key)
        )
        raise ValueError(
            "Existing GGUF lineage report does not match current "
            + ", ".join(drifted)
        )
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
    selected_agents = _selected_agents(
        agents_arg=args.agents,
        config_paths=args.config,
    )
    for agent in selected_agents:
        if agent not in AGENTS:
            raise ValueError(f"Unsupported agent in --agents: {agent}")

    config_dir = _resolve_config_dir(
        config_paths=args.config,
        config_dir=args.config_dir,
        release_bake=args.release_bake,
    )
    configs = gather_configs(
        args.config,
        config_dir,
        selected_agents,
        require_release_bake_lineage=args.release_bake,
    )
    if not args.release_bake:
        manifest_path = _write_manifest(args.manifest_output, _release_bake_skipped_manifest(configs, args))
        _emit(f"Skipped GGUF release bake by default. Wrote adapter-first manifest: {manifest_path}")
        _emit("Pass --release-bake to explicitly merge adapters into GGUF artifacts.")
        return

    if args.hf_repo_id and not args.skip_upload:
        raise ValueError(
            "Direct Hugging Face upload is unsupported in the GGUF exporter. "
            "Use --skip-upload, verify the completed manifest, then publish with "
            "the isolated credential-scoped Ubuntu uploader"
        )

    verified_lineages = {
        str(cfg["agent"]).strip().lower(): _verified_release_bake_lineage(cfg)
        for cfg in configs
    }
    qualification_evidence = _verified_release_bake_qualification(
        configs,
        verified_lineages,
    )
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "mode": "optional_release_bake",
        "release_bake_requested": True,
        "repo_id": args.hf_repo_id,
        "quantization_override": args.quantization,
        **qualification_evidence,
        "agents": {},
    }

    for cfg in configs:
        summary = None
        if args.skip_existing:
            summary = existing_summary_for_agent(
                cfg,
                output_root=output_root,
                quantization_override=args.quantization,
                verified_lineage=verified_lineages[
                    str(cfg["agent"]).strip().lower()
                ],
                qualification_evidence=qualification_evidence,
            )
        if summary is None:
            summary = export_agent_gguf(
                cfg,
                output_root=output_root,
                quantization_override=args.quantization,
                max_memory_usage_override=args.max_memory_usage,
                verified_lineage=verified_lineages[
                    str(cfg["agent"]).strip().lower()
                ],
                qualification_evidence=qualification_evidence,
            )
        agent = summary["agent"]
        manifest["agents"][agent] = summary

    manifest_path = _write_manifest(args.manifest_output, manifest)
    _emit(f"Wrote GGUF release-bake manifest: {manifest_path}")


if __name__ == "__main__":
    main()
