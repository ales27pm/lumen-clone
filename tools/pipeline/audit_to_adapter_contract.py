#!/usr/bin/env python3
"""Canonical audit -> dataset -> adapter -> app runtime contract for Lumen.

This module is intentionally dependency-light and importable from scripts, CI, and
static drift guards. It is the single place where the Qwen3 adapter-first artifact
shape is described in machine-readable form.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import glob
import json
import re


LIVE_RUNTIME_SLOTS: tuple[str, ...] = ("cortex", "executor", "mouth", "mimicry", "rem")
TRAINED_ADAPTER_ROLES: tuple[str, ...] = (*LIVE_RUNTIME_SLOTS, "fleet")
EMBEDDING_SLOT = "embedding"

SHARED_BASE_REPO_ID = "ales27pm/lumen-qwen3-bootstrap-gguf"
SHARED_BASE_FILE_NAME = "lumen-qwen3-fast-shared-q4_k_m.gguf"
SHARED_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"

EMBEDDING_REPO_ID = "Qwen/Qwen3-Embedding-0.6B-GGUF"
EMBEDDING_FILE_NAME = "Qwen3-Embedding-0.6B-Q8_0.gguf"

ADAPTER_REPO_ID = "ales27pm/lumen-qwen3-bootstrap-adapters-gguf"
ADAPTER_FILE_TEMPLATE = "lumen-{role}-lora.gguf"

QWEN3_CONFIG_DIR = Path("tools/fine_tuning/unsloth/configs_qwen3_bootstrap")
FINE_TUNING_OUTPUT_DIR = Path("generated/fine_tuning")
TRAINED_ADAPTER_DIR = Path("models/lora_qwen3_bootstrap")
ADAPTER_GGUF_DIR = Path("models/lora_qwen3_gguf")
SHARED_BASE_GGUF_DIR = Path("models/base_qwen3_fast")
PIPELINE_STATE_FILE = Path("generated/agent_improvement_loop/pipeline_state.json")
LOOP_OUTPUT_DIR = Path("generated/agent_improvement_loop")
MANIFEST_OUTPUT_DIR = Path("generated/agent_manifest")

RUNTIME_AUDIT_GLOBS: tuple[str, ...] = (
    "exports/*.json",
    "runtime-audits/**/*.json",
    "generated/runtime_audits/*.json",
    "generated/runtime_audit/*.json",
    "generated/testflight_exports/*.json",
    "generated/agent_improvement_loop/runtime_audits/*.json",
    "Diagnostics/LumenDatasetExports/*.json",
)
RUNTIME_AUDIT_TIMESTAMP_RE = re.compile(r"(?P<prefix>.*?)(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)(?:[-_].*)?$")
RUNTIME_AUDIT_LEADING_INDEX_RE = re.compile(r"^\d+-")
RUNTIME_AUDIT_TRAILING_INDEX_RE = re.compile(r"-\d+$")

IN_APP_DATASET_PACKAGE_SCHEMA_VERSIONS: tuple[str, ...] = ("1.0.0", "1.1.0", "1.2.0")
IN_APP_DATASET_EXPORT_FORMAT = "agent-grounding-runtime-json-package"
IN_APP_DATASET_SOURCE_LAYER = "agentGroundingRuntimeAudit"


@dataclass(frozen=True)
class ArtifactSpec:
    role: str
    repo_id: str
    file_name: str
    kind: str
    local_path: str
    runtime_slot: str | None = None
    base_model_id: str | None = None
    required_for_default_runtime: bool = True


@dataclass(frozen=True)
class PipelineStageSpec:
    id: str
    owner: str
    command_hint: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    required_before_app_validation: bool


@dataclass(frozen=True)
class AuditToAdapterPipelineContract:
    schema_version: str
    family: str
    mode: str
    live_runtime_slots: tuple[str, ...]
    trained_adapter_roles: tuple[str, ...]
    runtime_audit_globs: tuple[str, ...]
    artifacts: tuple[ArtifactSpec, ...]
    stages: tuple[PipelineStageSpec, ...]
    invariants: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def artifact_by_role(self, role: str) -> ArtifactSpec | None:
        return next((artifact for artifact in self.artifacts if artifact.role == role), None)

    def adapter_file_name(self, role: str) -> str:
        return adapter_file_name(role)


def adapter_file_name(role: str) -> str:
    if role not in TRAINED_ADAPTER_ROLES:
        raise ValueError(f"Unknown adapter role: {role}")
    return ADAPTER_FILE_TEMPLATE.format(role=role)


def expected_adapter_file_names() -> tuple[str, ...]:
    return tuple(adapter_file_name(role) for role in TRAINED_ADAPTER_ROLES)


def training_config_path(role: str) -> Path:
    return QWEN3_CONFIG_DIR / f"{role}.json"


def trained_adapter_path(role: str) -> Path:
    return TRAINED_ADAPTER_DIR / role


def adapter_gguf_path(role: str) -> Path:
    return ADAPTER_GGUF_DIR / ADAPTER_FILE_TEMPLATE.format(role=role)


def _resolve_user_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def _audit_stem(path: Path) -> str:
    return RUNTIME_AUDIT_LEADING_INDEX_RE.sub("", path.stem)


def _audit_family_key(path: Path) -> str:
    stem = _audit_stem(path)
    match = RUNTIME_AUDIT_TIMESTAMP_RE.match(stem)
    if match:
        return match.group("prefix").rstrip("-_") or stem
    return RUNTIME_AUDIT_TRAILING_INDEX_RE.sub("", stem)


def _audit_recency_key(path: Path) -> tuple[int, str, int]:
    stem = _audit_stem(path)
    match = RUNTIME_AUDIT_TIMESTAMP_RE.match(stem)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    if match:
        return (1, match.group("timestamp"), mtime_ns)
    return (0, "", mtime_ns)


def latest_runtime_audit_paths(paths: Iterable[Path]) -> list[Path]:
    latest_by_family: dict[str, Path] = {}
    for path in sorted({candidate.resolve() for candidate in paths if candidate.is_file()}):
        family = _audit_family_key(path)
        current = latest_by_family.get(family)
        if current is None or _audit_recency_key(path) > _audit_recency_key(current):
            latest_by_family[family] = path
    return sorted(latest_by_family.values())


def all_runtime_audit_candidates(root: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in RUNTIME_AUDIT_GLOBS:
        found.extend(root.glob(pattern))
    return sorted({path.resolve() for path in found if path.is_file()})


def runtime_audit_candidates(root: Path) -> list[Path]:
    return latest_runtime_audit_paths(all_runtime_audit_candidates(root))


def expand_runtime_audit_paths(root: Path, explicit: Iterable[str | Path]) -> list[Path]:
    if not explicit:
        return runtime_audit_candidates(root)
    found: list[Path] = []
    for raw in explicit:
        pattern = str(_resolve_user_path(root, raw))
        matches = [Path(path).resolve() for path in glob.glob(pattern, recursive=True)]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern).resolve()]
        found.extend(path for path in matches if path.is_file())
    return latest_runtime_audit_paths(found)


CONTRACT = AuditToAdapterPipelineContract(
    schema_version="lumen.audit_to_adapter_pipeline/1.1.0",
    family="qwen3",
    mode="adapter-first",
    live_runtime_slots=LIVE_RUNTIME_SLOTS,
    trained_adapter_roles=TRAINED_ADAPTER_ROLES,
    runtime_audit_globs=RUNTIME_AUDIT_GLOBS,
    artifacts=(
        ArtifactSpec(
            role="shared_chat_base",
            repo_id=SHARED_BASE_REPO_ID,
            file_name=SHARED_BASE_FILE_NAME,
            kind="chat",
            local_path=str(SHARED_BASE_GGUF_DIR / SHARED_BASE_FILE_NAME),
            runtime_slot="shared",
            base_model_id=SHARED_BASE_MODEL_ID,
        ),
        ArtifactSpec(
            role=EMBEDDING_SLOT,
            repo_id=EMBEDDING_REPO_ID,
            file_name=EMBEDDING_FILE_NAME,
            kind="embedding",
            local_path=str(Path("models/embedding_qwen3") / EMBEDDING_FILE_NAME),
            runtime_slot=EMBEDDING_SLOT,
        ),
        *(
            ArtifactSpec(
                role=role,
                repo_id=ADAPTER_REPO_ID,
                file_name=ADAPTER_FILE_TEMPLATE.format(role=role),
                kind="roleAdapter",
                local_path=str(ADAPTER_GGUF_DIR / ADAPTER_FILE_TEMPLATE.format(role=role)),
                runtime_slot=role if role in LIVE_RUNTIME_SLOTS else None,
                base_model_id=SHARED_BASE_MODEL_ID,
                required_for_default_runtime=role in LIVE_RUNTIME_SLOTS,
            )
            for role in TRAINED_ADAPTER_ROLES
        ),
    ),
    stages=(
        PipelineStageSpec(
            id="runtime_audit_ingest",
            owner="ios/Lumen/Services/AgentGrounding/InAppDatasetPackageExporter.swift + tools/lumen_manifest_crawler/dataset/runtime_ingest.py",
            command_hint="python -m lumen_manifest_crawler improve-loop --runtime-audit <audit.json> --generate-agent-fine-tuning",
            inputs=("ios/Lumen", *RUNTIME_AUDIT_GLOBS),
            outputs=(str(LOOP_OUTPUT_DIR / "loop_state.json"), str(MANIFEST_OUTPUT_DIR / "dataset_manifest.json"), str(FINE_TUNING_OUTPUT_DIR)),
            required_before_app_validation=True,
        ),
        PipelineStageSpec(
            id="adapter_sft_training",
            owner="tools/fine_tuning/unsloth/train_sft.py",
            command_hint="python tools/fine_tuning/unsloth/train_sft.py --config tools/fine_tuning/unsloth/configs_qwen3_bootstrap/<role>.json --assistant-only-loss",
            inputs=(str(QWEN3_CONFIG_DIR), str(FINE_TUNING_OUTPUT_DIR)),
            outputs=tuple(str(trained_adapter_path(role)) for role in TRAINED_ADAPTER_ROLES),
            required_before_app_validation=True,
        ),
        PipelineStageSpec(
            id="adapter_lora_to_gguf",
            owner="~/.unsloth/llama.cpp/convert_lora_to_gguf.py",
            command_hint="python ~/.unsloth/llama.cpp/convert_lora_to_gguf.py models/lora_qwen3_bootstrap/<role> --outfile models/lora_qwen3_gguf/lumen-<role>-lora.gguf --base-model-id Qwen/Qwen3-1.7B",
            inputs=tuple(str(trained_adapter_path(role)) for role in TRAINED_ADAPTER_ROLES),
            outputs=tuple(str(adapter_gguf_path(role)) for role in TRAINED_ADAPTER_ROLES),
            required_before_app_validation=True,
        ),
        PipelineStageSpec(
            id="hf_upload_adapters",
            owner="hf CLI",
            command_hint="hf upload ales27pm/lumen-qwen3-bootstrap-adapters-gguf models/lora_qwen3_gguf . --repo-type model",
            inputs=(str(ADAPTER_GGUF_DIR),),
            outputs=(f"hf://{ADAPTER_REPO_ID}",),
            required_before_app_validation=True,
        ),
        PipelineStageSpec(
            id="hf_upload_shared_base",
            owner="hf CLI",
            command_hint="hf upload-large-folder ales27pm/lumen-qwen3-bootstrap-gguf models/base_qwen3_fast --repo-type model",
            inputs=(str(SHARED_BASE_GGUF_DIR),),
            outputs=(f"hf://{SHARED_BASE_REPO_ID}",),
            required_before_app_validation=True,
        ),
    ),
    invariants=(
        "All live runtime slots use Qwen3 GGUF with adapter-first inference.",
        "The app adapter runtime contract must match generated GGUF adapter file names.",
        "Training configs, local adapter export paths, HF adapter repo paths, and app runtime contract entries must drift together through this contract.",
        "Runtime audits are the required feedback source for generated training examples and regression scenarios.",
    ),
)


def validate_repository_alignment(
    root: Path,
    *,
    require_generated_artifacts: bool = False,
    require_training_datasets: bool = False,
) -> list[str]:
    errors: list[str] = []
    runtime_contract = root / "ios/Lumen/Services/ModelAdapterRuntimeContract.swift"
    bootstrap = root / "ios/Lumen/LumenApp.swift"
    required_files = [runtime_contract, bootstrap]
    texts: dict[Path, str] = {}
    for path in required_files:
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(root)}")
            continue
        texts[path] = path.read_text(encoding="utf-8")

    if len(texts) == len(required_files):
        runtime_contract_text = texts[runtime_contract]
        bootstrap_text = texts[bootstrap]
        for artifact in CONTRACT.artifacts:
            if artifact.role == "shared_chat_base":
                needles = [artifact.file_name, artifact.repo_id, artifact.base_model_id or ""]
            elif artifact.role == EMBEDDING_SLOT:
                needles = [artifact.file_name, artifact.repo_id]
            else:
                needles = [artifact.file_name, artifact.repo_id]
            for needle in filter(None, needles):
                if needle not in runtime_contract_text and needle not in bootstrap_text:
                    errors.append(f"artifact reference missing from app runtime contract/bootstrap: {artifact.role} -> {needle}")
        if "LumenModelSlotContract.validateCompletenessAtStartup" not in bootstrap_text:
            errors.append("LumenApp.swift no longer validates LumenModelSlotContract at startup")
        if "LumenTrainedModelRuntimeRegistry" not in runtime_contract_text:
            errors.append("ModelAdapterRuntimeContract.swift no longer defines LumenTrainedModelRuntimeRegistry")

    for role in TRAINED_ADAPTER_ROLES:
        cfg = root / training_config_path(role)
        if not cfg.exists():
            errors.append(f"missing training config for {role}: {cfg.relative_to(root)}")
            continue
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON in {cfg.relative_to(root)}: {exc}")
            continue
        output_dir = Path(str(data.get("output_dir", "")))
        if output_dir != trained_adapter_path(role):
            errors.append(f"{cfg.relative_to(root)} output_dir={output_dir} expected={trained_adapter_path(role)}")
        datasets = data.get("datasets") or []
        if require_training_datasets and not datasets:
            errors.append(f"{cfg.relative_to(root)} has no datasets")

    if require_generated_artifacts:
        shared_base = root / SHARED_BASE_GGUF_DIR / SHARED_BASE_FILE_NAME
        if not shared_base.is_file():
            errors.append(f"missing shared base GGUF file: {shared_base.relative_to(root)}")
        for role in TRAINED_ADAPTER_ROLES:
            trained_path = root / trained_adapter_path(role)
            gguf_path = root / adapter_gguf_path(role)
            if not trained_path.is_dir():
                errors.append(f"missing trained adapter directory for {role}: {trained_path.relative_to(root)}")
            if not gguf_path.is_file():
                errors.append(f"missing GGUF adapter file for {role}: {gguf_path.relative_to(root)}")
    return errors


def write_contract_json(path: Path) -> None:
    """Write the canonical pipeline contract as deterministic UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            CONTRACT.as_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate Lumen audit-to-adapter pipeline contract alignment.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--require-generated-artifacts", action="store_true")
    parser.add_argument("--require-training-datasets", action="store_true")
    args = parser.parse_args()
    issues = validate_repository_alignment(
        args.root.resolve(),
        require_generated_artifacts=args.require_generated_artifacts,
        require_training_datasets=args.require_training_datasets,
    )
    if issues:
        print("Pipeline contract validation failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"PASS {CONTRACT.schema_version} ({CONTRACT.family}/{CONTRACT.mode})")
