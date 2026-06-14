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


def runtime_audit_candidates(root: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in RUNTIME_AUDIT_GLOBS:
        found.extend(root.glob(pattern))
    return sorted({path.resolve() for path in found if path.is_file()})


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
    return sorted({path for path in found})


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
            outputs=(f"hf://{SHARED_BASE_REPO_ID}/{SHARED_BASE_FILE_NAME}",),
            required_before_app_validation=True,
        ),
        PipelineStageSpec(
            id="ios_install_and_runtime_resolution",
            owner="ios/Lumen",
            command_hint="Models -> Download / repair Qwen3; then runtime audit must show runtimePath=sharedAdapter and adapterApplied=true for role turns",
            inputs=(f"hf://{SHARED_BASE_REPO_ID}/{SHARED_BASE_FILE_NAME}", f"hf://{ADAPTER_REPO_ID}/*.gguf", f"hf://{EMBEDDING_REPO_ID}/{EMBEDDING_FILE_NAME}"),
            outputs=("SwiftData StoredModel rows", "SlotModelRuntimeCoordinator assignments", "runtime audit JSON"),
            required_before_app_validation=True,
        ),
    ),
    invariants=(
        "Qwen3 default runtime loads one shared chat base and switches exactly one LoRA adapter per live role slot.",
        "Runtime audit JSONs are first-class training inputs and must carry adapterApplied/adapterSlot/adapterFailureReason when available.",
        "In-app dataset packages must include exportPolicy.format=agent-grounding-runtime-json-package and sourceLayer=agentGroundingRuntimeAudit.",
        "Training produces PEFT adapter directories under models/lora_qwen3_bootstrap/<role>; GGUF conversion is a separate explicit stage.",
        "Generated adapter export plans must point to models/lora_qwen3_bootstrap and models/lora_qwen3_gguf, not stale models/lora paths.",
        "Role adapters are stored as ModelRole.roleAdapter in iOS and are never directly activatable as chat or embedding models.",
        "Release-baked full GGUFs are manual fallback artifacts only and must not appear in the default Qwen3 app catalog.",
        "Fleet is a trained/downloadable role adapter artifact but not a live runtime slot until a dedicated slot contract is added.",
    ),
)


def _read(root: Path, relative: str) -> str:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(relative)
    return path.read_text(encoding="utf-8")


def _require_contains(errors: list[str], text: str, needle: str, label: str) -> None:
    if needle not in text:
        errors.append(f"{label}: missing {needle!r}")


def validate_repository_alignment(root: Path, *, require_generated_artifacts: bool = False) -> list[str]:
    """Return alignment errors. Empty list means the repo matches the contract."""
    errors: list[str] = []

    required_files = {
        "iOS runtime contract": "ios/Lumen/Services/ModelAdapterRuntimeContract.swift",
        "iOS model family catalog": "ios/Lumen/Services/ModelFamilySelection.swift",
        "iOS fleet resolver": "ios/Lumen/Services/ModelFleet.swift",
        "iOS llama runtime": "ios/Lumen/Services/LlamaService.swift",
        "iOS model bootstrap": "ios/Lumen/Services/ModelLaunchBootstrap.swift",
        "in-app dataset exporter": "ios/Lumen/Services/AgentGrounding/InAppDatasetPackageExporter.swift",
        "runtime ingest normalizer": "tools/lumen_manifest_crawler/lumen_manifest_crawler/dataset/runtime_ingest.py",
        "adapter export planner": "tools/lumen_manifest_crawler/lumen_manifest_crawler/dataset/adapter_export.py",
        "terminal pipeline": "tools/lumen_terminal_improve_loop.py",
        "training script": "tools/fine_tuning/unsloth/train_sft.py",
    }
    texts: dict[str, str] = {}
    for label, relative in required_files.items():
        try:
            texts[relative] = _read(root, relative)
        except FileNotFoundError:
            errors.append(f"{label}: missing {relative}")

    if len(texts) == len(required_files):
        contract_text = texts["ios/Lumen/Services/ModelAdapterRuntimeContract.swift"]
        for needle in (
            f'sharedBaseModelID: "{SHARED_BASE_MODEL_ID}"',
            f'sharedBaseRepoID: "{SHARED_BASE_REPO_ID}"',
            f'sharedBaseFileName: "{SHARED_BASE_FILE_NAME}"',
            f'embeddingRepoID: "{EMBEDDING_REPO_ID}"',
            f'embeddingFileName: "{EMBEDDING_FILE_NAME}"',
            f'adapterRepoID: "{ADAPTER_REPO_ID}"',
            "loadBaseModelOnce: true",
            "selectAdapterByAgentSlot: true",
            "mergeAdaptersByDefault: false",
            "releaseBakeEnabledByDefault: false",
            "releaseBakeManualOnly: true",
        ):
            _require_contains(errors, contract_text, needle, "ModelAdapterRuntimeContract")
        for file_name in expected_adapter_file_names():
            _require_contains(errors, contract_text, file_name, "ModelAdapterRuntimeContract adapters")

        catalog_text = texts["ios/Lumen/Services/ModelFamilySelection.swift"]
        for needle in (
            "static let defaultFamily: LumenModelFamily = .qwen3",
            "qwen3BootstrapModels",
            "contract.sharedBaseRepoID",
            "contract.embeddingRepoID",
            "contract.adapterRoles.map",
            "role: .roleAdapter",
        ):
            _require_contains(errors, catalog_text, needle, "ModelFamilySelection")

        fleet_text = texts["ios/Lumen/Services/ModelFleet.swift"]
        for needle in (
            "qwen3AdapterBase",
            "preferredAdapter(for: slot",
            "missingAdapterSlots",
            "mode == .qwen3AdapterRuntime",
            "artifactKind == .chat && adapterPath != nil",
            "hasAdapterMarker",
        ):
            _require_contains(errors, fleet_text, needle, "ModelFleet")

        llama_text = texts["ios/Lumen/Services/LlamaService.swift"]
        for needle in (
            "private actor AdapterChatRuntime",
            "private var sharedChatRuntime: AdapterChatRuntime?",
            "private var roleAdapters: [LumenModelSlot: LoadedRoleAdapter]",
            "context.removeAllLoraAdapters()",
            "context.apply(loraAdapter: adapter, scale: scale)",
            "adapterApplied",
            "adapterFailureReason",
            "runtimePath",
        ):
            _require_contains(errors, llama_text, needle, "LlamaService")

        bootstrap_text = texts["ios/Lumen/Services/ModelLaunchBootstrap.swift"]
        for needle in (
            "fleetModelsForInstall(family:",
            "LumenModelFleetCatalog.bootstrapModels(for: family)",
            "case .roleAdapter:",
            "ModelLoader.ensureChatLoaded",
        ):
            _require_contains(errors, bootstrap_text, needle, "ModelLaunchBootstrap")

        exporter_text = texts["ios/Lumen/Services/AgentGrounding/InAppDatasetPackageExporter.swift"]
        for needle in (
            'static let schemaVersion = "1.2.0"',
            "agent-grounding-runtime-json-package",
            "agentGroundingRuntimeAudit",
            "recentTraces",
            "ImproveLoopSampleGate.buildDataset",
            "accepted_training",
            "quarantined_samples",
            "regression_tests",
        ):
            _require_contains(errors, exporter_text, needle, "InAppDatasetPackageExporter")

        runtime_ingest_text = texts["tools/lumen_manifest_crawler/lumen_manifest_crawler/dataset/runtime_ingest.py"]
        for needle in (
            "_is_in_app_package",
            "_flatten_in_app_package",
            "agent-grounding-runtime-json-package",
            "recentTraces",
            "traceSelectedToolAllowedCount",
            "traceParseErrorCount",
        ):
            _require_contains(errors, runtime_ingest_text, needle, "runtime_ingest")

        adapter_export_text = texts["tools/lumen_manifest_crawler/lumen_manifest_crawler/dataset/adapter_export.py"]
        for needle in (
            'DEFAULT_LORA_OUTPUT_ROOT = "models/lora_qwen3_bootstrap"',
            'DEFAULT_ADAPTER_GGUF_OUTPUT_ROOT = "models/lora_qwen3_gguf"',
            f'DEFAULT_ADAPTER_REPO_ID = "{ADAPTER_REPO_ID}"',
            f'DEFAULT_SHARED_BASE_REPO_ID = "{SHARED_BASE_REPO_ID}"',
            f'DEFAULT_SHARED_BASE_FILE_NAME = "{SHARED_BASE_FILE_NAME}"',
            "adapterGGUFArtifact",
        ):
            _require_contains(errors, adapter_export_text, needle, "adapter_export")

        terminal_text = texts["tools/lumen_terminal_improve_loop.py"]
        for needle in (
            "discover_runtime_jsons",
            "--generate-agent-fine-tuning",
            "tools/fine_tuning/unsloth/train_sft.py",
            "convert_lora_to_gguf.py",
            "hf", "upload",
            "pipeline_state.json",
        ):
            _require_contains(errors, terminal_text, needle, "terminal improve loop")

        train_text = texts["tools/fine_tuning/unsloth/train_sft.py"]
        for needle in (
            "save_pretrained(str(output_dir))",
            "train_manifest.json",
            "--assistant-only-loss",
            "--resume-from-checkpoint",
            "--seed",
        ):
            _require_contains(errors, train_text, needle, "train_sft")

    config_dir = root / QWEN3_CONFIG_DIR
    if not config_dir.exists():
        errors.append(f"missing Qwen3 config dir: {QWEN3_CONFIG_DIR}")
    else:
        for role in TRAINED_ADAPTER_ROLES:
            cfg_path = config_dir / f"{role}.json"
            if not cfg_path.exists():
                errors.append(f"missing Qwen3 training config: {cfg_path.relative_to(root)}")
                continue
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"invalid JSON in {cfg_path.relative_to(root)}: {exc}")
                continue
            base = str(cfg.get("base_model_name", "")).lower()
            if "qwen3" not in base or "qwen2" in base or "qwen-2" in base:
                errors.append(f"{cfg_path.relative_to(root)} base_model_name must reference Qwen3 only: {cfg.get('base_model_name')!r}")
            expected_output = str(trained_adapter_path(role))
            if str(cfg.get("output_dir")) != expected_output:
                errors.append(f"{cfg_path.relative_to(root)} output_dir must be {expected_output!r}; got {cfg.get('output_dir')!r}")
            if cfg.get("merge_adapters_by_default", False) is not False:
                errors.append(f"{cfg_path.relative_to(root)} merge_adapters_by_default must be false")
            if cfg.get("release_bake_enabled_by_default", False) is not False:
                errors.append(f"{cfg_path.relative_to(root)} release_bake_enabled_by_default must be false")

    if require_generated_artifacts:
        for role in TRAINED_ADAPTER_ROLES:
            trained_path = root / trained_adapter_path(role)
            gguf_path = root / adapter_gguf_path(role)
            if not trained_path.is_dir():
                errors.append(f"missing trained adapter directory: {trained_adapter_path(role)}")
            if not gguf_path.is_file():
                errors.append(f"missing adapter GGUF: {adapter_gguf_path(role)}")
        shared_base_path = root / SHARED_BASE_GGUF_DIR / SHARED_BASE_FILE_NAME
        if not shared_base_path.is_file():
            errors.append(f"missing shared base GGUF: {SHARED_BASE_GGUF_DIR / SHARED_BASE_FILE_NAME}")

    return errors


def write_contract_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(CONTRACT.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
