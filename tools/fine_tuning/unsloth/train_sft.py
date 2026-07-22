from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from lumen_manifest_crawler.dataset.chat_template_contract import (
        apply_non_thinking_chat_template,
        verify_chat_template_contract,
    )
except ImportError:
    from tools.lumen_manifest_crawler.lumen_manifest_crawler.dataset.chat_template_contract import (
        apply_non_thinking_chat_template,
        verify_chat_template_contract,
    )

try:
    from .adapter_artifact import (
        portable_adapter_model_card,
        write_adapter_artifact_manifest,
        write_portable_adapter_model_card,
    )
    from .training_lineage import (
        build_resolved_training_environment,
        build_resolved_training_environment_snapshot,
        canonical_python_code_sha256,
        canonical_controlled_package_version,
        installed_distribution_python_callable_identity,
        installed_controlled_package_versions,
        repository_training_code_bundle,
        RUN_RESUME_LINEAGE_SCHEMA,
        TRAINING_VARIANT_ATTESTATION_SCHEMA,
        validate_runtime_source_audit,
        verify_private_base_model_conversion_snapshot,
        verify_private_base_model_tokenizer_snapshot,
        verify_resolved_training_environment,
        verify_resolved_training_environment_cache,
        verify_training_code_manifest,
        verify_training_dependency_lock,
        ZERO_GPU_ALLOWED_SIZES,
    )
except ImportError:
    from adapter_artifact import (
        portable_adapter_model_card,
        write_adapter_artifact_manifest,
        write_portable_adapter_model_card,
    )
    from training_lineage import (
        build_resolved_training_environment,
        build_resolved_training_environment_snapshot,
        canonical_python_code_sha256,
        canonical_controlled_package_version,
        installed_distribution_python_callable_identity,
        installed_controlled_package_versions,
        repository_training_code_bundle,
        RUN_RESUME_LINEAGE_SCHEMA,
        TRAINING_VARIANT_ATTESTATION_SCHEMA,
        validate_runtime_source_audit,
        verify_private_base_model_conversion_snapshot,
        verify_private_base_model_tokenizer_snapshot,
        verify_resolved_training_environment,
        verify_resolved_training_environment_cache,
        verify_training_code_manifest,
        verify_training_dependency_lock,
        ZERO_GPU_ALLOWED_SIZES,
    )


REQUIRED_CONFIG_KEYS = {
    "agent",
    "base_model_name",
    "baseModelID",
    "baseModelRevision",
    "baseModelIndexDigest",
    "baseModelIndexReferencedShardNames",
    "baseModelIndexShardBindingSHA256",
    "baseModelArtifactDigest",
    "baseModelWeightShards",
    "baseModelGenerationConfigFile",
    "baseModelTokenizerDigest",
    "baseModelTokenizerFiles",
    "baseModelTokenizerClosureSHA256",
    "baseModelTokenizerSnapshotPath",
    "baseModelTokenizerSnapshotVerification",
    "baseModelRuntimeSnapshotPath",
    "baseModelRuntimeSnapshotVerification",
    "chatTemplateContract",
    "trainingEnvironmentLock",
    "trainingContainerImageDigest",
    "trainingContainerImageDigestSource",
    "trainingRuntimeImageBindingStatus",
    "trainingRuntimeImageBindingVerified",
    "trainingEnvironmentSHA256",
    "trainingCodeManifest",
    "trainingCodeSHA256",
    "trainingDependencyLock",
    "trainingDependencyLockSHA256",
    "requirementsSHA256",
    "zeroGPUSize",
    "zeroGPUDurationSeconds",
    "observedAccelerator",
    "runtimeSourceKind",
    "runtimeSourceRevision",
    "expectedRuntimeSourceRevision",
    "observedRepositoryRevision",
    "observedRuntimeRevision",
    "runtimeSourceBindingStatus",
    "runtimeSourceBindingMethod",
    "max_seq_length",
    "load_in_4bit",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "learning_rate",
    "batch_size",
    "gradient_accumulation_steps",
    "num_train_epochs",
    "warmup_steps",
    "bf16",
    "fp16",
    "output_dir",
    "adapter_output_dir",
    "dataset_dir",
    "variant",
    "variantManifestSHA256",
    "publicCorpusLossShareContract",
    "seed",
}
AGENTS = {"cortex", "executor", "mouth", "mimicry", "rem", "fleet"}
FINETUNE_MARKERS = {"sft", "dpo", "orpo", "lora", "merged", "adapter", "finetune", "finetuned", "training"}
CHECKPOINT_LINEAGE_SCHEMA = "lumen.zerogpu.checkpoint_lineage/1.0.0"
CHECKPOINT_DIRECTORY_SCHEMA = "lumen.zerogpu.checkpoint_directory/1.0.0"
RUN_RESUME_LINEAGE_FIELDS = frozenset(
    {
        "schema",
        "runID",
        "datasetRepository",
        "datasetRevision",
        "datasetPath",
        "localDatasetSnapshot",
        "selectedAgents",
        "experimentVariant",
        "seed",
        "assistantOnlyLoss",
        "trainingCodeSHA256",
        "trainingDependencyLockSHA256",
        "requirementsSHA256",
        "resolvedTrainingEnvironment",
        "resolvedTrainingEnvironmentSHA256",
        "zeroGPUSize",
        "zeroGPUDurationSeconds",
        "observedAccelerator",
        "spaceConfigurationSHA256",
        "runtimeSourceKind",
        "runtimeSourceRevision",
        "expectedRuntimeSourceRevision",
        "observedRepositoryRevision",
        "observedRuntimeRevision",
        "runtimeSourceBindingStatus",
        "runtimeSourceBindingMethod",
        "agents",
        "runResumeLineageSHA256",
    }
)
RUN_RESUME_AGENT_LINEAGE_FIELDS = frozenset(
    {
        "agent",
        "sourceVariantManifestSHA256",
        "laneHashes",
        "datasetFileSHA256",
        "trainingCorpusSHA256",
        "controlledTrainingConfigSHA256",
        "trainingConfigInvariantSHA256",
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
        "baseModelTokenizerSnapshotPath",
        "baseModelGenerationConfigFile",
        "baseModelRuntimeSnapshotPath",
        "seed",
        "trainingEnvironmentLockSHA256",
        "configPath",
        "checkpointLineagePath",
        "checkpointRoot",
        "outputDirectory",
        "adapterOutputDirectory",
    }
)
RUN_RESUME_DATASET_FILES = frozenset(
    {"train_sft.jsonl", "val_sft.jsonl", "train_dpo.jsonl", "val_dpo.jsonl"}
)
SFT_CHECKPOINT_LINEAGE_SCHEMA = "lumen.sft_checkpoint_lineage/1.2.0"
SFT_CHECKPOINT_DIRECTORY_SCHEMA = "lumen.sft_checkpoint_directory/1.1.0"
SFT_CHECKPOINT_SAVE_STEPS = 10
SFT_CHECKPOINT_MINIMUM_RETENTION = 2
CHECKPOINT_SCALER_STATE_SCHEMA = "lumen.transformers_scaler_checkpoint/1.0.0"
CHECKPOINT_SCALER_FILENAME = "scaler.pt"
CHECKPOINT_SCALER_TRANSFORMERS_VERSION = "4.57.6"
CHECKPOINT_SCALER_ACCELERATE_VERSION = "1.14.0"
SFT_CHECKPOINT_REQUIRED_FILES = frozenset(
    {
        "adapter_config.json",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
)
SFT_TOKEN_LENGTH_PREFLIGHT_SCHEMA = "lumen.sft_token_length_preflight/1.4.0"
SFT_TOKENIZATION_TRANSCRIPT_SCHEMA = "lumen.sft-tokenization-transcript/1.0.0"
RUNTIME_MODEL_BINDING_SCHEMA = "lumen.runtime-model-binding/1.3.0"
RUNTIME_TOKENIZER_BINDING_SCHEMA = "lumen.runtime-tokenizer-binding/1.1.0"
ADAPTER_BASE_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
ADAPTER_DERIVED_TOKENIZER_FILES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "vocab.json",
    }
)
FLEET_LOSS_SHARE_CONTRACT_SCHEMA = "lumen.fleet-loss-share/1.9.0"
FLEET_LOSS_SHARE_EVIDENCE_SCHEMA = "lumen.fleet-loss-share-evidence/1.4.0"
FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_CONTRACT_SCHEMA = (
    "lumen.fleet-sft-optimizer-window-schedule-contract/1.0.0"
)
FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_SCHEMA = (
    "lumen.fleet-sft-optimizer-window-schedule/1.0.0"
)
FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_ALGORITHM = (
    "sha256_epoch_stratified_token_aware_family_round_robin/1.3.0"
)
FLEET_SFT_OPTIMIZER_WINDOW_CANDIDATE_COUNT = 256
FLEET_SFT_OPTIMIZER_RECORD_GEOMETRY_SCHEMA = (
    "lumen.fleet-sft-optimizer-record-geometry/1.0.0"
)
FLEET_SFT_OPTIMIZER_MAX_TRAIN_RECORDS = 615
FLEET_SFT_RUNTIME_LOSS_NORMALIZATION_SCHEMA = (
    "lumen.fleet-sft-runtime-loss-normalization/1.2.0"
)
FLEET_SFT_TRAINER_CLASS = "__main__._FleetSFTTrainer"
FLEET_SFT_MODEL_CLASS = "peft.peft_model.PeftModelForCausalLM"
FLEET_SFT_BASE_MODEL_CLASS = (
    "transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM"
)
FLEET_SFT_GET_BATCH_SAMPLES_MODULE = "unsloth_zoo.loss_utils"
FLEET_SFT_GET_BATCH_SAMPLES_NAME = "_unsloth_get_batch_samples"
FLEET_SFT_GET_BATCH_SAMPLES_SOURCE = "unsloth_zoo/loss_utils.py"
FLEET_OPTIMIZER_FAMILY_SHARE_SCHEMA = (
    "lumen.fleet-optimizer-family-share/1.1.0"
)
FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY = "fleet_orchestration_native"
FLEET_POLICY_VOCABULARY_SOURCE_FAMILY = "adapter_ultra_specific"
FLEET_POLICY_VOCABULARY_SFT_TASK_TYPE = (
    "fleet_contract_event_graph_vocabulary"
)
FLEET_NATIVE_ORCHESTRATION_TASK_TYPE_BY_LANE = {
    "sft": "fleet_orchestration_event_graph",
    "dpo": "fleet_orchestration_event_graph_preference",
}
FLEET_OPTIMIZER_FAMILY_SHARE_LANES = {
    "sft": {
        "basis": "assistant_mask_non_ignored_token_count",
        "numeratorEvidenceField": (
            "nativeOrchestrationAssistantTargetTokenCount"
        ),
        "denominatorEvidenceField": "assistantTargetTokenCount",
        "minimumBasisPoints": 5_000,
        "maximumBasisPoints": 6_000,
    },
    "dpo": {
        "basis": "preference_pair_count",
        "numeratorEvidenceField": "nativeOrchestrationPreferencePairCount",
        "denominatorEvidenceField": "preferencePairCount",
        "minimumBasisPoints": 1_800,
        "maximumBasisPoints": 2_200,
    },
}
FLEET_POLICY_SIGNAL_TOKEN_CLASSIFICATIONS = {
    "sft": (
        (
            FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY,
            FLEET_NATIVE_ORCHESTRATION_TASK_TYPE_BY_LANE["sft"],
        ),
        (
            FLEET_POLICY_VOCABULARY_SOURCE_FAMILY,
            FLEET_POLICY_VOCABULARY_SFT_TASK_TYPE,
        ),
    ),
    "dpo": (
        (
            FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY,
            FLEET_NATIVE_ORCHESTRATION_TASK_TYPE_BY_LANE["dpo"],
        ),
    ),
}
FLEET_POLICY_SIGNAL_TOKEN_SHARE_LANES = {
    "sft": {
        "basis": "assistant_mask_non_ignored_token_count",
        "numeratorEvidenceField": "allPolicyAssistantTargetTokenCount",
        "denominatorEvidenceField": "assistantTargetTokenCount",
        "minimumBasisPoints": 6_000,
        "maximumBasisPoints": 6_500,
    },
    "dpo": {
        "basis": "rendered_chosen_completion_token_count",
        "numeratorEvidenceField": (
            "nativeOrchestrationChosenTargetTokenCount"
        ),
        "denominatorEvidenceField": "chosenTargetTokenCount",
        "minimumBasisPoints": 7_500,
        "maximumBasisPoints": 8_500,
    },
}
FLEET_OPTIMIZER_FAMILY_SHARE_COMPARISON_RULES = {
    "minimum": (
        "numeratorCount*basisPointDenominator>="
        "denominatorCount*minimumBasisPoints"
    ),
    "maximum": (
        "numeratorCount*basisPointDenominator<="
        "denominatorCount*maximumBasisPoints"
    ),
}
FLEET_SOURCE_ROLE_SCHEMA = "lumen.fleet-source-role/1.0.0"
FLEET_LOSS_SHARE_BASIS_POINT_DENOMINATOR = 10_000
FLEET_SOURCE_ROLES = (
    "behavioral_primary",
    "public_behavioral",
    "supplemental_static",
)
FLEET_DPO_TOKENIZATION_POLICY = {
    "trainerImplementation": "trl.DPOTrainer.tokenize_row",
    "trlVersion": "0.24.0",
    "completionTokenization": "add_special_tokens_false",
    "completionSuffix": "append_tokenizer_eos_token_id",
    "appendedEOSTokensPerCompletion": 1,
}
FLEET_LOSS_SHARE_FIELD_NAMES = {
    "sft": {
        "denominatorTokenCount": "assistantTargetTokenCount",
        "supplementalNumeratorTokenCount": (
            "supplementalStaticAssistantTargetTokenCount"
        ),
        "publicNumeratorTokenCount": (
            "publicBehavioralAssistantTargetTokenCount"
        ),
        "policySignalNumeratorTokenCount": (
            "allPolicyAssistantTargetTokenCount"
        ),
        "perSourceFamilyNumeratorTokenCounts": (
            "supplementalStaticAssistantTargetTokenCountsBySourceFamily"
        ),
    },
    "dpo": {
        "denominatorTokenCount": "chosenTargetTokenCount",
        "supplementalNumeratorTokenCount": (
            "supplementalStaticChosenTargetTokenCount"
        ),
        "publicNumeratorTokenCount": (
            "publicBehavioralChosenTargetTokenCount"
        ),
        "policySignalNumeratorTokenCount": (
            "nativeOrchestrationChosenTargetTokenCount"
        ),
        "perSourceFamilyNumeratorTokenCounts": (
            "supplementalStaticChosenTargetTokenCountsBySourceFamily"
        ),
    },
}
PUBLIC_CORPUS_LOSS_SHARE_CONTRACT_SCHEMA = (
    "lumen.public-corpus-loss-share/1.0.0"
)
PUBLIC_CORPUS_LOSS_SHARE_EVIDENCE_SCHEMA = (
    "lumen.public-corpus-loss-share-evidence/1.0.0"
)
PUBLIC_CORPUS_LOSS_SHARE_BASIS_POINT_DENOMINATOR = 10_000
PUBLIC_CORPUS_DPO_TOKENIZATION_POLICY = dict(FLEET_DPO_TOKENIZATION_POLICY)
PUBLIC_CORPUS_LOSS_SHARE_FIELD_NAMES = {
    "sft": {
        "denominatorTokenCount": "assistantTargetTokenCount",
        "publicNumeratorTokenCount": "publicAssistantTargetTokenCount",
    },
    "dpo": {
        "denominatorTokenCount": "chosenTargetTokenCount",
        "publicNumeratorTokenCount": "publicChosenTargetTokenCount",
    },
}
TRAINING_COMPLETION_EVIDENCE_SCHEMA = "lumen.training_completion/1.1.0"
TRAINING_PRECISION_SCHEMA = "lumen.training-precision/1.0.0"
SFT_MINIMUM_SEQUENCE_MARGIN_TOKENS = 128
ZERO_GPU_LINEAGE_FIELDS = (
    "zeroGPUSize",
    "zeroGPUDurationSeconds",
    "observedAccelerator",
)


class _IncompleteSFTCheckpoint(RuntimeError):
    """A structurally unfinished checkpoint that can only be discarded when stale."""


def _runtime_accelerator_audit() -> dict[str, Any]:
    import torch  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA accelerator")
    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        raise RuntimeError("Training runtime exposed no CUDA devices")
    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "totalMemoryBytes": int(properties.total_memory),
                "computeCapability": [int(capability[0]), int(capability[1])],
            }
        )
    return {
        "bindingStatus": "runtime_observed_unverified",
        "backend": "cuda",
        "deviceCount": device_count,
        "devices": devices,
    }


def _validated_hardware_lineage(cfg: Mapping[str, Any]) -> dict[str, Any]:
    size = cfg.get("zeroGPUSize")
    duration = cfg.get("zeroGPUDurationSeconds")
    configured_accelerator = cfg.get("observedAccelerator")
    observed_accelerator = _runtime_accelerator_audit()
    if cfg.get("runtimeSourceKind") == "huggingface_space":
        if size not in ZERO_GPU_ALLOWED_SIZES:
            raise RuntimeError("ZeroGPU training requires a supported deployed size")
        if (
            type(duration) is not int
            or duration <= 0
        ):
            raise RuntimeError("ZeroGPU training requires a positive deployed duration")
        if configured_accelerator != observed_accelerator:
            raise RuntimeError("Observed accelerator drifted from the ZeroGPU lease")
    elif size is not None or duration is not None:
        raise RuntimeError("Non-Space training must not claim a ZeroGPU allocation")
    return {
        "zeroGPUSize": size,
        "zeroGPUDurationSeconds": duration,
        "observedAccelerator": observed_accelerator,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-agent SFT adapters with Unsloth.")
    parser.add_argument("--config", required=True, help="Path to agent Unsloth JSON config.")
    parser.add_argument(
        "--runtime-binding-smoke",
        action="store_true",
        help=(
            "Load the pinned private base model once and verify its model, "
            "generation-config, and tokenizer bindings without creating PEFT "
            "state or starting a trainer."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Controlled deterministic seed; any CLI or LUMEN_TRAIN_SEED value must match the config.",
    )
    parser.add_argument("--resume-from-checkpoint", action="store_true", help="Resume from the latest checkpoint in output_dir if present.")
    parser.add_argument("--assistant-only-loss", action="store_true", help="Compute loss only on assistant turns (TRL assistant_only_loss).")
    return parser.parse_args()


def _resolve_training_precision(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve an explicit, mutually exclusive mixed-precision contract."""

    bf16 = cfg.get("bf16")
    fp16 = cfg.get("fp16")
    if type(bf16) is not bool or type(fp16) is not bool:
        raise ValueError("bf16 and fp16 must both be explicit booleans")
    if bf16 == fp16:
        raise ValueError("Exactly one of bf16 and fp16 must be true")
    return {
        "schemaVersion": TRAINING_PRECISION_SCHEMA,
        "bf16": bf16,
        "fp16": fp16,
        "dtype": "bfloat16" if bf16 else "float16",
    }


def _controlled_torch_dtype(cfg: Mapping[str, Any]) -> Any:
    """Return the exact Torch dtype declared by the prepared precision contract."""

    import torch  # type: ignore

    precision = _resolve_training_precision(cfg)
    return torch.bfloat16 if precision["bf16"] else torch.float16


def _checkpoint_scaler_state_contract(
    precision_value: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the pinned Transformers/Accelerate native-AMP scaler contract."""

    precision = _resolve_training_precision(precision_value)
    return {
        "schemaVersion": CHECKPOINT_SCALER_STATE_SCHEMA,
        "filename": CHECKPOINT_SCALER_FILENAME,
        "required": precision["fp16"],
        "requirement": "required_for_fp16_cuda_native_amp",
        "transformersVersion": CHECKPOINT_SCALER_TRANSFORMERS_VERSION,
        "accelerateVersion": CHECKPOINT_SCALER_ACCELERATE_VERSION,
    }


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in sorted(REQUIRED_CONFIG_KEYS) if key not in cfg]
    if missing:
        raise ValueError(f"Config is missing required keys: {', '.join(missing)}")
    if cfg["baseModelID"] != cfg["base_model_name"]:
        raise ValueError("baseModelID must exactly match base_model_name")
    _resolve_training_precision(cfg)
    verify_chat_template_contract(cfg["chatTemplateContract"])
    validate_artifact_path_config(cfg)
    return cfg


def _tokenize_path(value: str) -> set[str]:
    return set("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def validate_artifact_path_config(cfg: dict[str, Any]) -> None:
    agent = str(cfg.get("agent", "")).strip().lower()
    if agent not in AGENTS:
        raise ValueError(f"Config has unsupported agent '{agent}'. Expected one of: {', '.join(sorted(AGENTS))}")

    for label in ("output_dir", "adapter_output_dir"):
        value = str(cfg.get(label, "")).strip()
        if not value:
            raise ValueError(f"Config {label} must be non-empty")

        tokens = _tokenize_path(value)
        if agent not in tokens:
            raise ValueError(
                f"{label} must include slot token '{agent}' in the artifact path. Got: {value}"
            )
        if not FINETUNE_MARKERS.intersection(tokens):
            raise ValueError(
                f"{label} must include a finetune marker token (one of: "
                + ", ".join(sorted(FINETUNE_MARKERS))
                + f"). Got: {value}"
            )

    validate_sft_artifact_paths(cfg)


def validate_sft_artifact_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(cfg["output_dir"]).resolve()
    adapter_output_dir = Path(cfg["adapter_output_dir"]).resolve()
    if (
        adapter_output_dir == output_dir
        or output_dir in adapter_output_dir.parents
        or adapter_output_dir in output_dir.parents
    ):
        raise ValueError(
            "adapter_output_dir must be separate from the training work/output directory"
        )
    return output_dir, adapter_output_dir


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return apply_non_thinking_chat_template(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def normalize_chat_messages(record: dict[str, Any], *, row_index: int, path: Path) -> list[dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path}:{row_index + 1} must contain a non-empty messages array")

    normalized: list[dict[str, str]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{path}:{row_index + 1}.messages[{message_index}] must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(
                f"{path}:{row_index + 1}.messages[{message_index}].role must be one of system, user, assistant, tool"
            )
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        normalized.append({"role": role, "content": content})

    if not any(message["role"] == "assistant" for message in normalized):
        raise ValueError(f"{path}:{row_index + 1} must contain at least one assistant message")
    return normalized


def build_sft_rows(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    assistant_only_loss: bool,
    path: Path,
    max_seq_length: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        messages = normalize_chat_messages(record, row_index=index, path=path)
        if assistant_only_loss:
            rows.append(
                tokenize_assistant_only_row(
                    tokenizer,
                    messages,
                    path=path,
                    row_index=index,
                    max_seq_length=max_seq_length,
                )
            )
        else:
            rows.append({"text": render_messages(tokenizer, messages)})
    return rows


def _flatten_tokenizer_output(value: Any) -> list[int]:
    if hasattr(value, "data") and isinstance(value.data, dict) and "input_ids" in value.data:
        value = value.data["input_ids"]
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise ValueError("Tokenizer chat template must return a list of token ids")
    return [int(token) for token in value]


def _chat_template_input_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    rendered = apply_non_thinking_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(rendered, "get"):
        rendered = rendered.get("input_ids")
    return _flatten_tokenizer_output(rendered)


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def _tokenizer_supports_assistant_masks(tokenizer: Any) -> bool:
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    return "{% generation" in chat_template


def _derive_assistant_masks_from_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    input_ids: list[int],
    *,
    path: Path,
    row_index: int,
) -> list[int]:
    assistant_masks = [0] * len(input_ids)
    for message_index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        if message_index == 0:
            start = 0
        else:
            start_ids = _chat_template_input_ids(
                tokenizer,
                messages[:message_index],
                add_generation_prompt=True,
            )
            start = (
                len(start_ids)
                if input_ids[: len(start_ids)] == start_ids
                else _common_prefix_length(input_ids, start_ids)
            )

        end_ids = _chat_template_input_ids(
            tokenizer,
            messages[: message_index + 1],
            add_generation_prompt=False,
        )
        end = (
            len(end_ids)
            if input_ids[: len(end_ids)] == end_ids
            else _common_prefix_length(input_ids, end_ids)
        )
        if end <= start:
            raise RuntimeError(
                f"{path}:{row_index + 1} could not derive assistant token span from the tokenizer chat template"
            )
        for token_index in range(start, min(end, len(assistant_masks))):
            assistant_masks[token_index] = 1
    return assistant_masks


def tokenize_assistant_only_row(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    path: Path,
    row_index: int,
    max_seq_length: int | None,
) -> dict[str, list[int]]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "Assistant-only loss requires a tokenizer with apply_chat_template(..., "
            "return_assistant_tokens_mask=True)."
        )

    chat_template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
        "enable_thinking": False,
    }
    if _tokenizer_supports_assistant_masks(tokenizer):
        chat_template_kwargs["return_assistant_tokens_mask"] = True
    processed = apply_non_thinking_chat_template(
        tokenizer,
        messages,
        **chat_template_kwargs,
    )
    if not hasattr(processed, "get"):
        raise RuntimeError(
            "Assistant-only loss requires apply_chat_template(..., return_dict=True) "
            "to return input_ids and assistant token masks."
        )

    input_ids = _flatten_tokenizer_output(processed.get("input_ids"))
    assistant_mask_value = processed.get("assistant_masks", processed.get("assistant_tokens_mask"))
    if assistant_mask_value is None:
        assistant_masks = _derive_assistant_masks_from_template(
            tokenizer,
            messages,
            input_ids,
            path=path,
            row_index=row_index,
        )
    else:
        assistant_masks = _flatten_tokenizer_output(assistant_mask_value)
        if 1 not in assistant_masks:
            assistant_masks = _derive_assistant_masks_from_template(
                tokenizer,
                messages,
                input_ids,
                path=path,
                row_index=row_index,
            )
    if len(input_ids) != len(assistant_masks):
        raise RuntimeError(
            f"{path}:{row_index + 1} produced mismatched input_ids and assistant mask lengths "
            f"({len(input_ids)} != {len(assistant_masks)})"
        )

    attention_mask_value = processed.get("attention_mask")
    if attention_mask_value is None:
        attention_mask = [1] * len(input_ids)
    else:
        attention_mask = _flatten_tokenizer_output(attention_mask_value)
        if len(attention_mask) != len(input_ids):
            raise RuntimeError(
                f"{path}:{row_index + 1} produced mismatched input_ids and attention_mask lengths "
                f"({len(input_ids)} != {len(attention_mask)})"
            )

    if max_seq_length is not None:
        if type(max_seq_length) is not int or max_seq_length <= 0:
            raise ValueError("max_seq_length must be a positive integer")
        if len(input_ids) > max_seq_length:
            raise RuntimeError(
                f"{path}:{row_index + 1} renders to {len(input_ids)} tokens, "
                f"exceeding max_seq_length {max_seq_length}; SFT truncation is forbidden"
            )

    labels = [token_id if mask else -100 for token_id, mask in zip(input_ids, assistant_masks)]
    if all(label == -100 for label in labels):
        raise RuntimeError(
            f"{path}:{row_index + 1} has no assistant tokens after chat-template masking. "
            "Check that the tokenizer chat template supports assistant token masks and "
            "that max_seq_length does not truncate away the assistant response."
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _shifted_sft_target_token_count(
    tokenized: Mapping[str, Any],
) -> int:
    """Count the exact causal-LM targets consumed after the one-token shift."""

    labels = tokenized.get("labels")
    attention_mask = tokenized.get("attention_mask")
    if (
        not isinstance(labels, list)
        or not isinstance(attention_mask, list)
        or len(labels) != len(attention_mask)
        or any(type(label) is not int for label in labels)
        or any(type(mask) is not int for mask in attention_mask)
    ):
        raise RuntimeError("SFT target-token accounting received malformed labels")
    return sum(
        1
        for label, attended in zip(labels[1:], attention_mask[1:])
        if label != -100 and attended != 0
    )


def _sft_nearest_rank(values: list[int], percentile: int) -> int:
    if not values:
        raise ValueError("SFT token-length statistics require at least one value")
    ordered = sorted(values)
    return ordered[max(0, math.ceil((percentile / 100) * len(ordered)) - 1)]


def _sft_token_length_statistics(values: list[int]) -> dict[str, int]:
    if not values:
        raise ValueError("SFT token-length statistics require at least one value")
    return {
        "min": min(values),
        "p50": _sft_nearest_rank(values, 50),
        "p95": _sft_nearest_rank(values, 95),
        "max": max(values),
    }


def _preflight_sft_token_lengths(
    splits: Mapping[str, tuple[list[dict[str, Any]], Path]],
    *,
    tokenizer: Any,
    max_sequence_length: int,
    minimum_sequence_margin_tokens: int = SFT_MINIMUM_SEQUENCE_MARGIN_TOKENS,
    agent: str | None = None,
    fleet_loss_share_contract: Any = None,
    public_corpus_loss_share_contract: Any = None,
    fleet_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if type(max_sequence_length) is not int or max_sequence_length <= 0:
        raise ValueError("max_seq_length must be a positive integer")
    if (
        type(minimum_sequence_margin_tokens) is not int
        or minimum_sequence_margin_tokens < SFT_MINIMUM_SEQUENCE_MARGIN_TOKENS
    ):
        raise ValueError(
            "minimum_sequence_margin_tokens cannot weaken the controlled 128-token margin"
        )
    validated_fleet_contract: dict[str, Any] | None = None
    if agent == "fleet":
        validated_fleet_contract = _validated_fleet_loss_share_contract(
            fleet_loss_share_contract,
            lane="sft",
            config=fleet_config,
        )
    elif fleet_loss_share_contract is not None:
        raise RuntimeError("Fleet loss-share contract is forbidden for non-Fleet SFT")
    if agent is not None:
        _validated_public_corpus_loss_share_contract(
            public_corpus_loss_share_contract,
            lane="sft",
            config=fleet_config,
        )
    elif public_corpus_loss_share_contract is not None:
        raise RuntimeError(
            "Public-corpus loss-share contract requires a controlled agent"
        )
    if agent == "fleet":
        if validated_fleet_contract is None or not isinstance(
            fleet_config,
            Mapping,
        ):
            raise RuntimeError(
                "Fleet SFT optimizer-record geometry requires its training "
                "config"
            )
        geometry = validated_fleet_contract[
            "sftOptimizerRecordGeometryContract"
        ]
        maximum_train_records = geometry["maximumTrainRecords"]
        optimization_policy = fleet_config.get("optimizationStepPolicy")
        sft_policy = (
            optimization_policy.get("sft")
            if isinstance(optimization_policy, Mapping)
            else None
        )
        declared_train_records = (
            sft_policy.get("trainRecordCount")
            if isinstance(sft_policy, Mapping)
            else None
        )
        train_split = splits.get("train")
        train_records = (
            train_split[0]
            if isinstance(train_split, tuple)
            and len(train_split) == 2
            and isinstance(train_split[0], list)
            else None
        )
        if train_records is None:
            raise RuntimeError(
                "Fleet SFT optimizer-record geometry requires a valid train "
                "split"
            )
        actual_train_records = len(train_records)
        if actual_train_records > maximum_train_records:
            raise RuntimeError(
                "Fleet SFT optimizer-record geometry exceeds its calibrated "
                "train-record ceiling: "
                f"actual={actual_train_records} "
                f"maximum={maximum_train_records}"
            )
        if actual_train_records != declared_train_records:
            raise RuntimeError(
                "Fleet SFT optimizer-record geometry differs from its "
                "training config: "
                f"actual={actual_train_records} "
                f"declared={declared_train_records}"
            )
    aggregate_total: list[int] = []
    aggregate_assistant: list[int] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    transcript_rows: list[dict[str, Any]] = []
    fleet_target_rows: dict[
        str,
        list[tuple[Mapping[str, Any], int]],
    ] = {}
    public_corpus_target_rows: dict[
        str,
        list[tuple[Mapping[str, Any], int]],
    ] = {}
    for split, split_value in splits.items():
        if (
            not isinstance(split, str)
            or not split
            or not isinstance(split_value, tuple)
            or len(split_value) != 2
        ):
            raise ValueError("SFT preflight splits must bind records and their source path")
        records, path = split_value
        if not isinstance(records, list) or not isinstance(path, Path):
            raise TypeError("SFT preflight split values are invalid")
        total_lengths: list[int] = []
        assistant_lengths: list[int] = []
        split_fleet_target_rows: list[tuple[Mapping[str, Any], int]] = []
        split_public_corpus_target_rows: list[
            tuple[Mapping[str, Any], int]
        ] = []
        for row_index, record in enumerate(records):
            messages = normalize_chat_messages(
                record,
                row_index=row_index,
                path=path,
            )
            tokenized = tokenize_assistant_only_row(
                tokenizer,
                messages,
                path=path,
                row_index=row_index,
                max_seq_length=None,
            )
            total_tokens = len(tokenized["input_ids"])
            assistant_tokens = _shifted_sft_target_token_count(tokenized)
            row_transcript = {
                "schemaVersion": SFT_TOKENIZATION_TRANSCRIPT_SCHEMA,
                "split": split,
                "rowIndex": row_index,
                "inputIDs": tokenized["input_ids"],
                "attentionMask": tokenized["attention_mask"],
                "labels": tokenized["labels"],
            }
            transcript_rows.append(
                {
                    "split": split,
                    "rowIndex": row_index,
                    "rowSHA256": _canonical_sha256(row_transcript),
                }
            )
            if total_tokens > max_sequence_length:
                raise RuntimeError(
                    "SFT token-length preflight rejected "
                    f"{split} row {row_index}: full rendered row uses "
                    f"{total_tokens} tokens, exceeding max_seq_length "
                    f"{max_sequence_length}; truncation is forbidden"
                )
            total_lengths.append(total_tokens)
            assistant_lengths.append(assistant_tokens)
            aggregate_total.append(total_tokens)
            aggregate_assistant.append(assistant_tokens)
            if agent == "fleet":
                split_fleet_target_rows.append((record, assistant_tokens))
            if agent is not None:
                split_public_corpus_target_rows.append(
                    (record, assistant_tokens)
                )
        if agent == "fleet":
            fleet_target_rows[split] = split_fleet_target_rows
        if agent is not None:
            public_corpus_target_rows[split] = split_public_corpus_target_rows
        if records:
            split_summaries[split] = {
                "records": len(records),
                "totalTokens": _sft_token_length_statistics(total_lengths),
                "assistantTargetTokens": _sft_token_length_statistics(
                    assistant_lengths
                ),
                "smallestSequenceMarginTokens": (
                    max_sequence_length - max(total_lengths)
                ),
            }
        else:
            split_summaries[split] = {"records": 0}
    if not aggregate_total:
        raise RuntimeError("SFT token-length preflight requires at least one row")
    smallest_margin = max_sequence_length - max(aggregate_total)
    if smallest_margin < minimum_sequence_margin_tokens:
        raise RuntimeError(
            "SFT token-length preflight rejected the configured sequence limit: "
            f"the smallest exact-tokenizer margin is {smallest_margin} tokens, "
            f"below the controlled minimum of {minimum_sequence_margin_tokens}"
        )
    report = {
        "schemaVersion": SFT_TOKEN_LENGTH_PREFLIGHT_SCHEMA,
        "maxSequenceLength": max_sequence_length,
        "minimumSequenceMarginTokens": minimum_sequence_margin_tokens,
        "percentileMethod": "nearest_rank",
        "records": len(aggregate_total),
        "totalTokens": _sft_token_length_statistics(aggregate_total),
        "assistantTargetTokens": _sft_token_length_statistics(
            aggregate_assistant
        ),
        "smallestSequenceMarginTokens": smallest_margin,
        "truncationRequired": False,
        "splits": split_summaries,
        "tokenizationTranscriptSHA256": _canonical_sha256(
            {
                "schemaVersion": SFT_TOKENIZATION_TRANSCRIPT_SCHEMA,
                "rows": transcript_rows,
            }
        ),
    }
    if agent == "fleet":
        report["fleetLossShareEvidence"] = _build_fleet_loss_share_evidence(
            contract_value=fleet_loss_share_contract,
            lane="sft",
            split_target_rows=fleet_target_rows,
            config=fleet_config,
        )
    if agent is not None:
        report["publicCorpusLossShareEvidence"] = (
            _build_public_corpus_loss_share_evidence(
                contract_value=public_corpus_loss_share_contract,
                lane="sft",
                split_target_rows=public_corpus_target_rows,
                config=fleet_config,
            )
        )
    return report


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import transformers  # type: ignore

        transformers.set_seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort determinism. Note: PyTorch never guarantees absolute reproducibility
        # across hardware/CUDA versions; this just removes obvious sources of drift.
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def _require_unsloth_before_transformers() -> None:
    if "transformers" in sys.modules and "unsloth" not in sys.modules:
        raise RuntimeError(
            "Unsloth must be imported before Transformers so its runtime patches are applied"
        )


def _resolve_controlled_seed(
    cfg: Mapping[str, Any],
    *,
    cli_seed: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    controlled_seed = cfg.get("seed")
    if type(controlled_seed) is not int:
        raise ValueError("Config seed must be an integer controlled by the variant manifest")

    environment = os.environ if environ is None else environ
    env_text = environment.get("LUMEN_TRAIN_SEED")
    try:
        env_seed = int(env_text) if env_text is not None and env_text.strip() else None
    except ValueError as exc:
        raise ValueError("LUMEN_TRAIN_SEED must be an integer") from exc

    if cli_seed is not None and int(cli_seed) != controlled_seed:
        raise ValueError(
            f"CLI seed override would break controlled lineage: expected {controlled_seed}, got {cli_seed}"
        )
    if env_seed is not None and env_seed != controlled_seed:
        raise ValueError(
            "LUMEN_TRAIN_SEED override would break controlled lineage: "
            f"expected {controlled_seed}, got {env_seed}"
        )
    if cli_seed is not None:
        return controlled_seed, "cli_verified"
    if env_seed is not None:
        return controlled_seed, "env_verified"
    return controlled_seed, "config"


def _hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_mapping_keys(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError(f"{label} has an invalid schema")
    return value


def _validated_public_corpus_loss_share_contract(
    value: Any,
    *,
    lane: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the all-agent exact public-target loss-share contract."""

    if lane not in PUBLIC_CORPUS_LOSS_SHARE_FIELD_NAMES:
        raise RuntimeError(f"Unsupported public-corpus loss-share lane: {lane}")
    contract = _require_exact_mapping_keys(
        value,
        {
            "schemaVersion",
            "enforcementRequired",
            "enforcementPhase",
            "requiredLanes",
            "authoritativeCapEncoding",
            "basisPointDenominator",
            "capBasisPoints",
            "dpoTokenizationPolicy",
            "exactTokenEvidenceContract",
            "failurePolicy",
            "rowMetadataContract",
            "sourceSelectionProxy",
            "tokenizer",
            "tokenAccounting",
        },
        label="Public-corpus loss-share contract",
    )
    if (
        contract.get("schemaVersion")
        != PUBLIC_CORPUS_LOSS_SHARE_CONTRACT_SCHEMA
        or contract.get("enforcementRequired") is not True
        or contract.get("enforcementPhase")
        != "post_tokenizer_load_pre_optimizer"
        or contract.get("requiredLanes") != ["sft", "dpo"]
        or contract.get("authoritativeCapEncoding") != "integer_basis_points"
        or contract.get("basisPointDenominator")
        != PUBLIC_CORPUS_LOSS_SHARE_BASIS_POINT_DENOMINATOR
        or contract.get("failurePolicy") != "abort_before_optimizer"
    ):
        raise RuntimeError("Public-corpus loss-share contract controls drifted")

    caps = _require_exact_mapping_keys(
        contract.get("capBasisPoints"),
        {"requested", "hard"},
        label="Public-corpus loss-share caps",
    )
    requested_cap = caps.get("requested")
    hard_cap = caps.get("hard")
    if (
        type(requested_cap) is not int
        or type(hard_cap) is not int
        or not 0 <= requested_cap <= hard_cap
        or hard_cap != 3_500
    ):
        raise RuntimeError("Public-corpus loss-share caps drifted")

    row_contract = _require_exact_mapping_keys(
        contract.get("rowMetadataContract"),
        {
            "publicSourceFamilyPrefix",
            "publicCorpusField",
            "classificationRule",
            "mismatch",
        },
        label="Public-corpus row-metadata contract",
    )
    if row_contract != {
        "publicSourceFamilyPrefix": "public_adapter_corpus_",
        "publicCorpusField": "publicCorpus",
        "classificationRule": "prefix_and_nonempty_lineage_required",
        "mismatch": "hard_fail",
    }:
        raise RuntimeError("Public-corpus row-metadata contract drifted")

    source_selection_proxy = _require_exact_mapping_keys(
        contract.get("sourceSelectionProxy"),
        {"status", "maximumPublicShareBasisPoints", "contract"},
        label="Public-corpus source-selection proxy",
    )
    proxy_cap = source_selection_proxy.get("maximumPublicShareBasisPoints")
    source_proxy_contract = _require_exact_mapping_keys(
        source_selection_proxy.get("contract"),
        {
            "schemaVersion",
            "status",
            "strategy",
            "maxCharsPerToken",
            "exactPinnedTokenizerAuthoritative",
            "authoritativeEnforcementPhase",
        },
        label="Public-corpus source-token proxy contract",
    )
    if (
        source_selection_proxy.get("status")
        != "safety_budget_not_exact_token_count"
        or type(proxy_cap) is not int
        or proxy_cap != min(requested_cap, 3_000)
        or source_proxy_contract.get("schemaVersion")
        != "lumen.source-token-proxy/1.0.0"
        or source_proxy_contract.get("status")
        != "source_side_selection_proxy_not_exact_token_count"
        or source_proxy_contract.get("strategy")
        != "max_whitespace_terms_utf8_byte_ceiling"
        or type(source_proxy_contract.get("maxCharsPerToken")) is not int
        or source_proxy_contract["maxCharsPerToken"] <= 0
        or source_proxy_contract.get("exactPinnedTokenizerAuthoritative") is not True
        or source_proxy_contract.get("authoritativeEnforcementPhase")
        != "post_tokenizer_load_pre_optimizer"
    ):
        raise RuntimeError("Public-corpus source-selection proxy drifted")

    dpo_policy = _require_exact_mapping_keys(
        contract.get("dpoTokenizationPolicy"),
        set(PUBLIC_CORPUS_DPO_TOKENIZATION_POLICY),
        label="Public-corpus DPO tokenization policy",
    )
    if dict(dpo_policy) != PUBLIC_CORPUS_DPO_TOKENIZATION_POLICY:
        raise RuntimeError("Public-corpus DPO tokenization policy drifted")
    accounting = _require_exact_mapping_keys(
        contract.get("tokenAccounting"),
        {"sft", "dpo"},
        label="Public-corpus token-accounting contract",
    )
    if accounting != {
        "sft": "assistant_mask_non_ignored_token_count",
        "dpo": (
            "rendered_chosen_completion_tokens_add_special_tokens_false_"
            "plus_one_trl_0_24_0_appended_eos"
        ),
    }:
        raise RuntimeError("Public-corpus token-accounting contract drifted")

    tokenizer_binding = _require_exact_mapping_keys(
        contract.get("tokenizer"),
        {
            "baseModelID",
            "baseModelRevision",
            "tokenizerSHA256",
            "tokenizerClosureSHA256",
        },
        label="Public-corpus tokenizer binding",
    )
    if (
        not isinstance(tokenizer_binding.get("baseModelID"), str)
        or not tokenizer_binding["baseModelID"]
        or re.fullmatch(
            r"[0-9a-f]{40}", str(tokenizer_binding.get("baseModelRevision") or "")
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(tokenizer_binding.get("tokenizerSHA256") or "")
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(tokenizer_binding.get("tokenizerClosureSHA256") or ""),
        )
        is None
    ):
        raise RuntimeError("Public-corpus tokenizer binding is malformed")
    if config is not None and (
        tokenizer_binding.get("baseModelID") != config.get("base_model_name")
        or tokenizer_binding.get("baseModelRevision")
        != config.get("baseModelRevision")
        or tokenizer_binding.get("tokenizerSHA256")
        != config.get("baseModelTokenizerDigest")
        or tokenizer_binding.get("tokenizerClosureSHA256")
        != config.get("baseModelTokenizerClosureSHA256")
    ):
        raise RuntimeError(
            "Public-corpus tokenizer binding drifted from the training config"
        )

    exact = _require_exact_mapping_keys(
        contract.get("exactTokenEvidenceContract"),
        {
            "required",
            "schemaVersion",
            "statusAtGeneration",
            "tokenizer",
            "comparisonRule",
            "lanes",
        },
        label="Public-corpus exact-token evidence contract",
    )
    if (
        exact.get("required") is not True
        or exact.get("schemaVersion")
        != PUBLIC_CORPUS_LOSS_SHARE_EVIDENCE_SCHEMA
        or exact.get("statusAtGeneration")
        != "pending_exact_tokenizer_preflight"
        or exact.get("tokenizer") != "pinned_qwen_tokenizer"
        or exact.get("comparisonRule")
        != (
            "numeratorTokenCount*basisPointDenominator<="
            "denominatorTokenCount*capBasisPoints"
        )
    ):
        raise RuntimeError("Public-corpus exact-token evidence contract drifted")
    lanes = _require_exact_mapping_keys(
        exact.get("lanes"),
        {"sft", "dpo"},
        label="Public-corpus exact-token evidence lanes",
    )
    for expected_lane, expected_fields in (
        PUBLIC_CORPUS_LOSS_SHARE_FIELD_NAMES.items()
    ):
        fields = _require_exact_mapping_keys(
            lanes.get(expected_lane),
            set(expected_fields),
            label=f"Public-corpus {expected_lane} exact-token fields",
        )
        if dict(fields) != expected_fields:
            raise RuntimeError(
                f"Public-corpus {expected_lane} exact-token fields drifted"
            )
    return dict(contract)


def _public_corpus_row_classification(
    record: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> tuple[str, bool]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Public-corpus loss-share rows require object metadata")
    source_family = metadata.get("sourceFamily")
    if (
        not isinstance(source_family, str)
        or not source_family
        or source_family.strip() != source_family
    ):
        raise RuntimeError(
            "Public-corpus loss-share rows require canonical metadata.sourceFamily"
        )
    row_contract = contract["rowMetadataContract"]
    has_public_prefix = source_family.startswith(
        row_contract["publicSourceFamilyPrefix"]
    )
    lineage = metadata.get(row_contract["publicCorpusField"])
    has_public_lineage = isinstance(lineage, Mapping) and bool(lineage)
    if has_public_prefix != has_public_lineage:
        raise RuntimeError(
            "Public-corpus row metadata prefix and lineage classification disagree"
        )
    return source_family, has_public_prefix


def _public_corpus_cap_passes(
    *,
    numerator: int,
    denominator: int,
    cap_basis_points: int,
) -> bool:
    return (
        type(numerator) is int
        and numerator >= 0
        and type(denominator) is int
        and denominator > 0
        and type(cap_basis_points) is int
        and 0
        <= cap_basis_points
        <= PUBLIC_CORPUS_LOSS_SHARE_BASIS_POINT_DENOMINATOR
        and numerator * PUBLIC_CORPUS_LOSS_SHARE_BASIS_POINT_DENOMINATOR
        <= denominator * cap_basis_points
    )


def _build_public_corpus_loss_share_evidence(
    *,
    contract_value: Any,
    lane: str,
    split_target_rows: Mapping[
        str,
        list[tuple[Mapping[str, Any], int]],
    ],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _validated_public_corpus_loss_share_contract(
        contract_value,
        lane=lane,
        config=config,
    )
    if set(split_target_rows) != {"train", "validation"}:
        raise RuntimeError(
            "Public-corpus exact-token enforcement requires train and validation splits"
        )
    fields = PUBLIC_CORPUS_LOSS_SHARE_FIELD_NAMES[lane]
    split_evidence: dict[str, Any] = {}
    for split in ("train", "validation"):
        rows = split_target_rows[split]
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(
                "Public-corpus exact-token enforcement requires a non-empty "
                f"{split} split"
            )
        denominator = 0
        public = 0
        row_evidence: list[dict[str, Any]] = []
        for row_index, row_value in enumerate(rows):
            if (
                not isinstance(row_value, tuple)
                or len(row_value) != 2
                or not isinstance(row_value[0], Mapping)
            ):
                raise RuntimeError("Public-corpus exact-token row evidence is malformed")
            record, target_tokens = row_value
            if type(target_tokens) is not int or target_tokens <= 0:
                raise RuntimeError(
                    "Public-corpus exact-token rows require positive target counts"
                )
            source_family, is_public = _public_corpus_row_classification(
                record,
                contract=contract,
            )
            denominator += target_tokens
            if is_public:
                public += target_tokens
            row_evidence.append(
                {
                    "rowIndex": row_index,
                    "sourceRowSHA256": _canonical_sha256(record),
                    "sourceFamily": source_family,
                    "isPublicCorpus": is_public,
                    "targetTokenCount": target_tokens,
                }
            )
        enforce = split == "train"
        for label, cap in contract["capBasisPoints"].items():
            if enforce and not _public_corpus_cap_passes(
                numerator=public,
                denominator=denominator,
                cap_basis_points=cap,
            ):
                raise RuntimeError(
                    f"Public-corpus {lane} {split} {label} exact-token cap failed: "
                    f"{public}*10000 > {denominator}*{cap}"
                )
        row_hashes = [item["sourceRowSHA256"] for item in row_evidence]
        split_evidence[split] = {
            "records": len(rows),
            "capEnforcementStatus": (
                "optimizer_enforced"
                if enforce
                else "observed_non_optimizer_split"
            ),
            "sourceRowsSHA256": _canonical_sha256(row_hashes),
            "rowTokenEvidence": row_evidence,
            fields["denominatorTokenCount"]: denominator,
            fields["publicNumeratorTokenCount"]: public,
        }
    return {
        "schemaVersion": PUBLIC_CORPUS_LOSS_SHARE_EVIDENCE_SCHEMA,
        "status": "passed",
        "lane": lane,
        "enforcementScope": "optimizer_train_with_validation_observation",
        "basisPointDenominator": contract["basisPointDenominator"],
        "capBasisPoints": contract["capBasisPoints"],
        "tokenizer": contract["tokenizer"],
        "tokenAccounting": contract["tokenAccounting"][lane],
        "dpoTokenizationPolicy": (
            contract["dpoTokenizationPolicy"] if lane == "dpo" else None
        ),
        "contractSHA256": _canonical_sha256(contract),
        "splits": split_evidence,
    }


def _validated_fleet_sft_optimizer_window_schedule_contract(
    value: Any,
    *,
    sft_family_band: Mapping[str, Any],
) -> dict[str, Any]:
    schedule_contract = _require_exact_mapping_keys(
        value,
        {
            "schemaVersion",
            "evidenceSchemaVersion",
            "lane",
            "split",
            "enforcementRequired",
            "enforcementPhase",
            "basis",
            "basisPointDenominator",
            "minimumBasisPoints",
            "maximumBasisPoints",
            "algorithm",
            "candidateSearchCount",
            "permutationPolicy",
            "packing",
            "distributedSamplingPolicy",
            "resumePolicy",
            "failurePolicy",
        },
        label="Fleet SFT optimizer-window schedule contract",
    )
    expected = {
        "schemaVersion": (
            FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_CONTRACT_SCHEMA
        ),
        "evidenceSchemaVersion": (
            FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_SCHEMA
        ),
        "lane": "sft",
        "split": "train",
        "enforcementRequired": True,
        "enforcementPhase": "post_tokenizer_load_pre_optimizer",
        "basis": (
            "mean_of_floor_per_optimizer_window_native_assistant_target_"
            "token_share_basis_points"
        ),
        "basisPointDenominator": (
            FLEET_LOSS_SHARE_BASIS_POINT_DENOMINATOR
        ),
        "minimumBasisPoints": 5_000,
        "maximumBasisPoints": 6_000,
        "algorithm": FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_ALGORITHM,
        "candidateSearchCount": (
            FLEET_SFT_OPTIMIZER_WINDOW_CANDIDATE_COUNT
        ),
        "permutationPolicy": "each_source_row_exactly_once_per_epoch",
        "packing": False,
        "distributedSamplingPolicy": "single_process_only",
        "resumePolicy": (
            "trainer_set_epoch_with_monotonic_sampler_guard_through_"
            "skip_first_batches"
        ),
        "failurePolicy": "abort_before_optimizer",
    }
    if dict(schedule_contract) != expected:
        raise RuntimeError(
            "Fleet SFT optimizer-window schedule contract drifted"
        )
    if (
        schedule_contract["minimumBasisPoints"]
        != sft_family_band.get("minimumBasisPoints")
        or schedule_contract["maximumBasisPoints"]
        != sft_family_band.get("maximumBasisPoints")
    ):
        raise RuntimeError(
            "Fleet SFT optimizer-window band differs from family-share band"
        )
    return dict(schedule_contract)


def _validated_fleet_loss_share_contract(
    value: Any,
    *,
    lane: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the complete deny-by-default Fleet loss-share contract.

    This deliberately does not import the dataset compiler. Training must
    fail closed if either the generated contract or its row-role registry is
    malformed, even when the compiler is unavailable in the runtime image.
    """

    if lane not in FLEET_LOSS_SHARE_FIELD_NAMES:
        raise RuntimeError(f"Unsupported Fleet loss-share lane: {lane}")
    contract = _require_exact_mapping_keys(
        value,
        {
            "schemaVersion",
            "enforcementRequired",
            "enforcementPhase",
            "requiredLanes",
            "authoritativeCapEncoding",
            "basisPointDenominator",
            "capsBasisPoints",
            "dpoTokenizationPolicy",
            "exactTokenEvidenceContract",
            "failurePolicy",
            "optimizerFamilyShareBands",
            "sftOptimizerRecordGeometryContract",
            "sftOptimizerWindowScheduleContract",
            "rowMetadataContract",
            "sourceSelectionProxy",
            "sourceRoleRegistry",
            "tokenizer",
            "tokenAccounting",
        },
        label="Fleet loss-share contract",
    )
    if (
        contract.get("schemaVersion") != FLEET_LOSS_SHARE_CONTRACT_SCHEMA
        or contract.get("enforcementRequired") is not True
        or contract.get("enforcementPhase")
        != "post_tokenizer_load_pre_optimizer"
        or contract.get("requiredLanes") != ["sft", "dpo"]
        or contract.get("authoritativeCapEncoding") != "integer_basis_points"
        or type(contract.get("basisPointDenominator")) is not int
        or contract.get("basisPointDenominator")
        != FLEET_LOSS_SHARE_BASIS_POINT_DENOMINATOR
        or contract.get("failurePolicy") != "abort_before_optimizer"
    ):
        raise RuntimeError("Fleet loss-share contract control fields drifted")

    caps = _require_exact_mapping_keys(
        contract.get("capsBasisPoints"),
        {
            "supplementalStaticTotal",
            "publicBehavioralTotal",
            "eachSupplementalSourceFamily",
        },
        label="Fleet loss-share caps",
    )
    supplemental_caps = _require_exact_mapping_keys(
        caps.get("supplementalStaticTotal"),
        {"requested", "hard"},
        label="Fleet supplemental-static caps",
    )
    public_caps = _require_exact_mapping_keys(
        caps.get("publicBehavioralTotal"),
        {"requested", "hard"},
        label="Fleet public-behavioral caps",
    )
    family_caps = _require_exact_mapping_keys(
        caps.get("eachSupplementalSourceFamily"),
        {"hard"},
        label="Fleet per-source-family caps",
    )
    if (
        any(type(value) is not int for value in supplemental_caps.values())
        or any(type(value) is not int for value in public_caps.values())
        or any(type(value) is not int for value in family_caps.values())
        or supplemental_caps != {"requested": 2_500, "hard": 3_000}
        or public_caps != {"requested": 3_500, "hard": 3_500}
        or family_caps != {"hard": 1_000}
    ):
        raise RuntimeError("Fleet loss-share basis-point caps drifted")

    row_contract = _require_exact_mapping_keys(
        contract.get("rowMetadataContract"),
        {"requiredCanonicalFields", "missingOrUnknown"},
        label="Fleet row-metadata contract",
    )
    if row_contract != {
        "requiredCanonicalFields": ["sourceFamily", "taskType"],
        "missingOrUnknown": "hard_fail",
    }:
        raise RuntimeError("Fleet row-metadata contract drifted")

    source_selection_proxy = _require_exact_mapping_keys(
        contract.get("sourceSelectionProxy"),
        {
            "status",
            "maximumPublicBehavioralShareBasisPoints",
            "maximumSupplementalStaticShareBasisPoints",
            "optimizerFamilySafetyBand",
            "contract",
        },
        label="Fleet source-selection proxy",
    )
    source_family_safety_band = _require_exact_mapping_keys(
        source_selection_proxy.get("optimizerFamilySafetyBand"),
        {
            "schemaVersion",
            "lane",
            "basis",
            "sourceFamily",
            "taskType",
            "minimumBasisPoints",
            "maximumBasisPoints",
            "selectionPolicy",
            "authoritativeExactBandBasisPoints",
        },
        label="Fleet optimizer-family source-proxy safety band",
    )
    authoritative_exact_band = _require_exact_mapping_keys(
        source_family_safety_band.get("authoritativeExactBandBasisPoints"),
        {"minimum", "maximum"},
        label="Fleet authoritative optimizer-family band reference",
    )
    source_proxy_contract = _require_exact_mapping_keys(
        source_selection_proxy.get("contract"),
        {
            "schemaVersion",
            "status",
            "strategy",
            "maxCharsPerToken",
            "exactPinnedTokenizerAuthoritative",
            "authoritativeEnforcementPhase",
        },
        label="Fleet source-token proxy contract",
    )
    if (
        source_selection_proxy.get("status")
        != "safety_budget_not_exact_token_count"
        or source_selection_proxy.get(
            "maximumPublicBehavioralShareBasisPoints"
        )
        != 3_000
        or source_selection_proxy.get("maximumSupplementalStaticShareBasisPoints")
        != 1_500
        or type(source_family_safety_band.get("minimumBasisPoints")) is not int
        or type(source_family_safety_band.get("maximumBasisPoints")) is not int
        or type(authoritative_exact_band.get("minimum")) is not int
        or type(authoritative_exact_band.get("maximum")) is not int
        or source_family_safety_band
        != {
            "schemaVersion": (
                "lumen.fleet-optimizer-family-source-proxy/1.2.0"
            ),
            "lane": "sft",
            "basis": "assistant_target_source_token_proxy_count",
            "sourceFamily": "fleet_orchestration_native",
            "taskType": "fleet_orchestration_event_graph",
            "minimumBasisPoints": 5_200,
            "maximumBasisPoints": 5_800,
            "selectionPolicy": (
                "retain_non_public_then_bound_public_behavioral"
            ),
            "authoritativeExactBandBasisPoints": authoritative_exact_band,
        }
        or authoritative_exact_band != {"minimum": 5_000, "maximum": 6_000}
        or source_proxy_contract.get("schemaVersion")
        != "lumen.source-token-proxy/1.0.0"
        or source_proxy_contract.get("status")
        != "source_side_selection_proxy_not_exact_token_count"
        or source_proxy_contract.get("strategy")
        != "max_whitespace_terms_utf8_byte_ceiling"
        or type(source_proxy_contract.get("maxCharsPerToken")) is not int
        or source_proxy_contract["maxCharsPerToken"] <= 0
        or source_proxy_contract.get("exactPinnedTokenizerAuthoritative") is not True
        or source_proxy_contract.get("authoritativeEnforcementPhase")
        != "post_tokenizer_load_pre_optimizer"
    ):
        raise RuntimeError("Fleet source-selection proxy contract drifted")

    dpo_tokenization_policy = _require_exact_mapping_keys(
        contract.get("dpoTokenizationPolicy"),
        set(FLEET_DPO_TOKENIZATION_POLICY),
        label="Fleet DPO tokenization policy",
    )
    if dict(dpo_tokenization_policy) != FLEET_DPO_TOKENIZATION_POLICY:
        raise RuntimeError("Fleet DPO tokenization policy drifted")

    family_share = _require_exact_mapping_keys(
        contract.get("optimizerFamilyShareBands"),
        {
            "schemaVersion",
            "enforcementScope",
            "classification",
            "lanes",
            "policySignalTokenClassificationByLane",
            "policySignalTokenLanes",
            "comparisonRules",
            "failurePolicy",
        },
        label="Fleet optimizer-family share bands",
    )
    classification = _require_exact_mapping_keys(
        family_share.get("classification"),
        {"sourceFamily", "taskTypeByLane"},
        label="Fleet optimizer-family classification",
    )
    task_types = _require_exact_mapping_keys(
        classification.get("taskTypeByLane"),
        {"sft", "dpo"},
        label="Fleet optimizer-family task types",
    )
    family_lanes = _require_exact_mapping_keys(
        family_share.get("lanes"),
        {"sft", "dpo"},
        label="Fleet optimizer-family lane bands",
    )
    policy_token_classifications = _require_exact_mapping_keys(
        family_share.get("policySignalTokenClassificationByLane"),
        {"sft", "dpo"},
        label="Fleet policy-signal token classifications",
    )
    policy_token_lanes = _require_exact_mapping_keys(
        family_share.get("policySignalTokenLanes"),
        {"sft", "dpo"},
        label="Fleet policy-signal token lane bands",
    )
    if (
        family_share.get("schemaVersion")
        != FLEET_OPTIMIZER_FAMILY_SHARE_SCHEMA
        or family_share.get("enforcementScope") != "optimizer_train_only"
        or classification.get("sourceFamily")
        != FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY
        or dict(task_types) != FLEET_NATIVE_ORCHESTRATION_TASK_TYPE_BY_LANE
        or family_share.get("comparisonRules")
        != FLEET_OPTIMIZER_FAMILY_SHARE_COMPARISON_RULES
        or family_share.get("failurePolicy") != "abort_before_optimizer"
    ):
        raise RuntimeError("Fleet optimizer-family share contract drifted")
    for expected_lane, expected_band in (
        FLEET_OPTIMIZER_FAMILY_SHARE_LANES.items()
    ):
        actual_band = _require_exact_mapping_keys(
            family_lanes.get(expected_lane),
            set(expected_band),
            label=f"Fleet {expected_lane} optimizer-family share band",
        )
        if any(
            type(actual_band[field]) is not type(expected_value)
            or actual_band[field] != expected_value
            for field, expected_value in expected_band.items()
        ):
            raise RuntimeError(
                f"Fleet {expected_lane} optimizer-family share band drifted"
            )
    for expected_lane, expected_pairs in (
        FLEET_POLICY_SIGNAL_TOKEN_CLASSIFICATIONS.items()
    ):
        actual_pairs = policy_token_classifications.get(expected_lane)
        if not isinstance(actual_pairs, list):
            raise RuntimeError(
                f"Fleet {expected_lane} policy-signal classifications drifted"
            )
        normalized_pairs: list[tuple[str, str]] = []
        for index, pair in enumerate(actual_pairs):
            bound_pair = _require_exact_mapping_keys(
                pair,
                {"sourceFamily", "taskType"},
                label=(
                    f"Fleet {expected_lane} policy-signal classification "
                    f"{index}"
                ),
            )
            source_family = bound_pair.get("sourceFamily")
            task_type = bound_pair.get("taskType")
            if not isinstance(source_family, str) or not isinstance(
                task_type,
                str,
            ):
                raise RuntimeError(
                    f"Fleet {expected_lane} policy-signal classification drifted"
                )
            normalized_pairs.append((source_family, task_type))
        if tuple(normalized_pairs) != expected_pairs:
            raise RuntimeError(
                f"Fleet {expected_lane} policy-signal classifications drifted"
            )
    for expected_lane, expected_band in (
        FLEET_POLICY_SIGNAL_TOKEN_SHARE_LANES.items()
    ):
        actual_band = _require_exact_mapping_keys(
            policy_token_lanes.get(expected_lane),
            set(expected_band),
            label=f"Fleet {expected_lane} policy-signal token share band",
        )
        if any(
            type(actual_band[field]) is not type(expected_value)
            or actual_band[field] != expected_value
            for field, expected_value in expected_band.items()
        ):
            raise RuntimeError(
                f"Fleet {expected_lane} policy-signal token share band drifted"
            )

    _validated_fleet_sft_optimizer_window_schedule_contract(
        contract.get("sftOptimizerWindowScheduleContract"),
        sft_family_band=family_lanes["sft"],
    )

    record_geometry = _require_exact_mapping_keys(
        contract.get("sftOptimizerRecordGeometryContract"),
        {
            "schemaVersion",
            "lane",
            "maximumTrainRecords",
            "protectedSourceRole",
            "removableSourceRoles",
            "selectionPolicy",
            "failurePolicy",
        },
        label="Fleet SFT optimizer-record geometry contract",
    )
    if dict(record_geometry) != {
        "schemaVersion": FLEET_SFT_OPTIMIZER_RECORD_GEOMETRY_SCHEMA,
        "lane": "sft",
        "maximumTrainRecords": FLEET_SFT_OPTIMIZER_MAX_TRAIN_RECORDS,
        "protectedSourceRole": "behavioral_primary",
        "removableSourceRoles": [
            "supplemental_static",
            "public_behavioral",
        ],
        "selectionPolicy": (
            "largest_deterministic_cap_valid_cohort_from_immutable_"
            "candidates"
        ),
        "failurePolicy": "abort_generation_before_optimizer",
    }:
        raise RuntimeError(
            "Fleet SFT optimizer-record geometry contract drifted"
        )
    if config is not None:
        optimization_policy = config.get("optimizationStepPolicy")
        sft_policy = (
            optimization_policy.get("sft")
            if isinstance(optimization_policy, Mapping)
            else None
        )
        train_record_count = (
            sft_policy.get("trainRecordCount")
            if isinstance(sft_policy, Mapping)
            else None
        )
        if (
            type(train_record_count) is not int
            or train_record_count <= 0
            or train_record_count
            > FLEET_SFT_OPTIMIZER_MAX_TRAIN_RECORDS
        ):
            raise RuntimeError(
                "Fleet SFT optimizer-record geometry drifted from the "
                "training config"
            )

    accounting = _require_exact_mapping_keys(
        contract.get("tokenAccounting"),
        {"sft", "dpo"},
        label="Fleet token-accounting contract",
    )
    if accounting != {
        "sft": "assistant_mask_non_ignored_token_count",
        "dpo": (
            "rendered_chosen_completion_tokens_add_special_tokens_false_"
            "plus_one_trl_0_24_0_appended_eos"
        ),
    }:
        raise RuntimeError("Fleet token-accounting contract drifted")

    tokenizer_binding = _require_exact_mapping_keys(
        contract.get("tokenizer"),
        {
            "baseModelID",
            "baseModelRevision",
            "tokenizerSHA256",
            "tokenizerClosureSHA256",
        },
        label="Fleet tokenizer binding",
    )
    if (
        not isinstance(tokenizer_binding.get("baseModelID"), str)
        or not tokenizer_binding["baseModelID"]
        or not isinstance(tokenizer_binding.get("baseModelRevision"), str)
        or re.fullmatch(
            r"[0-9a-f]{40}", str(tokenizer_binding["baseModelRevision"])
        )
        is None
        or re.fullmatch(r"[0-9a-f]{64}", str(tokenizer_binding.get("tokenizerSHA256")))
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(tokenizer_binding.get("tokenizerClosureSHA256")),
        )
        is None
    ):
        raise RuntimeError("Fleet tokenizer binding is malformed")
    if config is not None and (
        tokenizer_binding.get("baseModelID") != config.get("base_model_name")
        or tokenizer_binding.get("baseModelRevision")
        != config.get("baseModelRevision")
        or tokenizer_binding.get("tokenizerSHA256")
        != config.get("baseModelTokenizerDigest")
        or tokenizer_binding.get("tokenizerClosureSHA256")
        != config.get("baseModelTokenizerClosureSHA256")
    ):
        raise RuntimeError("Fleet tokenizer binding drifted from the training config")

    evidence_contract = _require_exact_mapping_keys(
        contract.get("exactTokenEvidenceContract"),
        {
            "required",
            "schemaVersion",
            "statusAtGeneration",
            "tokenizer",
            "comparisonRule",
            "lanes",
        },
        label="Fleet exact-token evidence contract",
    )
    if (
        evidence_contract.get("required") is not True
        or evidence_contract.get("schemaVersion")
        != FLEET_LOSS_SHARE_EVIDENCE_SCHEMA
        or evidence_contract.get("statusAtGeneration")
        != "pending_exact_tokenizer_preflight"
        or evidence_contract.get("tokenizer") != "pinned_qwen_tokenizer"
        or evidence_contract.get("comparisonRule")
        != (
            "numeratorTokenCount*basisPointDenominator<="
            "denominatorTokenCount*capBasisPoints"
        )
    ):
        raise RuntimeError("Fleet exact-token evidence contract drifted")
    lanes = _require_exact_mapping_keys(
        evidence_contract.get("lanes"),
        {"sft", "dpo"},
        label="Fleet exact-token evidence lanes",
    )
    for expected_lane, expected_fields in FLEET_LOSS_SHARE_FIELD_NAMES.items():
        actual_fields = _require_exact_mapping_keys(
            lanes.get(expected_lane),
            set(expected_fields),
            label=f"Fleet {expected_lane} exact-token evidence fields",
        )
        if dict(actual_fields) != expected_fields:
            raise RuntimeError(
                f"Fleet {expected_lane} exact-token evidence field names drifted"
            )

    registry = _require_exact_mapping_keys(
        contract.get("sourceRoleRegistry"),
        {
            "schemaVersion",
            "unknownPairs",
            "categories",
            "registeredPairs",
            "publicBehavioralRule",
        },
        label="Fleet source-role registry",
    )
    if (
        registry.get("schemaVersion") != FLEET_SOURCE_ROLE_SCHEMA
        or registry.get("unknownPairs") != "hard_fail"
        or registry.get("categories") != list(FLEET_SOURCE_ROLES)
    ):
        raise RuntimeError("Fleet source-role registry control fields drifted")
    registered_pairs = registry.get("registeredPairs")
    if not isinstance(registered_pairs, list) or not registered_pairs:
        raise RuntimeError("Fleet source-role registry has no registered pairs")
    observed_pairs: set[tuple[str, str]] = set()
    observed_registered_categories: set[str] = set()
    for index, item in enumerate(registered_pairs):
        pair = _require_exact_mapping_keys(
            item,
            {"sourceFamily", "taskType", "category"},
            label=f"Fleet registered source-role pair {index}",
        )
        source_family = pair.get("sourceFamily")
        task_type = pair.get("taskType")
        category = pair.get("category")
        if (
            not isinstance(source_family, str)
            or not source_family
            or source_family.strip() != source_family
            or not isinstance(task_type, str)
            or not task_type
            or task_type.strip() != task_type
            or category not in FLEET_SOURCE_ROLES
            or category == "public_behavioral"
            or source_family.startswith("public_adapter_corpus_")
            or (source_family, task_type) in observed_pairs
        ):
            raise RuntimeError("Fleet source-role registry contains an invalid pair")
        observed_pairs.add((source_family, task_type))
        observed_registered_categories.add(category)
    if observed_registered_categories != {
        "behavioral_primary",
        "supplemental_static",
    }:
        raise RuntimeError(
            "Fleet source-role registry must contain primary and static pairs"
        )
    public_rule = _require_exact_mapping_keys(
        registry.get("publicBehavioralRule"),
        {
            "sourceFamilyPrefix",
            "taskType",
            "requiresPublicCorpusLineage",
        },
        label="Fleet public-behavioral source rule",
    )
    if public_rule != {
        "sourceFamilyPrefix": "public_adapter_corpus_",
        "taskType": "public_capability_delegation",
        "requiresPublicCorpusLineage": True,
    }:
        raise RuntimeError("Fleet public-behavioral source rule drifted")
    return dict(contract)


def _fleet_source_role_from_contract(
    record: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> tuple[str, str, str]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Fleet loss-share rows require object metadata")
    source_family = metadata.get("sourceFamily")
    task_type = metadata.get("taskType")
    if (
        not isinstance(source_family, str)
        or not source_family
        or source_family.strip() != source_family
        or not isinstance(task_type, str)
        or not task_type
        or task_type.strip() != task_type
    ):
        raise RuntimeError(
            "Fleet loss-share rows require canonical metadata.sourceFamily and "
            "metadata.taskType"
        )
    registry = contract["sourceRoleRegistry"]
    registered = {
        (item["sourceFamily"], item["taskType"]): item["category"]
        for item in registry["registeredPairs"]
    }
    category = registered.get((source_family, task_type))
    if category is None:
        public_rule = registry["publicBehavioralRule"]
        public_corpus = metadata.get("publicCorpus")
        if (
            source_family.startswith(public_rule["sourceFamilyPrefix"])
            and task_type == public_rule["taskType"]
            and isinstance(public_corpus, Mapping)
            and bool(public_corpus)
        ):
            category = "public_behavioral"
        else:
            raise RuntimeError(
                "Unregistered Fleet source-role pair: "
                f"sourceFamily={source_family!r}, taskType={task_type!r}"
            )
    return source_family, task_type, category


def _fleet_cap_passes(
    *,
    numerator: int,
    denominator: int,
    cap_basis_points: int,
) -> bool:
    return (
        type(numerator) is int
        and numerator >= 0
        and type(denominator) is int
        and denominator > 0
        and type(cap_basis_points) is int
        and 0 <= cap_basis_points <= FLEET_LOSS_SHARE_BASIS_POINT_DENOMINATOR
        and numerator * FLEET_LOSS_SHARE_BASIS_POINT_DENOMINATOR
        <= denominator * cap_basis_points
    )


def _fleet_optimizer_family_band_passes(
    *,
    numerator: int,
    denominator: int,
    minimum_basis_points: int,
    maximum_basis_points: int,
) -> bool:
    return (
        type(numerator) is int
        and numerator >= 0
        and type(denominator) is int
        and denominator > 0
        and type(minimum_basis_points) is int
        and type(maximum_basis_points) is int
        and 0 <= minimum_basis_points <= maximum_basis_points
        <= FLEET_LOSS_SHARE_BASIS_POINT_DENOMINATOR
        and numerator * FLEET_LOSS_SHARE_BASIS_POINT_DENOMINATOR
        >= denominator * minimum_basis_points
        and numerator * FLEET_LOSS_SHARE_BASIS_POINT_DENOMINATOR
        <= denominator * maximum_basis_points
    )


def _fleet_sft_schedule_controls(
    config: Mapping[str, Any] | None,
) -> tuple[int, int, int, int]:
    if not isinstance(config, Mapping):
        raise RuntimeError(
            "Fleet SFT optimizer-window scheduling requires training config"
        )
    seed = config.get("seed")
    batch_size = config.get("batch_size")
    gradient_accumulation_steps = config.get(
        "gradient_accumulation_steps"
    )
    configured_epochs = config.get("num_train_epochs")
    if type(seed) is not int:
        raise RuntimeError("Fleet SFT schedule requires an integer seed")
    if type(batch_size) is not int or batch_size != 1:
        raise RuntimeError(
            "Fleet SFT schedule requires batch_size=1"
        )
    if (
        type(gradient_accumulation_steps) is not int
        or gradient_accumulation_steps <= 0
    ):
        raise RuntimeError(
            "Fleet SFT schedule requires positive integer gradient accumulation"
        )
    if type(configured_epochs) is not int or configured_epochs <= 0:
        raise RuntimeError(
            "Fleet SFT schedule requires a positive integer epoch count"
        )
    if config.get("packing", False) is not False:
        raise RuntimeError("Fleet SFT optimizer-window scheduling forbids packing")
    return seed, batch_size, gradient_accumulation_steps, configured_epochs


def _fleet_sft_schedule_rank(
    *,
    seed: int,
    role: str,
    source_row_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "algorithm": FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_ALGORITHM,
            "seed": seed,
            "role": role,
            "sourceRowSHA256": source_row_sha256,
        }
    )


def _fleet_sft_cycle_take(
    values: list[int],
    *,
    count: int,
    epoch_index: int,
    candidate_index: int,
) -> list[int]:
    if not values or count <= 0:
        return []
    offset = (epoch_index * count + candidate_index) % len(values)
    return [values[(offset + index) % len(values)] for index in range(count)]


def _fleet_sft_epoch_order(
    *,
    row_token_evidence: list[Mapping[str, Any]],
    native_indices: list[int],
    non_native_indices: list[int],
    optimizer_window_record_capacity: int,
    seed: int,
    epoch_index: int,
    candidate_index: int,
) -> list[int]:
    record_count = len(row_token_evidence)
    native_samples = _fleet_sft_cycle_take(
        native_indices,
        count=len(native_indices),
        epoch_index=epoch_index,
        candidate_index=candidate_index,
    )
    non_native_samples = _fleet_sft_cycle_take(
        non_native_indices,
        count=len(non_native_indices),
        epoch_index=epoch_index,
        candidate_index=candidate_index,
    )
    window_sizes = [
        min(optimizer_window_record_capacity, record_count - start)
        for start in range(0, record_count, optimizer_window_record_capacity)
    ]
    window_order = sorted(
        range(len(window_sizes)),
        key=lambda window_index: (
            _canonical_sha256(
                {
                    "algorithm": (
                        FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_ALGORITHM
                    ),
                    "seed": seed,
                    "epochIndex": epoch_index,
                    "candidateIndex": candidate_index,
                    "role": "optimizer_window",
                    "windowIndex": window_index,
                }
            ),
            window_index,
        ),
    )
    windows: list[list[int]] = [[] for _ in window_sizes]

    cursor = 0
    for row_index in native_samples:
        for _ in window_order:
            window_index = window_order[cursor % len(window_order)]
            cursor += 1
            if len(windows[window_index]) < window_sizes[window_index]:
                windows[window_index].append(row_index)
                break
        else:  # pragma: no cover - guarded by exact schedule geometry
            raise RuntimeError("Fleet SFT native schedule exceeded window capacity")

    non_native_window_cursor = 0
    for row_index in non_native_samples:
        for _ in window_order:
            window_index = window_order[
                non_native_window_cursor % len(window_order)
            ]
            non_native_window_cursor += 1
            if len(windows[window_index]) < window_sizes[window_index]:
                windows[window_index].append(row_index)
                break
        else:  # pragma: no cover - guarded by exact schedule geometry
            raise RuntimeError(
                "Fleet SFT non-native schedule exceeded window capacity"
            )

    for window_index, window in enumerate(windows):
        ranked_occurrences = sorted(
            enumerate(window),
            key=lambda item: (
                _canonical_sha256(
                    {
                        "algorithm": (
                            FLEET_SFT_OPTIMIZER_WINDOW_SCHEDULE_ALGORITHM
                        ),
                        "seed": seed,
                        "epochIndex": epoch_index,
                        "candidateIndex": candidate_index,
                        "role": "window_record",
                        "windowIndex": window_index,
                        "sourceRowSHA256": row_token_evidence[item[1]][
                            "sourceRowSHA256"
                        ],
                    }
                ),
                str(
                    row_token_evidence[item[1]]["sourceRowSHA256"]
                ),
                item[0],
            ),
        )
        windows[window_index] = [row_index for _, row_index in ranked_occurrences]
    order = [row_index for window in windows for row_index in window]
    if (
        len(order) != record_count
        or len(set(order)) != record_count
        or set(order) != set(range(record_count))
    ):
        raise RuntimeError(
            "Fleet SFT schedule is not a strict source-row permutation"
        )
    return order


def _fleet_sft_epoch_window_evidence(
    *,
    row_token_evidence: list[Mapping[str, Any]],
    record_indices: list[int],
    optimizer_window_record_capacity: int,
    native_source_family: str,
    native_task_type: str,
    epoch_index: int,
    candidate_index: int,
) -> dict[str, Any]:
    native_source_indices = {
        int(row["rowIndex"])
        for row in row_token_evidence
        if row.get("sourceFamily") == native_source_family
        and row.get("taskType") == native_task_type
    }
    non_native_source_indices = set(range(len(row_token_evidence))) - (
        native_source_indices
    )
    sampled_native = [
        row_index
        for row_index in record_indices
        if row_index in native_source_indices
    ]
    sampled_non_native = [
        row_index
        for row_index in record_indices
        if row_index in non_native_source_indices
    ]
    windows: list[dict[str, Any]] = []
    for window_index, start in enumerate(
        range(0, len(record_indices), optimizer_window_record_capacity)
    ):
        window_indices = record_indices[
            start : start + optimizer_window_record_capacity
        ]
        native_target_tokens = sum(
            int(row_token_evidence[row_index]["targetTokenCount"])
            for row_index in window_indices
            if row_index in native_source_indices
        )
        all_target_tokens = sum(
            int(row_token_evidence[row_index]["targetTokenCount"])
            for row_index in window_indices
        )
        if all_target_tokens <= 0:
            raise RuntimeError("Fleet SFT schedule contains an empty-loss window")
        share_basis_points = (
            native_target_tokens * FLEET_LOSS_SHARE_BASIS_POINT_DENOMINATOR
        ) // all_target_tokens
        windows.append(
            {
                "windowIndex": window_index,
                "rowIndices": window_indices,
                "recordCount": len(window_indices),
                "nativeTargetTokenCount": native_target_tokens,
                "assistantTargetTokenCount": all_target_tokens,
                "nativeShareBasisPoints": share_basis_points,
            }
        )
    window_share_sum = sum(
        window["nativeShareBasisPoints"] for window in windows
    )
    unique_native = set(sampled_native)
    unique_non_native = set(sampled_non_native)
    shares = [window["nativeShareBasisPoints"] for window in windows]
    return {
        "epochIndex": epoch_index,
        "candidateIndex": candidate_index,
        "recordIndicesSHA256": _canonical_sha256(record_indices),
        "windowEvidenceSHA256": _canonical_sha256(windows),
        "firstOptimizerWindowRecordIndicesSHA256": _canonical_sha256(
            windows[0]["rowIndices"]
        ),
        "firstOptimizerWindowTargetTokenCount": windows[0][
            "assistantTargetTokenCount"
        ],
        "sampledRecordCount": len(record_indices),
        "nativeSampleCount": len(sampled_native),
        "nonNativeSampleCount": len(sampled_non_native),
        "uniqueNativeSourceRecordCount": len(unique_native),
        "uniqueNonNativeSourceRecordCount": len(unique_non_native),
        "repeatedNativeSampleCount": len(sampled_native) - len(unique_native),
        "repeatedNonNativeSampleCount": (
            len(sampled_non_native) - len(unique_non_native)
        ),
        "omittedNativeSourceRecordCount": (
            len(native_source_indices) - len(unique_native)
        ),
        "omittedNonNativeSourceRecordCount": (
            len(non_native_source_indices) - len(unique_non_native)
        ),
        "optimizerWindowCount": len(windows),
        "optimizerWindowsWithNativeSamples": sum(
            1 for window in windows if window["nativeTargetTokenCount"] > 0
        ),
        "optimizerWindowsWithoutNativeSamples": sum(
            1 for window in windows if window["nativeTargetTokenCount"] == 0
        ),
        "windowShareBasisPointSum": window_share_sum,
        "windowShareBasisPointCount": len(windows),
        "windowNormalizedNativeShareBasisPoints": (
            window_share_sum // len(windows)
        ),
        "minimumWindowNativeShareBasisPoints": min(shares),
        "maximumWindowNativeShareBasisPoints": max(shares),
    }


def _build_fleet_sft_optimizer_window_schedule(
    *,
    row_token_evidence: list[Mapping[str, Any]],
    config: Mapping[str, Any] | None,
    schedule_contract: Mapping[str, Any],
    minimum_basis_points: int,
    maximum_basis_points: int,
) -> tuple[dict[str, Any], list[list[int]]]:
    validated_schedule_contract = (
        _validated_fleet_sft_optimizer_window_schedule_contract(
            schedule_contract,
            sft_family_band={
                "minimumBasisPoints": minimum_basis_points,
                "maximumBasisPoints": maximum_basis_points,
            },
        )
    )
    seed, batch_size, gradient_accumulation_steps, configured_epochs = (
        _fleet_sft_schedule_controls(config)
    )
    if (
        not isinstance(row_token_evidence, list)
        or len(row_token_evidence) < 2
        or type(minimum_basis_points) is not int
        or type(maximum_basis_points) is not int
        or not 0
        <= minimum_basis_points
        <= maximum_basis_points
        <= FLEET_LOSS_SHARE_BASIS_POINT_DENOMINATOR
    ):
        raise RuntimeError("Fleet SFT schedule inputs are invalid")
    for row_index, row in enumerate(row_token_evidence):
        if (
            not isinstance(row, Mapping)
            or row.get("rowIndex") != row_index
            or type(row.get("targetTokenCount")) is not int
            or row["targetTokenCount"] <= 0
            or not isinstance(row.get("sourceRowSHA256"), str)
        ):
            raise RuntimeError("Fleet SFT schedule row evidence is malformed")
    native_indices = [
        row_index
        for row_index, row in enumerate(row_token_evidence)
        if row.get("sourceFamily") == FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY
        and row.get("taskType")
        == FLEET_NATIVE_ORCHESTRATION_TASK_TYPE_BY_LANE["sft"]
    ]
    native_index_set = set(native_indices)
    non_native_indices = [
        row_index
        for row_index in range(len(row_token_evidence))
        if row_index not in native_index_set
    ]
    if not native_indices or not non_native_indices:
        raise RuntimeError(
            "Fleet SFT schedule requires native and non-native source rows"
        )
    native_indices.sort(
        key=lambda row_index: (
            _fleet_sft_schedule_rank(
                seed=seed,
                role="native_source_record",
                source_row_sha256=str(
                    row_token_evidence[row_index]["sourceRowSHA256"]
                ),
            ),
            str(row_token_evidence[row_index]["sourceRowSHA256"]),
            row_index,
        )
    )
    non_native_indices.sort(
        key=lambda row_index: (
            int(row_token_evidence[row_index]["targetTokenCount"]),
            _fleet_sft_schedule_rank(
                seed=seed,
                role="non_native_source_record",
                source_row_sha256=str(
                    row_token_evidence[row_index]["sourceRowSHA256"]
                ),
            ),
            str(row_token_evidence[row_index]["sourceRowSHA256"]),
            row_index,
        )
    )
    optimizer_window_record_capacity = (
        batch_size * gradient_accumulation_steps
    )
    epoch_evidence: list[dict[str, Any]] = []
    epoch_orders: list[list[int]] = []
    sampled_across_epochs: set[int] = set()
    record_count = len(row_token_evidence)
    midpoint_basis_point_sum_factor = (
        minimum_basis_points + maximum_basis_points
    )
    for epoch_index in range(configured_epochs):
        candidates: list[
            tuple[tuple[int, int], dict[str, Any], list[int]]
        ] = []
        for candidate_index in range(
            validated_schedule_contract["candidateSearchCount"]
        ):
            record_indices = _fleet_sft_epoch_order(
                row_token_evidence=row_token_evidence,
                native_indices=native_indices,
                non_native_indices=non_native_indices,
                optimizer_window_record_capacity=(
                    optimizer_window_record_capacity
                ),
                seed=seed,
                epoch_index=epoch_index,
                candidate_index=candidate_index,
            )
            observed = _fleet_sft_epoch_window_evidence(
                row_token_evidence=row_token_evidence,
                record_indices=record_indices,
                optimizer_window_record_capacity=(
                    optimizer_window_record_capacity
                ),
                native_source_family=(
                    FLEET_NATIVE_ORCHESTRATION_SOURCE_FAMILY
                ),
                native_task_type=(
                    FLEET_NATIVE_ORCHESTRATION_TASK_TYPE_BY_LANE["sft"]
                ),
                epoch_index=epoch_index,
                candidate_index=candidate_index,
            )
            if (
                observed["nativeSampleCount"] != len(native_indices)
                or observed["nonNativeSampleCount"]
                != len(non_native_indices)
                or observed["uniqueNativeSourceRecordCount"]
                != len(native_indices)
                or observed["uniqueNonNativeSourceRecordCount"]
                != len(non_native_indices)
                or observed["repeatedNativeSampleCount"] != 0
                or observed["repeatedNonNativeSampleCount"] != 0
                or observed["omittedNativeSourceRecordCount"] != 0
                or observed["omittedNonNativeSourceRecordCount"] != 0
            ):
                raise RuntimeError(
                    "Fleet SFT schedule violated strict permutation evidence"
                )
            share_sum = observed["windowShareBasisPointSum"]
            share_count = observed["windowShareBasisPointCount"]
            if not (
                share_sum >= minimum_basis_points * share_count
                and share_sum <= maximum_basis_points * share_count
            ):
                continue
            candidates.append(
                (
                    (
                        abs(
                            2 * share_sum
                            - midpoint_basis_point_sum_factor * share_count
                        ),
                        candidate_index,
                    ),
                    observed,
                    record_indices,
                )
            )
        if not candidates:
            raise RuntimeError(
                "Fleet SFT optimizer-window native share cannot satisfy the "
                "controlled band with a strict source-row permutation"
            )
        _, selected_evidence, selected_order = min(
            candidates,
            key=lambda candidate: candidate[0],
        )
        epoch_evidence.append(selected_evidence)
        epoch_orders.append(selected_order)
        sampled_across_epochs.update(selected_order)

    schedule: dict[str, Any] = {
        "schemaVersion": validated_schedule_contract[
            "evidenceSchemaVersion"
        ],
        "scheduleContractSchemaVersion": validated_schedule_contract[
            "schemaVersion"
        ],
        "scheduleContractSHA256": _canonical_sha256(
            validated_schedule_contract
        ),
        "status": "passed",
        "lane": validated_schedule_contract["lane"],
        "split": validated_schedule_contract["split"],
        "enforcementRequired": validated_schedule_contract[
            "enforcementRequired"
        ],
        "enforcementPhase": validated_schedule_contract[
            "enforcementPhase"
        ],
        "basis": validated_schedule_contract["basis"],
        "basisPointDenominator": validated_schedule_contract[
            "basisPointDenominator"
        ],
        "algorithm": validated_schedule_contract["algorithm"],
        "comparisonRule": (
            "windowShareBasisPointSum between minimumBasisPoints*"
            "windowShareBasisPointCount and maximumBasisPoints*"
            "windowShareBasisPointCount inclusive"
        ),
        "permutationPolicy": validated_schedule_contract[
            "permutationPolicy"
        ],
        "candidateSearchCount": validated_schedule_contract[
            "candidateSearchCount"
        ],
        "distributedSamplingPolicy": validated_schedule_contract[
            "distributedSamplingPolicy"
        ],
        "resumePolicy": validated_schedule_contract["resumePolicy"],
        "packing": validated_schedule_contract["packing"],
        "failurePolicy": validated_schedule_contract["failurePolicy"],
        "seed": seed,
        "perDeviceTrainBatchSize": batch_size,
        "gradientAccumulationSteps": gradient_accumulation_steps,
        "optimizerWindowRecordCapacity": optimizer_window_record_capacity,
        "configuredEpochs": configured_epochs,
        "datasetRecordsPerEpoch": record_count,
        "nativeSourceRecordCount": len(native_indices),
        "nonNativeSourceRecordCount": len(non_native_indices),
        "minimumBasisPoints": validated_schedule_contract[
            "minimumBasisPoints"
        ],
        "maximumBasisPoints": validated_schedule_contract[
            "maximumBasisPoints"
        ],
        "sourceRowsSHA256": _canonical_sha256(
            [row["sourceRowSHA256"] for row in row_token_evidence]
        ),
        "uniqueSourceRecordsAcrossConfiguredEpochs": len(
            sampled_across_epochs
        ),
        "allSourceRecordsCoveredAcrossConfiguredEpochs": (
            len(sampled_across_epochs) == record_count
        ),
        "samplerSetEpochRequired": True,
        "epochs": epoch_evidence,
    }
    schedule["scheduleSHA256"] = _canonical_sha256(schedule)
    return schedule, epoch_orders


class _FleetEpochStratifiedSampler:
    def __init__(self, epoch_orders: list[list[int]]) -> None:
        expected_indices = (
            set(range(len(epoch_orders[0]))) if epoch_orders else set()
        )
        if (
            not epoch_orders
            or not epoch_orders[0]
            or any(len(order) != len(epoch_orders[0]) for order in epoch_orders)
            or any(set(order) != expected_indices for order in epoch_orders)
        ):
            raise RuntimeError("Fleet SFT sampler received invalid epoch schedules")
        self._epoch_orders = tuple(tuple(order) for order in epoch_orders)
        self._epoch = 0
        self._set_epoch_request_count = 0
        self._accepted_epoch_transition_count = 0
        self._idempotent_epoch_request_count = 0
        self._suppressed_lower_epoch_reset_count = 0
        self._last_requested_epoch: int | None = None
        self._last_suppressed_lower_epoch: int | None = None

    def __len__(self) -> int:
        return len(self._epoch_orders[0])

    def __iter__(self):
        return iter(self._epoch_orders[self._epoch])

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or not 0 <= epoch < len(self._epoch_orders):
            raise RuntimeError("Fleet SFT sampler received an invalid epoch")
        self._set_epoch_request_count += 1
        self._last_requested_epoch = epoch
        if epoch < self._epoch:
            # Transformers selects the resumed epoch on the prepared training
            # dataloader before Accelerate's skip_first_batches constructs a
            # replacement DataLoaderShard. In the pinned Accelerate runtime,
            # that replacement starts with iteration == 0 and calls
            # set_epoch(0) on this same sampler from __iter__. Preserve the
            # already-selected epoch; epoch progression within one Trainer
            # invocation is monotonic.
            self._suppressed_lower_epoch_reset_count += 1
            self._last_suppressed_lower_epoch = epoch
            return
        if epoch == self._epoch:
            self._idempotent_epoch_request_count += 1
            return
        self._epoch = epoch
        self._accepted_epoch_transition_count += 1

    def audit_state(self) -> dict[str, int | None]:
        return {
            "configuredEpochCount": len(self._epoch_orders),
            "activeEpoch": self._epoch,
            "setEpochRequestCount": self._set_epoch_request_count,
            "acceptedEpochTransitionCount": (
                self._accepted_epoch_transition_count
            ),
            "idempotentEpochRequestCount": (
                self._idempotent_epoch_request_count
            ),
            "suppressedLowerEpochResetCount": (
                self._suppressed_lower_epoch_reset_count
            ),
            "lastRequestedEpoch": self._last_requested_epoch,
            "lastSuppressedLowerEpoch": (
                self._last_suppressed_lower_epoch
            ),
        }

    def _snapshot_runtime_state(self) -> dict[str, int | None]:
        return dict(self.audit_state())

    def _restore_runtime_state(
        self,
        snapshot: Mapping[str, Any],
    ) -> None:
        expected_keys = set(self.audit_state())
        if (
            not isinstance(snapshot, Mapping)
            or set(snapshot) != expected_keys
            or snapshot.get("configuredEpochCount") != len(self._epoch_orders)
        ):
            raise RuntimeError("Fleet SFT sampler runtime snapshot is invalid")
        integer_fields = (
            "activeEpoch",
            "setEpochRequestCount",
            "acceptedEpochTransitionCount",
            "idempotentEpochRequestCount",
            "suppressedLowerEpochResetCount",
        )
        if any(
            type(snapshot.get(field)) is not int
            or int(snapshot[field]) < 0
            for field in integer_fields
        ):
            raise RuntimeError("Fleet SFT sampler runtime snapshot is malformed")
        optional_epoch_fields = (
            "lastRequestedEpoch",
            "lastSuppressedLowerEpoch",
        )
        if any(
            value is not None
            and (
                type(value) is not int
                or not 0 <= value < len(self._epoch_orders)
            )
            for value in (snapshot.get(field) for field in optional_epoch_fields)
        ):
            raise RuntimeError("Fleet SFT sampler epoch snapshot is malformed")
        if int(snapshot["activeEpoch"]) >= len(self._epoch_orders):
            raise RuntimeError("Fleet SFT sampler active epoch snapshot is invalid")
        self._epoch = int(snapshot["activeEpoch"])
        self._set_epoch_request_count = int(snapshot["setEpochRequestCount"])
        self._accepted_epoch_transition_count = int(
            snapshot["acceptedEpochTransitionCount"]
        )
        self._idempotent_epoch_request_count = int(
            snapshot["idempotentEpochRequestCount"]
        )
        self._suppressed_lower_epoch_reset_count = int(
            snapshot["suppressedLowerEpochResetCount"]
        )
        self._last_requested_epoch = snapshot["lastRequestedEpoch"]
        self._last_suppressed_lower_epoch = snapshot[
            "lastSuppressedLowerEpoch"
        ]
        if self.audit_state() != dict(snapshot):
            raise RuntimeError("Fleet SFT sampler runtime restoration failed")


def _capture_fleet_runtime_rng_state() -> dict[str, Any]:
    import torch  # type: ignore

    numpy_state: Any = None
    try:
        import numpy as np  # type: ignore

        numpy_state = np.random.get_state()
    except ImportError:
        pass
    return {
        "python": random.getstate(),
        "numpy": numpy_state,
        "torchCPU": torch.random.get_rng_state().clone(),
        "torchCUDA": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _restore_fleet_runtime_rng_state(state: Mapping[str, Any]) -> None:
    import torch  # type: ignore

    if not isinstance(state, Mapping) or set(state) != {
        "python",
        "numpy",
        "torchCPU",
        "torchCUDA",
    }:
        raise RuntimeError("Fleet runtime RNG snapshot is invalid")
    random.setstate(state["python"])
    if state["numpy"] is not None:
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - pinned image includes NumPy
            raise RuntimeError("Fleet runtime RNG restoration requires NumPy") from exc
        np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torchCPU"])
    torch.cuda.set_rng_state_all(state["torchCUDA"])


def _fleet_runtime_rng_states_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    import torch  # type: ignore

    if left.get("python") != right.get("python"):
        return False
    left_numpy = left.get("numpy")
    right_numpy = right.get("numpy")
    if (left_numpy is None) != (right_numpy is None):
        return False
    if left_numpy is not None:
        import numpy as np  # type: ignore

        if (
            left_numpy[0] != right_numpy[0]
            or not np.array_equal(left_numpy[1], right_numpy[1])
            or left_numpy[2:] != right_numpy[2:]
        ):
            return False
    if not torch.equal(left.get("torchCPU"), right.get("torchCPU")):
        return False
    left_cuda = left.get("torchCUDA")
    right_cuda = right.get("torchCUDA")
    return (
        isinstance(left_cuda, list)
        and isinstance(right_cuda, list)
        and len(left_cuda) == len(right_cuda)
        and all(torch.equal(a, b) for a, b in zip(left_cuda, right_cuda))
    )


def _runtime_nested_integer_rows(value: Any, *, label: str) -> list[list[int]]:
    for operation in ("detach", "cpu", "tolist"):
        method = getattr(value, operation, None)
        if callable(method):
            value = method()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"Fleet runtime {label} must be a non-empty tensor")
    if all(type(item) is int for item in value):
        return [list(value)]
    rows: list[list[int]] = []
    for row in value:
        if isinstance(row, tuple):
            row = list(row)
        if (
            not isinstance(row, list)
            or not row
            or any(type(item) is not int for item in row)
        ):
            raise RuntimeError(f"Fleet runtime {label} tensor shape is invalid")
        rows.append(list(row))
    return rows


def _runtime_shifted_sft_target_token_count(
    batch: Mapping[str, Any],
    *,
    expected_batch_size: int | None = None,
) -> int:
    if not isinstance(batch, Mapping) or "labels" not in batch:
        raise RuntimeError("Fleet runtime batch lacks labels")
    if (
        batch.get("packed_seq_lengths") is not None
        or batch.get("position_ids") is not None
    ):
        raise RuntimeError(
            "Fleet runtime normalization forbids padding-free or packed "
            "sequences"
        )
    labels = _runtime_nested_integer_rows(batch["labels"], label="labels")
    raw_attention = batch.get("attention_mask")
    if raw_attention is None:
        raise RuntimeError("Fleet runtime batch lacks attention_mask")
    attention = _runtime_nested_integer_rows(
        raw_attention,
        label="attention_mask",
    )
    if (
        expected_batch_size is not None
        and (
            type(expected_batch_size) is not int
            or expected_batch_size <= 0
            or len(labels) != expected_batch_size
        )
    ):
        raise RuntimeError("Fleet runtime batch size differs from config")
    if len(labels) != len(attention) or any(
        len(label_row) != len(attention_row)
        for label_row, attention_row in zip(labels, attention)
    ):
        raise RuntimeError("Fleet runtime labels and attention mask differ")
    return sum(
        1
        for label_row, attention_row in zip(labels, attention)
        for target, attended in zip(label_row[1:], attention_row[1:])
        if target != -100 and attended != 0
    )


def _runtime_positive_scalar_integer(value: Any, *, label: str) -> int:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numel = getattr(value, "numel", None)
    if callable(numel) and numel() != 1:
        raise RuntimeError(f"{label} must be scalar")
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _runtime_callable_code_sha256(value: Any, *, label: str) -> str:
    function = getattr(value, "__func__", value)
    code = getattr(function, "__code__", None)
    if code is None:
        raise RuntimeError(f"Fleet runtime {label} lacks inspectable code")
    try:
        return canonical_python_code_sha256(code)
    except ValueError as exc:
        raise RuntimeError(
            f"Fleet runtime {label} code object cannot be canonicalized"
        ) from exc


def _attest_fleet_sft_runtime_loss_normalization(
    trainer: Any,
    *,
    assistant_only_loss: bool,
    sft_config_padding_free: bool,
    config: Mapping[str, Any],
    row_token_evidence: list[Mapping[str, Any]],
    epoch_orders: list[list[int]],
    fleet_sampler: _FleetEpochStratifiedSampler,
) -> dict[str, Any]:
    """Prove the pinned trainer uses one token denominator per GA window."""

    _, batch_size, gradient_accumulation_steps, _ = (
        _fleet_sft_schedule_controls(config)
    )
    trainer_args = getattr(trainer, "args", None)
    accelerator = getattr(trainer, "accelerator", None)
    data_collator = getattr(trainer, "data_collator", None)
    get_batch_samples = getattr(trainer, "get_batch_samples", None)
    get_batch_samples_function = getattr(
        get_batch_samples,
        "__func__",
        get_batch_samples,
    )
    trainer_class = f"{type(trainer).__module__}.{type(trainer).__name__}"
    if (
        assistant_only_loss is not True
        or sft_config_padding_free is not False
        or trainer_class != FLEET_SFT_TRAINER_CLASS
        or trainer_args is None
        or type(
            getattr(trainer_args, "gradient_accumulation_steps", None)
        )
        is not int
        or trainer_args.gradient_accumulation_steps
        != gradient_accumulation_steps
        or type(getattr(trainer_args, "per_device_train_batch_size", None))
        is not int
        or trainer_args.per_device_train_batch_size != batch_size
        or type(getattr(trainer_args, "world_size", None)) is not int
        or trainer_args.world_size != 1
        or getattr(trainer_args, "packing", None) is not False
        or getattr(trainer_args, "padding_free", None) is not False
        or getattr(trainer_args, "dataloader_drop_last", None) is not False
        or getattr(trainer_args, "loss_type", None) != "nll"
        or getattr(trainer, "compute_loss_func", None) is not None
        or getattr(trainer, "model_accepts_loss_kwargs", None) is not True
        or getattr(trainer, "padding_free", None) is not False
        or data_collator is None
        or getattr(data_collator, "padding_free", None) is not False
        or accelerator is None
        or type(getattr(accelerator, "gradient_accumulation_steps", None))
        is not int
        or accelerator.gradient_accumulation_steps != 1
        or not callable(get_batch_samples)
        or getattr(get_batch_samples_function, "__name__", None)
        != FLEET_SFT_GET_BATCH_SAMPLES_NAME
        or getattr(get_batch_samples_function, "__module__", None)
        != FLEET_SFT_GET_BATCH_SAMPLES_MODULE
        or getattr(trainer, "optimizer", None) is not None
        or getattr(trainer, "lr_scheduler", None) is not None
    ):
        raise RuntimeError(
            "Fleet SFT runtime lacks the pinned token-normalized gradient "
            "accumulation path"
        )
    model = getattr(trainer, "model", None)
    get_base_model = getattr(model, "get_base_model", None)
    if not callable(get_base_model):
        raise RuntimeError("Fleet SFT runtime model is not PEFT-wrapped")
    base_model = get_base_model()
    base_config = getattr(base_model, "config", None)
    model_class = f"{type(model).__module__}.{type(model).__name__}"
    base_model_class = (
        f"{type(base_model).__module__}.{type(base_model).__name__}"
    )
    if (
        model_class != FLEET_SFT_MODEL_CLASS
        or base_model_class != FLEET_SFT_BASE_MODEL_CLASS
        or getattr(base_config, "model_type", None) != "qwen3"
    ):
        raise RuntimeError("Fleet SFT runtime base model is not pinned Qwen3")
    if (
        len(row_token_evidence) != len(getattr(trainer, "train_dataset", []))
        or not epoch_orders
        or len(epoch_orders[0]) != len(row_token_evidence)
        or len(row_token_evidence) < batch_size * gradient_accumulation_steps
    ):
        raise RuntimeError("Fleet SFT runtime normalization inputs are incomplete")
    for row_index, row in enumerate(row_token_evidence):
        if (
            not isinstance(row, Mapping)
            or row.get("rowIndex") != row_index
            or type(row.get("targetTokenCount")) is not int
            or row["targetTokenCount"] <= 0
        ):
            raise RuntimeError("Fleet SFT runtime row evidence is malformed")

    optimizer_window_capacity = batch_size * gradient_accumulation_steps
    scheduled_indices = epoch_orders[0][:optimizer_window_capacity]
    expected_micro_batch_counts = [
        sum(
            int(row_token_evidence[row_index]["targetTokenCount"])
            for row_index in scheduled_indices[start : start + batch_size]
        )
        for start in range(0, optimizer_window_capacity, batch_size)
    ]
    sampler_before = fleet_sampler._snapshot_runtime_state()
    rng_before = _capture_fleet_runtime_rng_state()
    epoch_iterator: Any = None
    batch_samples: list[Any] = []
    observed_denominator: Any = None
    restoration_verified = False
    try:
        train_dataloader = trainer.get_train_dataloader()
        epoch_iterator = iter(train_dataloader)
        batch_samples, observed_denominator = get_batch_samples(
            epoch_iterator,
            gradient_accumulation_steps,
            trainer_args.device,
        )
    finally:
        shutdown_workers = getattr(epoch_iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
        fleet_sampler._restore_runtime_state(sampler_before)
        _restore_fleet_runtime_rng_state(rng_before)
        rng_after = _capture_fleet_runtime_rng_state()
        restoration_verified = (
            fleet_sampler._snapshot_runtime_state() == sampler_before
            and _fleet_runtime_rng_states_equal(rng_before, rng_after)
        )
        if not restoration_verified:
            raise RuntimeError(
                "Fleet SFT runtime normalization probe changed training state"
            )

    if len(batch_samples) != gradient_accumulation_steps:
        raise RuntimeError(
            "Fleet SFT runtime probe did not materialize one full optimizer window"
        )
    observed_micro_batch_counts = [
        _runtime_shifted_sft_target_token_count(
            batch,
            expected_batch_size=batch_size,
        )
        for batch in batch_samples
    ]
    observed_micro_batch_sizes = [
        len(_runtime_nested_integer_rows(batch["labels"], label="labels"))
        for batch in batch_samples
    ]
    reconstructed_denominator = sum(observed_micro_batch_counts)
    reported_denominator = _runtime_positive_scalar_integer(
        observed_denominator,
        label="Fleet SFT num_items_in_batch",
    )
    expected_denominator = sum(expected_micro_batch_counts)
    if (
        observed_micro_batch_counts != expected_micro_batch_counts
        or reconstructed_denominator != expected_denominator
        or reported_denominator != reconstructed_denominator
    ):
        raise RuntimeError(
            "Fleet SFT runtime token denominator differs from the scheduled "
            "shifted-label evidence"
        )

    get_batch_samples_code_sha256 = _runtime_callable_code_sha256(
        get_batch_samples_function,
        label="get_batch_samples",
    )
    try:
        installed_callable_identity = (
            installed_distribution_python_callable_identity(
                distribution_name="unsloth-zoo",
                source_logical_path=(
                    FLEET_SFT_GET_BATCH_SAMPLES_SOURCE
                ),
                callable_name=FLEET_SFT_GET_BATCH_SAMPLES_NAME,
                resolved_environment=config.get(
                    "resolvedTrainingEnvironment"
                ),
            )
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Fleet SFT runtime callable identity cannot be reconstructed"
        ) from exc
    live_code = getattr(get_batch_samples_function, "__code__", None)
    if (
        live_code is None
        or installed_callable_identity.get("codeSHA256")
        != get_batch_samples_code_sha256
        or installed_callable_identity.get("callableQualname")
        != getattr(get_batch_samples_function, "__qualname__", None)
        or installed_callable_identity.get("callableFirstLineNumber")
        != getattr(live_code, "co_firstlineno", None)
    ):
        raise RuntimeError(
            "Fleet SFT live callable drifted from the installed wheel"
        )
    training_code_sha256 = _require_sha256(
        config.get("trainingCodeSHA256"),
        name="trainingCodeSHA256",
    )
    resolved_environment_sha256 = _require_sha256(
        config.get("resolvedTrainingEnvironmentSHA256"),
        name="resolvedTrainingEnvironmentSHA256",
    )
    training_environment_sha256 = _require_sha256(
        config.get("trainingEnvironmentSHA256"),
        name="trainingEnvironmentSHA256",
    )
    container_image_digest = _require_sha256(
        config.get("trainingContainerImageDigest"),
        name="trainingContainerImageDigest",
        prefix=True,
    )
    if (
        installed_callable_identity.get(
            "resolvedTrainingEnvironmentSHA256"
        )
        != resolved_environment_sha256
    ):
        raise RuntimeError(
            "Fleet SFT callable identity drifted from the resolved environment"
        )
    unsigned = {
        "schemaVersion": FLEET_SFT_RUNTIME_LOSS_NORMALIZATION_SCHEMA,
        "status": "passed",
        "enforcementPhase": "post_trainer_init_pre_optimizer",
        "countingRule": (
            "sum_shifted_non_ignored_labels_intersect_shifted_attention_mask"
        ),
        "trainingCodeSHA256": training_code_sha256,
        "resolvedTrainingEnvironmentSHA256": (
            resolved_environment_sha256
        ),
        "trainingEnvironmentSHA256": training_environment_sha256,
        "trainingContainerImageDigest": container_image_digest,
        "trainerClass": trainer_class,
        "modelClass": model_class,
        "baseModelClass": base_model_class,
        "baseModelType": "qwen3",
        "getBatchSamples": {
            "module": FLEET_SFT_GET_BATCH_SAMPLES_MODULE,
            "name": FLEET_SFT_GET_BATCH_SAMPLES_NAME,
            "installedCallableIdentity": installed_callable_identity,
        },
        "modelAcceptsLossKwargs": True,
        "lossType": "nll",
        "packing": False,
        "sftConfigPaddingFree": False,
        "trainerArgsPaddingFree": False,
        "trainerPaddingFree": False,
        "dataCollatorPaddingFree": False,
        "batchCollationMode": "padded_attention_mask",
        "packedSequenceLengthsPresent": False,
        "positionIDsPresent": False,
        "attentionMaskPresent": True,
        "observedMicroBatchSizes": observed_micro_batch_sizes,
        "worldSize": 1,
        "perDeviceTrainBatchSize": batch_size,
        "trainerGradientAccumulationSteps": gradient_accumulation_steps,
        "acceleratorGradientAccumulationSteps": 1,
        "optimizerWindowRecordCapacity": optimizer_window_capacity,
        "optimizerWindowMicroBatchCount": len(batch_samples),
        "scheduledRowIndicesSHA256": _canonical_sha256(scheduled_indices),
        "expectedMicroBatchTargetTokenCounts": expected_micro_batch_counts,
        "observedMicroBatchTargetTokenCounts": observed_micro_batch_counts,
        "expectedTargetTokenCount": expected_denominator,
        "reconstructedTargetTokenCount": reconstructed_denominator,
        "reportedNumItemsInBatch": reported_denominator,
        "samplerStateSHA256": _canonical_sha256(sampler_before),
        "samplerStatePreserved": restoration_verified,
        "rngStateRestored": restoration_verified,
        "preOptimizerStateVerified": True,
    }
    return {
        **unsigned,
        "runtimeLossNormalizationSHA256": _canonical_sha256(unsigned),
    }


def _validate_fleet_sft_trainer_args(training_args: Any) -> None:
    world_size = getattr(training_args, "world_size", None)
    per_device_train_batch_size = getattr(
        training_args,
        "per_device_train_batch_size",
        None,
    )
    if (
        type(world_size) is not int
        or world_size != 1
        or getattr(training_args, "ignore_data_skip", None) is not False
        or getattr(training_args, "dataloader_drop_last", None) is not False
        or type(per_device_train_batch_size) is not int
        or per_device_train_batch_size != 1
        or getattr(training_args, "packing", None) is not False
        or getattr(training_args, "padding_free", None) is not False
    ):
        raise RuntimeError(
            "Fleet SFT attested scheduling requires single-process Trainer "
            "resume semantics, per_device_train_batch_size=1, "
            "packing=False, padding_free=False, and dataloader_drop_last=False"
        )


def _apply_fleet_sft_batching_policy(
    sft_kwargs: dict[str, Any],
    *,
    enabled: bool,
) -> None:
    if type(enabled) is not bool:
        raise RuntimeError("Fleet SFT batching policy requires a boolean mode")
    if enabled:
        # The pinned Unsloth runtime changes an omitted padding_free value to
        # True. Fleet requires one record per micro-batch, so explicit padded
        # collation adds no within-batch padding and preserves an unambiguous
        # one-record denominator for the pre-optimizer loss probe.
        sft_kwargs["padding_free"] = False


def _build_fleet_loss_share_evidence(
    *,
    contract_value: Any,
    lane: str,
    split_target_rows: Mapping[
        str,
        list[tuple[Mapping[str, Any], int]],
    ],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _validated_fleet_loss_share_contract(
        contract_value,
        lane=lane,
        config=config,
    )
    if set(split_target_rows) != {"train", "validation"}:
        raise RuntimeError(
            "Fleet exact-token enforcement requires train and validation splits"
        )
    field_names = FLEET_LOSS_SHARE_FIELD_NAMES[lane]
    family_share_contract = contract["optimizerFamilyShareBands"]
    selected_family_band = family_share_contract["lanes"][lane]
    selected_policy_token_band = family_share_contract[
        "policySignalTokenLanes"
    ][lane]
    policy_signal_pairs = {
        (item["sourceFamily"], item["taskType"])
        for item in family_share_contract[
            "policySignalTokenClassificationByLane"
        ][lane]
    }
    native_source_family = family_share_contract["classification"][
        "sourceFamily"
    ]
    native_task_type = family_share_contract["classification"][
        "taskTypeByLane"
    ][lane]
    caps = contract["capsBasisPoints"]
    split_evidence: dict[str, Any] = {}
    for split in ("train", "validation"):
        rows = split_target_rows[split]
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(
                f"Fleet exact-token enforcement requires a non-empty {split} split"
            )
        target_by_category = {category: 0 for category in FLEET_SOURCE_ROLES}
        supplemental_by_family: dict[str, int] = {}
        native_target_tokens = 0
        native_preference_pairs = 0
        policy_signal_target_tokens = 0
        row_evidence: list[dict[str, Any]] = []
        for row_index, row_value in enumerate(rows):
            if (
                not isinstance(row_value, tuple)
                or len(row_value) != 2
                or not isinstance(row_value[0], Mapping)
            ):
                raise RuntimeError("Fleet exact-token row evidence is malformed")
            record, target_tokens = row_value
            if type(target_tokens) is not int or target_tokens <= 0:
                raise RuntimeError(
                    "Fleet exact-token rows require positive integer target counts"
                )
            source_family, task_type, category = _fleet_source_role_from_contract(
                record,
                contract=contract,
            )
            target_by_category[category] += target_tokens
            if (
                source_family == native_source_family
                and task_type == native_task_type
            ):
                native_target_tokens += target_tokens
                native_preference_pairs += 1
            if (source_family, task_type) in policy_signal_pairs:
                policy_signal_target_tokens += target_tokens
            if category == "supplemental_static":
                supplemental_by_family[source_family] = (
                    supplemental_by_family.get(source_family, 0) + target_tokens
                )
            row_evidence.append(
                {
                    "rowIndex": row_index,
                    "sourceRowSHA256": _canonical_sha256(record),
                    "sourceFamily": source_family,
                    "taskType": task_type,
                    "category": category,
                    "targetTokenCount": target_tokens,
                }
            )

        denominator = sum(target_by_category.values())
        supplemental = target_by_category["supplemental_static"]
        public = target_by_category["public_behavioral"]
        family_numerator = (
            native_target_tokens
            if lane == "sft"
            else native_preference_pairs
        )
        family_denominator = denominator if lane == "sft" else len(rows)
        cap_checks = (
            (
                "supplemental-static requested",
                supplemental,
                caps["supplementalStaticTotal"]["requested"],
            ),
            (
                "supplemental-static hard",
                supplemental,
                caps["supplementalStaticTotal"]["hard"],
            ),
            (
                "public-behavioral requested",
                public,
                caps["publicBehavioralTotal"]["requested"],
            ),
            (
                "public-behavioral hard",
                public,
                caps["publicBehavioralTotal"]["hard"],
            ),
        )
        # The contract bounds optimization loss. Validation rows never enter
        # optimizer steps (there is no early stopping or best-checkpoint
        # selection), so their exact shares are recorded and independently
        # reconstructed without imposing a quantized small-split cap.
        enforce_optimizer_caps = split == "train"
        for label, numerator, cap in cap_checks:
            if enforce_optimizer_caps and not _fleet_cap_passes(
                numerator=numerator,
                denominator=denominator,
                cap_basis_points=cap,
            ):
                raise RuntimeError(
                    f"Fleet {lane} {split} {label} exact-token cap failed: "
                    f"{numerator}*10000 > {denominator}*{cap}"
                )
        family_cap = caps["eachSupplementalSourceFamily"]["hard"]
        for source_family, numerator in sorted(supplemental_by_family.items()):
            if enforce_optimizer_caps and not _fleet_cap_passes(
                numerator=numerator,
                denominator=denominator,
                cap_basis_points=family_cap,
            ):
                raise RuntimeError(
                    f"Fleet {lane} {split} supplemental source family "
                    f"{source_family!r} exact-token cap failed: "
                    f"{numerator}*10000 > {denominator}*{family_cap}"
                )

        if enforce_optimizer_caps and not _fleet_optimizer_family_band_passes(
            numerator=family_numerator,
            denominator=family_denominator,
            minimum_basis_points=selected_family_band["minimumBasisPoints"],
            maximum_basis_points=selected_family_band["maximumBasisPoints"],
        ):
            raise RuntimeError(
                f"Fleet {lane} {split} optimizer-family share band failed: "
                f"{family_numerator}*10000 must be between "
                f"{family_denominator}*"
                f"{selected_family_band['minimumBasisPoints']} and "
                f"{family_denominator}*"
                f"{selected_family_band['maximumBasisPoints']}"
            )
        if enforce_optimizer_caps and not _fleet_optimizer_family_band_passes(
            numerator=policy_signal_target_tokens,
            denominator=denominator,
            minimum_basis_points=selected_policy_token_band[
                "minimumBasisPoints"
            ],
            maximum_basis_points=selected_policy_token_band[
                "maximumBasisPoints"
            ],
        ):
            raise RuntimeError(
                f"Fleet {lane} {split} policy-signal token share band failed: "
                f"{policy_signal_target_tokens}*10000 must be between "
                f"{denominator}*"
                f"{selected_policy_token_band['minimumBasisPoints']} and "
                f"{denominator}*"
                f"{selected_policy_token_band['maximumBasisPoints']}"
            )

        row_hashes = [item["sourceRowSHA256"] for item in row_evidence]
        split_evidence[split] = {
            "records": len(rows),
            "capEnforcementStatus": (
                "optimizer_enforced"
                if enforce_optimizer_caps
                else "observed_non_optimizer_split"
            ),
            "sourceRowsSHA256": _canonical_sha256(row_hashes),
            "rowTokenEvidence": row_evidence,
            "targetTokenCountsByCategory": target_by_category,
            "optimizerFamilyBandEnforcementStatus": (
                "optimizer_enforced"
                if enforce_optimizer_caps
                else "observed_non_optimizer_split"
            ),
            "policySignalTokenBandEnforcementStatus": (
                "optimizer_enforced"
                if enforce_optimizer_caps
                else "observed_non_optimizer_split"
            ),
            selected_family_band["numeratorEvidenceField"]: family_numerator,
            selected_family_band["denominatorEvidenceField"]: (
                family_denominator
            ),
            selected_policy_token_band[
                "numeratorEvidenceField"
            ]: policy_signal_target_tokens,
            field_names["denominatorTokenCount"]: denominator,
            field_names["supplementalNumeratorTokenCount"]: supplemental,
            field_names["publicNumeratorTokenCount"]: public,
            field_names["perSourceFamilyNumeratorTokenCounts"]: dict(
                sorted(supplemental_by_family.items())
            ),
        }
        if lane == "sft" and split == "train":
            optimizer_window_schedule, _ = (
                _build_fleet_sft_optimizer_window_schedule(
                    row_token_evidence=row_evidence,
                    config=config,
                    schedule_contract=contract[
                        "sftOptimizerWindowScheduleContract"
                    ],
                    minimum_basis_points=selected_family_band[
                        "minimumBasisPoints"
                    ],
                    maximum_basis_points=selected_family_band[
                        "maximumBasisPoints"
                    ],
                )
            )
            split_evidence[split]["optimizerWindowSchedule"] = (
                optimizer_window_schedule
            )

    return {
        "schemaVersion": FLEET_LOSS_SHARE_EVIDENCE_SCHEMA,
        "status": "passed",
        "lane": lane,
        "enforcementScope": "optimizer_train_with_validation_observation",
        "basisPointDenominator": contract["basisPointDenominator"],
        "capsBasisPoints": contract["capsBasisPoints"],
        "tokenizer": contract["tokenizer"],
        "tokenAccounting": contract["tokenAccounting"][lane],
        "dpoTokenizationPolicy": (
            contract["dpoTokenizationPolicy"] if lane == "dpo" else None
        ),
        "optimizerFamilyShareBand": selected_family_band,
        "policySignalTokenShareBand": selected_policy_token_band,
        "contractSHA256": _canonical_sha256(contract),
        "sourceRoleRegistrySHA256": _canonical_sha256(
            contract["sourceRoleRegistry"]
        ),
        "splits": split_evidence,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_self_hashed_json(
    path: Path,
    *,
    schema: str,
    hash_field: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing required lineage manifest: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise RuntimeError(f"Invalid lineage manifest contract: {path.name}")
    expected = payload.get(hash_field)
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    if (
        not isinstance(expected, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        or _canonical_sha256(unsigned) != expected
    ):
        raise RuntimeError(f"Lineage manifest integrity check failed: {path.name}")
    return payload


def _checkpoint_directory_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise RuntimeError("Checkpoint directory is missing")
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    if not files:
        raise RuntimeError("Checkpoint directory is empty")
    entries = [
        {
            "path": candidate.relative_to(path).as_posix(),
            "sizeBytes": candidate.stat().st_size,
            "sha256": _hash_file(candidate),
        }
        for candidate in sorted(
            files,
            key=lambda value: value.relative_to(path).as_posix(),
        )
    ]
    payload = {
        "schema": CHECKPOINT_DIRECTORY_SCHEMA,
        "files": entries,
    }
    return {**payload, "checkpointSHA256": _canonical_sha256(payload)}


def _checkpoint_step(value: str) -> int:
    match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", value)
    if match is None:
        raise RuntimeError("Checkpoint lineage contains an invalid checkpoint path")
    return int(match.group(1))


def _agent_resume_lineage(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    run_lineage = cfg.get("runResumeLineage")
    if not isinstance(run_lineage, dict) or run_lineage.get("schema") != RUN_RESUME_LINEAGE_SCHEMA:
        raise RuntimeError("runResumeLineage must be a canonical run-resume lineage object")
    if set(run_lineage) != RUN_RESUME_LINEAGE_FIELDS:
        raise RuntimeError("runResumeLineage fields do not match its schema")
    expected_digest = cfg.get("runResumeLineageSHA256")
    unsigned = dict(run_lineage)
    embedded_digest = unsigned.pop("runResumeLineageSHA256", None)
    actual_digest = _canonical_sha256(unsigned)
    if (
        not isinstance(expected_digest, str)
        or expected_digest != embedded_digest
        or expected_digest != actual_digest
    ):
        raise RuntimeError("runResumeLineageSHA256 does not match the run-resume lineage")
    agents = run_lineage.get("agents")
    if not isinstance(agents, list):
        raise RuntimeError("runResumeLineage.agents must be a list")
    if any(
        not isinstance(item, dict)
        or set(item) != RUN_RESUME_AGENT_LINEAGE_FIELDS
        for item in agents
    ):
        raise RuntimeError("runResumeLineage agent fields do not match its schema")
    agent_names = [item["agent"] for item in agents]
    if (
        any(not isinstance(name, str) or name not in AGENTS for name in agent_names)
        or len(agent_names) != len(set(agent_names))
        or run_lineage.get("selectedAgents") != agent_names
        or cfg.get("agent") not in agent_names
    ):
        raise RuntimeError("runResumeLineage selected-agent contract is invalid")
    matches = [item for item in agents if isinstance(item, dict) and item.get("agent") == cfg.get("agent")]
    if len(matches) != 1:
        raise RuntimeError("runResumeLineage must contain exactly one entry for this agent")
    return run_lineage, matches[0]


def _validate_run_resume_config(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
    assistant_only_loss: bool,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_lineage, agent_lineage = _agent_resume_lineage(cfg)
    checkpoint_path = Path(str(cfg.get("checkpointLineagePath") or "")).resolve()
    expected_static = {
        "runID": cfg_path.resolve().parent.parent.name,
        "datasetRepository": cfg.get("datasetRepository"),
        "datasetRevision": cfg.get("datasetRevision"),
        "datasetPath": cfg.get("datasetPath"),
        "localDatasetSnapshot": str(
            Path(str(cfg["dataset_dir"])).resolve().parents[2]
        ),
        "experimentVariant": cfg.get("variant"),
        "seed": cfg.get("seed"),
        "trainingCodeSHA256": cfg.get("trainingCodeSHA256"),
        "trainingDependencyLockSHA256": cfg.get("trainingDependencyLockSHA256"),
        "requirementsSHA256": cfg.get("requirementsSHA256"),
        "resolvedTrainingEnvironment": cfg.get("resolvedTrainingEnvironment"),
        "resolvedTrainingEnvironmentSHA256": cfg.get(
            "resolvedTrainingEnvironmentSHA256"
        ),
        "zeroGPUSize": cfg.get("zeroGPUSize"),
        "zeroGPUDurationSeconds": cfg.get("zeroGPUDurationSeconds"),
        "observedAccelerator": cfg.get("observedAccelerator"),
        "spaceConfigurationSHA256": cfg.get("spaceConfigurationSHA256"),
        "runtimeSourceKind": cfg.get("runtimeSourceKind"),
        "runtimeSourceRevision": cfg.get("runtimeSourceRevision"),
        "expectedRuntimeSourceRevision": cfg.get(
            "expectedRuntimeSourceRevision"
        ),
        "observedRepositoryRevision": cfg.get("observedRepositoryRevision"),
        "observedRuntimeRevision": cfg.get("observedRuntimeRevision"),
        "runtimeSourceBindingStatus": cfg.get("runtimeSourceBindingStatus"),
        "runtimeSourceBindingMethod": cfg.get("runtimeSourceBindingMethod"),
        "assistantOnlyLoss": bool(assistant_only_loss),
    }
    if any(run_lineage.get(key) != value for key, value in expected_static.items()):
        raise RuntimeError("Training config drifted from the run-resume lineage")
    variant_attestation = cfg.get("variantAttestation")
    if (
        not isinstance(variant_attestation, dict)
        or variant_attestation.get("schema")
        != TRAINING_VARIANT_ATTESTATION_SCHEMA
    ):
        raise RuntimeError("Training config is missing its variant attestation")
    for field in (
        "effectiveTrainingConfigSHA256",
        "trainingConfigInvariantSHA256",
    ):
        if re.fullmatch(
            r"[0-9a-f]{64}",
            str(variant_attestation.get(field) or ""),
        ) is None:
            raise RuntimeError(
                f"Training config variant attestation lacks a valid {field}"
            )
    expected_agent = {
        "sourceVariantManifestSHA256": cfg.get("variantManifestSHA256"),
        "laneHashes": variant_attestation.get("laneHashes"),
        "trainingCorpusSHA256": variant_attestation.get("trainingCorpusSHA256"),
        "controlledTrainingConfigSHA256": variant_attestation.get(
            "effectiveTrainingConfigSHA256"
        ),
        "trainingConfigInvariantSHA256": variant_attestation.get(
            "trainingConfigInvariantSHA256"
        ),
        "baseModelID": cfg.get("baseModelID", cfg.get("base_model_name")),
        "baseModelRevision": cfg.get("baseModelRevision"),
        "baseModelIndexDigest": cfg.get("baseModelIndexDigest"),
        "baseModelIndexReferencedShardNames": cfg.get(
            "baseModelIndexReferencedShardNames"
        ),
        "baseModelIndexShardBindingSHA256": cfg.get(
            "baseModelIndexShardBindingSHA256"
        ),
        "baseModelArtifactDigest": cfg.get("baseModelArtifactDigest"),
        "baseModelWeightShards": cfg.get("baseModelWeightShards"),
        "baseModelTokenizerDigest": cfg.get("baseModelTokenizerDigest"),
        "baseModelTokenizerFiles": cfg.get("baseModelTokenizerFiles"),
        "baseModelTokenizerClosureSHA256": cfg.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "baseModelTokenizerSnapshotPath": str(
            Path(str(cfg["baseModelTokenizerSnapshotPath"])).resolve()
        ),
        "baseModelGenerationConfigFile": cfg.get(
            "baseModelGenerationConfigFile"
        ),
        "baseModelRuntimeSnapshotPath": str(
            Path(str(cfg["baseModelRuntimeSnapshotPath"])).resolve()
        ),
        "seed": cfg.get("seed"),
        "trainingEnvironmentLockSHA256": variant_attestation.get(
            "trainingEnvironmentLockSHA256"
        ),
        "configPath": str(cfg_path),
        "checkpointLineagePath": str(checkpoint_path),
        "checkpointRoot": str(Path(str(cfg["output_dir"])).resolve()),
        "outputDirectory": str(Path(str(cfg["output_dir"])).resolve()),
        "adapterOutputDirectory": str(Path(str(cfg["adapter_output_dir"])).resolve()),
    }
    if any(agent_lineage.get(key) != value for key, value in expected_agent.items()):
        raise RuntimeError("Agent training config drifted from the run-resume lineage")
    dataset_dir = Path(str(cfg["dataset_dir"])).resolve()
    expected_dataset_files = agent_lineage.get("datasetFileSHA256")
    if (
        not isinstance(expected_dataset_files, dict)
        or set(expected_dataset_files) != RUN_RESUME_DATASET_FILES
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is None
            for value in expected_dataset_files.values()
        )
    ):
        raise RuntimeError("Run-resume lineage is missing dataset file hashes")
    actual_dataset_files = {
        filename: _hash_file(dataset_dir / filename)
        for filename in sorted(expected_dataset_files)
    }
    if actual_dataset_files != expected_dataset_files:
        raise RuntimeError("Dataset snapshot drifted from the run-resume lineage")
    return run_lineage, agent_lineage, checkpoint_path


def _validate_checkpoint_lineage(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
    require_checkpoint: bool,
    assistant_only_loss: bool = False,
) -> tuple[Path | None, Path | None]:
    has_lineage = any(
        key in cfg
        for key in (
            "runResumeLineage",
            "runResumeLineageSHA256",
            "checkpointLineagePath",
        )
    )
    if not has_lineage:
        if require_checkpoint:
            raise RuntimeError(
                "--resume-from-checkpoint requires run and checkpoint lineage evidence"
            )
        return None, None
    run_lineage, agent_lineage, record_path = _validate_run_resume_config(
        cfg,
        cfg_path=cfg_path,
        assistant_only_loss=assistant_only_loss,
    )
    record = _read_self_hashed_json(
        record_path,
        schema=CHECKPOINT_LINEAGE_SCHEMA,
        hash_field="checkpointLineageSHA256",
    )
    dataset_files = agent_lineage["datasetFileSHA256"]
    expected_record = {
        "agent": cfg.get("agent"),
        "runResumeLineageSHA256": run_lineage["runResumeLineageSHA256"],
        "configSHA256": _hash_file(cfg_path),
        "datasetFileSHA256": dataset_files,
        "laneHashes": agent_lineage["laneHashes"],
        "resolvedTrainingEnvironmentSHA256": run_lineage.get(
            "resolvedTrainingEnvironmentSHA256"
        ),
        "zeroGPUSize": run_lineage.get("zeroGPUSize"),
        "zeroGPUDurationSeconds": run_lineage.get("zeroGPUDurationSeconds"),
        "observedAccelerator": run_lineage.get("observedAccelerator"),
        "spaceConfigurationSHA256": run_lineage.get(
            "spaceConfigurationSHA256"
        ),
        "runtimeSourceBinding": {
            field: run_lineage[field]
            for field in (
                "runtimeSourceKind",
                "runtimeSourceRevision",
                "expectedRuntimeSourceRevision",
                "observedRepositoryRevision",
                "observedRuntimeRevision",
                "runtimeSourceBindingStatus",
                "runtimeSourceBindingMethod",
            )
        },
        "checkpointRoot": agent_lineage["checkpointRoot"],
        "outputDirectory": agent_lineage["outputDirectory"],
    }
    if any(record.get(key) != value for key, value in expected_record.items()):
        raise RuntimeError("Checkpoint lineage drifted from the training config")
    checkpoints = record.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise RuntimeError("Checkpoint lineage checkpoints must be a list")
    if not require_checkpoint:
        if checkpoints:
            raise RuntimeError(
                "Fresh training cannot reuse recorded checkpoints; select resume or reset the run"
            )
        return None, record_path
    if not checkpoints:
        raise RuntimeError("Resume requires at least one checkpoint bound to the run")
    root = Path(agent_lineage["checkpointRoot"]).resolve()
    validated: list[tuple[int, Path]] = []
    for entry in checkpoints:
        if not isinstance(entry, dict) or set(entry) != {"path", "checkpointSHA256"}:
            raise RuntimeError("Checkpoint lineage entries must contain only path and digest")
        relative = str(entry["path"])
        step = _checkpoint_step(relative)
        checkpoint = (root / relative).resolve()
        if checkpoint.parent != root:
            raise RuntimeError("Checkpoint lineage escapes the checkpoint root")
        manifest = _checkpoint_directory_manifest(checkpoint)
        if manifest["checkpointSHA256"] != entry["checkpointSHA256"]:
            raise RuntimeError("Checkpoint contents do not match checkpoint lineage")
        validated.append((step, checkpoint))
    if [step for step, _ in validated] != sorted({step for step, _ in validated}):
        raise RuntimeError("Checkpoint lineage entries must be unique and step-sorted")
    return validated[-1][1], record_path


def _record_checkpoint(record_path: Path, checkpoint: Path) -> None:
    record = _read_self_hashed_json(
        record_path,
        schema=CHECKPOINT_LINEAGE_SCHEMA,
        hash_field="checkpointLineageSHA256",
    )
    checkpoint_root = Path(str(record.get("checkpointRoot") or "")).resolve()
    checkpoint = checkpoint.resolve()
    if checkpoint.parent != checkpoint_root:
        raise RuntimeError("Saved checkpoint escapes the recorded checkpoint root")
    relative = checkpoint.name
    _checkpoint_step(relative)
    if not isinstance(record.get("checkpoints"), list):
        raise RuntimeError("Checkpoint lineage checkpoints must be a list")
    current_checkpoints = sorted(
        (
            candidate
            for candidate in checkpoint_root.glob("checkpoint-*")
            if candidate.is_dir()
        ),
        key=lambda candidate: _checkpoint_step(candidate.name),
    )
    if checkpoint not in current_checkpoints:
        raise RuntimeError("Saved checkpoint is absent from the checkpoint root")
    entries = [
        {
            "path": candidate.name,
            "checkpointSHA256": _checkpoint_directory_manifest(candidate)[
                "checkpointSHA256"
            ],
        }
        for candidate in current_checkpoints
    ]
    unsigned = dict(record)
    unsigned.pop("checkpointLineageSHA256", None)
    # Trainer may rotate old checkpoints before invoking on_save. Rebuild the
    # record from the currently present checkpoint set so resume never points
    # at a checkpoint that save_total_limit has already removed.
    unsigned["checkpoints"] = entries
    updated = {
        **unsigned,
        "checkpointLineageSHA256": _canonical_sha256(unsigned),
    }
    _atomic_write_json(record_path, updated)


def _checkpoint_lineage_callback(
    trainer_callback_type: type,
    *,
    record_path: Path,
) -> Any:
    class CheckpointLineageCallback(trainer_callback_type):
        def on_save(self, args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            checkpoint = Path(args.output_dir).resolve() / f"checkpoint-{state.global_step}"
            _record_checkpoint(record_path, checkpoint)
            return control

    return CheckpointLineageCallback()


def _sft_checkpoint_policy(cfg: Mapping[str, Any]) -> tuple[int, int]:
    save_steps = cfg.get("sft_checkpoint_save_steps", SFT_CHECKPOINT_SAVE_STEPS)
    save_total_limit = cfg.get("save_total_limit", 2)
    if type(save_steps) is not int or save_steps <= 0:
        raise ValueError("sft_checkpoint_save_steps must be a positive integer")
    if (
        type(save_total_limit) is not int
        or save_total_limit < SFT_CHECKPOINT_MINIMUM_RETENTION
    ):
        raise ValueError("save_total_limit must retain at least two SFT checkpoints")
    return save_steps, save_total_limit


def _sft_checkpoint_dataset_sha256(cfg: Mapping[str, Any]) -> dict[str, str]:
    dataset_root = Path(str(cfg.get("dataset_dir") or "")).resolve()
    filenames = ("train_sft.jsonl", "val_sft.jsonl", "variant_manifest.json")
    hashes: dict[str, str] = {}
    for filename in filenames:
        path = dataset_root / filename
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"SFT checkpoint lineage requires a regular {filename}")
        hashes[filename] = _require_sha256(
            _hash_file(path),
            name=f"{filename} SHA-256",
        )
    return hashes


def _sft_training_code_sha256(cfg: Mapping[str, Any]) -> str:
    phase_digests = cfg.get("trainingCodeSHA256ByPhase")
    digest = (
        phase_digests.get("sft")
        if isinstance(phase_digests, Mapping)
        else cfg.get("trainingCodeSHA256")
    )
    return _require_sha256(digest, name="SFT training-code SHA-256")


def _sft_checkpoint_static_contract(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
) -> dict[str, Any]:
    output_dir = Path(str(cfg.get("output_dir") or "")).resolve()
    record_value = str(cfg.get("sftCheckpointLineagePath") or "")
    if not record_value:
        raise RuntimeError("SFT checkpoint lineage path is missing")
    record_path = Path(record_value).resolve()
    if record_path == output_dir or output_dir in record_path.parents:
        raise RuntimeError("SFT checkpoint lineage record must be outside its checkpoint root")
    preflight_value = str(cfg.get("sftTokenLengthPreflightPath") or "")
    if not preflight_value:
        raise RuntimeError("SFT token-length preflight path is missing")
    preflight_path = Path(preflight_value).resolve()
    if output_dir not in preflight_path.parents:
        raise RuntimeError("SFT token-length preflight must be inside its output directory")
    save_steps, save_total_limit = _sft_checkpoint_policy(cfg)
    precision = _resolve_training_precision(cfg)
    return {
        "schema": SFT_CHECKPOINT_LINEAGE_SCHEMA,
        "agent": cfg.get("agent"),
        "configPath": str(cfg_path.resolve()),
        "configSHA256": _require_sha256(
            _hash_file(cfg_path.resolve()),
            name="SFT config SHA-256",
        ),
        "sourceVariantManifestSHA256": _require_sha256(
            cfg.get("variantManifestSHA256"),
            name="variantManifestSHA256",
        ),
        "datasetFileSHA256": _sft_checkpoint_dataset_sha256(cfg),
        "trainingCodeSHA256": _sft_training_code_sha256(cfg),
        "resolvedTrainingEnvironmentSHA256": _require_sha256(
            cfg.get("resolvedTrainingEnvironmentSHA256"),
            name="resolvedTrainingEnvironmentSHA256",
        ),
        "precision": precision,
        "checkpointScalerState": _checkpoint_scaler_state_contract(precision),
        "checkpointRoot": str(output_dir),
        "outputDirectory": str(output_dir),
        "adapterOutputDirectory": str(
            Path(str(cfg.get("adapter_output_dir") or "")).resolve()
        ),
        "tokenLengthPreflightPath": str(
            preflight_path
        ),
        "saveStrategy": "steps",
        "saveSteps": save_steps,
        "saveTotalLimit": save_total_limit,
    }


def _self_hashed_sft_checkpoint_record(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("checkpointLineageSHA256", None)
    return {
        **unsigned,
        "checkpointLineageSHA256": _canonical_sha256(unsigned),
    }


def _initial_sft_checkpoint_lineage(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
) -> dict[str, Any]:
    return _self_hashed_sft_checkpoint_record(
        {
            **_sft_checkpoint_static_contract(cfg, cfg_path=cfg_path),
            "assistantOnlyLoss": None,
            "tokenLengthPreflightSHA256": None,
            "checkpoints": [],
        }
    )


def _read_sft_checkpoint_lineage(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Missing regular SFT checkpoint lineage record")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SFT_CHECKPOINT_LINEAGE_SCHEMA
    ):
        raise RuntimeError("Invalid SFT checkpoint lineage contract")
    expected = payload.get("checkpointLineageSHA256")
    unsigned = dict(payload)
    unsigned.pop("checkpointLineageSHA256", None)
    if (
        not isinstance(expected, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        or _canonical_sha256(unsigned) != expected
    ):
        raise RuntimeError("SFT checkpoint lineage integrity check failed")
    return payload


def _validate_sft_checkpoint_lineage_static(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
) -> dict[str, Any]:
    record = _read_sft_checkpoint_lineage(
        Path(str(cfg.get("sftCheckpointLineagePath") or "")).resolve()
    )
    expected = _sft_checkpoint_static_contract(cfg, cfg_path=cfg_path)
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("SFT checkpoint lineage drifted from the config")
    assistant_only_loss = record.get("assistantOnlyLoss")
    if assistant_only_loss is not None and type(assistant_only_loss) is not bool:
        raise RuntimeError("SFT checkpoint assistant-only-loss binding is invalid")
    if not isinstance(record.get("checkpoints"), list):
        raise RuntimeError("SFT checkpoint lineage checkpoints must be a list")
    preflight_sha256 = record.get("tokenLengthPreflightSHA256")
    if preflight_sha256 is not None:
        _require_sha256(
            preflight_sha256,
            name="SFT token-length preflight SHA-256",
        )
    if assistant_only_loss is None and (
        record.get("checkpoints") != [] or preflight_sha256 is not None
    ):
        raise RuntimeError("Unbound SFT checkpoint lineage cannot contain checkpoints")
    return record


def _reset_sft_checkpoint_lineage(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
) -> None:
    _write_json_atomic(
        Path(str(cfg.get("sftCheckpointLineagePath") or "")).resolve(),
        _initial_sft_checkpoint_lineage(cfg, cfg_path=cfg_path),
    )


def _bind_sft_token_length_preflight(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    if preflight.get("schemaVersion") != SFT_TOKEN_LENGTH_PREFLIGHT_SCHEMA:
        raise RuntimeError("SFT token-length preflight contract is invalid")
    record_path = Path(str(cfg.get("sftCheckpointLineagePath") or "")).resolve()
    record = _validate_sft_checkpoint_lineage_static(cfg, cfg_path=cfg_path)
    if record.get("assistantOnlyLoss") is None:
        raise RuntimeError("SFT token-length preflight requires bound training semantics")
    unsigned = {
        **dict(preflight),
        "agent": cfg.get("agent"),
        "variant": cfg.get("variant"),
        "configPath": str(cfg_path.resolve()),
        "configSHA256": record["configSHA256"],
        "datasetFileSHA256": record["datasetFileSHA256"],
        "trainingCodeSHA256": record["trainingCodeSHA256"],
        "baseModelID": cfg.get("base_model_name"),
        "baseModelRevision": cfg.get("baseModelRevision"),
        "baseModelTokenizerDigest": cfg.get("baseModelTokenizerDigest"),
        "baseModelTokenizerClosureSHA256": cfg.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "chatTemplateContract": cfg.get("chatTemplateContract"),
    }
    evidence = {
        **unsigned,
        "preflightSHA256": _canonical_sha256(unsigned),
    }
    evidence_path = Path(str(record["tokenLengthPreflightPath"])).resolve()
    if evidence_path.is_symlink():
        raise RuntimeError("SFT token-length preflight path is unsafe")
    if evidence_path.exists():
        if not evidence_path.is_file():
            raise RuntimeError("SFT token-length preflight path is not a regular file")
        existing = json.loads(evidence_path.read_text(encoding="utf-8"))
        if existing != evidence:
            raise RuntimeError("SFT token-length preflight evidence drifted")
    else:
        _write_json_atomic(evidence_path, evidence)
    recorded_digest = record.get("tokenLengthPreflightSHA256")
    if recorded_digest is None:
        if record.get("checkpoints") != []:
            raise RuntimeError("SFT checkpoints predate token-length preflight evidence")
        updated = dict(record)
        updated["tokenLengthPreflightSHA256"] = evidence["preflightSHA256"]
        _write_json_atomic(
            record_path,
            _self_hashed_sft_checkpoint_record(updated),
        )
    elif recorded_digest != evidence["preflightSHA256"]:
        raise RuntimeError("SFT token-length preflight lineage drifted")
    return evidence


def _verify_prepared_global_tokenizer_preflight(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
    phase: str,
    bound_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the actual Unsloth tokenizer with the prepared all-agent audit.

    Callers invoke this only after their authoritative phase evidence is bound
    and before constructing a PEFT adapter, policy adapter, or trainer.
    """

    agent = cfg.get("agent")
    if not isinstance(agent, str) or agent not in AGENTS:
        raise RuntimeError("Prepared tokenizer preflight requires a valid agent")
    resolved_config = cfg_path.resolve()
    run_root = resolved_config.parent.parent
    expected_config = (run_root / "configs" / f"{agent}.json").resolve()
    if (
        resolved_config != expected_config
        or resolved_config.parent != (run_root / "configs").resolve()
    ):
        raise RuntimeError(
            "Prepared tokenizer preflight config path drifted from the run root"
        )
    try:
        from .ubuntu_pipeline import _verified_global_tokenizer_preflight
    except ImportError:
        from ubuntu_pipeline import _verified_global_tokenizer_preflight

    audit = _verified_global_tokenizer_preflight(
        run_root=run_root,
        agent=agent,
        config=cfg,
        phase=phase,
        bound_preflight=bound_preflight,
    )
    closure = audit.get("tokenizerClosure")
    if (
        not isinstance(closure, Mapping)
        or closure.get("schemaVersion")
        != "lumen.global-tokenizer-snapshot/1.1.0"
    ):
        raise RuntimeError(
            "Production training rejects injected global-tokenizer test evidence"
        )
    return audit


def _sft_checkpoint_directory_manifest(
    checkpoint: Path,
    *,
    expected_base_model: str,
    expected_base_revision: str,
    precision: Mapping[str, Any],
) -> dict[str, Any]:
    if checkpoint.is_symlink():
        raise RuntimeError("SFT checkpoint directory is missing or unsafe")
    checkpoint = checkpoint.resolve()
    step = _checkpoint_step(checkpoint.name)
    if not checkpoint.is_dir():
        raise RuntimeError("SFT checkpoint directory is missing or unsafe")
    files: set[str] = set()
    entries: list[dict[str, Any]] = []
    for candidate in sorted(
        checkpoint.rglob("*"),
        key=lambda path: path.relative_to(checkpoint).as_posix(),
    ):
        if candidate.is_symlink():
            raise RuntimeError("SFT checkpoint contains a symlink")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise RuntimeError("SFT checkpoint contains a non-regular entry")
        relative = candidate.relative_to(checkpoint).as_posix()
        files.add(relative)
        entries.append(
            {
                "path": relative,
                "sizeBytes": candidate.stat().st_size,
                "sha256": _require_sha256(
                    _hash_file(candidate),
                    name=f"SFT checkpoint file {relative}",
                ),
            }
        )
    scaler_state = _checkpoint_scaler_state_contract(precision)
    required_files = set(SFT_CHECKPOINT_REQUIRED_FILES)
    if scaler_state["required"]:
        required_files.add(CHECKPOINT_SCALER_FILENAME)
    missing = sorted(required_files - files)
    adapter_weights = files & {"adapter_model.bin", "adapter_model.safetensors"}
    nested_adapters = sorted(
        path
        for path in files
        if path.endswith("/adapter_config.json")
        or path.endswith("/adapter_model.bin")
        or path.endswith("/adapter_model.safetensors")
    )
    if missing:
        raise _IncompleteSFTCheckpoint(
            "SFT checkpoint is incomplete: missing " + ", ".join(missing)
        )
    if not adapter_weights:
        raise _IncompleteSFTCheckpoint(
            "SFT checkpoint is incomplete: missing a root policy adapter"
        )
    if len(adapter_weights) != 1 or nested_adapters:
        raise RuntimeError("SFT checkpoint must contain exactly one root policy adapter")
    try:
        trainer_state = json.loads(
            (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
        adapter_config = json.loads(
            (checkpoint / "adapter_config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("SFT checkpoint metadata is unreadable") from exc
    if (
        not isinstance(trainer_state, Mapping)
        or trainer_state.get("global_step") != step
    ):
        raise RuntimeError("SFT checkpoint trainer state does not match its directory step")
    if (
        not isinstance(adapter_config, Mapping)
        or adapter_config.get("base_model_name_or_path") != expected_base_model
        or adapter_config.get("revision") != expected_base_revision
    ):
        raise RuntimeError("SFT checkpoint base-model lineage drifted")
    payload = {
        "schema": SFT_CHECKPOINT_DIRECTORY_SCHEMA,
        "globalStep": step,
        "scalerState": scaler_state,
        "files": entries,
    }
    return {**payload, "checkpointSHA256": _canonical_sha256(payload)}


def _bound_sft_checkpoint_entries(
    record: Mapping[str, Any],
    *,
    expected_base_model: str,
    expected_base_revision: str,
) -> tuple[list[tuple[int, Path]], set[str], list[Path]]:
    root = Path(str(record.get("checkpointRoot") or "")).resolve()
    entries = record.get("checkpoints")
    if not isinstance(entries, list):
        raise RuntimeError("SFT checkpoint lineage checkpoints must be a list")
    prepared: list[tuple[int, Path, str]] = []
    declared_names: set[str] = set()
    steps: list[int] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "path",
            "checkpointSHA256",
        }:
            raise RuntimeError("SFT checkpoint lineage entry is not canonical")
        relative = str(entry.get("path") or "")
        step = _checkpoint_step(relative)
        if relative in declared_names:
            raise RuntimeError("SFT checkpoint lineage contains duplicates")
        declared_names.add(relative)
        steps.append(step)
        expected_digest = _require_sha256(
            entry.get("checkpointSHA256"),
            name=f"SFT checkpoint lineage digest for {relative}",
        )
        unresolved = root / relative
        if unresolved.is_symlink():
            raise RuntimeError("SFT checkpoint lineage points to a symlink")
        checkpoint = unresolved.resolve()
        if checkpoint.parent != root:
            raise RuntimeError("SFT checkpoint lineage escapes its root")
        prepared.append((step, checkpoint, expected_digest))
    if steps != sorted(set(steps)):
        raise RuntimeError("SFT checkpoint lineage entries must be unique and step-sorted")

    # Transformers may rotate the oldest checkpoint before the on_save callback
    # binds the newly completed checkpoint. Validate from newest to oldest so a
    # newer signed recovery point can prove that a structurally unfinished older
    # directory is only a stale rotation remnant. A missing or incomplete newest
    # signed entry remains fatal, as does digest drift at any age.
    validated_descending: list[tuple[int, Path]] = []
    stale_partials: list[Path] = []
    newer_checkpoint_validated = False
    for step, checkpoint, expected_digest in reversed(prepared):
        if not checkpoint.exists():
            if not newer_checkpoint_validated:
                raise RuntimeError("Newest bound SFT checkpoint is missing")
            continue
        try:
            manifest = _sft_checkpoint_directory_manifest(
                checkpoint,
                expected_base_model=expected_base_model,
                expected_base_revision=expected_base_revision,
                precision=record["precision"],
            )
        except _IncompleteSFTCheckpoint:
            if not newer_checkpoint_validated:
                raise
            stale_partials.append(checkpoint)
            continue
        if manifest["checkpointSHA256"] != expected_digest:
            raise RuntimeError("SFT checkpoint contents drifted from lineage")
        validated_descending.append((step, checkpoint))
        newer_checkpoint_validated = True
    return (
        list(reversed(validated_descending)),
        declared_names,
        sorted(stale_partials, key=lambda path: _checkpoint_step(path.name)),
    )


def _unbound_sft_checkpoint_directories(
    root: Path,
    *,
    declared_names: set[str],
) -> list[Path]:
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("SFT checkpoint root is unsafe")
    unbound: list[Path] = []
    for candidate in root.glob("checkpoint-*"):
        _checkpoint_step(candidate.name)
        if candidate.is_symlink() or not candidate.is_dir():
            raise RuntimeError("SFT checkpoint candidate is unsafe")
        if candidate.name not in declared_names:
            unbound.append(candidate)
    return sorted(unbound, key=lambda path: _checkpoint_step(path.name))


def _bind_and_validate_sft_checkpoint_lineage(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
    assistant_only_loss: bool,
    require_checkpoint: bool,
) -> tuple[Path | None, Path, list[Path]]:
    record_path = Path(str(cfg.get("sftCheckpointLineagePath") or "")).resolve()
    record = _validate_sft_checkpoint_lineage_static(cfg, cfg_path=cfg_path)
    recorded_assistant_only = record.get("assistantOnlyLoss")
    if recorded_assistant_only is None:
        updated = dict(record)
        updated["assistantOnlyLoss"] = bool(assistant_only_loss)
        record = _self_hashed_sft_checkpoint_record(updated)
        _write_json_atomic(record_path, record)
    elif recorded_assistant_only is not bool(assistant_only_loss):
        raise RuntimeError("SFT checkpoint assistant-only-loss setting drifted")
    validated, declared_names, stale_partials = _bound_sft_checkpoint_entries(
        record,
        expected_base_model=str(cfg.get("base_model_name") or ""),
        expected_base_revision=str(cfg.get("baseModelRevision") or ""),
    )
    root = Path(str(record["checkpointRoot"])).resolve()
    unbound = _unbound_sft_checkpoint_directories(
        root,
        declared_names=declared_names,
    )
    discardable = sorted(
        [*stale_partials, *unbound],
        key=lambda path: _checkpoint_step(path.name),
    )
    if not require_checkpoint:
        if record.get("checkpoints") or discardable:
            raise RuntimeError("Fresh SFT training cannot reuse checkpoint state")
        return None, record_path, []
    if not validated:
        if record.get("checkpoints") == [] and unbound:
            return None, record_path, unbound
        raise RuntimeError("Resume requires a complete bound SFT checkpoint")
    return validated[-1][1], record_path, discardable


def _prune_unbound_sft_checkpoints(paths: list[Path]) -> None:
    for path in paths:
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError("Refusing to prune an unsafe SFT checkpoint")
        shutil.rmtree(path)


def _record_sft_checkpoint(record_path: Path, checkpoint: Path) -> None:
    record = _read_sft_checkpoint_lineage(record_path)
    root = Path(str(record.get("checkpointRoot") or "")).resolve()
    checkpoint = checkpoint.resolve()
    if checkpoint.parent != root:
        raise RuntimeError("Saved SFT checkpoint escapes its recorded root")
    candidates = sorted(
        (
            candidate
            for candidate in root.glob("checkpoint-*")
            if candidate.is_dir() and not candidate.is_symlink()
        ),
        key=lambda path: _checkpoint_step(path.name),
    )
    if checkpoint not in candidates:
        raise RuntimeError("Saved SFT checkpoint is absent from its root")
    config_path = Path(str(record["configPath"]))
    if (
        config_path.is_symlink()
        or not config_path.is_file()
        or _hash_file(config_path) != record.get("configSHA256")
    ):
        raise RuntimeError("SFT checkpoint config drifted during training")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_base_model = str(config.get("base_model_name") or "")
    expected_base_revision = str(config.get("baseModelRevision") or "")
    precision = _resolve_training_precision(config)
    updated = dict(record)
    updated["checkpoints"] = [
        {
            "path": candidate.name,
            "checkpointSHA256": _sft_checkpoint_directory_manifest(
                candidate,
                expected_base_model=expected_base_model,
                expected_base_revision=expected_base_revision,
                precision=precision,
            )["checkpointSHA256"],
        }
        for candidate in candidates
    ]
    _write_json_atomic(record_path, _self_hashed_sft_checkpoint_record(updated))


def _sft_checkpoint_lineage_callback(
    trainer_callback_type: type,
    *,
    record_path: Path,
) -> Any:
    class SFTCheckpointLineageCallback(trainer_callback_type):
        def on_save(self, args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            checkpoint = Path(args.output_dir).resolve() / f"checkpoint-{state.global_step}"
            _record_sft_checkpoint(record_path, checkpoint)
            return control

    return SFTCheckpointLineageCallback()


def _require_sha256(value: Any, *, name: str, prefix: bool = False) -> str:
    text = str(value or "")
    pattern = r"sha256:[0-9a-f]{64}" if prefix else r"[0-9a-f]{64}"
    if re.fullmatch(pattern, text) is None:
        raise RuntimeError(f"{name} must be an immutable lowercase SHA-256 digest")
    return text


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _require_finite_training_metrics(
    value: Any,
    *,
    name: str,
    required_fields: frozenset[str],
) -> None:
    """Reject empty, non-numeric, or non-finite trainer metric evidence."""

    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(f"{name} must be a non-empty metric object")
    missing = sorted(required_fields - set(value))
    if missing:
        raise RuntimeError(f"{name} is missing required metrics: {', '.join(missing)}")

    def verify_metric(metric: Any, *, path: str) -> None:
        if isinstance(metric, Mapping):
            if not metric:
                raise RuntimeError(f"{path} must not be empty")
            for key, nested in metric.items():
                if not isinstance(key, str) or not key:
                    raise RuntimeError(f"{path} contains an invalid metric name")
                verify_metric(nested, path=f"{path}.{key}")
            return
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
        ):
            raise RuntimeError(f"{path} must be a finite numeric metric")

    verify_metric(value, path=name)


def _verified_training_completion_evidence(
    trainer: Any,
    training_args: Any,
    train_result: Any,
    evaluation_metrics: Mapping[str, Any],
    *,
    has_eval_dataset: bool,
    train_record_count: int,
    expected_precision: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that Trainer reached its controlled terminal step and epoch."""

    canonical_expected_precision = _resolve_training_precision(
        {
            "bf16": expected_precision.get("bf16"),
            "fp16": expected_precision.get("fp16"),
        }
    )
    if dict(expected_precision) != canonical_expected_precision:
        raise RuntimeError("Expected training precision contract is not canonical")
    try:
        resolved_precision = _resolve_training_precision(
            {
                "bf16": getattr(training_args, "bf16", None),
                "fp16": getattr(training_args, "fp16", None),
            }
        )
    except ValueError as exc:
        raise RuntimeError("Trainer precision arguments are invalid") from exc
    if resolved_precision != canonical_expected_precision:
        raise RuntimeError("Trainer precision drifted from the controlled config")

    state = getattr(trainer, "state", None)
    global_step = getattr(state, "global_step", None)
    max_steps = getattr(state, "max_steps", None)
    train_result_global_step = getattr(train_result, "global_step", None)
    if (
        type(global_step) is not int
        or type(max_steps) is not int
        or type(train_result_global_step) is not int
        or max_steps <= 0
        or global_step != max_steps
        or train_result_global_step != global_step
    ):
        raise RuntimeError(
            "Training did not reach the exact positive terminal global step"
        )

    configured_epochs_raw = getattr(training_args, "num_train_epochs", None)
    observed_epoch_raw = getattr(state, "epoch", None)
    if (
        isinstance(configured_epochs_raw, bool)
        or not isinstance(configured_epochs_raw, (int, float))
        or isinstance(observed_epoch_raw, bool)
        or not isinstance(observed_epoch_raw, (int, float))
    ):
        raise RuntimeError("Training epoch completion evidence is not numeric")
    configured_epochs = float(configured_epochs_raw)
    observed_epoch = float(observed_epoch_raw)
    per_device_batch_size = getattr(
        training_args,
        "per_device_train_batch_size",
        None,
    )
    gradient_accumulation_steps = getattr(
        training_args,
        "gradient_accumulation_steps",
        None,
    )
    world_size = getattr(training_args, "world_size", None)
    if (
        configured_epochs <= 0
        or not math.isfinite(configured_epochs)
        or not math.isfinite(observed_epoch)
        or not math.isclose(
            observed_epoch,
            configured_epochs,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise RuntimeError("Training did not complete the configured epoch count")
    if (
        type(train_record_count) is not int
        or train_record_count <= 0
        or type(per_device_batch_size) is not int
        or per_device_batch_size <= 0
        or type(gradient_accumulation_steps) is not int
        or gradient_accumulation_steps <= 0
        or type(world_size) is not int
        or world_size != 1
    ):
        raise RuntimeError("Training step reconstruction inputs are invalid")
    train_dataloader_batch_count = math.ceil(
        train_record_count / (per_device_batch_size * world_size)
    )
    update_steps_per_epoch = max(
        math.ceil(train_dataloader_batch_count / gradient_accumulation_steps),
        1,
    )
    expected_max_steps = math.ceil(configured_epochs * update_steps_per_epoch)
    if max_steps != expected_max_steps:
        raise RuntimeError("Training terminal step drifted from controlled inputs")

    train_metrics = getattr(train_result, "metrics", None)
    _require_finite_training_metrics(
        train_metrics,
        name="training metrics",
        required_fields=frozenset({"train_loss", "epoch"}),
    )
    train_metric_epoch = float(train_metrics["epoch"])
    if not math.isclose(
        train_metric_epoch,
        observed_epoch,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError("Training metric epoch drifted from Trainer state")

    if has_eval_dataset:
        _require_finite_training_metrics(
            evaluation_metrics,
            name="evaluation metrics",
            required_fields=frozenset({"eval_loss", "epoch"}),
        )
        if not math.isclose(
            float(evaluation_metrics["epoch"]),
            observed_epoch,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError("Evaluation metric epoch drifted from Trainer state")
    elif evaluation_metrics:
        raise RuntimeError("Evaluation metrics exist without an evaluation dataset")

    return {
        "schema": TRAINING_COMPLETION_EVIDENCE_SCHEMA,
        "status": "completed",
        "globalStep": global_step,
        "maxSteps": max_steps,
        "expectedMaxSteps": expected_max_steps,
        "trainResultGlobalStep": train_result_global_step,
        "configuredNumTrainEpochs": configured_epochs,
        "observedEpoch": observed_epoch,
        "trainRecordCount": train_record_count,
        "perDeviceTrainBatchSize": per_device_batch_size,
        "gradientAccumulationSteps": gradient_accumulation_steps,
        "worldSize": world_size,
        "trainDataloaderBatchCount": train_dataloader_batch_count,
        "updateStepsPerEpoch": update_steps_per_epoch,
        "trainMetricsVerified": True,
        "evaluationMetricsVerified": has_eval_dataset,
        "resolvedPrecision": resolved_precision,
    }


def _training_environment(
    cfg: dict[str, Any],
    *,
    runtime_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    lock = cfg.get("trainingEnvironmentLock")
    if not isinstance(lock, dict):
        raise RuntimeError("trainingEnvironmentLock must be an object")
    if (
        lock.get("schemaVersion")
        != "lumen.adapter-training-environment-lock/1.1.0"
        or lock.get("baseTokenizerSHA256")
        != cfg.get("baseModelTokenizerDigest")
        or lock.get("baseTokenizerClosureSHA256")
        != cfg.get("baseModelTokenizerClosureSHA256")
    ):
        raise RuntimeError(
            "trainingEnvironmentLock tokenizer lineage drifted from the config"
        )
    container_digest = _require_sha256(
        cfg.get("trainingContainerImageDigest"),
        name="trainingContainerImageDigest",
        prefix=True,
    )
    digest_source = cfg.get("trainingContainerImageDigestSource")
    binding_status = cfg.get("trainingRuntimeImageBindingStatus")
    binding_verified = cfg.get("trainingRuntimeImageBindingVerified")
    if (
        digest_source != "operator_declared"
        or binding_status != "manual_validation_required"
        or binding_verified is not False
    ):
        raise RuntimeError(
            "Training container provenance must remain operator-declared and manually unverified"
        )
    expected_python = str(lock.get("pythonVersion") or "")
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        raise RuntimeError(f"Training Python drifted from lock: expected {expected_python}, got {actual_python}")

    expected_packages = lock.get("packageVersions")
    if not isinstance(expected_packages, dict) or not expected_packages:
        raise RuntimeError("trainingEnvironmentLock.packageVersions must be non-empty")
    observed_packages = {
        name: _package_version(name) for name in sorted(expected_packages)
    }
    actual_packages = {
        name: canonical_controlled_package_version(
            name,
            version,
            expected_version=str(expected_packages[name]),
            cuda_version=str(lock.get("cudaVersion") or ""),
        )
        for name, version in observed_packages.items()
    }
    if actual_packages != expected_packages:
        raise RuntimeError(
            "Training package versions drifted from lock: "
            + json.dumps(
                {"expected": expected_packages, "actual": observed_packages},
                sort_keys=True,
            )
        )

    import importlib.metadata as metadata

    direct_url_text = metadata.distribution("unsloth").read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError("Installed Unsloth lacks PEP 610 VCS provenance")
    direct_url = json.loads(direct_url_text)
    vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
    unsloth_revision = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if unsloth_revision != lock.get("unslothRevision"):
        raise RuntimeError(
            f"Installed Unsloth revision drifted from lock: expected {lock.get('unslothRevision')}, "
            f"got {unsloth_revision or '<unattested>'}"
        )

    import torch  # type: ignore

    expected_cuda = str(lock.get("cudaVersion") or "")
    actual_cuda = str(torch.version.cuda or "")
    if actual_cuda != expected_cuda:
        raise RuntimeError(f"Training CUDA drifted from lock: expected {expected_cuda}, got {actual_cuda or '<none>'}")
    resolved_environment = (
        runtime_lineage.get("resolvedTrainingEnvironment")
        if runtime_lineage is not None
        else cfg.get("resolvedTrainingEnvironment")
    )
    if not isinstance(resolved_environment, Mapping):
        resolved_environment = build_resolved_training_environment()
    try:
        resolved_environment_digest = verify_resolved_training_environment(
            resolved_environment,
        )
    except ValueError as exc:
        raise RuntimeError("Resolved training environment verification failed") from exc
    configured_resolved_digest = (
        runtime_lineage.get("resolvedTrainingEnvironmentSHA256")
        if runtime_lineage is not None
        else cfg.get("resolvedTrainingEnvironmentSHA256")
    )
    if (
        configured_resolved_digest is not None
        and configured_resolved_digest != resolved_environment_digest
    ):
        raise RuntimeError("resolvedTrainingEnvironmentSHA256 drifted")
    payload = {
        "schemaVersion": "lumen.adapter-training-environment/1.0.0",
        "containerImageDigest": container_digest,
        "containerImageDigestSource": digest_source,
        "runtimeImageBindingStatus": binding_status,
        "runtimeImageBindingVerified": binding_verified,
        "effectiveSeed": int(cfg["seed"]),
        "environmentLock": lock,
        "zeroGPUSize": (
            runtime_lineage.get("zeroGPUSize")
            if runtime_lineage is not None
            else cfg.get("zeroGPUSize")
        ),
        "zeroGPUDurationSeconds": (
            runtime_lineage.get("zeroGPUDurationSeconds")
            if runtime_lineage is not None
            else cfg.get("zeroGPUDurationSeconds")
        ),
        "observedAccelerator": (
            runtime_lineage.get("observedAccelerator")
            if runtime_lineage is not None
            else cfg.get("observedAccelerator")
        ),
        "trainingCodeSHA256": _require_sha256(
            cfg.get("trainingCodeSHA256"), name="trainingCodeSHA256"
        ),
        "trainingDependencyLockSHA256": _require_sha256(
            cfg.get("trainingDependencyLockSHA256"),
            name="trainingDependencyLockSHA256",
        ),
        "requirementsSHA256": _require_sha256(
            cfg.get("requirementsSHA256"), name="requirementsSHA256"
        ),
        "resolvedTrainingEnvironment": dict(resolved_environment),
        "resolvedTrainingEnvironmentSHA256": resolved_environment_digest,
    }
    digest = _canonical_sha256(payload)
    if (
        cfg.get("resolvedTrainingEnvironmentSHA256") is not None
        and cfg.get("trainingEnvironmentSHA256") is not None
        and digest != cfg.get("trainingEnvironmentSHA256")
    ):
        raise RuntimeError("trainingEnvironmentSHA256 is not bound to the recorded environment payload")
    return {**payload, "trainingEnvironmentSHA256": digest}


def _installed_unsloth_revision() -> str:
    import importlib.metadata as metadata

    direct_url_text = metadata.distribution("unsloth").read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError("Installed Unsloth lacks PEP 610 VCS provenance")
    direct_url = json.loads(direct_url_text)
    vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
    revision = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if not isinstance(revision, str):
        raise RuntimeError("Installed Unsloth lacks an immutable VCS revision")
    return revision


def _resolved_environment_runtime_lineage(
    cfg: Mapping[str, Any],
    *,
    deployed_space: bool,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    configured = cfg.get("resolvedTrainingEnvironment")
    if configured is not None and not isinstance(configured, Mapping):
        raise RuntimeError("resolvedTrainingEnvironment must be an object")
    cache_attestation = cfg.get("resolvedTrainingEnvironmentCacheAttestation")
    runtime_cache_attestation = os.environ.get(
        "LUMEN_ZERO_GPU_RESOLVED_ENVIRONMENT_CACHE_ATTESTATION",
        "",
    )
    if runtime_cache_attestation:
        try:
            parsed_cache_attestation = json.loads(runtime_cache_attestation)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Resolved training environment cache authorization is invalid"
            ) from exc
        if not isinstance(parsed_cache_attestation, Mapping):
            raise RuntimeError(
                "Resolved training environment cache authorization is invalid"
            )
        cache_attestation = parsed_cache_attestation
    cache_key_hex = os.environ.get(
        "LUMEN_ZERO_GPU_RESOLVED_ENVIRONMENT_CACHE_HMAC_KEY",
        "",
    )
    if deployed_space and cache_key_hex:
        if (
            not isinstance(configured, Mapping)
            or not isinstance(cache_attestation, Mapping)
            or re.fullmatch(r"[0-9a-f]{64}", cache_key_hex) is None
        ):
            raise RuntimeError(
                "Resolved training environment cache authorization is invalid"
            )
        try:
            scan = verify_resolved_training_environment_cache(
                configured,
                cache_attestation,
                key=bytes.fromhex(cache_key_hex),
            )
        except ValueError as exc:
            raise RuntimeError(
                "Resolved training environment cache verification failed"
            ) from exc
        resolved = dict(configured)
    else:
        resolved, scan = build_resolved_training_environment_snapshot()
    try:
        digest = verify_resolved_training_environment(resolved)
        if configured is not None:
            configured_digest = verify_resolved_training_environment(configured)
            if configured_digest != digest or dict(configured) != resolved:
                raise ValueError("Installed resolved training environment drifted")
        configured_scan = cfg.get("resolvedTrainingEnvironmentScanAudit")
        if isinstance(configured_scan, Mapping) and any(
            configured_scan.get(field) != scan.get(field)
            for field in (
                "schemaVersion",
                "resolvedTrainingEnvironmentSHA256",
                "distributionCount",
                "installedFileCount",
                "totalHashedBytes",
            )
        ):
            raise ValueError("Resolved training environment scan audit drifted")
    except ValueError as exc:
        raise RuntimeError("Resolved training environment verification failed") from exc
    configured_digest = cfg.get("resolvedTrainingEnvironmentSHA256")
    if configured_digest is not None and configured_digest != digest:
        raise RuntimeError("resolvedTrainingEnvironmentSHA256 drifted")
    return resolved, digest, scan


def _training_runtime_lineage(
    cfg: Mapping[str, Any],
    *,
    phase: str = "sft",
) -> dict[str, Any]:
    code_manifest = cfg.get("trainingCodeManifest")
    if not isinstance(code_manifest, dict) or code_manifest.get("phase") != phase:
        raise RuntimeError(
            f"trainingCodeManifest must be the {phase} phase manifest"
        )
    expected_code_digest = _require_sha256(
        cfg.get("trainingCodeSHA256"),
        name="trainingCodeSHA256",
    )
    source_path = Path(__file__).resolve()
    deployed_root = source_path.parents[1]
    deployed_requirements = deployed_root / "requirements.txt"
    local_repository_root: Path | None = None
    deployed_space = (
        source_path.parent.name == "lumen_training"
        and deployed_requirements.is_file()
    )
    if deployed_space:
        actual_code_digest = verify_training_code_manifest(
            code_manifest,
            root=deployed_root,
        )
        requirements_path = deployed_requirements
    else:
        local_repository_root = source_path.parents[3]
        current_manifest = repository_training_code_bundle(local_repository_root)[
            "phases"
        ][phase]
        if current_manifest != code_manifest:
            raise RuntimeError("Repository training-code files drifted from the config")
        actual_code_digest = verify_training_code_manifest(code_manifest)
        requirements_path = (
            local_repository_root
            / "tools/hf_zerogpu/space_template/requirements.txt"
        )
    if actual_code_digest != expected_code_digest:
        raise RuntimeError(
            f"trainingCodeSHA256 does not match the verified {phase} code"
        )

    dependency_lock = cfg.get("trainingDependencyLock")
    if not isinstance(dependency_lock, dict):
        raise RuntimeError("trainingDependencyLock must be an object")
    expected_dependency_digest = _require_sha256(
        cfg.get("trainingDependencyLockSHA256"),
        name="trainingDependencyLockSHA256",
    )
    try:
        installed_versions = installed_controlled_package_versions(dependency_lock)
        dependency_digest = verify_training_dependency_lock(
            dependency_lock,
            requirements_path=requirements_path,
            installed_versions=installed_versions,
            installed_unsloth_revision=_installed_unsloth_revision(),
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise RuntimeError(
            f"Training dependency lineage verification failed: {exc}"
        ) from exc
    if (
        dependency_digest != expected_dependency_digest
        or dependency_lock.get("requirementsSHA256")
        != _require_sha256(cfg.get("requirementsSHA256"), name="requirementsSHA256")
    ):
        raise RuntimeError("Training dependency or requirements digest mismatch")
    (
        resolved_environment,
        resolved_environment_digest,
        resolved_environment_scan,
    ) = _resolved_environment_runtime_lineage(
        cfg,
        deployed_space=deployed_space,
    )
    observed_local_revision: str | None = None
    if cfg.get("runtimeSourceKind") == "git":
        if local_repository_root is None:
            raise RuntimeError(
                "Local Git runtime source requires execution from the repository checkout"
            )
        if cfg.get("runtimeSourceBindingMethod") == (
            "git_clean_worktree_plus_ubuntu_orchestration_manifest"
        ):
            from tools.fine_tuning.unsloth.ubuntu_source_integrity import (
                load_verified_attestation,
            )

            source_integrity = load_verified_attestation(local_repository_root)
            expected_integrity = cfg.get("ubuntuSourceIntegrity")
            if (
                not isinstance(expected_integrity, Mapping)
                or dict(expected_integrity) != source_integrity
                or cfg.get("workingTreeDigest")
                != source_integrity["workingTreeDigest"]
                or cfg.get("ubuntuOrchestrationCodeSHA256")
                != source_integrity["ubuntuOrchestrationCodeSHA256"]
                or cfg.get("ubuntuSourceIntegritySHA256")
                != source_integrity["sourceIntegritySHA256"]
            ):
                raise RuntimeError(
                    "Training config does not match the verified image source attestation"
                )
            observed_local_revision = source_integrity["baseCommit"]
        else:
            try:
                observed_local_revision = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=local_repository_root,
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                raise RuntimeError("Unable to observe the local Git runtime source") from exc
    try:
        runtime_source = validate_runtime_source_audit(
            cfg,
            observed_local_revision=observed_local_revision,
        )
    except ValueError as exc:
        raise RuntimeError("Training runtime source lineage is invalid") from exc
    space_configuration_sha256 = cfg.get("spaceConfigurationSHA256")
    if runtime_source["runtimeSourceKind"] == "huggingface_space":
        space_configuration_sha256 = _require_sha256(
            space_configuration_sha256,
            name="spaceConfigurationSHA256",
        )
    elif space_configuration_sha256 is not None:
        raise RuntimeError(
            "Local training must not claim a Hugging Face Space configuration"
        )
    hardware_lineage = _validated_hardware_lineage(cfg)
    return {
        "trainingCodeManifest": code_manifest,
        "trainingCodeSHA256": actual_code_digest,
        "trainingDependencyLock": dependency_lock,
        "trainingDependencyLockSHA256": dependency_digest,
        "requirementsSHA256": dependency_lock["requirementsSHA256"],
        "resolvedTrainingEnvironment": resolved_environment,
        "resolvedTrainingEnvironmentSHA256": resolved_environment_digest,
        "resolvedTrainingEnvironmentScanAudit": resolved_environment_scan,
        **hardware_lineage,
        "spaceConfigurationSHA256": space_configuration_sha256,
        **runtime_source,
    }


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(
        header + payload,
        usedforsecurity=False,
    ).hexdigest()


def _validated_base_model_tokenizer_closure(
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from lumen_manifest_crawler.dataset.adapter_evaluation import (
            DEFAULT_BASE_MODEL_ID,
            DEFAULT_BASE_MODEL_REVISION,
            DEFAULT_BASE_MODEL_TOKENIZER_CLOSURE_SHA256,
            DEFAULT_BASE_MODEL_TOKENIZER_FILES,
            canonical_base_model_tokenizer_closure,
        )
    except ImportError:
        from tools.lumen_manifest_crawler.lumen_manifest_crawler.dataset.adapter_evaluation import (
            DEFAULT_BASE_MODEL_ID,
            DEFAULT_BASE_MODEL_REVISION,
            DEFAULT_BASE_MODEL_TOKENIZER_CLOSURE_SHA256,
            DEFAULT_BASE_MODEL_TOKENIZER_FILES,
            canonical_base_model_tokenizer_closure,
        )

    base_model_id = cfg.get("baseModelID")
    base_model_name = cfg.get("base_model_name")
    if (
        not isinstance(base_model_id, str)
        or not base_model_id
        or base_model_id != base_model_name
    ):
        raise RuntimeError(
            "baseModelID must exactly match base_model_name"
        )
    files = cfg.get("baseModelTokenizerFiles")
    if not isinstance(files, list):
        raise RuntimeError("Base-model tokenizer file closure is missing")
    try:
        closure = canonical_base_model_tokenizer_closure(
            base_model_id=base_model_id,
            base_model_revision=cfg.get("baseModelRevision"),
            files=files,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Base-model tokenizer file closure is invalid") from exc
    declared = _require_sha256(
        cfg.get("baseModelTokenizerClosureSHA256"),
        name="baseModelTokenizerClosureSHA256",
    )
    tokenizer_json = next(
        item for item in closure["files"] if item["path"] == "tokenizer.json"
    )
    if (
        _canonical_sha256(closure) != declared
        or tokenizer_json["sha256"]
        != _require_sha256(
            cfg.get("baseModelTokenizerDigest"),
            name="baseModelTokenizerDigest",
        )
    ):
        raise RuntimeError("Base-model tokenizer closure digest drifted")
    if (
        closure["baseModelID"] == DEFAULT_BASE_MODEL_ID
        and closure["baseModelRevision"] == DEFAULT_BASE_MODEL_REVISION
        and (
            closure["files"] != DEFAULT_BASE_MODEL_TOKENIZER_FILES
            or declared != DEFAULT_BASE_MODEL_TOKENIZER_CLOSURE_SHA256
        )
    ):
        raise RuntimeError(
            "Pinned Qwen tokenizer closure drifted from the trusted registry"
        )
    return {
        **closure,
        "baseModelTokenizerClosureSHA256": declared,
    }


def _verified_private_runtime_model_snapshot(
    cfg: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Re-verify the self-contained private base view used by model runtime."""

    raw_path = cfg.get("baseModelRuntimeSnapshotPath")
    declared = cfg.get("baseModelRuntimeSnapshotVerification")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(declared, Mapping):
        raise RuntimeError(
            "Training config lacks the private base-model runtime snapshot binding"
        )
    try:
        snapshot_path = Path(raw_path).resolve(strict=True)
        observed = verify_private_base_model_conversion_snapshot(
            snapshot_path,
            base_model_id=str(cfg.get("baseModelID") or ""),
            base_model_name=str(cfg.get("base_model_name") or ""),
            base_model_revision=str(cfg.get("baseModelRevision") or ""),
            tokenizer_files=cfg.get("baseModelTokenizerFiles"),
            tokenizer_digest=str(cfg.get("baseModelTokenizerDigest") or ""),
            tokenizer_closure_sha256=str(
                cfg.get("baseModelTokenizerClosureSHA256") or ""
            ),
            generation_config_file=cfg.get("baseModelGenerationConfigFile"),
            model_index_digest=str(cfg.get("baseModelIndexDigest") or ""),
            index_referenced_shard_names=cfg.get(
                "baseModelIndexReferencedShardNames"
            ),
            index_shard_binding_sha256=str(
                cfg.get("baseModelIndexShardBindingSHA256") or ""
            ),
            model_artifact_digest=str(cfg.get("baseModelArtifactDigest") or ""),
            weight_shards=cfg.get("baseModelWeightShards"),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Private base-model runtime snapshot verification failed"
        ) from exc
    if (
        observed != dict(declared)
        or observed.get("snapshotPath") != str(snapshot_path)
        or Path(str(observed.get("snapshotPath") or "")).resolve(strict=True)
        != snapshot_path
    ):
        raise RuntimeError(
            "Private base-model runtime snapshot verification drifted from the config"
        )
    return snapshot_path, observed


def _verify_base_model_lineage(cfg: dict[str, Any]) -> None:
    revision = str(cfg.get("baseModelRevision") or "")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("baseModelRevision must be a full lowercase Hugging Face commit SHA")
    _validated_base_model_tokenizer_closure(cfg)
    _verified_private_runtime_tokenizer_snapshot(cfg)
    _verified_private_runtime_model_snapshot(cfg)


def _verified_private_runtime_tokenizer_snapshot(
    cfg: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Re-verify the exact private tokenizer closure consumed by runtime code."""

    raw_path = cfg.get("baseModelTokenizerSnapshotPath")
    declared = cfg.get("baseModelTokenizerSnapshotVerification")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(declared, Mapping):
        raise RuntimeError(
            "Training config lacks the private base-model tokenizer snapshot binding"
        )
    try:
        snapshot_path = Path(raw_path).resolve(strict=True)
        observed = verify_private_base_model_tokenizer_snapshot(
            snapshot_path,
            base_model_id=str(cfg.get("baseModelID") or ""),
            base_model_name=str(cfg.get("base_model_name") or ""),
            base_model_revision=str(cfg.get("baseModelRevision") or ""),
            tokenizer_files=cfg.get("baseModelTokenizerFiles"),
            tokenizer_digest=str(cfg.get("baseModelTokenizerDigest") or ""),
            tokenizer_closure_sha256=str(
                cfg.get("baseModelTokenizerClosureSHA256") or ""
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Private base-model tokenizer snapshot verification failed"
        ) from exc
    if (
        observed != dict(declared)
        or observed.get("snapshotPath") != str(snapshot_path)
        or Path(str(observed.get("snapshotPath") or "")).resolve(strict=True)
        != snapshot_path
    ):
        raise RuntimeError(
            "Private base-model tokenizer snapshot verification drifted from the config"
        )
    return snapshot_path, observed


def _load_verified_runtime_tokenizer_source(
    cfg: Mapping[str, Any],
) -> tuple[Any, Path, dict[str, Any]]:
    """Load a reference fast tokenizer from the self-contained runtime base view.

    Unsloth must already be imported before this function is called because importing
    ``AutoTokenizer`` imports Transformers.
    """

    snapshot_path, before = _verified_private_runtime_model_snapshot(cfg)
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    _, after = _verified_private_runtime_model_snapshot(cfg)
    if before != after:
        raise RuntimeError(
            "Private base-model runtime snapshot changed while loading the tokenizer"
        )
    if getattr(tokenizer, "is_fast", None) is not True:
        raise RuntimeError("Runtime requires the pinned fast tokenizer implementation")
    return tokenizer, snapshot_path, before


def _tokenizer_backend_contract(tokenizer: Any) -> dict[str, Any]:
    """Describe tokenization behavior without mutable runtime-only length settings."""

    if getattr(tokenizer, "is_fast", None) is not True:
        raise RuntimeError("Runtime tokenizer must be a fast tokenizer")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend, "to_str", None)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if not callable(backend_to_str) or not callable(get_vocab) or not callable(get_added_vocab):
        raise RuntimeError("Runtime tokenizer does not expose its fast-tokenizer state")
    backend_json = backend_to_str()
    vocab = get_vocab()
    added_vocab = get_added_vocab()
    if (
        not isinstance(backend_json, str)
        or not backend_json
        or not isinstance(vocab, Mapping)
        or not isinstance(added_vocab, Mapping)
        or any(not isinstance(key, str) or type(value) is not int for key, value in vocab.items())
        or any(
            not isinstance(key, str) or type(value) is not int
            for key, value in added_vocab.items()
        )
    ):
        raise RuntimeError("Runtime tokenizer exposes non-canonical backend state")
    try:
        parsed_backend = json.loads(backend_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Runtime tokenizer backend serialization is invalid") from exc

    probe_texts = (
        "",
        "Lumen routes a precise tool request.",
        'JSON {"selectedToolID":"files.read","requiresApproval":true}',
        "Unicode café 東京 🔐 and whitespace\n\tboundary",
        "<|im_start|>assistant\n/no_think<|im_end|>",
    )
    text_probes: list[dict[str, Any]] = []
    for text in probe_texts:
        for add_special_tokens in (False, True):
            encoded = tokenizer(text, add_special_tokens=add_special_tokens)
            input_ids = _flatten_tokenizer_output(encoded.get("input_ids"))
            if any(type(token_id) is not int or token_id < 0 for token_id in input_ids):
                raise RuntimeError("Runtime tokenizer probe returned invalid token IDs")
            text_probes.append(
                {
                    "textSHA256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "addSpecialTokens": add_special_tokens,
                    "inputIDs": input_ids,
                }
            )

    messages = [
        {"role": "system", "content": "Follow the exact Lumen contract."},
        {"role": "user", "content": "Read report.json and summarize it."},
    ]
    chat_probes: list[dict[str, Any]] = []
    for add_generation_prompt in (False, True):
        rendered = apply_non_thinking_chat_template(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        tokenized = apply_non_thinking_chat_template(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if not isinstance(rendered, str):
            raise RuntimeError("Runtime tokenizer chat probe did not render text")
        input_ids = _flatten_tokenizer_output(tokenized)
        chat_probes.append(
            {
                "addGenerationPrompt": add_generation_prompt,
                "renderedSHA256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "inputIDs": input_ids,
            }
        )

    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise RuntimeError("Runtime tokenizer lacks the controlled chat template")
    special_token_fields = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "mask_token_id",
    )
    payload = {
        "schemaVersion": "lumen.tokenizer-backend-contract/1.0.0",
        "backendSHA256": _canonical_sha256(parsed_backend),
        "vocabularySHA256": _canonical_sha256(dict(vocab)),
        "addedVocabularySHA256": _canonical_sha256(dict(added_vocab)),
        "specialTokenIDs": {
            field: getattr(tokenizer, field, None) for field in special_token_fields
        },
        "allSpecialTokenIDs": list(getattr(tokenizer, "all_special_ids", []) or []),
        "allSpecialTokens": list(getattr(tokenizer, "all_special_tokens", []) or []),
        "chatTemplateSHA256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "textProbes": text_probes,
        "chatProbes": chat_probes,
    }
    return {**payload, "contractSHA256": _canonical_sha256(payload)}


def _verify_runtime_tokenizer_binding(
    cfg: Mapping[str, Any],
    *,
    expected_tokenizer: Any,
    runtime_tokenizer: Any,
    snapshot_path: Path,
    snapshot_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove Unsloth used the verified bytes before any PEFT state is created."""

    actual_name = getattr(runtime_tokenizer, "name_or_path", None)
    try:
        actual_path = Path(str(actual_name or "")).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            "Unsloth runtime tokenizer is not bound to the private snapshot path"
        ) from exc
    if actual_path != snapshot_path:
        raise RuntimeError(
            "Unsloth runtime tokenizer ignored the private snapshot path"
        )
    expected_contract = _tokenizer_backend_contract(expected_tokenizer)
    runtime_contract = _tokenizer_backend_contract(runtime_tokenizer)
    if runtime_contract != expected_contract:
        raise RuntimeError(
            "Unsloth runtime tokenizer behavior drifted from the verified snapshot"
        )
    configured_max_length = cfg.get("max_seq_length")
    expected_runtime_max_length = _expected_unsloth_runtime_model_max_length(
        cfg,
        snapshot_path=snapshot_path,
    )
    runtime_max_length = getattr(runtime_tokenizer, "model_max_length", None)
    if (
        type(configured_max_length) is not int
        or configured_max_length <= 0
        or type(runtime_max_length) is not int
        or runtime_max_length != expected_runtime_max_length
        or getattr(runtime_tokenizer, "padding_side", None) != "left"
        or getattr(runtime_tokenizer, "truncation_side", None)
        != getattr(expected_tokenizer, "truncation_side", None)
    ):
        raise RuntimeError(
            "Unsloth runtime tokenizer applied an unapproved runtime transformation"
        )
    _, after = _verified_private_runtime_model_snapshot(cfg)
    if dict(snapshot_verification) != after:
        raise RuntimeError(
            "Private base-model runtime snapshot changed during Unsloth loading"
        )
    unsigned = {
        "schemaVersion": RUNTIME_TOKENIZER_BINDING_SCHEMA,
        "baseModelID": cfg.get("baseModelID"),
        "baseModelRevision": cfg.get("baseModelRevision"),
        "baseModelTokenizerClosureSHA256": cfg.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "runtimeSnapshotVerificationSHA256": after.get(
            "snapshotVerificationSHA256"
        ),
        "runtimeSnapshotPath": str(snapshot_path),
        "backendContractSHA256": runtime_contract["contractSHA256"],
        "allowedRuntimeTransformations": {
            "modelMaxLength": expected_runtime_max_length,
            "paddingSide": "left",
            "truncationSide": getattr(runtime_tokenizer, "truncation_side", None),
        },
    }
    return {**unsigned, "runtimeTokenizerBindingSHA256": _canonical_sha256(unsigned)}


def _expected_unsloth_runtime_model_max_length(
    cfg: Mapping[str, Any],
    *,
    snapshot_path: Path,
) -> int:
    """Reconstruct the pinned Unsloth tokenizer length from attested bytes.

    The controlled Unsloth revision passes
    ``max(requested length, config.max_position_embeddings)`` to the tokenizer
    loader.  Qwen3's bound config advertises 40,960 positions while Lumen trains
    with 4,096-token examples, so treating the requested length as the expected
    tokenizer property would reject the genuine pinned runtime before training.
    """

    configured_max_length = cfg.get("max_seq_length")
    if type(configured_max_length) is not int or configured_max_length <= 0:
        raise RuntimeError("Configured maximum sequence length is invalid")
    return max(
        configured_max_length,
        _bound_base_model_max_position_embeddings(
            cfg,
            snapshot_path=snapshot_path,
        ),
    )


def _bound_base_model_max_position_embeddings(
    cfg: Mapping[str, Any],
    *,
    snapshot_path: Path,
) -> int:
    """Read the original model context directly from the bound config bytes."""

    tokenizer_files = cfg.get("baseModelTokenizerFiles")
    if not isinstance(tokenizer_files, list):
        raise RuntimeError("Base-model tokenizer closure is missing config.json")
    config_records = [
        item
        for item in tokenizer_files
        if isinstance(item, Mapping) and item.get("path") == "config.json"
    ]
    if len(config_records) != 1:
        raise RuntimeError("Base-model tokenizer closure must bind one config.json")
    config_record = config_records[0]
    size_bytes = config_record.get("sizeBytes")
    config_sha256 = config_record.get("sha256")
    if (
        type(size_bytes) is not int
        or size_bytes <= 0
        or not isinstance(config_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", config_sha256) is None
    ):
        raise RuntimeError("Base-model config.json binding is invalid")
    payload = _read_verified_snapshot_file(
        snapshot_path / "config.json",
        expected_size=size_bytes,
        expected_sha256=config_sha256,
    )
    try:
        model_config = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Bound base-model config.json is invalid") from exc
    max_position_embeddings = (
        model_config.get("max_position_embeddings")
        if isinstance(model_config, Mapping)
        else None
    )
    if (
        type(max_position_embeddings) is not int
        or max_position_embeddings <= 0
    ):
        raise RuntimeError(
            "Bound base-model config lacks a valid max_position_embeddings"
        )
    return max_position_embeddings


def _runtime_4bit_materialization_evidence(
    runtime_model: Any,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove every Qwen target projection is materialized as CUDA BnB NF4."""

    if cfg.get("load_in_4bit") is not True:
        raise RuntimeError("The controlled runtime requires load_in_4bit=true")
    precision = _resolve_training_precision(cfg)
    expected_compute_dtype = precision["dtype"]
    configured_max_seq_length = cfg.get("max_seq_length")
    runtime_max_seq_length = getattr(runtime_model, "max_seq_length", None)
    if (
        type(configured_max_seq_length) is not int
        or configured_max_seq_length <= 0
        or type(runtime_max_seq_length) is not int
        or runtime_max_seq_length != configured_max_seq_length
    ):
        raise RuntimeError(
            "Unsloth runtime model max_seq_length drifted from the prepared config"
        )
    if (
        getattr(runtime_model, "is_loaded_in_4bit", None) is not True
        or getattr(runtime_model, "is_quantized", None) is not True
        or getattr(runtime_model, "quantization_method", None) != "bitsandbytes"
    ):
        raise RuntimeError("Unsloth runtime model was not materialized with 4-bit BitsAndBytes")

    model_config = getattr(runtime_model, "config", None)
    quantization_payload = getattr(model_config, "quantization_config", None)
    if type(quantization_payload) is not dict:
        raise RuntimeError("Runtime model lacks its plain BitsAndBytes config contract")
    quantizer = getattr(runtime_model, "hf_quantizer", None)
    quantizer_config = getattr(quantizer, "quantization_config", None)
    quantizer_config_to_dict = getattr(quantizer_config, "to_dict", None)
    quantizer_class = type(quantizer).__module__ + "." + type(quantizer).__name__
    quantizer_config_class = (
        type(quantizer_config).__module__ + "." + type(quantizer_config).__name__
    )
    if (
        quantizer_class
        != "transformers.quantizers.quantizer_bnb_4bit.Bnb4BitHfQuantizer"
        or quantizer_config_class
        != "transformers.utils.quantization_config.BitsAndBytesConfig"
        or not callable(quantizer_config_to_dict)
    ):
        raise RuntimeError("Runtime model lacks its active BitsAndBytes quantizer")
    active_quantization_payload = quantizer_config_to_dict()
    if not isinstance(active_quantization_payload, Mapping):
        raise RuntimeError("Active BitsAndBytes configuration is not serializable")

    def dtype_name(value: Any) -> str:
        return str(value or "").removeprefix("torch.")

    quant_method = active_quantization_payload.get("quant_method")
    quant_method = getattr(quant_method, "value", quant_method)
    expected_outer_config = {
        "load_in_4bit": True,
        "load_in_8bit": False,
        "quant_method": "bitsandbytes",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": expected_compute_dtype,
        "llm_int8_enable_fp32_cpu_offload": False,
        "llm_int8_has_fp16_weight": False,
        "llm_int8_threshold": 6.0,
        "llm_int8_skip_modules": None,
    }
    expected_active_config = {
        **expected_outer_config,
        "llm_int8_skip_modules": [
            "lm_head",
            "multi_modal_projector",
            "merger",
            "modality_projection",
            "router",
            "mlp.gate",
            "block_sparse_moe.gate",
            "mamba",
            "audio_tower",
            "vision_tower",
            "vision_embedder",
            "embed_vision",
            "embed_audio",
            "score",
            "classifier",
            "qa_outputs",
        ],
    }
    observed_config = {
        **{
            field: quantization_payload.get(field)
            for field in expected_outer_config
        },
        "quant_method": getattr(
            quantization_payload.get("quant_method"),
            "value",
            quantization_payload.get("quant_method"),
        ),
        "bnb_4bit_compute_dtype": dtype_name(
            quantization_payload.get("bnb_4bit_compute_dtype")
        ),
    }
    active_config = {
        **{
            field: active_quantization_payload.get(field)
            for field in expected_active_config
        },
        "quant_method": quant_method,
        "bnb_4bit_compute_dtype": dtype_name(
            active_quantization_payload.get("bnb_4bit_compute_dtype")
        ),
    }
    if (
        observed_config != expected_outer_config
        or active_config != expected_active_config
    ):
        raise RuntimeError(
            "Runtime BitsAndBytes configuration drifted from NF4 double-quant QLoRA"
        )

    config_to_dict = getattr(model_config, "to_dict", None)
    config_payload = config_to_dict() if callable(config_to_dict) else None
    layer_count = (
        config_payload.get("num_hidden_layers")
        if isinstance(config_payload, Mapping)
        else None
    )
    if (
        type(layer_count) is not int
        or layer_count <= 0
        or config_payload.get("model_type") != "qwen3"
        or config_payload.get("attention_bias") is not False
        or type(config_payload.get("tie_word_embeddings")) is not bool
    ):
        raise RuntimeError("Runtime model config lacks a valid Qwen layer count")
    expected_names = {
        *(
            f"model.layers.{layer}.self_attn.{projection}"
            for layer in range(layer_count)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        ),
        *(
            f"model.layers.{layer}.mlp.{projection}"
            for layer in range(layer_count)
            for projection in ("gate_proj", "up_proj", "down_proj")
        ),
    }
    named_modules = getattr(runtime_model, "named_modules", None)
    if not callable(named_modules):
        raise RuntimeError("Runtime model does not expose its quantized modules")
    observed_modules: dict[str, Any] = {}
    all_linear4bit_names: set[str] = set()
    module_records: list[dict[str, Any]] = []
    for module_name, module in named_modules():
        module_lineage = {
            f"{candidate.__module__}.{candidate.__name__}"
            for candidate in type(module).__mro__
        }
        if "bitsandbytes.nn.modules.Linear4bit" not in module_lineage:
            continue
        all_linear4bit_names.add(str(module_name))
        if module_name not in expected_names:
            continue
        if module_name in observed_modules:
            raise RuntimeError("Runtime model contains duplicate target modules")
        weight = getattr(module, "weight", None)
        parameter_lineage = {
            f"{candidate.__module__}.{candidate.__name__}"
            for candidate in type(weight).__mro__
        } if weight is not None else set()
        quant_state = getattr(weight, "quant_state", None)
        nested_quant_state = getattr(quant_state, "state2", None)
        device_type = getattr(getattr(weight, "device", None), "type", None)
        record = {
            "moduleName": str(module_name),
            "moduleClass": type(module).__module__ + "." + type(module).__name__,
            "parameterClass": type(weight).__module__ + "." + type(weight).__name__,
            "deviceType": device_type,
            "storageDType": dtype_name(getattr(weight, "dtype", None)),
            "computeDType": dtype_name(getattr(module, "compute_dtype", None)),
            "quantType": getattr(weight, "quant_type", None),
            "doubleQuantized": getattr(weight, "compress_statistics", None),
            "bnbQuantized": getattr(weight, "bnb_quantized", None),
            "requiresGrad": getattr(weight, "requires_grad", None),
            "quantStatePresent": quant_state is not None,
            "quantStateClass": (
                type(quant_state).__module__ + "." + type(quant_state).__name__
            ),
            "quantStateQuantType": getattr(quant_state, "quant_type", None),
            "quantStateNested": getattr(quant_state, "nested", None),
            "quantStateBlocksize": getattr(quant_state, "blocksize", None),
            "quantStateDType": dtype_name(getattr(quant_state, "dtype", None)),
            "quantStateNestedBlocksize": getattr(
                nested_quant_state,
                "blocksize",
                None,
            ),
        }
        if (
            "bitsandbytes.nn.modules.Params4bit" not in parameter_lineage
            or record
            != {
                **record,
                "deviceType": "cuda",
                "storageDType": "uint8",
                "computeDType": expected_compute_dtype,
                "quantType": "nf4",
                "doubleQuantized": True,
                "bnbQuantized": True,
                "requiresGrad": False,
                "quantStatePresent": True,
                "quantStateClass": "bitsandbytes.functional.QuantState",
                "quantStateQuantType": "nf4",
                "quantStateNested": True,
                "quantStateBlocksize": 64,
                "quantStateDType": expected_compute_dtype,
                "quantStateNestedBlocksize": 256,
            }
        ):
            raise RuntimeError(
                "Runtime Linear4bit target is not a materialized CUDA NF4 parameter"
            )
        observed_modules[str(module_name)] = module
        module_records.append(record)
    if set(observed_modules) != expected_names or all_linear4bit_names != expected_names:
        raise RuntimeError(
            "Runtime model does not contain the exact fully quantized Qwen projection set"
        )
    module_records.sort(key=lambda item: item["moduleName"])
    ordered_names = sorted(expected_names)
    vocab_size = config_payload.get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 4:
        raise RuntimeError("Runtime model config lacks a valid vocabulary size")
    representative_weight = getattr(
        observed_modules[ordered_names[0]],
        "weight",
        None,
    )
    expected_parameter_names = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        *(
            f"model.layers.{layer}.{name}"
            for layer in range(layer_count)
            for name in (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            )
        ),
    }
    if config_payload.get("tie_word_embeddings") is False:
        expected_parameter_names.add("lm_head.weight")
    named_parameters = getattr(runtime_model, "named_parameters", None)
    if not callable(named_parameters):
        raise RuntimeError("Runtime model does not expose parameter placement")
    observed_parameter_names: set[str] = set()
    for parameter_name, parameter in named_parameters():
        if (
            not isinstance(parameter_name, str)
            or not parameter_name
            or parameter_name in observed_parameter_names
            or getattr(getattr(parameter, "device", None), "type", None) != "cuda"
        ):
            raise RuntimeError(
                "Runtime model contains duplicate or non-CUDA parameter placement"
            )
        observed_parameter_names.add(parameter_name)
    if observed_parameter_names != expected_parameter_names:
        raise RuntimeError(
            "Runtime model parameter inventory drifted from the bound Qwen config"
        )
    ordered_parameter_names = sorted(expected_parameter_names)
    placement_unsigned = {
        "schemaVersion": "lumen.runtime-parameter-placement/1.0.0",
        "status": "passed",
        "totalParameterCount": len(ordered_parameter_names),
        "cudaParameterCount": len(ordered_parameter_names),
        "deviceTypeCounts": {"cuda": len(ordered_parameter_names)},
        "parameterNamesSHA256": _canonical_sha256(ordered_parameter_names),
        "allParametersOnCUDA": True,
    }
    parameter_placement = {
        **placement_unsigned,
        "runtimeParameterPlacementSHA256": _canonical_sha256(
            placement_unsigned
        ),
    }
    forward_probe = _runtime_cuda_forward_probe(
        runtime_model,
        compute_dtype=expected_compute_dtype,
        device=getattr(representative_weight, "device", None),
        vocab_size=vocab_size,
    )
    return {
        "requestedMaxSequenceLength": configured_max_seq_length,
        "runtimeMaxSequenceLength": runtime_max_seq_length,
        "requestedComputeDType": expected_compute_dtype,
        "runtimeIsLoadedIn4Bit": True,
        "runtimeIsQuantized": True,
        "runtimeQuantizationMethod": "bitsandbytes",
        "quantizerClass": quantizer_class,
        "activeQuantizationConfigClass": quantizer_config_class,
        "outerQuantizationConfig": expected_outer_config,
        "activeQuantizationConfig": expected_active_config,
        "expectedTargetModuleCount": len(expected_names),
        "materializedTargetModuleCount": len(observed_modules),
        "targetModuleNamesSHA256": _canonical_sha256(ordered_names),
        "materializedTargetModulesSHA256": _canonical_sha256(module_records),
        "representativeMaterializedTarget": module_records[0],
        "parameterPlacement": parameter_placement,
        "forwardKernelProbe": forward_probe,
    }


def _runtime_cuda_forward_probe(
    runtime_model: Any,
    *,
    compute_dtype: str,
    device: Any,
    vocab_size: int,
) -> dict[str, Any]:
    """Execute one bounded forward to prove patched Qwen and BnB CUDA kernels."""

    import torch  # type: ignore

    fixed_token_ids = (1, 2, 3, 4)
    fixed_attention_mask = (1, 1, 1, 1)
    input_contract = {
        "inputIDs": [list(fixed_token_ids)],
        "attentionMask": [list(fixed_attention_mask)],
        "useCache": False,
    }
    if (
        getattr(device, "type", None) != "cuda"
        or type(vocab_size) is not int
        or vocab_size <= max(fixed_token_ids)
    ):
        raise RuntimeError("Runtime forward probe lacks a valid CUDA model contract")
    eval_model = getattr(runtime_model, "eval", None)
    train_model = getattr(runtime_model, "train", None)
    was_training = getattr(runtime_model, "training", None)
    if (
        not callable(runtime_model)
        or not callable(eval_model)
        or not callable(train_model)
        or was_training is not False
    ):
        raise RuntimeError("Runtime model cannot execute a controlled forward probe")

    input_ids = torch.tensor(
        [fixed_token_ids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.tensor(
        [fixed_attention_mask],
        dtype=torch.long,
        device=device,
    )
    eval_model()
    try:
        with torch.inference_mode():
            outputs = runtime_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    except Exception as exc:
        raise RuntimeError(
            "Runtime Qwen/BitsAndBytes CUDA forward probe failed"
        ) from exc
    finally:
        if was_training:
            train_model()

    logits = getattr(outputs, "logits", None)
    logits_shape = tuple(getattr(logits, "shape", ()))
    logits_device_type = getattr(getattr(logits, "device", None), "type", None)
    logits_dtype = str(getattr(logits, "dtype", "")).removeprefix("torch.")
    try:
        all_finite = bool(torch.isfinite(logits).all().item())
    except Exception as exc:
        raise RuntimeError(
            "Runtime forward probe did not return inspectable logits"
        ) from exc
    if (
        logits_shape != (1, len(fixed_token_ids), vocab_size)
        or logits_device_type != "cuda"
        or logits_dtype != compute_dtype
        or getattr(logits, "requires_grad", None) is not False
        or not all_finite
    ):
        raise RuntimeError(
            "Runtime forward probe logits violate the controlled CUDA contract"
        )
    unsigned = {
        "schemaVersion": "lumen.runtime-forward-kernel-probe/1.0.0",
        "status": "passed",
        "fixedInputSHA256": _canonical_sha256(input_contract),
        "batchSize": 1,
        "tokenCount": len(fixed_token_ids),
        "logitsShape": list(logits_shape),
        "logitsDType": logits_dtype,
        "logitsDeviceType": logits_device_type,
        "allFinite": True,
        "requiresGrad": False,
        "useCache": False,
    }
    return {
        **unsigned,
        "runtimeForwardKernelProbeSHA256": _canonical_sha256(unsigned),
    }


def _verify_runtime_model_binding(
    cfg: Mapping[str, Any],
    *,
    runtime_model: Any,
    snapshot_path: Path,
    snapshot_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the model and controlled generation defaults use the private view."""

    model_config = getattr(runtime_model, "config", None)
    config_to_dict = getattr(model_config, "to_dict", None)
    actual_name = getattr(model_config, "_name_or_path", None)
    try:
        actual_path = Path(str(actual_name or "")).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            "Unsloth runtime model config is not bound to the private snapshot"
        ) from exc
    if actual_path != snapshot_path or not callable(config_to_dict):
        raise RuntimeError("Unsloth runtime model ignored the private snapshot path")
    quantization_evidence = _runtime_4bit_materialization_evidence(
        runtime_model,
        cfg,
    )

    # generation_config.json is a separately bound regular file in the private
    # view. Load it independently and require the model's effective defaults to
    # match, so no implicit cache or model-config fallback can change generation.
    from transformers import GenerationConfig  # type: ignore

    expected_generation_config = GenerationConfig.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
    )
    runtime_generation_config = getattr(runtime_model, "generation_config", None)
    expected_generation_to_dict = getattr(expected_generation_config, "to_dict", None)
    runtime_generation_to_dict = getattr(runtime_generation_config, "to_dict", None)
    if not callable(expected_generation_to_dict) or not callable(
        runtime_generation_to_dict
    ):
        raise RuntimeError(
            "Runtime generation configuration is not bound to the verified file"
        )
    model_config_payload = config_to_dict()
    expected_generation_config_payload = expected_generation_to_dict()
    generation_config_payload = runtime_generation_to_dict()
    if not isinstance(model_config_payload, Mapping) or not isinstance(
        generation_config_payload, Mapping
    ) or not isinstance(expected_generation_config_payload, Mapping):
        raise RuntimeError("Runtime model configuration is not serializable")
    bound_base_max_position_embeddings = (
        _bound_base_model_max_position_embeddings(
            cfg,
            snapshot_path=snapshot_path,
        )
    )
    configured_max_length = cfg.get("max_seq_length")
    if type(configured_max_length) is not int or configured_max_length <= 0:
        raise RuntimeError("Configured maximum sequence length is invalid")
    expected_runtime_model_max_position_embeddings = max(
        configured_max_length,
        bound_base_max_position_embeddings,
    )
    runtime_model_max_position_embeddings = model_config_payload.get(
        "max_position_embeddings"
    )
    if (
        type(runtime_model_max_position_embeddings) is not int
        or runtime_model_max_position_embeddings
        != expected_runtime_model_max_position_embeddings
    ):
        raise RuntimeError(
            "Runtime model maximum context drifted from the pinned Unsloth load"
        )
    expected_runtime_max_length = runtime_model_max_position_embeddings
    approved_generation_config_payload = dict(expected_generation_config_payload)
    source_max_length = approved_generation_config_payload.get("max_length")
    approved_generation_config_payload["max_length"] = expected_runtime_max_length
    if dict(generation_config_payload) != approved_generation_config_payload:
        raise RuntimeError(
            "Runtime generation configuration contains an unapproved transformation"
        )

    _, after = _verified_private_runtime_model_snapshot(cfg)
    if dict(snapshot_verification) != after:
        raise RuntimeError(
            "Private base-model runtime snapshot changed during model loading"
        )
    unsigned = {
        "schemaVersion": RUNTIME_MODEL_BINDING_SCHEMA,
        "baseModelID": cfg.get("baseModelID"),
        "baseModelRevision": cfg.get("baseModelRevision"),
        "baseModelIndexDigest": cfg.get("baseModelIndexDigest"),
        "baseModelIndexShardBindingSHA256": cfg.get(
            "baseModelIndexShardBindingSHA256"
        ),
        "baseModelArtifactDigest": cfg.get("baseModelArtifactDigest"),
        "baseModelTokenizerClosureSHA256": cfg.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "baseModelGenerationConfigFile": cfg.get(
            "baseModelGenerationConfigFile"
        ),
        "runtimeSnapshotVerificationSHA256": after.get(
            "snapshotVerificationSHA256"
        ),
        "runtimeSnapshotPath": str(snapshot_path),
        "modelConfigSHA256": _canonical_sha256(dict(model_config_payload)),
        "modelConfigVerificationStatus": (
            "attested_runtime_observation_not_independently_reconstructed"
        ),
        "sourceGenerationConfigSHA256": _canonical_sha256(
            dict(expected_generation_config_payload)
        ),
        "generationConfigSHA256": _canonical_sha256(
            dict(generation_config_payload)
        ),
        "generationConfigSource": "verified_private_generation_config_file",
        "allowedGenerationConfigTransformations": {
            "maxLength": {
                "source": "verified_runtime_model.config.max_position_embeddings",
                "sourceValue": expected_runtime_max_length,
                "originalValue": source_max_length,
                "runtimeValue": generation_config_payload.get("max_length"),
            }
        },
        "runtimeLoadMaterialization": quantization_evidence,
        "localFilesOnly": True,
    }
    return {**unsigned, "runtimeModelBindingSHA256": _canonical_sha256(unsigned)}


def _normalize_peft_base_model_identity(
    model: Any,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Prevent private runtime paths or floating revisions in saved adapters."""

    base_model_id = cfg.get("baseModelID")
    revision = cfg.get("baseModelRevision")
    peft_configs = getattr(model, "peft_config", None)
    get_base_model = getattr(model, "get_base_model", None)
    peft_base_model = getattr(model, "base_model", None)
    if (
        not isinstance(base_model_id, str)
        or not base_model_id
        or base_model_id != cfg.get("base_model_name")
        or not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or not isinstance(peft_configs, Mapping)
        or not peft_configs
        or not callable(get_base_model)
        or peft_base_model is None
    ):
        raise RuntimeError("PEFT adapter base-model identity is unavailable")
    try:
        portable_adapter_model_card(base_model_id, revision)
    except ValueError as exc:
        raise RuntimeError(
            "PEFT adapter base-model identity is not canonical and immutable"
        ) from exc
    runtime_base_model = get_base_model()
    runtime_base_config = getattr(runtime_base_model, "config", None)
    if runtime_base_config is None:
        raise RuntimeError("PEFT runtime base-model config is unavailable")
    try:
        # PEFT 0.19.1 independently reads these two sources while creating
        # README.md: model.config.to_dict()["_name_or_path"] and
        # model.base_model.name_or_path. Normalize both before Trainer can
        # write any checkpoint, not just before the final adapter save.
        setattr(runtime_base_config, "_name_or_path", base_model_id)
        setattr(runtime_base_config, "name_or_path", base_model_id)
        setattr(runtime_base_model, "name_or_path", base_model_id)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "PEFT model-card base-model identity normalization failed"
        ) from exc
    adapter_names: list[str] = []
    for adapter_name, peft_config in peft_configs.items():
        if not isinstance(adapter_name, str) or not adapter_name:
            raise RuntimeError("PEFT adapter name is invalid")
        setattr(peft_config, "base_model_name_or_path", base_model_id)
        setattr(peft_config, "revision", revision)
        if (
            getattr(peft_config, "base_model_name_or_path", None) != base_model_id
            or getattr(peft_config, "revision", None) != revision
        ):
            raise RuntimeError("PEFT adapter base-model identity normalization failed")
        adapter_names.append(adapter_name)
    model_card_config = getattr(model, "config", None)
    to_dict = getattr(model_card_config, "to_dict", None)
    if not callable(to_dict):
        raise RuntimeError("PEFT model-card config identity is unavailable")
    model_card_config_payload = to_dict()
    if (
        not isinstance(model_card_config_payload, Mapping)
        or model_card_config_payload.get("_name_or_path") != base_model_id
        or getattr(peft_base_model, "name_or_path", None) != base_model_id
        or getattr(runtime_base_model, "name_or_path", None) != base_model_id
    ):
        raise RuntimeError(
            "PEFT model-card base-model identity normalization failed"
        )
    unsigned = {
        "schemaVersion": "lumen.peft-base-model-identity/1.0.0",
        "baseModelID": base_model_id,
        "baseModelRevision": revision,
        "adapterNames": sorted(adapter_names),
        "privateRuntimePathPersisted": False,
    }
    return {**unsigned, "peftBaseModelIdentitySHA256": _canonical_sha256(unsigned)}


def _save_portable_peft_adapter(
    model: Any,
    output_dir: Path,
    cfg: Mapping[str, Any],
    *,
    selected_adapters: list[str] | None = None,
    expected_adapter_names: list[str] | None = None,
) -> dict[str, Any]:
    """Save one normalized PEFT artifact and replace its generated model card."""

    identity = _normalize_peft_base_model_identity(model, cfg)
    if (
        expected_adapter_names is not None
        and identity["adapterNames"] != sorted(expected_adapter_names)
    ):
        raise RuntimeError(
            "Final PEFT artifact contains an unexpected adapter set"
        )
    save_kwargs: dict[str, Any] = {"safe_serialization": True}
    if selected_adapters is not None:
        if (
            not selected_adapters
            or len(selected_adapters) != len(set(selected_adapters))
            or any(
                adapter_name not in identity["adapterNames"]
                for adapter_name in selected_adapters
            )
        ):
            raise RuntimeError("Selected PEFT adapter set is invalid")
        save_kwargs["selected_adapters"] = selected_adapters
    model.save_pretrained(str(output_dir), **save_kwargs)
    write_portable_adapter_model_card(
        output_dir,
        base_model_id=identity["baseModelID"],
        base_model_revision=identity["baseModelRevision"],
    )
    return identity


def _runtime_tokenizer_evidence(
    cfg: Mapping[str, Any],
    *,
    snapshot_path: Path,
    snapshot_verification: Mapping[str, Any],
    runtime_model_binding: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact, independently reconstructable runtime-tokenizer evidence."""

    try:
        configured_runtime_path = Path(
            str(cfg.get("baseModelRuntimeSnapshotPath") or "")
        ).resolve(strict=True)
        tokenizer_snapshot_path, tokenizer_snapshot_verification = (
            _verified_private_runtime_tokenizer_snapshot(cfg)
        )
    except OSError as exc:
        raise RuntimeError(
            "Runtime-tokenizer evidence lacks its verified private snapshot"
        ) from exc
    verification = dict(snapshot_verification)
    model_binding = dict(runtime_model_binding)
    binding = dict(runtime_binding)
    unsigned_model_binding = dict(model_binding)
    model_binding_sha256 = unsigned_model_binding.pop(
        "runtimeModelBindingSHA256", None
    )
    unsigned_binding = dict(binding)
    binding_sha256 = unsigned_binding.pop("runtimeTokenizerBindingSHA256", None)
    if (
        configured_runtime_path != snapshot_path
        or verification
        != dict(cfg.get("baseModelRuntimeSnapshotVerification") or {})
        or model_binding.get("schemaVersion") != RUNTIME_MODEL_BINDING_SCHEMA
        or not isinstance(model_binding_sha256, str)
        or model_binding_sha256 != _canonical_sha256(unsigned_model_binding)
        or model_binding.get("baseModelID") != cfg.get("baseModelID")
        or model_binding.get("baseModelRevision") != cfg.get("baseModelRevision")
        or model_binding.get("runtimeSnapshotVerificationSHA256")
        != verification.get("snapshotVerificationSHA256")
        or model_binding.get("runtimeSnapshotPath") != str(snapshot_path)
        or binding.get("schemaVersion") != RUNTIME_TOKENIZER_BINDING_SCHEMA
        or not isinstance(binding_sha256, str)
        or binding_sha256 != _canonical_sha256(unsigned_binding)
        or binding.get("baseModelID") != cfg.get("baseModelID")
        or binding.get("baseModelRevision") != cfg.get("baseModelRevision")
        or binding.get("baseModelTokenizerClosureSHA256")
        != cfg.get("baseModelTokenizerClosureSHA256")
        or binding.get("runtimeSnapshotVerificationSHA256")
        != verification.get("snapshotVerificationSHA256")
        or binding.get("runtimeSnapshotPath") != str(snapshot_path)
    ):
        raise RuntimeError("Runtime-tokenizer evidence is not self-consistent")
    return {
        "baseModelTokenizerDigest": cfg.get("baseModelTokenizerDigest"),
        "baseModelTokenizerFiles": cfg.get("baseModelTokenizerFiles"),
        "baseModelTokenizerClosureSHA256": cfg.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "baseModelGenerationConfigFile": cfg.get(
            "baseModelGenerationConfigFile"
        ),
        "baseModelTokenizerSnapshotPath": str(tokenizer_snapshot_path),
        "baseModelTokenizerSnapshotVerification": tokenizer_snapshot_verification,
        "baseModelRuntimeSnapshotPath": str(snapshot_path),
        "baseModelRuntimeSnapshotVerification": verification,
        "runtimeModelBinding": model_binding,
        "runtimeTokenizerBinding": binding,
    }


def _load_verified_unsloth_runtime(
    cfg: Mapping[str, Any],
    *,
    fast_language_model: Any,
) -> tuple[Any, Any, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load and verify the exact base runtime shared by training and smoke checks."""

    (
        expected_runtime_tokenizer,
        runtime_snapshot_path,
        runtime_snapshot_verification,
    ) = _load_verified_runtime_tokenizer_source(cfg)
    model, tokenizer = fast_language_model.from_pretrained(
        model_name=str(runtime_snapshot_path),
        revision=cfg["baseModelRevision"],
        tokenizer_name=str(runtime_snapshot_path),
        max_seq_length=int(cfg["max_seq_length"]),
        dtype=_controlled_torch_dtype(cfg),
        load_in_4bit=bool(cfg["load_in_4bit"]),
        local_files_only=True,
        trust_remote_code=False,
        use_exact_model_name=True,
    )
    runtime_model_binding = _verify_runtime_model_binding(
        cfg,
        runtime_model=model,
        snapshot_path=runtime_snapshot_path,
        snapshot_verification=runtime_snapshot_verification,
    )
    runtime_tokenizer_binding = _verify_runtime_tokenizer_binding(
        cfg,
        expected_tokenizer=expected_runtime_tokenizer,
        runtime_tokenizer=tokenizer,
        snapshot_path=runtime_snapshot_path,
        snapshot_verification=runtime_snapshot_verification,
    )
    verify_chat_template_contract(cfg["chatTemplateContract"], tokenizer=tokenizer)
    return (
        model,
        tokenizer,
        runtime_snapshot_path,
        runtime_snapshot_verification,
        runtime_model_binding,
        runtime_tokenizer_binding,
    )


def _run_runtime_binding_smoke(
    cfg: Mapping[str, Any],
    *,
    fast_language_model: Any,
) -> dict[str, Any]:
    """Exercise the real pinned Unsloth loader without PEFT or trainer setup."""

    (
        model,
        tokenizer,
        snapshot_path,
        snapshot_verification,
        runtime_model_binding,
        runtime_tokenizer_binding,
    ) = _load_verified_unsloth_runtime(
        cfg,
        fast_language_model=fast_language_model,
    )
    evidence = _runtime_tokenizer_evidence(
        cfg,
        snapshot_path=snapshot_path,
        snapshot_verification=snapshot_verification,
        runtime_model_binding=runtime_model_binding,
        runtime_binding=runtime_tokenizer_binding,
    )
    # Keep the smoke artifact independent of object reprs and GPU addresses.
    del model, tokenizer
    unsigned = {
        "schemaVersion": "lumen.runtime-binding-smoke/1.0.0",
        **evidence,
    }
    return {
        **unsigned,
        "runtimeBindingSmokeSHA256": _canonical_sha256(unsigned),
    }


def _read_verified_snapshot_file(
    source: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> bytes:
    def signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise RuntimeError("Adapter tokenizer publication requires O_NOFOLLOW")
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("Adapter tokenizer source is not a regular file")
        chunks: list[bytes] = []
        offset = 0
        while True:
            chunk = os.pread(descriptor, 1 << 20, offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        rebound = source.stat(follow_symlinks=False)
        if (
            signature(before) != signature(after)
            or signature(before) != signature(rebound)
        ):
            raise RuntimeError("Adapter tokenizer source changed during copying")
    finally:
        os.close(descriptor)
    if (
        len(payload) != expected_size
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise RuntimeError("Adapter tokenizer source drifted from the base closure")
    return payload


def _write_regular_file_atomic(path: Path, payload: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError("Adapter tokenizer destination is unsafe")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600, follow_symlinks=False)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _publish_exact_base_tokenizer_subset(
    cfg: Mapping[str, Any],
    *,
    adapter_output_dir: Path,
    snapshot_path: Path,
    snapshot_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish only exact base tokenizer bytes, never Unsloth's mutable derivative."""

    unexpected = sorted(
        filename
        for filename in ADAPTER_DERIVED_TOKENIZER_FILES
        if (adapter_output_dir / filename).exists()
        or (adapter_output_dir / filename).is_symlink()
    )
    if unexpected:
        raise RuntimeError(
            "Adapter output contains unapproved derived tokenizer files: "
            + ", ".join(unexpected)
        )
    expected_by_path = {
        str(item["path"]): item
        for item in cfg.get("baseModelTokenizerFiles", [])
        if isinstance(item, Mapping)
    }
    published: list[dict[str, Any]] = []
    for filename in ADAPTER_BASE_TOKENIZER_FILES:
        expected = expected_by_path.get(filename)
        if not isinstance(expected, Mapping):
            raise RuntimeError(
                f"Base tokenizer closure lacks adapter publication file {filename}"
            )
        payload = _read_verified_snapshot_file(
            snapshot_path / filename,
            expected_size=int(expected["sizeBytes"]),
            expected_sha256=str(expected["sha256"]),
        )
        destination = adapter_output_dir / filename
        _write_regular_file_atomic(destination, payload)
        if destination.read_bytes() != payload:
            raise RuntimeError("Adapter tokenizer publication changed after writing")
        published.append(
            {
                "path": filename,
                "sizeBytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    _, after = _verified_private_runtime_model_snapshot(cfg)
    if after != dict(snapshot_verification):
        raise RuntimeError(
            "Private base-model runtime snapshot changed during adapter publication"
        )
    unsigned = {
        "schemaVersion": "lumen.adapter-base-tokenizer-binding/1.0.0",
        "baseModelTokenizerClosureSHA256": cfg.get(
            "baseModelTokenizerClosureSHA256"
        ),
        "runtimeSnapshotVerificationSHA256": after.get(
            "snapshotVerificationSHA256"
        ),
        "files": published,
        "transformation": "exact_byte_subset_no_derived_tokenizer",
    }
    return {**unsigned, "adapterTokenizerBindingSHA256": _canonical_sha256(unsigned)}


def _base_model_weight_shard_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, list) or not value:
        raise RuntimeError("baseModelWeightShards must be a non-empty list")
    shards: list[dict[str, Any]] = []
    filenames: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise RuntimeError("baseModelWeightShards entries must be objects")
        filename = item.get("filename")
        size = item.get("size")
        digest = item.get("sha256")
        if (
            not isinstance(filename, str)
            or filename != filename.rsplit("/", 1)[-1]
            or not filename.endswith(".safetensors")
            or filename in filenames
            or type(size) is not int
            or size <= 0
        ):
            raise RuntimeError("baseModelWeightShards contains invalid shard metadata")
        filenames.add(filename)
        shards.append(
            {
                "filename": filename,
                "size": size,
                "sha256": _require_sha256(digest, name=f"baseModelWeightShards[{filename}].sha256"),
            }
        )
    return {
        "schemaVersion": "lumen.base-model-weight-shards/1.0.0",
        "shards": sorted(shards, key=lambda item: item["filename"]),
    }


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _package_version(name: str) -> str:
    try:
        import importlib.metadata as md  # type: ignore

        return md.version(name)
    except Exception:
        return ""


def _precision_flags(cfg: Mapping[str, Any]) -> tuple[bool, bool]:
    precision = _resolve_training_precision(cfg)
    return precision["bf16"], precision["fp16"]


def _limit_records(records: list[dict[str, Any]], value: Any) -> list[dict[str, Any]]:
    limit = int(value or 0)
    if limit <= 0:
        return records
    return records[:limit]


def _finalize_variant_manifest(
    cfg: dict[str, Any],
    *,
    adapter_artifact_manifest: dict[str, Any],
    training_environment: dict[str, Any],
    training_runtime_lineage: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    source_path = Path(cfg["dataset_dir"]).resolve() / "variant_manifest.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Expected experiment variant manifest: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise RuntimeError("Experiment variant manifest must be a JSON object")
    if (
        source.get("variant") != cfg["variant"]
        or source.get("agent") != cfg["agent"]
        or source.get("variantManifestSHA256") != cfg["variantManifestSHA256"]
    ):
        raise RuntimeError("Training config is not bound to the selected experiment variant manifest")

    repo_root = Path(__file__).resolve().parents[3]
    crawler_root = repo_root / "tools" / "lumen_manifest_crawler"
    if crawler_root.is_dir() and str(crawler_root) not in sys.path:
        sys.path.insert(0, str(crawler_root))
    try:
        from lumen_manifest_crawler.dataset.adapter_evaluation import (
            finalize_experiment_variant_manifest,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The experiment-manifest finalizer must be bundled with the training runtime"
        ) from exc

    finalized = finalize_experiment_variant_manifest(
        source,
        adapter_sha256=adapter_artifact_manifest["adapterSHA256"],
        adapter_artifact_manifest=adapter_artifact_manifest,
        training_environment={
            **training_environment,
            "trainingCodeSHA256": training_runtime_lineage[
                "trainingCodeSHA256"
            ],
            "trainingDependencyLockSHA256": training_runtime_lineage[
                "trainingDependencyLockSHA256"
            ],
            "requirementsSHA256": training_runtime_lineage[
                "requirementsSHA256"
            ],
            "resolvedTrainingEnvironment": training_runtime_lineage[
                "resolvedTrainingEnvironment"
            ],
            "resolvedTrainingEnvironmentSHA256": training_runtime_lineage[
                "resolvedTrainingEnvironmentSHA256"
            ],
            "spaceConfigurationSHA256": training_runtime_lineage[
                "spaceConfigurationSHA256"
            ],
            "runtimeSourceKind": training_runtime_lineage["runtimeSourceKind"],
            "runtimeSourceRevision": training_runtime_lineage[
                "runtimeSourceRevision"
            ],
            "expectedRuntimeSourceRevision": training_runtime_lineage[
                "expectedRuntimeSourceRevision"
            ],
            "observedRepositoryRevision": training_runtime_lineage[
                "observedRepositoryRevision"
            ],
            "observedRuntimeRevision": training_runtime_lineage[
                "observedRuntimeRevision"
            ],
            "runtimeSourceBindingStatus": training_runtime_lineage[
                "runtimeSourceBindingStatus"
            ],
            "runtimeSourceBindingMethod": training_runtime_lineage[
                "runtimeSourceBindingMethod"
            ],
        },
        training_phase="sft",
    )
    destination = output_dir / "finalized_variant_manifest.json"
    _write_json_atomic(destination, finalized)
    return finalized


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    seed, seed_source = _resolve_controlled_seed(cfg, cli_seed=args.seed)

    if args.runtime_binding_smoke:
        training_runtime_lineage = _training_runtime_lineage(cfg)
        _training_environment(
            cfg,
            runtime_lineage=training_runtime_lineage,
        )
        _verify_base_model_lineage(cfg)
        _require_unsloth_before_transformers()
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise RuntimeError(
                "Runtime-binding smoke requires the pinned Unsloth environment"
            ) from exc
        _seed_everything(seed)
        smoke = _run_runtime_binding_smoke(
            cfg,
            fast_language_model=FastLanguageModel,
        )
        sys.stdout.write(json.dumps(smoke, sort_keys=True, separators=(",", ":")))
        sys.stdout.write("\n")
        return

    dataset_dir = Path(cfg["dataset_dir"]).resolve()
    train_path = dataset_dir / "train_sft.jsonl"
    val_path = dataset_dir / "val_sft.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Expected {train_path} and {val_path}")
    assistant_only_loss = bool(
        args.assistant_only_loss or cfg.get("assistant_only_loss", False)
    )
    unbound_checkpoints: list[Path] = []
    ubuntu_sft_checkpoint_lineage = bool(cfg.get("sftCheckpointLineagePath"))
    if ubuntu_sft_checkpoint_lineage:
        (
            resume_checkpoint,
            checkpoint_lineage_path,
            unbound_checkpoints,
        ) = _bind_and_validate_sft_checkpoint_lineage(
            cfg,
            cfg_path=cfg_path,
            assistant_only_loss=assistant_only_loss,
            require_checkpoint=bool(args.resume_from_checkpoint),
        )
        if unbound_checkpoints:
            _prune_unbound_sft_checkpoints(unbound_checkpoints)
    else:
        resume_checkpoint, checkpoint_lineage_path = _validate_checkpoint_lineage(
            cfg,
            cfg_path=cfg_path,
            require_checkpoint=bool(args.resume_from_checkpoint),
            assistant_only_loss=assistant_only_loss,
        )

    training_runtime_lineage = _training_runtime_lineage(cfg)
    training_environment = _training_environment(
        cfg,
        runtime_lineage=training_runtime_lineage,
    )
    _verify_base_model_lineage(cfg)

    _require_unsloth_before_transformers()
    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from transformers import TrainerCallback
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for Unsloth SFT training. Install: unsloth, trl, datasets, transformers, peft, accelerate, bitsandbytes."
        ) from exc
    except AssertionError as exc:
        if "CUDA" in str(exc).upper():
            raise RuntimeError(
                "Unsloth SFT training requires a CUDA-enabled PyTorch runtime. "
                "This host imported Unsloth, but Torch is not compiled with CUDA enabled."
            ) from exc
        raise

    # Unsloth must patch Transformers before the shared seed helper imports it.
    _seed_everything(seed)

    (
        model,
        tokenizer,
        runtime_tokenizer_snapshot_path,
        runtime_tokenizer_snapshot_verification,
        runtime_model_binding,
        runtime_tokenizer_binding,
    ) = _load_verified_unsloth_runtime(
        cfg,
        fast_language_model=FastLanguageModel,
    )

    train_records = _limit_records(load_jsonl(train_path), cfg.get("max_train_records"))
    val_records = _limit_records(load_jsonl(val_path), cfg.get("max_val_records"))
    max_sequence_length = cfg["max_seq_length"]
    token_length_preflight = _preflight_sft_token_lengths(
        {
            "train": (train_records, train_path),
            "validation": (val_records, val_path),
        },
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        minimum_sequence_margin_tokens=cfg.get(
            "sft_minimum_sequence_margin_tokens",
            SFT_MINIMUM_SEQUENCE_MARGIN_TOKENS,
        ),
        agent=cfg.get("agent"),
        fleet_loss_share_contract=cfg.get("fleetLossShareContract"),
        public_corpus_loss_share_contract=cfg.get(
            "publicCorpusLossShareContract"
        ),
        fleet_config=cfg,
    )
    fleet_row_evidence: list[Mapping[str, Any]] | None = None
    fleet_epoch_orders: list[list[int]] | None = None
    fleet_sampler: _FleetEpochStratifiedSampler | None = None
    if cfg.get("agent") == "fleet":
        fleet_evidence = token_length_preflight.get("fleetLossShareEvidence")
        if not isinstance(fleet_evidence, Mapping):
            raise RuntimeError("Fleet SFT preflight is missing loss-share evidence")
        fleet_splits = fleet_evidence.get("splits")
        fleet_train = (
            fleet_splits.get("train")
            if isinstance(fleet_splits, Mapping)
            else None
        )
        if not isinstance(fleet_train, Mapping):
            raise RuntimeError("Fleet SFT preflight is missing train evidence")
        fleet_row_evidence = fleet_train.get("rowTokenEvidence")
        optimizer_family_band = fleet_evidence.get(
            "optimizerFamilyShareBand"
        )
        fleet_contract = cfg.get("fleetLossShareContract")
        schedule_contract = (
            fleet_contract.get("sftOptimizerWindowScheduleContract")
            if isinstance(fleet_contract, Mapping)
            else None
        )
        if not isinstance(fleet_row_evidence, list) or not isinstance(
            optimizer_family_band,
            Mapping,
        ) or not isinstance(schedule_contract, Mapping):
            raise RuntimeError("Fleet SFT preflight schedule inputs are missing")
        rebuilt_schedule, fleet_epoch_orders = (
            _build_fleet_sft_optimizer_window_schedule(
                row_token_evidence=fleet_row_evidence,
                config=cfg,
                schedule_contract=schedule_contract,
                minimum_basis_points=optimizer_family_band[
                    "minimumBasisPoints"
                ],
                maximum_basis_points=optimizer_family_band[
                    "maximumBasisPoints"
                ],
            )
        )
        if fleet_train.get("optimizerWindowSchedule") != rebuilt_schedule:
            raise RuntimeError("Fleet SFT optimizer-window schedule drifted")
    token_length_preflight_evidence = (
        _bind_sft_token_length_preflight(
            cfg,
            cfg_path=cfg_path,
            preflight=token_length_preflight,
        )
        if ubuntu_sft_checkpoint_lineage
        else token_length_preflight
    )
    if ubuntu_sft_checkpoint_lineage:
        _verify_prepared_global_tokenizer_preflight(
            cfg,
            cfg_path=cfg_path,
            phase="sft",
            bound_preflight=token_length_preflight_evidence,
        )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        target_modules=cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        use_gradient_checkpointing="unsloth" if cfg.get("gradient_checkpointing", True) else False,
        random_state=seed,
    )
    peft_base_model_identity = _normalize_peft_base_model_identity(model, cfg)

    output_dir, adapter_output_dir = validate_sft_artifact_paths(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_sft_rows(
        train_records,
        tokenizer=tokenizer,
        assistant_only_loss=assistant_only_loss,
        path=train_path,
        max_seq_length=int(cfg["max_seq_length"]),
    )
    val_rows = build_sft_rows(
        val_records,
        tokenizer=tokenizer,
        assistant_only_loss=assistant_only_loss,
        path=val_path,
        max_seq_length=int(cfg["max_seq_length"]),
    )

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(val_rows) if val_rows else None
    precision = _resolve_training_precision(cfg)
    bf16, fp16 = precision["bf16"], precision["fp16"]
    checkpoint_save_steps, checkpoint_save_total_limit = _sft_checkpoint_policy(cfg)

    sft_kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=max(1, int(cfg["batch_size"])),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        warmup_steps=int(cfg["warmup_steps"]),
        logging_steps=int(cfg.get("logging_steps", 10)),
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="steps",
        save_steps=checkpoint_save_steps,
        save_total_limit=checkpoint_save_total_limit,
        save_only_model=False,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        max_length=int(cfg["max_seq_length"]),
        packing=bool(cfg.get("packing", False)),
        seed=seed,
        data_seed=seed,
        dataloader_drop_last=False,
    )
    # Assistant-only rows are pre-tokenized with masked labels, so TRL should
    # treat them as an already-processed dataset instead of rebuilding masks.
    if not assistant_only_loss:
        sft_kwargs["dataset_text_field"] = "text"
    _apply_fleet_sft_batching_policy(
        sft_kwargs,
        enabled=fleet_epoch_orders is not None,
    )

    training_args = SFTConfig(**sft_kwargs)
    sft_config_padding_free = getattr(training_args, "padding_free", None)
    if fleet_epoch_orders is not None:
        _validate_fleet_sft_trainer_args(training_args)
        fleet_sampler = _FleetEpochStratifiedSampler(fleet_epoch_orders)

        class _FleetSFTTrainer(SFTTrainer):
            def _get_train_sampler(self, train_dataset=None):
                selected_dataset = (
                    self.train_dataset
                    if train_dataset is None
                    else train_dataset
                )
                if len(selected_dataset) != len(fleet_sampler):
                    raise RuntimeError(
                        "Fleet SFT sampler and train dataset lengths differ"
                    )
                return fleet_sampler

        trainer_class = _FleetSFTTrainer
    else:
        trainer_class = SFTTrainer

    trainer = trainer_class(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    fleet_runtime_loss_normalization: dict[str, Any] | None = None
    if fleet_epoch_orders is not None:
        if fleet_row_evidence is None or fleet_sampler is None:
            raise RuntimeError(
                "Fleet SFT runtime normalization lacks schedule evidence"
            )
        fleet_runtime_loss_normalization = (
            _attest_fleet_sft_runtime_loss_normalization(
                trainer,
                assistant_only_loss=assistant_only_loss,
                sft_config_padding_free=sft_config_padding_free,
                config=cfg,
                row_token_evidence=fleet_row_evidence,
                epoch_orders=fleet_epoch_orders,
                fleet_sampler=fleet_sampler,
            )
        )
    if checkpoint_lineage_path is not None:
        trainer.add_callback(
            (
                _sft_checkpoint_lineage_callback(
                    TrainerCallback,
                    record_path=checkpoint_lineage_path,
                )
                if ubuntu_sft_checkpoint_lineage
                else _checkpoint_lineage_callback(
                    TrainerCallback,
                    record_path=checkpoint_lineage_path,
                )
            )
        )

    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint is not None else None
        )
    )
    evaluation_metrics = trainer.evaluate() if eval_dataset is not None else {}
    training_completion = _verified_training_completion_evidence(
        trainer,
        training_args,
        train_result,
        evaluation_metrics,
        has_eval_dataset=eval_dataset is not None,
        train_record_count=len(train_rows),
        expected_precision=precision,
    )
    peft_base_model_identity = _save_portable_peft_adapter(
        trainer.model,
        adapter_output_dir,
        cfg,
        expected_adapter_names=["default"],
    )
    adapter_tokenizer_binding = _publish_exact_base_tokenizer_subset(
        cfg,
        adapter_output_dir=adapter_output_dir,
        snapshot_path=runtime_tokenizer_snapshot_path,
        snapshot_verification=runtime_tokenizer_snapshot_verification,
    )
    adapter_artifact_manifest = write_adapter_artifact_manifest(
        adapter_output_dir,
        training_phase="sft",
        expected_base_model=cfg["baseModelID"],
        expected_base_revision=cfg["baseModelRevision"],
    )
    finalized_variant_manifest = _finalize_variant_manifest(
        cfg,
        adapter_artifact_manifest=adapter_artifact_manifest,
        training_environment=training_environment,
        training_runtime_lineage=training_runtime_lineage,
        output_dir=output_dir,
    )

    repo_root = Path(__file__).resolve().parents[3]
    manifest = {
        "schema": "lumen.train_sft.manifest/1.2.0",
        "agent": cfg["agent"],
        "base_model_name": cfg["base_model_name"],
        "baseModelID": cfg["baseModelID"],
        "baseModelRevision": cfg["baseModelRevision"],
        "baseModelIndexDigest": cfg["baseModelIndexDigest"],
        "baseModelIndexReferencedShardNames": cfg[
            "baseModelIndexReferencedShardNames"
        ],
        "baseModelIndexShardBindingSHA256": cfg[
            "baseModelIndexShardBindingSHA256"
        ],
        "baseModelArtifactDigest": cfg["baseModelArtifactDigest"],
        "baseModelWeightShards": cfg["baseModelWeightShards"],
        "baseModelTokenizerDigest": cfg["baseModelTokenizerDigest"],
        "baseModelTokenizerFiles": cfg["baseModelTokenizerFiles"],
        "baseModelTokenizerClosureSHA256": cfg[
            "baseModelTokenizerClosureSHA256"
        ],
        "baseModelGenerationConfigFile": cfg["baseModelGenerationConfigFile"],
        "baseModelTokenizerSnapshotVerification": (
            cfg["baseModelTokenizerSnapshotVerification"]
        ),
        "baseModelTokenizerSnapshotPath": cfg["baseModelTokenizerSnapshotPath"],
        "baseModelRuntimeSnapshotPath": str(runtime_tokenizer_snapshot_path),
        "baseModelRuntimeSnapshotVerification": (
            runtime_tokenizer_snapshot_verification
        ),
        "runtimeModelBinding": runtime_model_binding,
        "runtimeTokenizerBinding": runtime_tokenizer_binding,
        "peftBaseModelIdentity": peft_base_model_identity,
        "adapterTokenizerBinding": adapter_tokenizer_binding,
        "trainingEnvironment": training_environment,
        "trainingEnvironmentSHA256": training_environment["trainingEnvironmentSHA256"],
        **training_runtime_lineage,
        "config_path": str(cfg_path),
        "config_sha256": _hash_file(cfg_path),
        "dataset_dir": str(dataset_dir),
        "datasetRepository": cfg.get("datasetRepository"),
        "datasetRevision": cfg.get("datasetRevision"),
        "runResumeLineageSHA256": cfg.get("runResumeLineageSHA256"),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_sha256": _hash_file(train_path),
        "val_sha256": _hash_file(val_path),
        "train_records": len(train_rows),
        "val_records": len(val_rows),
        "max_seq_length": cfg["max_seq_length"],
        "load_in_4bit": cfg["load_in_4bit"],
        "packing": bool(cfg.get("packing", False)),
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", True)),
        "precision": precision,
        "assistant_only_loss": assistant_only_loss,
        "resume_from_checkpoint": resume_checkpoint is not None,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "checkpoint_lineage": (
            str(checkpoint_lineage_path) if checkpoint_lineage_path else None
        ),
        "checkpoint_save_steps": int(training_args.save_steps),
        "checkpoint_save_total_limit": int(training_args.save_total_limit),
        "checkpoint_recovery_discarded_unbound": [
            str(path) for path in unbound_checkpoints
        ],
        "token_length_preflight": token_length_preflight_evidence,
        "token_length_preflight_path": cfg.get("sftTokenLengthPreflightPath"),
        "token_length_preflight_sha256": token_length_preflight_evidence.get(
            "preflightSHA256"
        ),
        "seed": seed,
        "seed_source": seed_source,
        "output_dir": str(output_dir),
        "adapter_output_dir": str(adapter_output_dir),
        "adapterSHA256": adapter_artifact_manifest["adapterSHA256"],
        "adapterArtifactManifest": str(
            adapter_output_dir / "adapter_artifact_manifest.json"
        ),
        "finalizedVariantManifest": str(output_dir / "finalized_variant_manifest.json"),
        "finalizedVariantManifestSHA256": finalized_variant_manifest[
            "variantManifestSHA256"
        ],
        "trainingPhase": "sft",
        "git_sha": _git_sha(repo_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "package_versions": {
            name: _package_version(name)
            for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes", "unsloth")
        },
    }

    report = {
        **manifest,
        **(
            {
                "fleetRuntimeLossNormalization": (
                    fleet_runtime_loss_normalization
                )
            }
            if fleet_runtime_loss_normalization is not None
            else {}
        ),
        "trainingCompletion": training_completion,
        "metrics": train_result.metrics,
        "evaluation_metrics": evaluation_metrics,
    }
    _write_json_atomic(output_dir / "training_report.json", report)
    _write_json_atomic(output_dir / "train_manifest.json", manifest)


if __name__ == "__main__":
    main()
