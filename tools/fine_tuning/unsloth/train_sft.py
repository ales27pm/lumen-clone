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
    from .adapter_artifact import write_adapter_artifact_manifest
    from .training_lineage import (
        build_resolved_training_environment,
        build_resolved_training_environment_snapshot,
        canonical_controlled_package_version,
        installed_controlled_package_versions,
        repository_training_code_bundle,
        validate_runtime_source_audit,
        verify_resolved_training_environment,
        verify_resolved_training_environment_cache,
        verify_training_code_manifest,
        verify_training_dependency_lock,
        ZERO_GPU_ALLOWED_SIZES,
    )
except ImportError:
    from adapter_artifact import write_adapter_artifact_manifest
    from training_lineage import (
        build_resolved_training_environment,
        build_resolved_training_environment_snapshot,
        canonical_controlled_package_version,
        installed_controlled_package_versions,
        repository_training_code_bundle,
        validate_runtime_source_audit,
        verify_resolved_training_environment,
        verify_resolved_training_environment_cache,
        verify_training_code_manifest,
        verify_training_dependency_lock,
        ZERO_GPU_ALLOWED_SIZES,
    )


REQUIRED_CONFIG_KEYS = {
    "agent",
    "base_model_name",
    "baseModelRevision",
    "baseModelIndexDigest",
    "baseModelIndexReferencedShardNames",
    "baseModelIndexShardBindingSHA256",
    "baseModelArtifactDigest",
    "baseModelWeightShards",
    "baseModelTokenizerDigest",
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
    "seed",
}
AGENTS = {"cortex", "executor", "mouth", "mimicry", "rem", "fleet"}
FINETUNE_MARKERS = {"sft", "dpo", "orpo", "lora", "merged", "adapter", "finetune", "finetuned", "training"}
CHECKPOINT_LINEAGE_SCHEMA = "lumen.zerogpu.checkpoint_lineage/1.0.0"
CHECKPOINT_DIRECTORY_SCHEMA = "lumen.zerogpu.checkpoint_directory/1.0.0"
RUN_RESUME_LINEAGE_SCHEMA = "lumen.zerogpu.run_resume_lineage/1.0.0"
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
SFT_TOKEN_LENGTH_PREFLIGHT_SCHEMA = "lumen.sft_token_length_preflight/1.1.0"
FLEET_LOSS_SHARE_CONTRACT_SCHEMA = "lumen.fleet-loss-share/1.1.0"
FLEET_LOSS_SHARE_EVIDENCE_SCHEMA = "lumen.fleet-loss-share-evidence/1.1.0"
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
        "perSourceFamilyNumeratorTokenCounts": (
            "supplementalStaticChosenTargetTokenCountsBySourceFamily"
        ),
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
    if agent == "fleet":
        _validated_fleet_loss_share_contract(
            fleet_loss_share_contract,
            lane="sft",
            config=fleet_config,
        )
    elif fleet_loss_share_contract is not None:
        raise RuntimeError("Fleet loss-share contract is forbidden for non-Fleet SFT")
    aggregate_total: list[int] = []
    aggregate_assistant: list[int] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    fleet_target_rows: dict[
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
            assistant_tokens = sum(
                1 for label in tokenized["labels"] if label != -100
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
        if agent == "fleet":
            fleet_target_rows[split] = split_fleet_target_rows
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
    }
    if agent == "fleet":
        report["fleetLossShareEvidence"] = _build_fleet_loss_share_evidence(
            contract_value=fleet_loss_share_contract,
            lane="sft",
            split_target_rows=fleet_target_rows,
            config=fleet_config,
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
            "maximumSupplementalStaticShareBasisPoints",
            "contract",
        },
        label="Fleet source-selection proxy",
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
        or source_selection_proxy.get("maximumSupplementalStaticShareBasisPoints")
        != 1_500
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
        {"baseModelID", "baseModelRevision", "tokenizerSHA256"},
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
    ):
        raise RuntimeError("Fleet tokenizer binding is malformed")
    if config is not None and (
        tokenizer_binding.get("baseModelID") != config.get("base_model_name")
        or tokenizer_binding.get("baseModelRevision")
        != config.get("baseModelRevision")
        or tokenizer_binding.get("tokenizerSHA256")
        != config.get("baseModelTokenizerDigest")
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
            field_names["denominatorTokenCount"]: denominator,
            field_names["supplementalNumeratorTokenCount"]: supplemental,
            field_names["publicNumeratorTokenCount"]: public,
            field_names["perSourceFamilyNumeratorTokenCounts"]: dict(
                sorted(supplemental_by_family.items())
            ),
        }

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
    if not isinstance(variant_attestation, dict):
        raise RuntimeError("Training config is missing its variant attestation")
    expected_agent = {
        "sourceVariantManifestSHA256": cfg.get("variantManifestSHA256"),
        "laneHashes": variant_attestation.get("laneHashes"),
        "trainingCorpusSHA256": variant_attestation.get("trainingCorpusSHA256"),
        "controlledTrainingConfigSHA256": variant_attestation.get(
            "effectiveTrainingConfigSHA256"
        ),
        "baseModelID": cfg.get("baseModelID", cfg.get("base_model_name")),
        "baseModelRevision": cfg.get("baseModelRevision"),
        "baseModelIndexDigest": cfg.get("baseModelIndexDigest"),
        "baseModelIndexShardBindingSHA256": cfg.get(
            "baseModelIndexShardBindingSHA256"
        ),
        "baseModelArtifactDigest": cfg.get("baseModelArtifactDigest"),
        "baseModelWeightShards": cfg.get("baseModelWeightShards"),
        "baseModelTokenizerDigest": cfg.get("baseModelTokenizerDigest"),
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
    if not isinstance(expected_dataset_files, dict):
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


def _sft_checkpoint_directory_manifest(
    checkpoint: Path,
    *,
    expected_base_model: str,
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
    precision = _resolve_training_precision(config)
    updated = dict(record)
    updated["checkpoints"] = [
        {
            "path": candidate.name,
            "checkpointSHA256": _sft_checkpoint_directory_manifest(
                candidate,
                expected_base_model=expected_base_model,
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


def _verify_base_model_lineage(cfg: dict[str, Any]) -> None:
    revision = str(cfg.get("baseModelRevision") or "")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("baseModelRevision must be a full lowercase Hugging Face commit SHA")
    expected = {
        "model.safetensors.index.json": _require_sha256(
            cfg.get("baseModelIndexDigest"), name="baseModelIndexDigest"
        ),
        "tokenizer.json": _require_sha256(
            cfg.get("baseModelTokenizerDigest"), name="baseModelTokenizerDigest"
        ),
    }
    from huggingface_hub import hf_hub_download  # type: ignore

    downloaded: dict[str, Path] = {}
    for filename, digest in expected.items():
        path = Path(hf_hub_download(repo_id=cfg["base_model_name"], filename=filename, revision=revision))
        downloaded[filename] = path
        if _hash_file(path) != digest:
            raise RuntimeError(f"Pinned base-model artifact digest mismatch: {filename}")

    shard_contract = _base_model_weight_shard_contract(cfg.get("baseModelWeightShards"))
    if _canonical_sha256(shard_contract) != _require_sha256(
        cfg.get("baseModelArtifactDigest"), name="baseModelArtifactDigest"
    ):
        raise RuntimeError("baseModelArtifactDigest does not match baseModelWeightShards")
    try:
        index = json.loads(downloaded["model.safetensors.index.json"].read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict) or any(
            not isinstance(filename, str) for filename in weight_map.values()
        ):
            raise TypeError
        referenced_shards = sorted(set(weight_map.values()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Pinned base-model index has an invalid weight_map") from exc
    expected_shards = [item["filename"] for item in shard_contract["shards"]]
    if referenced_shards != expected_shards:
        raise RuntimeError("Pinned base-model index shard set does not match baseModelWeightShards")
    declared_referenced_shards = cfg.get("baseModelIndexReferencedShardNames")
    if declared_referenced_shards != referenced_shards:
        raise RuntimeError(
            "Pinned base-model index shard set does not match baseModelIndexReferencedShardNames"
        )
    index_shard_binding = {
        "schemaVersion": "lumen.base-model-index-shard-binding/1.0.0",
        "indexDigest": expected["model.safetensors.index.json"],
        "referencedShardNames": referenced_shards,
        "shardContractDigest": cfg["baseModelArtifactDigest"],
    }
    if _canonical_sha256(index_shard_binding) != _require_sha256(
        cfg.get("baseModelIndexShardBindingSHA256"),
        name="baseModelIndexShardBindingSHA256",
    ):
        raise RuntimeError("baseModelIndexShardBindingSHA256 does not bind the verified index and shards")
    for item in shard_contract["shards"]:
        path = Path(
            hf_hub_download(
                repo_id=cfg["base_model_name"],
                filename=item["filename"],
                revision=revision,
            )
        )
        if path.stat().st_size != item["size"] or _hash_file(path) != item["sha256"]:
            raise RuntimeError(f"Pinned base-model weight shard digest mismatch: {item['filename']}")


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

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model_name"],
        revision=cfg["baseModelRevision"],
        max_seq_length=int(cfg["max_seq_length"]),
        load_in_4bit=bool(cfg["load_in_4bit"]),
        use_exact_model_name=True,
    )
    verify_chat_template_contract(cfg["chatTemplateContract"], tokenizer=tokenizer)

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
        fleet_config=cfg,
    )
    token_length_preflight_evidence = (
        _bind_sft_token_length_preflight(
            cfg,
            cfg_path=cfg_path,
            preflight=token_length_preflight,
        )
        if ubuntu_sft_checkpoint_lineage
        else token_length_preflight
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
    )
    # Assistant-only rows are pre-tokenized with masked labels, so TRL should
    # treat them as an already-processed dataset instead of rebuilding masks.
    if not assistant_only_loss:
        sft_kwargs["dataset_text_field"] = "text"

    training_args = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
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
    trainer.model.save_pretrained(str(adapter_output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_output_dir), legacy_format=False)
    adapter_artifact_manifest = write_adapter_artifact_manifest(
        adapter_output_dir,
        training_phase="sft",
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
        "schema": "lumen.train_sft.manifest/1.0.0",
        "agent": cfg["agent"],
        "base_model_name": cfg["base_model_name"],
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
        "trainingCompletion": training_completion,
        "metrics": train_result.metrics,
        "evaluation_metrics": evaluation_metrics,
    }
    _write_json_atomic(output_dir / "training_report.json", report)
    _write_json_atomic(output_dir / "train_manifest.json", manifest)


if __name__ == "__main__":
    main()
