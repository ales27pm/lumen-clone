from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
ZERO_GPU_LINEAGE_FIELDS = (
    "zeroGPUSize",
    "zeroGPUDurationSeconds",
    "observedAccelerator",
)


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


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in sorted(REQUIRED_CONFIG_KEYS) if key not in cfg]
    if missing:
        raise ValueError(f"Config is missing required keys: {', '.join(missing)}")
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
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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
    rendered = tokenizer.apply_chat_template(
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
    }
    if _tokenizer_supports_assistant_masks(tokenizer):
        chat_template_kwargs["return_assistant_tokens_mask"] = True
    processed = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
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

    if max_seq_length is not None and max_seq_length > 0:
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        assistant_masks = assistant_masks[:max_seq_length]

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
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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


def _precision_flags(cfg: dict[str, Any]) -> tuple[bool, bool]:
    has_bf16 = "bf16" in cfg
    has_fp16 = "fp16" in cfg
    if has_bf16 or has_fp16:
        bf16 = bool(cfg.get("bf16", False))
        fp16 = bool(cfg.get("fp16", not bf16))
        if bf16 and fp16:
            fp16 = False
        return bf16, fp16

    try:
        import torch  # type: ignore

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return True, False
    except Exception:
        pass
    return False, True


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
    resume_checkpoint, checkpoint_lineage_path = _validate_checkpoint_lineage(
        cfg,
        cfg_path=cfg_path,
        require_checkpoint=bool(args.resume_from_checkpoint),
        assistant_only_loss=bool(
            args.assistant_only_loss or cfg.get("assistant_only_loss", False)
        ),
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

    train_records = _limit_records(load_jsonl(train_path), cfg.get("max_train_records"))
    val_records = _limit_records(load_jsonl(val_path), cfg.get("max_val_records"))

    output_dir, adapter_output_dir = validate_sft_artifact_paths(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_output_dir.mkdir(parents=True, exist_ok=True)

    assistant_only_loss = bool(args.assistant_only_loss or cfg.get("assistant_only_loss", False))
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
    bf16, fp16 = _precision_flags(cfg)

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
        save_strategy="epoch",
        save_total_limit=int(cfg.get("save_total_limit", 2)),
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
            _checkpoint_lineage_callback(
                TrainerCallback,
                record_path=checkpoint_lineage_path,
            )
        )

    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint is not None else None
        )
    )
    evaluation_metrics = trainer.evaluate() if eval_dataset is not None else {}
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
        "assistant_only_loss": assistant_only_loss,
        "resume_from_checkpoint": resume_checkpoint is not None,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "checkpoint_lineage": (
            str(checkpoint_lineage_path) if checkpoint_lineage_path else None
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
        "metrics": train_result.metrics,
        "evaluation_metrics": evaluation_metrics,
    }
    _write_json_atomic(output_dir / "training_report.json", report)
    _write_json_atomic(output_dir / "train_manifest.json", manifest)


if __name__ == "__main__":
    main()
