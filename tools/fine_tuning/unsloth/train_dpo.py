from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from lumen_manifest_crawler.dataset.chat_template_contract import (
        canonical_non_thinking_messages,
        non_thinking_template_kwargs,
        verify_chat_template_contract,
    )
except ImportError:
    from tools.lumen_manifest_crawler.lumen_manifest_crawler.dataset.chat_template_contract import (
        canonical_non_thinking_messages,
        non_thinking_template_kwargs,
        verify_chat_template_contract,
    )

try:
    from .adapter_artifact import verify_adapter_artifact, write_adapter_artifact_manifest
    from .training_lineage import (
        build_resolved_training_environment,
        TRAINING_VARIANT_ATTESTATION_SCHEMA,
        verify_resolved_training_environment,
    )
    from .train_sft import (
        CHECKPOINT_SCALER_FILENAME,
        CHECKPOINT_SCALER_STATE_SCHEMA,
        _build_fleet_loss_share_evidence,
        _build_public_corpus_loss_share_evidence,
        _checkpoint_scaler_state_contract,
        _controlled_torch_dtype,
        _resolve_training_precision,
        _resolve_controlled_seed,
        _load_verified_runtime_tokenizer_source,
        _normalize_peft_base_model_identity,
        _publish_exact_base_tokenizer_subset,
        _require_unsloth_before_transformers,
        _save_portable_peft_adapter,
        _seed_everything,
        _training_environment,
        _training_runtime_lineage,
        _verify_prepared_global_tokenizer_preflight,
        _verify_base_model_lineage as _verify_sft_base_model_lineage,
        _verify_runtime_model_binding,
        _verify_runtime_tokenizer_binding,
        _validated_fleet_loss_share_contract,
        _validated_public_corpus_loss_share_contract,
        _verified_training_completion_evidence,
        _write_json_atomic,
    )
except ImportError:
    from adapter_artifact import verify_adapter_artifact, write_adapter_artifact_manifest
    from training_lineage import (
        build_resolved_training_environment,
        TRAINING_VARIANT_ATTESTATION_SCHEMA,
        verify_resolved_training_environment,
    )
    from train_sft import (
        CHECKPOINT_SCALER_FILENAME,
        CHECKPOINT_SCALER_STATE_SCHEMA,
        _build_fleet_loss_share_evidence,
        _build_public_corpus_loss_share_evidence,
        _checkpoint_scaler_state_contract,
        _controlled_torch_dtype,
        _resolve_training_precision,
        _resolve_controlled_seed,
        _load_verified_runtime_tokenizer_source,
        _normalize_peft_base_model_identity,
        _publish_exact_base_tokenizer_subset,
        _require_unsloth_before_transformers,
        _save_portable_peft_adapter,
        _seed_everything,
        _training_environment,
        _training_runtime_lineage,
        _verify_prepared_global_tokenizer_preflight,
        _verify_base_model_lineage as _verify_sft_base_model_lineage,
        _verify_runtime_model_binding,
        _verify_runtime_tokenizer_binding,
        _validated_fleet_loss_share_contract,
        _validated_public_corpus_loss_share_contract,
        _verified_training_completion_evidence,
        _write_json_atomic,
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
    "trainingCodeManifestsByPhase",
    "trainingCodeSHA256ByPhase",
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
    "preference_trainer",
    "dpo_learning_rate",
    "dpo_num_train_epochs",
    "dpo_beta",
    "max_prompt_length",
    "use_logits_to_keep",
    "precompute_ref_log_probs",
    "precompute_ref_batch_size",
    "gradient_checkpointing",
    "output_dir",
    "adapter_output_dir",
    "dpo_output_dir",
    "dataset_dir",
    "variant",
    "variantManifestSHA256",
    "publicCorpusLossShareContract",
    "seed",
}
AGENTS = {"cortex", "executor", "mouth", "mimicry", "rem", "fleet"}
FINETUNE_MARKERS = {"sft", "dpo", "orpo", "lora", "merged", "adapter", "finetune", "finetuned", "training"}
POLICY_ADAPTER_NAME = "default"
REFERENCE_ADAPTER_NAME = "lumen_sft_reference"
CUDA_ALLOCATOR_CONFIG_ENV = "PYTORCH_CUDA_ALLOC_CONF"
PREFERENCE_CHECKPOINT_LINEAGE_SCHEMA = (
    "lumen.preference_checkpoint_lineage/1.3.0"
)
PREFERENCE_CHECKPOINT_DIRECTORY_SCHEMA = (
    "lumen.preference_checkpoint_directory/1.1.0"
)
PREFERENCE_CHECKPOINT_SAVE_STEPS = 5
PREFERENCE_CHECKPOINT_MINIMUM_RETENTION = 2
PREFERENCE_CHECKPOINT_REQUIRED_FILES = frozenset(
    {
        "adapter_config.json",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
)
PREFERENCE_TOKEN_LENGTH_PREFLIGHT_SCHEMA = (
    "lumen.preference_token_length_preflight/1.4.0"
)
PREFERENCE_TOKENIZATION_TRANSCRIPT_SCHEMA = (
    "lumen.preference-tokenization-transcript/1.0.0"
)
PREFERENCE_MINIMUM_PROMPT_MARGIN_TOKENS = 64
PREFERENCE_MINIMUM_SEQUENCE_MARGIN_TOKENS = 128
DPO_TRL_VERSION = "0.24.0"
DPO_COMPLETION_APPENDED_EOS_TOKENS = 1
DPO_COMPLETION_TOKENIZATION_POLICY = {
    "trainerImplementation": "trl.DPOTrainer.tokenize_row",
    "trlVersion": DPO_TRL_VERSION,
    "completionTokenization": "add_special_tokens_false",
    "completionSuffix": "append_tokenizer_eos_token_id",
    "appendedEOSTokensPerCompletion": DPO_COMPLETION_APPENDED_EOS_TOKENS,
}
DPO_REFERENCE_LOG_PROB_EVIDENCE_SCHEMA = (
    "lumen.dpo_reference_log_prob_evidence/1.0.0"
)
DPO_REFERENCE_LOG_PROB_COLUMNS = (
    "ref_chosen_logps",
    "ref_rejected_logps",
)
DPO_REFERENCE_LOG_PROB_EVIDENCE_FILENAME = "reference_log_prob_evidence.json"


class _IncompletePreferenceCheckpoint(RuntimeError):
    """A structurally unfinished checkpoint that can only be discarded when stale."""


def _latch_expandable_cuda_allocator(
    *,
    environ: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Initialize and verify the CUDA allocator before Unsloth rewrites its env."""

    environment = os.environ if environ is None else environ
    raw_config = environment.get(CUDA_ALLOCATOR_CONFIG_ENV)
    if not isinstance(raw_config, str) or not raw_config.strip():
        raise RuntimeError(
            f"DPO requires {CUDA_ALLOCATOR_CONFIG_ENV}=expandable_segments:True"
        )
    settings: dict[str, str] = {}
    for raw_setting in raw_config.split(","):
        key, separator, value = raw_setting.strip().partition(":")
        if not separator or not key or key in settings:
            raise RuntimeError(f"Invalid {CUDA_ALLOCATOR_CONFIG_ENV} setting")
        settings[key] = value
    if settings.get("expandable_segments") != "True":
        raise RuntimeError(
            f"DPO requires {CUDA_ALLOCATOR_CONFIG_ENV}=expandable_segments:True"
        )

    if torch_module is None:
        import torch as torch_module  # type: ignore

    if not torch_module.cuda.is_available():
        raise RuntimeError("DPO allocator verification requires a CUDA accelerator")

    probe = torch_module.empty(1, device="cuda")
    try:
        probe_address = int(probe.data_ptr())
        matching_segments = [
            segment
            for segment in torch_module.cuda.memory_snapshot()
            if (
                type(segment.get("address")) is int
                and type(segment.get("total_size")) is int
                and segment["address"] <= probe_address
                < segment["address"] + segment["total_size"]
            )
        ]
        if (
            len(matching_segments) != 1
            or matching_segments[0].get("is_expandable") is not True
        ):
            raise RuntimeError(
                "DPO CUDA allocator did not enable expandable segments before Unsloth import"
            )
    finally:
        del probe
        torch_module.cuda.empty_cache()

    return {
        "configurationEnvironmentVariable": CUDA_ALLOCATOR_CONFIG_ENV,
        "configuration": raw_config,
        "expandableSegmentsVerified": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-agent DPO/ORPO adapters with Unsloth.")
    parser.add_argument("--config", required=True, help="Path to agent Unsloth JSON config.")
    parser.add_argument("--sft-adapter-dir", required=True, help="Canonical finalized SFT adapter directory.")
    parser.add_argument(
        "--sft-finalized-variant-manifest",
        required=True,
        help="Finalized SFT variant manifest bound to --sft-adapter-dir.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help=(
            "Resume from the latest cryptographically bound preference checkpoint. "
            "Unbound checkpoint directories are never selected."
        ),
    )
    return parser.parse_args()


def _required_preference_trainer(cfg: Mapping[str, Any]) -> str:
    preference_trainer = cfg.get("preference_trainer")
    if type(preference_trainer) is not str or preference_trainer != "dpo":
        raise ValueError("preference_trainer must be exactly 'dpo'")
    return preference_trainer


def _required_positive_number(
    cfg: Mapping[str, Any],
    field: str,
    *,
    maximum: float,
) -> float:
    value = cfg.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        or float(value) > maximum
    ):
        raise ValueError(f"{field} must be finite and in the range (0, {maximum}]")
    return float(value)


def _validate_preference_training_config(
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on every loss-affecting preference-training control."""

    preference_trainer = _required_preference_trainer(cfg)
    learning_rate = _required_positive_number(
        cfg,
        "dpo_learning_rate",
        maximum=0.001,
    )
    num_train_epochs = _required_positive_number(
        cfg,
        "dpo_num_train_epochs",
        maximum=8.0,
    )
    beta = _required_positive_number(cfg, "dpo_beta", maximum=1.0)

    max_sequence_length = cfg.get("max_seq_length")
    max_prompt_length = cfg.get("max_prompt_length")
    if type(max_sequence_length) is not int or max_sequence_length <= 1:
        raise ValueError("max_seq_length must be an integer greater than one")
    if (
        type(max_prompt_length) is not int
        or max_prompt_length <= 0
        or max_prompt_length >= max_sequence_length
    ):
        raise ValueError(
            "max_prompt_length must be a positive integer smaller than max_seq_length"
        )

    for field in ("batch_size", "gradient_accumulation_steps"):
        value = cfg.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    warmup_steps = cfg.get("warmup_steps")
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative integer")

    gradient_checkpointing = cfg.get("gradient_checkpointing")
    use_logits_to_keep = cfg.get("use_logits_to_keep")
    precompute_ref_log_probs = cfg.get("precompute_ref_log_probs")
    precompute_ref_batch_size = cfg.get("precompute_ref_batch_size")
    if type(gradient_checkpointing) is not bool:
        raise ValueError("gradient_checkpointing must be a boolean")
    if type(use_logits_to_keep) is not bool:
        raise ValueError("use_logits_to_keep must be a boolean")
    if type(precompute_ref_log_probs) is not bool:
        raise ValueError("precompute_ref_log_probs must be a boolean")
    if type(precompute_ref_batch_size) is not int or precompute_ref_batch_size <= 0:
        raise ValueError("precompute_ref_batch_size must be a positive integer")
    if gradient_checkpointing is not True:
        raise ValueError("DPO requires gradient_checkpointing=true")
    if use_logits_to_keep is not True:
        raise ValueError("DPO requires use_logits_to_keep=true")
    if precompute_ref_log_probs is not True:
        raise ValueError("DPO requires precompute_ref_log_probs=true")

    precision = _resolve_training_precision(cfg)
    return {
        "preferenceTrainer": preference_trainer,
        "learningRate": learning_rate,
        "numTrainEpochs": num_train_epochs,
        "beta": beta,
        "maxPromptLength": max_prompt_length,
        "gradientCheckpointing": gradient_checkpointing,
        "useLogitsToKeep": use_logits_to_keep,
        "precomputeRefLogProbs": precompute_ref_log_probs,
        "precomputeRefBatchSize": precompute_ref_batch_size,
        "precision": precision,
    }


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in sorted(REQUIRED_CONFIG_KEYS) if key not in cfg]
    if missing:
        raise ValueError(f"Config is missing required keys: {', '.join(missing)}")
    _validate_preference_training_config(cfg)
    verify_chat_template_contract(cfg["chatTemplateContract"])
    validate_artifact_path_config(cfg)
    return cfg


def _tokenize_path(value: str) -> set[str]:
    return set("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def validate_artifact_path_config(cfg: dict[str, Any]) -> None:
    agent = str(cfg.get("agent", "")).strip().lower()
    if agent not in AGENTS:
        raise ValueError(f"Config has unsupported agent '{agent}'. Expected one of: {', '.join(sorted(AGENTS))}")

    output_dir = str(cfg.get("output_dir", "")).strip()
    if not output_dir:
        raise ValueError("Config output_dir must be non-empty")

    tokens = _tokenize_path(output_dir)
    if agent not in tokens:
        raise ValueError(
            f"output_dir must include slot token '{agent}' in the artifact path. Got: {output_dir}"
        )
    if not FINETUNE_MARKERS.intersection(tokens):
        raise ValueError(
            "output_dir must include a finetune marker token (one of: "
            + ", ".join(sorted(FINETUNE_MARKERS))
            + f"). Got: {output_dir}"
        )


def validate_dpo_artifact_paths(
    cfg: dict[str, Any],
    *,
    sft_adapter_dir: Path,
) -> tuple[Path, Path]:
    agent = str(cfg["agent"]).strip().lower()
    configured_sft = Path(cfg["adapter_output_dir"]).resolve()
    dpo_adapter_dir = Path(cfg["dpo_output_dir"]).resolve()
    if sft_adapter_dir != configured_sft:
        raise ValueError("--sft-adapter-dir must match config adapter_output_dir")
    tokens = _tokenize_path(str(dpo_adapter_dir))
    if agent not in tokens or not FINETUNE_MARKERS.intersection(tokens):
        raise ValueError(
            "dpo_output_dir must include the agent role and a finetune marker"
        )
    work_dir = Path(cfg["output_dir"]).resolve() / "dpo"
    if (
        dpo_adapter_dir == sft_adapter_dir
        or dpo_adapter_dir == work_dir
        or dpo_adapter_dir in work_dir.parents
        or work_dir in dpo_adapter_dir.parents
    ):
        raise ValueError("DPO work, SFT adapter, and DPO adapter directories must be separate")
    return work_dir, dpo_adapter_dir


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    text = str(value or "")
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise RuntimeError(f"{name} must be an immutable lowercase SHA-256 digest")
    return text


def _verify_base_model_lineage(cfg: dict[str, Any]) -> None:
    _verify_sft_base_model_lineage(cfg)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _preference_checkpoint_policy(cfg: Mapping[str, Any]) -> tuple[int, int]:
    save_steps = cfg.get(
        "preference_checkpoint_save_steps",
        PREFERENCE_CHECKPOINT_SAVE_STEPS,
    )
    save_total_limit = cfg.get("save_total_limit", 2)
    if type(save_steps) is not int or save_steps <= 0:
        raise ValueError("preference_checkpoint_save_steps must be a positive integer")
    if (
        type(save_total_limit) is not int
        or save_total_limit < PREFERENCE_CHECKPOINT_MINIMUM_RETENTION
    ):
        raise ValueError(
            "save_total_limit must retain at least two preference checkpoints"
        )
    return save_steps, save_total_limit


def _preference_training_code_sha256(
    cfg: Mapping[str, Any],
    *,
    preference_trainer: str,
) -> str:
    phase_digests = cfg.get("trainingCodeSHA256ByPhase")
    digest = (
        phase_digests.get(preference_trainer)
        if isinstance(phase_digests, Mapping)
        else None
    )
    return _require_sha256(
        digest,
        name=f"trainingCodeSHA256ByPhase.{preference_trainer}",
    )


def _preference_dataset_file_sha256(cfg: Mapping[str, Any]) -> dict[str, str]:
    dataset_root = Path(str(cfg.get("dataset_dir") or "")).resolve()
    filenames = ("train_dpo.jsonl", "val_dpo.jsonl", "variant_manifest.json")
    hashes: dict[str, str] = {}
    for filename in filenames:
        path = dataset_root / filename
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"Preference checkpoint lineage requires a regular {filename}"
            )
        hashes[filename] = _require_sha256(
            _hash_file(path),
            name=f"{filename} SHA-256",
        )
    return hashes


def _preference_checkpoint_static_contract(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
    preference_trainer: str,
) -> dict[str, Any]:
    output_dir = Path(str(cfg.get("output_dir") or "")).resolve() / "dpo"
    record_path = Path(
        str(cfg.get("preferenceCheckpointLineagePath") or "")
    ).resolve()
    if not str(cfg.get("preferenceCheckpointLineagePath") or ""):
        raise RuntimeError("Preference checkpoint lineage path is missing")
    if record_path == output_dir or output_dir in record_path.parents:
        raise RuntimeError(
            "Preference checkpoint lineage record must be outside its checkpoint root"
        )
    preflight_value = str(cfg.get("preferenceTokenLengthPreflightPath") or "")
    if not preflight_value:
        raise RuntimeError("Preference token-length preflight path is missing")
    preflight_path = Path(preflight_value).resolve()
    if output_dir not in preflight_path.parents:
        raise RuntimeError(
            "Preference token-length preflight must be inside its output directory"
        )
    save_steps, save_total_limit = _preference_checkpoint_policy(cfg)
    precision = _resolve_training_precision(cfg)
    return {
        "schema": PREFERENCE_CHECKPOINT_LINEAGE_SCHEMA,
        "agent": cfg.get("agent"),
        "preferenceTrainer": preference_trainer,
        "configPath": str(cfg_path.resolve()),
        "configSHA256": _require_sha256(
            _hash_file(cfg_path.resolve()),
            name="preference config SHA-256",
        ),
        "sourceVariantManifestSHA256": _require_sha256(
            cfg.get("variantManifestSHA256"),
            name="variantManifestSHA256",
        ),
        "datasetFileSHA256": _preference_dataset_file_sha256(cfg),
        "trainingCodeSHA256": _preference_training_code_sha256(
            cfg,
            preference_trainer=preference_trainer,
        ),
        "resolvedTrainingEnvironmentSHA256": _require_sha256(
            cfg.get("resolvedTrainingEnvironmentSHA256"),
            name="resolvedTrainingEnvironmentSHA256",
        ),
        "precision": precision,
        "checkpointScalerState": _checkpoint_scaler_state_contract(precision),
        "checkpointRoot": str(output_dir),
        "outputDirectory": str(output_dir),
        "policyAdapterName": POLICY_ADAPTER_NAME,
        "referenceAdapterName": (
            REFERENCE_ADAPTER_NAME if preference_trainer == "dpo" else None
        ),
        "referenceLogProbEvidencePath": (
            str(output_dir / DPO_REFERENCE_LOG_PROB_EVIDENCE_FILENAME)
            if preference_trainer == "dpo"
            else None
        ),
        "tokenLengthPreflightPath": str(preflight_path),
        "saveStrategy": "steps",
        "saveSteps": save_steps,
        "saveTotalLimit": save_total_limit,
    }


def _self_hashed_preference_checkpoint_record(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("checkpointLineageSHA256", None)
    return {
        **unsigned,
        "checkpointLineageSHA256": _canonical_sha256(unsigned),
    }


def _initial_preference_checkpoint_lineage(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
) -> dict[str, Any]:
    preference_trainer = _required_preference_trainer(cfg)
    payload = {
        **_preference_checkpoint_static_contract(
            cfg,
            cfg_path=cfg_path,
            preference_trainer=preference_trainer,
        ),
        "parentSFTAdapterSHA256": None,
        "referenceSFTAdapterSHA256": None,
        "referenceLogProbEvidenceSHA256": None,
        "tokenLengthPreflightSHA256": None,
        "checkpoints": [],
    }
    return _self_hashed_preference_checkpoint_record(payload)


def _read_preference_checkpoint_lineage(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Missing regular preference checkpoint lineage record")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != PREFERENCE_CHECKPOINT_LINEAGE_SCHEMA
    ):
        raise RuntimeError("Invalid preference checkpoint lineage contract")
    expected = payload.get("checkpointLineageSHA256")
    unsigned = dict(payload)
    unsigned.pop("checkpointLineageSHA256", None)
    if (
        not isinstance(expected, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        or _canonical_sha256(unsigned) != expected
    ):
        raise RuntimeError("Preference checkpoint lineage integrity check failed")
    return payload


def _validate_preference_checkpoint_lineage_static(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
) -> dict[str, Any]:
    try:
        preference_trainer = _required_preference_trainer(cfg)
    except ValueError as exc:
        raise RuntimeError("Preference checkpoint trainer is invalid") from exc
    record_path = Path(
        str(cfg.get("preferenceCheckpointLineagePath") or "")
    ).resolve()
    record = _read_preference_checkpoint_lineage(record_path)
    expected = _preference_checkpoint_static_contract(
        cfg,
        cfg_path=cfg_path,
        preference_trainer=preference_trainer,
    )
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Preference checkpoint lineage drifted from the config")
    parent = record.get("parentSFTAdapterSHA256")
    reference = record.get("referenceSFTAdapterSHA256")
    reference_log_prob_sha256 = record.get("referenceLogProbEvidenceSHA256")
    preflight_sha256 = record.get("tokenLengthPreflightSHA256")
    if preflight_sha256 is not None:
        _require_sha256(
            preflight_sha256,
            name="preference token-length preflight SHA-256",
        )
    if reference_log_prob_sha256 is not None:
        _require_sha256(
            reference_log_prob_sha256,
            name="DPO reference log-probability evidence SHA-256",
        )
    if preference_trainer == "orpo" and reference_log_prob_sha256 is not None:
        raise RuntimeError("ORPO checkpoint lineage cannot bind DPO reference evidence")
    if parent is None:
        if (
            reference is not None
            or reference_log_prob_sha256 is not None
            or preflight_sha256 is not None
            or record.get("checkpoints") != []
        ):
            raise RuntimeError("Unbound preference checkpoint lineage is invalid")
    else:
        _require_sha256(parent, name="parent SFT adapter SHA-256")
        if preference_trainer == "dpo":
            if reference != parent:
                raise RuntimeError("DPO checkpoint reference SFT lineage drifted")
        elif reference is not None:
            raise RuntimeError("ORPO checkpoint lineage cannot declare a reference adapter")
        if not isinstance(record.get("checkpoints"), list):
            raise RuntimeError("Preference checkpoint lineage checkpoints must be a list")
        if (
            preference_trainer == "dpo"
            and record["checkpoints"]
            and reference_log_prob_sha256 is None
        ):
            raise RuntimeError(
                "DPO checkpoints require bound reference log-probability evidence"
            )
    return record


def _reset_preference_checkpoint_lineage(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
) -> None:
    path = Path(str(cfg.get("preferenceCheckpointLineagePath") or "")).resolve()
    _write_json_atomic(
        path,
        _initial_preference_checkpoint_lineage(cfg, cfg_path=cfg_path),
    )


def _preference_checkpoint_step(name: str) -> int:
    match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", name)
    if match is None:
        raise RuntimeError("Invalid preference checkpoint directory name")
    return int(match.group(1))


def _preference_checkpoint_directory_manifest(
    checkpoint: Path,
    *,
    expected_base_model: str,
    expected_base_revision: str,
    precision: Mapping[str, Any],
) -> dict[str, Any]:
    if checkpoint.is_symlink():
        raise RuntimeError("Preference checkpoint directory is missing or unsafe")
    checkpoint = checkpoint.resolve()
    step = _preference_checkpoint_step(checkpoint.name)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise RuntimeError("Preference checkpoint directory is missing or unsafe")
    entries: list[dict[str, Any]] = []
    relative_files: set[str] = set()
    for candidate in sorted(
        checkpoint.rglob("*"),
        key=lambda path: path.relative_to(checkpoint).as_posix(),
    ):
        if candidate.is_symlink():
            raise RuntimeError("Preference checkpoint contains a symlink")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise RuntimeError("Preference checkpoint contains a non-regular entry")
        relative = candidate.relative_to(checkpoint).as_posix()
        relative_files.add(relative)
        entries.append(
            {
                "path": relative,
                "sizeBytes": candidate.stat().st_size,
                "sha256": _require_sha256(
                    _hash_file(candidate),
                    name=f"preference checkpoint file {relative}",
                ),
            }
        )
    scaler_state = _checkpoint_scaler_state_contract(precision)
    required_files = set(PREFERENCE_CHECKPOINT_REQUIRED_FILES)
    if scaler_state["required"]:
        required_files.add(CHECKPOINT_SCALER_FILENAME)
    missing = sorted(required_files - relative_files)
    adapter_weight_files = relative_files & {
        "adapter_model.bin",
        "adapter_model.safetensors",
    }
    nested_adapter_configs = sorted(
        path for path in relative_files if path.endswith("/adapter_config.json")
    )
    nested_adapter_weights = sorted(
        path
        for path in relative_files
        if path.endswith("/adapter_model.bin")
        or path.endswith("/adapter_model.safetensors")
    )
    if missing:
        raise _IncompletePreferenceCheckpoint(
            "Preference checkpoint is incomplete: missing " + ", ".join(missing)
        )
    if not adapter_weight_files:
        raise _IncompletePreferenceCheckpoint(
            "Preference checkpoint is incomplete: missing a policy adapter weight file"
        )
    if len(adapter_weight_files) != 1:
        raise RuntimeError(
            "Preference checkpoint must contain exactly one policy adapter weight file"
        )
    if nested_adapter_configs or nested_adapter_weights:
        raise RuntimeError(
            "Preference checkpoint must not persist the frozen reference adapter"
        )
    try:
        trainer_state = json.loads(
            (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
        adapter_config = json.loads(
            (checkpoint / "adapter_config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Preference checkpoint metadata is unreadable") from exc
    if (
        not isinstance(trainer_state, Mapping)
        or trainer_state.get("global_step") != step
    ):
        raise RuntimeError(
            "Preference checkpoint trainer state does not match its directory step"
        )
    if (
        not isinstance(adapter_config, Mapping)
        or adapter_config.get("base_model_name_or_path") != expected_base_model
        or adapter_config.get("revision") != expected_base_revision
    ):
        raise RuntimeError("Preference checkpoint base-model lineage drifted")
    payload = {
        "schema": PREFERENCE_CHECKPOINT_DIRECTORY_SCHEMA,
        "globalStep": step,
        "scalerState": scaler_state,
        "files": entries,
    }
    return {
        **payload,
        "checkpointSHA256": _canonical_sha256(payload),
    }


def _bound_preference_checkpoint_entries(
    record: Mapping[str, Any],
    *,
    expected_base_model: str,
    expected_base_revision: str,
) -> tuple[list[tuple[int, Path]], set[str], list[Path]]:
    root = Path(str(record.get("checkpointRoot") or "")).resolve()
    entries = record.get("checkpoints")
    if not isinstance(entries, list):
        raise RuntimeError("Preference checkpoint lineage checkpoints must be a list")
    prepared: list[tuple[int, Path, str]] = []
    declared_names: set[str] = set()
    declared_steps: list[int] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "path",
            "checkpointSHA256",
        }:
            raise RuntimeError("Preference checkpoint lineage entry is not canonical")
        relative = str(entry.get("path") or "")
        step = _preference_checkpoint_step(relative)
        if relative in declared_names:
            raise RuntimeError("Preference checkpoint lineage contains duplicates")
        declared_names.add(relative)
        declared_steps.append(step)
        expected_digest = _require_sha256(
            entry.get("checkpointSHA256"),
            name=f"preference checkpoint lineage digest for {relative}",
        )
        unresolved_checkpoint = root / relative
        if unresolved_checkpoint.is_symlink():
            raise RuntimeError("Preference checkpoint lineage points to a symlink")
        checkpoint = unresolved_checkpoint.resolve()
        if checkpoint.parent != root:
            raise RuntimeError("Preference checkpoint lineage escapes its root")
        prepared.append((step, checkpoint, expected_digest))
    if declared_steps != sorted(set(declared_steps)):
        raise RuntimeError(
            "Preference checkpoint lineage entries must be unique and step-sorted"
        )

    # Rotation precedes the on_save lineage callback. Walk newest to oldest so
    # a newer signed recovery point can authorize discarding only a structurally
    # unfinished older directory. Missing/incomplete newest state and complete
    # content drift at any age are never accepted.
    validated_descending: list[tuple[int, Path]] = []
    stale_partials: list[Path] = []
    newer_checkpoint_validated = False
    for step, checkpoint, expected_digest in reversed(prepared):
        if not checkpoint.exists():
            if not newer_checkpoint_validated:
                raise RuntimeError("Newest bound preference checkpoint is missing")
            continue
        try:
            manifest = _preference_checkpoint_directory_manifest(
                checkpoint,
                expected_base_model=expected_base_model,
                expected_base_revision=expected_base_revision,
                precision=record["precision"],
            )
        except _IncompletePreferenceCheckpoint:
            if not newer_checkpoint_validated:
                raise
            stale_partials.append(checkpoint)
            continue
        if manifest["checkpointSHA256"] != expected_digest:
            raise RuntimeError("Preference checkpoint contents drifted from lineage")
        validated_descending.append((step, checkpoint))
        newer_checkpoint_validated = True
    return (
        list(reversed(validated_descending)),
        declared_names,
        sorted(
            stale_partials,
            key=lambda path: _preference_checkpoint_step(path.name),
        ),
    )


def _unbound_preference_checkpoint_directories(
    root: Path,
    *,
    declared_names: set[str],
) -> list[Path]:
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("Preference checkpoint root is unsafe")
    unbound: list[Path] = []
    for candidate in root.glob("checkpoint-*"):
        _preference_checkpoint_step(candidate.name)
        if candidate.is_symlink() or not candidate.is_dir():
            raise RuntimeError("Preference checkpoint candidate is unsafe")
        if candidate.name not in declared_names:
            unbound.append(candidate)
    return sorted(unbound, key=lambda path: _preference_checkpoint_step(path.name))


def _prune_unbound_preference_checkpoints(paths: list[Path]) -> None:
    for path in paths:
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError("Refusing to prune an unsafe checkpoint candidate")
        shutil.rmtree(path)


def _bind_and_validate_preference_checkpoint_lineage(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
    preference_trainer: str,
    parent_sft_adapter_sha256: str,
    require_checkpoint: bool,
) -> tuple[Path | None, Path | None, list[Path]]:
    declared_path = str(cfg.get("preferenceCheckpointLineagePath") or "")
    if not declared_path:
        if require_checkpoint:
            raise RuntimeError(
                "--resume-from-checkpoint requires preference checkpoint lineage"
            )
        return None, None, []
    record_path = Path(declared_path).resolve()
    record = _validate_preference_checkpoint_lineage_static(
        cfg,
        cfg_path=cfg_path,
    )
    expected_parent = _require_sha256(
        parent_sft_adapter_sha256,
        name="parent SFT adapter SHA-256",
    )
    expected_reference = expected_parent if preference_trainer == "dpo" else None
    recorded_parent = record.get("parentSFTAdapterSHA256")
    recorded_reference = record.get("referenceSFTAdapterSHA256")
    if recorded_parent is None and recorded_reference is None:
        if record.get("checkpoints") != []:
            raise RuntimeError("Unbound preference lineage cannot contain checkpoints")
        updated = dict(record)
        updated["parentSFTAdapterSHA256"] = expected_parent
        updated["referenceSFTAdapterSHA256"] = expected_reference
        record = _self_hashed_preference_checkpoint_record(updated)
        _write_json_atomic(record_path, record)
    elif (
        recorded_parent != expected_parent
        or recorded_reference != expected_reference
    ):
        raise RuntimeError("Preference checkpoint parent SFT lineage drifted")

    validated, declared_names, stale_partials = (
        _bound_preference_checkpoint_entries(
            record,
            expected_base_model=str(cfg.get("base_model_name") or ""),
            expected_base_revision=str(cfg.get("baseModelRevision") or ""),
        )
    )
    root = Path(str(record["checkpointRoot"])).resolve()
    unbound = _unbound_preference_checkpoint_directories(
        root,
        declared_names=declared_names,
    )
    discardable = sorted(
        [*stale_partials, *unbound],
        key=lambda path: _preference_checkpoint_step(path.name),
    )
    if not require_checkpoint:
        if (
            record.get("checkpoints")
            or discardable
            or record.get("referenceLogProbEvidenceSHA256") is not None
        ):
            raise RuntimeError(
                "Fresh preference training cannot reuse checkpoint state; select resume or reset"
            )
        return None, record_path, []
    if not validated:
        if record.get("checkpoints") == []:
            # A kill can land after Transformers has atomically completed the
            # first checkpoint directory but before our on_save callback binds
            # its digest, or before the first checkpoint exists at all. Never
            # trust unbound checkpoint state; discard it and either restore
            # separately bound reference evidence or restart from step zero.
            return None, record_path, discardable
        raise RuntimeError("Resume requires a complete bound preference checkpoint")
    return validated[-1][1], record_path, discardable


def _record_preference_checkpoint(
    record_path: Path,
    checkpoint: Path,
) -> None:
    record = _read_preference_checkpoint_lineage(record_path)
    root = Path(str(record.get("checkpointRoot") or "")).resolve()
    checkpoint = checkpoint.resolve()
    if checkpoint.parent != root:
        raise RuntimeError("Saved preference checkpoint escapes its recorded root")
    candidates = sorted(
        (
            candidate
            for candidate in root.glob("checkpoint-*")
            if candidate.is_dir() and not candidate.is_symlink()
        ),
        key=lambda path: _preference_checkpoint_step(path.name),
    )
    if checkpoint not in candidates:
        raise RuntimeError("Saved preference checkpoint is absent from its root")
    config_path = Path(str(record["configPath"]))
    if (
        config_path.is_symlink()
        or not config_path.is_file()
        or _hash_file(config_path) != record.get("configSHA256")
    ):
        raise RuntimeError("Preference checkpoint config drifted during training")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_base_model = str(config.get("base_model_name") or "")
    expected_base_revision = str(config.get("baseModelRevision") or "")
    precision = _resolve_training_precision(config)
    entries = [
        {
            "path": candidate.name,
            "checkpointSHA256": _preference_checkpoint_directory_manifest(
                candidate,
                expected_base_model=expected_base_model,
                expected_base_revision=expected_base_revision,
                precision=precision,
            )["checkpointSHA256"],
        }
        for candidate in candidates
    ]
    updated = dict(record)
    updated["checkpoints"] = entries
    _write_json_atomic(
        record_path,
        _self_hashed_preference_checkpoint_record(updated),
    )


def _preference_checkpoint_callback(
    trainer_callback_type: type,
    *,
    record_path: Path,
) -> Any:
    class PreferenceCheckpointLineageCallback(trainer_callback_type):
        def on_save(self, args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            checkpoint = (
                Path(args.output_dir).resolve()
                / f"checkpoint-{state.global_step}"
            )
            _record_preference_checkpoint(record_path, checkpoint)
            return control

    return PreferenceCheckpointLineageCallback()


_PROMPT_ROLES = {"system", "user", "assistant"}


def _preference_message(
    value: Any,
    *,
    field: str,
    allowed_roles: set[str],
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"role", "content"}:
        raise ValueError(f"Preference {field} must be a message with only role and content")
    role = value.get("role")
    content = value.get("content")
    if not isinstance(role, str) or role not in allowed_roles:
        raise ValueError(f"Preference {field} has an unsupported role")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Preference {field} content must be non-empty text")
    return {"role": role, "content": content}


def _completion_messages(value: Any, *, field: str) -> list[dict[str, str]]:
    if isinstance(value, dict):
        message = value
    elif isinstance(value, list) and len(value) == 1:
        message = value[0]
    else:
        raise ValueError(f"Preference {field} must contain exactly one assistant message")
    return [
        _preference_message(
            message,
            field=field,
            allowed_roles={"assistant"},
        )
    ]


def _normalized_completion_content(value: str) -> str:
    return " ".join(value.split())


def row_to_preference(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("Preference record must be a JSON object")
    prompt_value = row.get("prompt")
    if not isinstance(prompt_value, list) or not prompt_value:
        raise ValueError("Preference prompt must be a non-empty message list")

    prompt = [
        _preference_message(
            message,
            field=f"prompt[{index}]",
            allowed_roles=_PROMPT_ROLES,
        )
        for index, message in enumerate(prompt_value)
    ]
    if prompt[0]["role"] == "system":
        conversation = prompt[1:]
        if not conversation:
            raise ValueError("Preference prompt must contain a user message after the system message")
    else:
        conversation = prompt
    if any(message["role"] == "system" for message in conversation):
        raise ValueError("Preference system messages are only supported at the start of the prompt")
    expected_role = "user"
    for message in conversation:
        if message["role"] != expected_role:
            raise ValueError("Preference prompt user and assistant roles must alternate")
        expected_role = "assistant" if expected_role == "user" else "user"
    if prompt[-1]["role"] != "user":
        raise ValueError("Preference prompt must end with a user message before the assistant response")
    prompt = canonical_non_thinking_messages(prompt)

    if "chosen" not in row or "rejected" not in row:
        raise ValueError("Preference record must include chosen and rejected assistant messages")
    chosen = _completion_messages(row["chosen"], field="chosen")
    rejected = _completion_messages(row["rejected"], field="rejected")
    if _normalized_completion_content(chosen[0]["content"]) == _normalized_completion_content(
        rejected[0]["content"]
    ):
        raise ValueError("Preference chosen and rejected completions must differ")
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        # TRL 0.24 reads this controlled per-record field and forwards it to
        # every tokenizer.apply_chat_template call. Keep preference rendering
        # identical to SFT and frozen evaluation instead of inheriting the
        # Qwen tokenizer's thinking-mode default.
        "chat_template_kwargs": non_thinking_template_kwargs(),
    }


def _nearest_rank(values: list[int], percentile: int) -> int:
    if not values:
        raise ValueError("Token-length statistics require at least one value")
    if percentile <= 0 or percentile > 100:
        raise ValueError("Nearest-rank percentile must be in the range 1...100")
    ordered = sorted(values)
    return ordered[max(0, math.ceil((percentile / 100) * len(ordered)) - 1)]


def _token_length_statistics(values: list[int]) -> dict[str, int]:
    if not values:
        raise ValueError("Token-length statistics require at least one value")
    return {
        "min": min(values),
        "p50": _nearest_rank(values, 50),
        "p95": _nearest_rank(values, 95),
        "max": max(values),
    }


def _preference_text_token_ids(
    tokenizer: Any,
    value: Any,
    *,
    split: str,
    row_index: int,
    field: str,
) -> list[int]:
    if not isinstance(value, str):
        raise RuntimeError(
            "Preference token-length preflight received a non-text "
            f"{field} after chat rendering for {split} row {row_index}"
        )
    try:
        encoded = tokenizer(value, add_special_tokens=False)
        input_ids = encoded["input_ids"]
    except Exception as exc:
        raise RuntimeError(
            "Preference token-length preflight could not tokenize "
            f"{field} for {split} row {row_index}"
        ) from exc
    if (
        not isinstance(input_ids, list)
        or any(type(token_id) is not int for token_id in input_ids)
    ):
        raise RuntimeError(
            "Preference token-length preflight received invalid input_ids "
            f"for {field} in {split} row {row_index}"
        )
    return input_ids


def _preference_text_token_count(
    tokenizer: Any,
    value: Any,
    *,
    split: str,
    row_index: int,
    field: str,
) -> int:
    return len(
        _preference_text_token_ids(
            tokenizer,
            value,
            split=split,
            row_index=row_index,
            field=field,
        )
    )


def _dpo_completion_token_ids(
    tokenizer: Any,
    value: Any,
    *,
    split: str,
    row_index: int,
    field: str,
    appended_eos_token_id: int | None = None,
) -> list[int]:
    eos_token_id = (
        _dpo_appended_eos_token_id(tokenizer)
        if appended_eos_token_id is None
        else appended_eos_token_id
    )
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise RuntimeError("DPO completion transcript requires a valid EOS token ID")
    return [
        *_preference_text_token_ids(
            tokenizer,
            value,
            split=split,
            row_index=row_index,
            field=field,
        ),
        eos_token_id,
    ]


def _dpo_completion_token_count(
    tokenizer: Any,
    value: Any,
    *,
    split: str,
    row_index: int,
    field: str,
) -> int:
    """Mirror TRL 0.24's loss-bearing completion IDs exactly.

    ``DPOTrainer.tokenize_row`` tokenizes a rendered completion with
    ``add_special_tokens=False`` and then unconditionally appends one
    ``tokenizer.eos_token_id``. The rendered Qwen completion already ending in
    EOS does not suppress that append, so counting the suffix is required even
    when it produces two adjacent termination markers in the trainer input.
    """

    return len(
        _dpo_completion_token_ids(
            tokenizer,
            value,
            split=split,
            row_index=row_index,
            field=field,
        )
    )


def _dpo_appended_eos_token_id(tokenizer: Any) -> int:
    token_id = getattr(tokenizer, "eos_token_id", None)
    if type(token_id) is not int or token_id < 0:
        raise RuntimeError(
            "Preference token-length preflight requires a non-negative integer "
            "tokenizer.eos_token_id because TRL 0.24 appends it unconditionally"
        )
    return token_id


def _preflight_preference_token_lengths(
    splits: Mapping[str, list[dict[str, Any]]],
    *,
    tokenizer: Any,
    render_preference: Any,
    max_prompt_length: int,
    max_sequence_length: int,
    minimum_prompt_margin_tokens: int = PREFERENCE_MINIMUM_PROMPT_MARGIN_TOKENS,
    minimum_sequence_margin_tokens: int = PREFERENCE_MINIMUM_SEQUENCE_MARGIN_TOKENS,
    source_splits: Mapping[str, list[dict[str, Any]]] | None = None,
    agent: str | None = None,
    fleet_loss_share_contract: Any = None,
    public_corpus_loss_share_contract: Any = None,
    fleet_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove TRL's rendered preference rows need no prompt/sequence truncation.

    TRL 0.24 renders conversational rows with ``maybe_apply_chat_template``,
    tokenizes the prompt and each completion independently with
    ``add_special_tokens=False``, and then unconditionally appends one EOS ID
    to each completion. Repeating that exact contract before trainer
    construction prevents its default ``keep_end`` behavior from silently
    discarding the leading system contract and keeps Fleet target-token
    evidence aligned with the optimizer's loss-bearing completion IDs.
    """

    if type(max_prompt_length) is not int or max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be a positive integer")
    if type(max_sequence_length) is not int or max_sequence_length <= 0:
        raise ValueError("max_seq_length must be a positive integer")
    if max_prompt_length > max_sequence_length:
        raise ValueError("max_prompt_length cannot exceed max_seq_length")
    if (
        type(minimum_prompt_margin_tokens) is not int
        or minimum_prompt_margin_tokens < PREFERENCE_MINIMUM_PROMPT_MARGIN_TOKENS
    ):
        raise ValueError(
            "minimum_prompt_margin_tokens cannot weaken the controlled 64-token margin"
        )
    if (
        type(minimum_sequence_margin_tokens) is not int
        or minimum_sequence_margin_tokens
        < PREFERENCE_MINIMUM_SEQUENCE_MARGIN_TOKENS
    ):
        raise ValueError(
            "minimum_sequence_margin_tokens cannot weaken the controlled 128-token margin"
        )
    if not callable(render_preference):
        raise TypeError("Preference chat-template renderer must be callable")
    appended_eos_token_id = _dpo_appended_eos_token_id(tokenizer)
    if agent == "fleet":
        _validated_fleet_loss_share_contract(
            fleet_loss_share_contract,
            lane="dpo",
            config=fleet_config,
        )
    elif fleet_loss_share_contract is not None:
        raise RuntimeError(
            "Fleet preference loss-share input is forbidden for non-Fleet training"
        )
    if agent is not None:
        _validated_public_corpus_loss_share_contract(
            public_corpus_loss_share_contract,
            lane="dpo",
            config=fleet_config,
        )
        if not isinstance(source_splits, Mapping) or set(source_splits) != set(splits):
            raise RuntimeError(
                "Public-corpus preference token preflight requires aligned raw "
                "source splits"
            )
    elif public_corpus_loss_share_contract is not None or source_splits is not None:
        raise RuntimeError(
            "Public-corpus preference loss-share inputs require a controlled agent"
        )

    aggregate: dict[str, list[int]] = {
        "prompt": [],
        "chosenCompletion": [],
        "rejectedCompletion": [],
        "chosenTotal": [],
        "rejectedTotal": [],
        "maximumTotal": [],
    }
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
    for split, rows in splits.items():
        if not isinstance(split, str) or not split:
            raise ValueError("Preference token-length split names must be non-empty")
        if not isinstance(rows, list):
            raise TypeError("Preference token-length splits must contain row lists")
        raw_source_rows = source_splits[split] if source_splits is not None else None
        if agent is not None and (
            not isinstance(raw_source_rows, list)
            or len(raw_source_rows) != len(rows)
        ):
            raise RuntimeError(
                f"Public-corpus preference {split} source rows are not aligned"
            )
        lengths: dict[str, list[int]] = {
            key: [] for key in aggregate
        }
        split_fleet_target_rows: list[tuple[Mapping[str, Any], int]] = []
        split_public_corpus_target_rows: list[
            tuple[Mapping[str, Any], int]
        ] = []
        for row_index, row in enumerate(rows):
            try:
                rendered = render_preference(dict(row), tokenizer=tokenizer)
            except Exception as exc:
                raise RuntimeError(
                    "Preference token-length preflight could not apply the chat "
                    f"template for {split} row {row_index}"
                ) from exc
            if not isinstance(rendered, Mapping):
                raise RuntimeError(
                    "Preference token-length preflight received invalid rendered "
                    f"data for {split} row {row_index}"
                )
            prompt_token_ids = _preference_text_token_ids(
                tokenizer,
                rendered.get("prompt"),
                split=split,
                row_index=row_index,
                field="prompt",
            )
            chosen_completion_token_ids = _dpo_completion_token_ids(
                tokenizer,
                rendered.get("chosen"),
                split=split,
                row_index=row_index,
                field="chosen completion",
                appended_eos_token_id=appended_eos_token_id,
            )
            rejected_completion_token_ids = _dpo_completion_token_ids(
                tokenizer,
                rendered.get("rejected"),
                split=split,
                row_index=row_index,
                field="rejected completion",
                appended_eos_token_id=appended_eos_token_id,
            )
            prompt_tokens = len(prompt_token_ids)
            chosen_completion_tokens = len(chosen_completion_token_ids)
            rejected_completion_tokens = len(rejected_completion_token_ids)
            chosen_total_tokens = prompt_tokens + chosen_completion_tokens
            rejected_total_tokens = prompt_tokens + rejected_completion_tokens
            maximum_total_tokens = max(
                chosen_total_tokens,
                rejected_total_tokens,
            )
            if prompt_tokens > max_prompt_length:
                raise RuntimeError(
                    "Preference token-length preflight rejected "
                    f"{split} row {row_index}: prompt uses {prompt_tokens} tokens, "
                    f"exceeding max_prompt_length {max_prompt_length}; keep_end "
                    "truncation would discard the leading system contract"
                )
            if maximum_total_tokens > max_sequence_length:
                raise RuntimeError(
                    "Preference token-length preflight rejected "
                    f"{split} row {row_index}: prompt plus completion uses "
                    f"{maximum_total_tokens} tokens, exceeding max_seq_length "
                    f"{max_sequence_length}"
                )
            row_lengths = {
                "prompt": prompt_tokens,
                "chosenCompletion": chosen_completion_tokens,
                "rejectedCompletion": rejected_completion_tokens,
                "chosenTotal": chosen_total_tokens,
                "rejectedTotal": rejected_total_tokens,
                "maximumTotal": maximum_total_tokens,
            }
            row_transcript = {
                "schemaVersion": PREFERENCE_TOKENIZATION_TRANSCRIPT_SCHEMA,
                "split": split,
                "rowIndex": row_index,
                "promptInputIDs": prompt_token_ids,
                "chosenCompletionInputIDs": chosen_completion_token_ids,
                "rejectedCompletionInputIDs": rejected_completion_token_ids,
            }
            transcript_rows.append(
                {
                    "split": split,
                    "rowIndex": row_index,
                    "rowSHA256": _canonical_sha256(row_transcript),
                }
            )
            for key, value in row_lengths.items():
                lengths[key].append(value)
                aggregate[key].append(value)
            if agent is not None:
                raw_source_row = raw_source_rows[row_index]
                if not isinstance(raw_source_row, Mapping):
                    raise RuntimeError(
                        "Preference source rows must be JSON objects"
                    )
                split_public_corpus_target_rows.append(
                    (raw_source_row, chosen_completion_tokens)
                )
            if agent == "fleet":
                split_fleet_target_rows.append(
                    (raw_source_row, chosen_completion_tokens)
                )

        if agent == "fleet":
            fleet_target_rows[split] = split_fleet_target_rows
        if agent is not None:
            public_corpus_target_rows[split] = split_public_corpus_target_rows

        if rows:
            split_summaries[split] = {
                "records": len(rows),
                "promptTokens": _token_length_statistics(lengths["prompt"]),
                "chosenCompletionTokens": _token_length_statistics(
                    lengths["chosenCompletion"]
                ),
                "rejectedCompletionTokens": _token_length_statistics(
                    lengths["rejectedCompletion"]
                ),
                "chosenTotalTokens": _token_length_statistics(
                    lengths["chosenTotal"]
                ),
                "rejectedTotalTokens": _token_length_statistics(
                    lengths["rejectedTotal"]
                ),
                "maximumTotalTokens": _token_length_statistics(
                    lengths["maximumTotal"]
                ),
                "smallestPromptMarginTokens": (
                    max_prompt_length - max(lengths["prompt"])
                ),
                "smallestSequenceMarginTokens": (
                    max_sequence_length - max(lengths["maximumTotal"])
                ),
            }
        else:
            split_summaries[split] = {"records": 0}

    if not aggregate["prompt"]:
        raise RuntimeError("Preference token-length preflight requires at least one row")
    smallest_prompt_margin = max_prompt_length - max(aggregate["prompt"])
    smallest_sequence_margin = (
        max_sequence_length - max(aggregate["maximumTotal"])
    )
    if smallest_prompt_margin < minimum_prompt_margin_tokens:
        raise RuntimeError(
            "Preference token-length preflight rejected the configured prompt "
            f"limit: the smallest exact-tokenizer margin is {smallest_prompt_margin} "
            f"tokens, below the controlled minimum of {minimum_prompt_margin_tokens}"
        )
    if smallest_sequence_margin < minimum_sequence_margin_tokens:
        raise RuntimeError(
            "Preference token-length preflight rejected the configured sequence "
            f"limit: the smallest exact-tokenizer margin is {smallest_sequence_margin} "
            f"tokens, below the controlled minimum of {minimum_sequence_margin_tokens}"
        )
    renderer_module = getattr(render_preference, "__module__", "")
    renderer_name = getattr(
        render_preference,
        "__qualname__",
        getattr(render_preference, "__name__", ""),
    )
    renderer_identity = ".".join(
        part for part in (renderer_module, renderer_name) if part
    )
    report = {
        "schemaVersion": PREFERENCE_TOKEN_LENGTH_PREFLIGHT_SCHEMA,
        "renderer": renderer_identity,
        "addSpecialTokens": False,
        "completionTokenizationPolicy": DPO_COMPLETION_TOKENIZATION_POLICY,
        "appendedEOSTokenID": appended_eos_token_id,
        "percentileMethod": "nearest_rank",
        "maxPromptLength": max_prompt_length,
        "maxSequenceLength": max_sequence_length,
        "minimumPromptMarginTokens": minimum_prompt_margin_tokens,
        "minimumSequenceMarginTokens": minimum_sequence_margin_tokens,
        "records": len(aggregate["prompt"]),
        "promptTokens": _token_length_statistics(aggregate["prompt"]),
        "chosenCompletionTokens": _token_length_statistics(
            aggregate["chosenCompletion"]
        ),
        "rejectedCompletionTokens": _token_length_statistics(
            aggregate["rejectedCompletion"]
        ),
        "chosenTotalTokens": _token_length_statistics(aggregate["chosenTotal"]),
        "rejectedTotalTokens": _token_length_statistics(
            aggregate["rejectedTotal"]
        ),
        "maximumTotalTokens": _token_length_statistics(aggregate["maximumTotal"]),
        "smallestPromptMarginTokens": smallest_prompt_margin,
        "smallestSequenceMarginTokens": smallest_sequence_margin,
        "truncationRequired": False,
        "splits": split_summaries,
        "tokenizationTranscriptSHA256": _canonical_sha256(
            {
                "schemaVersion": PREFERENCE_TOKENIZATION_TRANSCRIPT_SCHEMA,
                "rows": transcript_rows,
            }
        ),
    }
    if agent == "fleet":
        report["fleetLossShareEvidence"] = _build_fleet_loss_share_evidence(
            contract_value=fleet_loss_share_contract,
            lane="dpo",
            split_target_rows=fleet_target_rows,
            config=fleet_config,
        )
    if agent is not None:
        report["publicCorpusLossShareEvidence"] = (
            _build_public_corpus_loss_share_evidence(
                contract_value=public_corpus_loss_share_contract,
                lane="dpo",
                split_target_rows=public_corpus_target_rows,
                config=fleet_config,
            )
        )
    return report


def _bind_preference_token_length_preflight(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    if preflight.get("schemaVersion") != PREFERENCE_TOKEN_LENGTH_PREFLIGHT_SCHEMA:
        raise RuntimeError("Preference token-length preflight contract is invalid")
    record_path = Path(
        str(cfg.get("preferenceCheckpointLineagePath") or "")
    ).resolve()
    record = _validate_preference_checkpoint_lineage_static(
        cfg,
        cfg_path=cfg_path,
    )
    parent_sha256 = record.get("parentSFTAdapterSHA256")
    if parent_sha256 is None:
        raise RuntimeError(
            "Preference token-length preflight requires bound parent SFT lineage"
        )
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
        "parentSFTAdapterSHA256": parent_sha256,
        "referenceSFTAdapterSHA256": record.get("referenceSFTAdapterSHA256"),
    }
    evidence = {
        **unsigned,
        "preflightSHA256": _canonical_sha256(unsigned),
    }
    evidence_path = Path(str(record["tokenLengthPreflightPath"])).resolve()
    if evidence_path.is_symlink():
        raise RuntimeError("Preference token-length preflight path is unsafe")
    if evidence_path.exists():
        if not evidence_path.is_file():
            raise RuntimeError(
                "Preference token-length preflight path is not a regular file"
            )
        existing = json.loads(evidence_path.read_text(encoding="utf-8"))
        if existing != evidence:
            raise RuntimeError("Preference token-length preflight evidence drifted")
    else:
        _write_json_atomic(evidence_path, evidence)
    recorded_digest = record.get("tokenLengthPreflightSHA256")
    if recorded_digest is None:
        if record.get("checkpoints") != []:
            raise RuntimeError(
                "Preference checkpoints predate token-length preflight evidence"
            )
        updated = dict(record)
        updated["tokenLengthPreflightSHA256"] = evidence["preflightSHA256"]
        _write_json_atomic(
            record_path,
            _self_hashed_preference_checkpoint_record(updated),
        )
    elif recorded_digest != evidence["preflightSHA256"]:
        raise RuntimeError("Preference token-length preflight lineage drifted")
    return evidence


def _load_sft_policy(
    base_model: Any,
    *,
    peft_model_class: Any,
    sft_adapter_dir: Path,
    preference_trainer: str,
) -> Any:
    model = peft_model_class.from_pretrained(
        base_model,
        str(sft_adapter_dir),
        adapter_name=POLICY_ADAPTER_NAME,
        is_trainable=True,
    )
    if preference_trainer == "dpo":
        model.load_adapter(
            str(sft_adapter_dir),
            adapter_name=REFERENCE_ADAPTER_NAME,
            is_trainable=False,
        )
        model.set_adapter(POLICY_ADAPTER_NAME)
    return model


def _build_preference_trainer(
    cfg: dict[str, Any],
    *,
    preference_trainer: str,
    seed: int,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    val_dataset: Any,
    output_dir: Path,
    dpo_config_class: Any,
    dpo_trainer_class: Any,
    orpo_config_class: Any,
    orpo_trainer_class: Any,
) -> tuple[Any, Any]:
    preference_config = _validate_preference_training_config(cfg)
    if preference_trainer != preference_config["preferenceTrainer"]:
        raise ValueError("Preference trainer argument drifted from the config")
    checkpoint_save_steps, checkpoint_save_total_limit = (
        _preference_checkpoint_policy(cfg)
    )
    gradient_checkpointing = preference_config["gradientCheckpointing"]
    use_logits_to_keep = preference_config["useLogitsToKeep"]
    precompute_ref_log_probs = preference_config["precomputeRefLogProbs"]
    precompute_ref_batch_size = preference_config["precomputeRefBatchSize"]
    precision = preference_config["precision"]
    common_config = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(cfg["batch_size"]),
        "per_device_eval_batch_size": max(1, int(cfg["batch_size"])),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "learning_rate": preference_config["learningRate"],
        "num_train_epochs": preference_config["numTrainEpochs"],
        "warmup_steps": int(cfg["warmup_steps"]),
        "logging_steps": int(cfg.get("logging_steps", 10)),
        "eval_strategy": "epoch" if val_dataset is not None else "no",
        "save_strategy": "steps",
        "save_steps": checkpoint_save_steps,
        "save_total_limit": checkpoint_save_total_limit,
        # Exact recovery requires optimizer, scheduler, scaler, trainer, and
        # RNG state in addition to the LoRA weights.
        "save_only_model": False,
        "bf16": precision["bf16"],
        "fp16": precision["fp16"],
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "max_length": int(cfg["max_seq_length"]),
        "max_prompt_length": preference_config["maxPromptLength"],
        # TRL 0.24 truncates this field independently during tokenization.
        # Leave it explicitly unbounded; the exact preflight proves the
        # prompt-plus-completion sequence fits max_length without truncation.
        "max_completion_length": None,
        "gradient_checkpointing": gradient_checkpointing,
    }
    if preference_trainer == "dpo":
        common_config["torch_empty_cache_steps"] = 1
        training_args = dpo_config_class(
            **common_config,
            beta=preference_config["beta"],
            model_adapter_name=POLICY_ADAPTER_NAME,
            ref_adapter_name=REFERENCE_ADAPTER_NAME,
            use_logits_to_keep=use_logits_to_keep,
            precompute_ref_log_probs=precompute_ref_log_probs,
            precompute_ref_batch_size=precompute_ref_batch_size,
        )
        trainer = dpo_trainer_class(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )
        return trainer, training_args
    raise ValueError("preference_trainer must be exactly 'dpo'")


def _save_policy_adapter(
    model: Any,
    output_dir: Path,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    model.set_adapter(POLICY_ADAPTER_NAME)
    return _save_portable_peft_adapter(
        model,
        output_dir,
        cfg,
        selected_adapters=[POLICY_ADAPTER_NAME],
        expected_adapter_names=[POLICY_ADAPTER_NAME],
    )


def _precompute_reference_log_probs_before_training(
    trainer: Any,
    training_args: Any,
    *,
    has_eval_dataset: bool,
) -> dict[str, bool]:
    """Precompute frozen-reference logps before policy checkpoint graphs exist."""

    if getattr(training_args, "precompute_ref_log_probs", None) is not True:
        raise RuntimeError("DPO reference log-probability precomputation is required")
    if getattr(training_args, "gradient_checkpointing", None) is not True:
        raise RuntimeError("DPO policy gradient checkpointing is required")
    model = trainer.model
    disable = getattr(model, "gradient_checkpointing_disable", None)
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(disable) or not callable(enable):
        raise RuntimeError("DPO model lacks explicit gradient-checkpointing controls")

    disable()
    try:
        trainer.get_train_dataloader()
        if has_eval_dataset:
            trainer.get_eval_dataloader()
    finally:
        enable(
            gradient_checkpointing_kwargs=getattr(
                training_args,
                "gradient_checkpointing_kwargs",
                None,
            )
        )

    train_precomputed = (
        getattr(trainer, "_precomputed_train_ref_log_probs", None) is True
    )
    eval_precomputed = (
        not has_eval_dataset
        or getattr(trainer, "_precomputed_eval_ref_log_probs", None) is True
    )
    if not train_precomputed or not eval_precomputed:
        raise RuntimeError(
            "DPO trainer did not bind precomputed frozen-reference log probabilities"
        )
    return {
        "train": train_precomputed,
        "evaluation": eval_precomputed,
    }


def _reference_log_prob_source_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("DPO reference evidence requires a non-empty dataset split")
    row_digests = [_canonical_sha256(row) for row in rows]
    return {
        "rowCount": len(rows),
        "sourceRowSHA256": row_digests,
        "sourceRowsSHA256": _canonical_sha256(row_digests),
    }


def _reference_log_prob_static_contract(
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
    parent_sft_adapter_sha256: str,
    source_rows_by_split: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if set(source_rows_by_split) != {"train", "validation"}:
        raise RuntimeError("DPO reference evidence requires train and validation splits")
    return {
        "schema": DPO_REFERENCE_LOG_PROB_EVIDENCE_SCHEMA,
        "agent": cfg.get("agent"),
        "preferenceTrainer": "dpo",
        "parentSFTAdapterSHA256": _require_sha256(
            parent_sft_adapter_sha256,
            name="DPO reference evidence parent SFT adapter SHA-256",
        ),
        "sourceVariantManifestSHA256": _require_sha256(
            cfg.get("variantManifestSHA256"),
            name="DPO reference evidence source variant SHA-256",
        ),
        "configSHA256": _require_sha256(
            _hash_file(cfg_path.resolve()),
            name="DPO reference evidence config SHA-256",
        ),
        "datasetFileSHA256": _preference_dataset_file_sha256(cfg),
        "trainingCodeSHA256": _preference_training_code_sha256(
            cfg,
            preference_trainer="dpo",
        ),
        "resolvedTrainingEnvironmentSHA256": _require_sha256(
            cfg.get("resolvedTrainingEnvironmentSHA256"),
            name="DPO reference evidence resolved environment SHA-256",
        ),
        "columns": list(DPO_REFERENCE_LOG_PROB_COLUMNS),
        "splits": {
            split: _reference_log_prob_source_split(source_rows_by_split[split])
            for split in ("train", "validation")
        },
    }


def _binary32_hex(value: Any, *, field: str) -> str:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be a finite IEEE-754 binary32 value")
    try:
        number = float(value)
        encoded = struct.pack(">f", number)
        restored = struct.unpack(">f", encoded)[0]
    except (OverflowError, TypeError, ValueError, struct.error) as exc:
        raise RuntimeError(
            f"{field} must be a finite IEEE-754 binary32 value"
        ) from exc
    if not math.isfinite(number) or not math.isfinite(restored) or restored != number:
        raise RuntimeError(f"{field} must be an exact finite binary32 value")
    return encoded.hex()


def _qualified_reference_log_prob_columns(
    dataset: Any,
    *,
    split: str,
    expected_row_count: int,
) -> tuple[list[str], list[str]]:
    column_names = getattr(dataset, "column_names", None)
    if not isinstance(column_names, list) or any(
        not isinstance(name, str) for name in column_names
    ):
        raise RuntimeError(f"DPO {split} dataset does not expose canonical columns")
    missing = [name for name in DPO_REFERENCE_LOG_PROB_COLUMNS if name not in column_names]
    if missing:
        raise RuntimeError(
            f"DPO {split} dataset lacks frozen reference columns: {', '.join(missing)}"
        )
    if type(expected_row_count) is not int or expected_row_count <= 0:
        raise RuntimeError(f"DPO {split} expected row count is invalid")
    if len(dataset) != expected_row_count:
        raise RuntimeError(f"DPO {split} reference columns have the wrong row count")

    encoded_columns: list[list[str]] = []
    for column_name in DPO_REFERENCE_LOG_PROB_COLUMNS:
        try:
            values = dataset[column_name]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"DPO {split} reference column {column_name} is unreadable"
            ) from exc
        if isinstance(values, (str, bytes, Mapping)):
            raise RuntimeError(
                f"DPO {split} reference column {column_name} is not a scalar sequence"
            )
        try:
            scalar_values = list(values)
        except TypeError as exc:
            raise RuntimeError(
                f"DPO {split} reference column {column_name} is unreadable"
            ) from exc
        if len(scalar_values) != expected_row_count:
            raise RuntimeError(
                f"DPO {split} reference column {column_name} has the wrong row count"
            )
        encoded_columns.append(
            [
                _binary32_hex(
                    value,
                    field=f"DPO {split} {column_name}[{index}]",
                )
                for index, value in enumerate(scalar_values)
            ]
        )
    return encoded_columns[0], encoded_columns[1]


def _reference_log_prob_split_evidence(
    dataset: Any,
    *,
    split: str,
    source_contract: Mapping[str, Any],
) -> dict[str, Any]:
    row_count = source_contract.get("rowCount")
    chosen, rejected = _qualified_reference_log_prob_columns(
        dataset,
        split=split,
        expected_row_count=row_count,
    )
    pairs = [
        {"chosen": chosen_value, "rejected": rejected_value}
        for chosen_value, rejected_value in zip(chosen, rejected, strict=True)
    ]
    return {
        **source_contract,
        "refChosenLogpsIEEE754Binary32": chosen,
        "refRejectedLogpsIEEE754Binary32": rejected,
        "referenceLogProbPairsSHA256": _canonical_sha256(pairs),
    }


def _self_hashed_reference_log_prob_evidence(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("referenceLogProbEvidenceSHA256", None)
    return {
        **unsigned,
        "referenceLogProbEvidenceSHA256": _canonical_sha256(unsigned),
    }


def _build_reference_log_prob_evidence(
    trainer: Any,
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
    parent_sft_adapter_sha256: str,
    source_rows_by_split: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    static = _reference_log_prob_static_contract(
        cfg,
        cfg_path=cfg_path,
        parent_sft_adapter_sha256=parent_sft_adapter_sha256,
        source_rows_by_split=source_rows_by_split,
    )
    payload = {
        **{key: value for key, value in static.items() if key != "splits"},
        "splits": {
            "train": _reference_log_prob_split_evidence(
                getattr(trainer, "train_dataset", None),
                split="train",
                source_contract=static["splits"]["train"],
            ),
            "validation": _reference_log_prob_split_evidence(
                getattr(trainer, "eval_dataset", None),
                split="validation",
                source_contract=static["splits"]["validation"],
            ),
        },
    }
    return verify_reference_log_prob_evidence(
        _self_hashed_reference_log_prob_evidence(payload)
    )


def verify_reference_log_prob_evidence(value: Any) -> dict[str, Any]:
    """Verify canonical source-bound DPO scalar evidence without a model."""

    required_top_level = {
        "schema",
        "agent",
        "preferenceTrainer",
        "parentSFTAdapterSHA256",
        "sourceVariantManifestSHA256",
        "configSHA256",
        "datasetFileSHA256",
        "trainingCodeSHA256",
        "resolvedTrainingEnvironmentSHA256",
        "columns",
        "splits",
        "referenceLogProbEvidenceSHA256",
    }
    if not isinstance(value, dict) or set(value) != required_top_level:
        raise RuntimeError("DPO reference log-probability evidence is not canonical")
    if (
        value.get("schema") != DPO_REFERENCE_LOG_PROB_EVIDENCE_SCHEMA
        or value.get("preferenceTrainer") != "dpo"
        or value.get("columns") != list(DPO_REFERENCE_LOG_PROB_COLUMNS)
        or not isinstance(value.get("agent"), str)
        or not value["agent"]
    ):
        raise RuntimeError("DPO reference log-probability evidence identity is invalid")
    for field in (
        "parentSFTAdapterSHA256",
        "sourceVariantManifestSHA256",
        "configSHA256",
        "trainingCodeSHA256",
        "resolvedTrainingEnvironmentSHA256",
    ):
        _require_sha256(value.get(field), name=f"DPO reference evidence {field}")
    dataset_hashes = value.get("datasetFileSHA256")
    if (
        not isinstance(dataset_hashes, dict)
        or set(dataset_hashes)
        != {"train_dpo.jsonl", "val_dpo.jsonl", "variant_manifest.json"}
    ):
        raise RuntimeError("DPO reference evidence dataset lineage is invalid")
    for filename, digest in dataset_hashes.items():
        _require_sha256(digest, name=f"DPO reference evidence {filename}")

    splits = value.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation"}:
        raise RuntimeError("DPO reference evidence split contract is invalid")
    split_fields = {
        "rowCount",
        "sourceRowSHA256",
        "sourceRowsSHA256",
        "refChosenLogpsIEEE754Binary32",
        "refRejectedLogpsIEEE754Binary32",
        "referenceLogProbPairsSHA256",
    }
    for split_name in ("train", "validation"):
        split = splits[split_name]
        if not isinstance(split, dict) or set(split) != split_fields:
            raise RuntimeError(f"DPO reference evidence {split_name} split is invalid")
        row_count = split.get("rowCount")
        source_rows = split.get("sourceRowSHA256")
        chosen = split.get("refChosenLogpsIEEE754Binary32")
        rejected = split.get("refRejectedLogpsIEEE754Binary32")
        if (
            type(row_count) is not int
            or row_count <= 0
            or not isinstance(source_rows, list)
            or not isinstance(chosen, list)
            or not isinstance(rejected, list)
            or len(source_rows) != row_count
            or len(chosen) != row_count
            or len(rejected) != row_count
        ):
            raise RuntimeError(
                f"DPO reference evidence {split_name} row counts are invalid"
            )
        for digest in source_rows:
            _require_sha256(
                digest,
                name=f"DPO reference evidence {split_name} source row SHA-256",
            )
        if split.get("sourceRowsSHA256") != _canonical_sha256(source_rows):
            raise RuntimeError(
                f"DPO reference evidence {split_name} source-row digest drifted"
            )
        for column_name, encoded_values in (
            ("chosen", chosen),
            ("rejected", rejected),
        ):
            for index, encoded in enumerate(encoded_values):
                if not isinstance(encoded, str) or re.fullmatch(
                    r"[0-9a-f]{8}", encoded
                ) is None:
                    raise RuntimeError(
                        f"DPO reference evidence {split_name} {column_name}[{index}] is invalid"
                    )
                if not math.isfinite(struct.unpack(">f", bytes.fromhex(encoded))[0]):
                    raise RuntimeError(
                        f"DPO reference evidence {split_name} contains non-finite log probabilities"
                    )
        pairs = [
            {"chosen": chosen_value, "rejected": rejected_value}
            for chosen_value, rejected_value in zip(chosen, rejected, strict=True)
        ]
        if split.get("referenceLogProbPairsSHA256") != _canonical_sha256(pairs):
            raise RuntimeError(
                f"DPO reference evidence {split_name} scalar digest drifted"
            )

    expected_digest = value.get("referenceLogProbEvidenceSHA256")
    unsigned = dict(value)
    unsigned.pop("referenceLogProbEvidenceSHA256", None)
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(expected_digest or "")) is None
        or _canonical_sha256(unsigned) != expected_digest
    ):
        raise RuntimeError("DPO reference log-probability evidence integrity failed")
    return value


def _read_reference_log_prob_evidence(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Missing regular DPO reference log-probability evidence")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            "Unable to read DPO reference log-probability evidence"
        ) from exc
    return verify_reference_log_prob_evidence(value)


def _verify_reference_log_prob_lineage(
    evidence: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    for field in (
        "schema",
        "agent",
        "preferenceTrainer",
        "parentSFTAdapterSHA256",
        "sourceVariantManifestSHA256",
        "configSHA256",
        "datasetFileSHA256",
        "trainingCodeSHA256",
        "resolvedTrainingEnvironmentSHA256",
        "columns",
    ):
        if evidence.get(field) != expected.get(field):
            raise RuntimeError(f"DPO reference evidence {field} drifted")
    evidence_splits = evidence.get("splits")
    expected_splits = expected.get("splits")
    if not isinstance(evidence_splits, Mapping) or not isinstance(
        expected_splits, Mapping
    ):
        raise RuntimeError("DPO reference evidence source splits are invalid")
    for split_name in ("train", "validation"):
        for field in ("rowCount", "sourceRowSHA256", "sourceRowsSHA256"):
            if evidence_splits[split_name].get(field) != expected_splits[
                split_name
            ].get(field):
                raise RuntimeError(
                    f"DPO reference evidence {split_name} {field} drifted"
                )


def _restore_reference_log_prob_columns(
    trainer: Any,
    evidence: Mapping[str, Any],
) -> None:
    for attribute, split_name in (
        ("train_dataset", "train"),
        ("eval_dataset", "validation"),
    ):
        dataset = getattr(trainer, attribute, None)
        column_names = getattr(dataset, "column_names", None)
        if not isinstance(column_names, list) or any(
            column in column_names for column in DPO_REFERENCE_LOG_PROB_COLUMNS
        ):
            raise RuntimeError(
                f"DPO {split_name} dataset is unsafe for reference-column restoration"
            )
        split = evidence["splits"][split_name]
        chosen = [
            struct.unpack(">f", bytes.fromhex(encoded))[0]
            for encoded in split["refChosenLogpsIEEE754Binary32"]
        ]
        rejected = [
            struct.unpack(">f", bytes.fromhex(encoded))[0]
            for encoded in split["refRejectedLogpsIEEE754Binary32"]
        ]
        restored = dataset.add_column(DPO_REFERENCE_LOG_PROB_COLUMNS[0], chosen)
        restored = restored.add_column(DPO_REFERENCE_LOG_PROB_COLUMNS[1], rejected)
        setattr(trainer, attribute, restored)
    trainer._precomputed_train_ref_log_probs = True
    trainer._precomputed_eval_ref_log_probs = True


def _bind_reference_log_prob_evidence(
    record_path: Path,
    evidence_path: Path,
    evidence: Mapping[str, Any],
) -> None:
    record = _read_preference_checkpoint_lineage(record_path)
    if (
        record.get("preferenceTrainer") != "dpo"
        or record.get("referenceLogProbEvidencePath") != str(evidence_path.resolve())
        or record.get("parentSFTAdapterSHA256")
        != evidence.get("parentSFTAdapterSHA256")
    ):
        raise RuntimeError("DPO checkpoint lineage cannot bind this reference evidence")
    digest = evidence.get("referenceLogProbEvidenceSHA256")
    current = record.get("referenceLogProbEvidenceSHA256")
    if current is None:
        if record.get("checkpoints"):
            raise RuntimeError(
                "DPO reference evidence cannot be bound after checkpoints exist"
            )
        updated = dict(record)
        updated["referenceLogProbEvidenceSHA256"] = digest
        _write_json_atomic(
            record_path,
            _self_hashed_preference_checkpoint_record(updated),
        )
    elif current != digest:
        raise RuntimeError("DPO checkpoint reference evidence digest drifted")


def _prepare_reference_log_prob_evidence(
    trainer: Any,
    training_args: Any,
    cfg: Mapping[str, Any],
    *,
    cfg_path: Path,
    parent_sft_adapter_sha256: str,
    source_rows_by_split: Mapping[str, list[dict[str, Any]]],
    evidence_path: Path,
    checkpoint_lineage_path: Path | None,
    reuse_existing: bool,
) -> tuple[dict[str, bool], dict[str, Any], bool]:
    static = _reference_log_prob_static_contract(
        cfg,
        cfg_path=cfg_path,
        parent_sft_adapter_sha256=parent_sft_adapter_sha256,
        source_rows_by_split=source_rows_by_split,
    )
    if reuse_existing:
        if checkpoint_lineage_path is None:
            raise RuntimeError("DPO reference evidence reuse requires checkpoint lineage")
        evidence = _read_reference_log_prob_evidence(evidence_path)
        record = _read_preference_checkpoint_lineage(checkpoint_lineage_path)
        if record.get("referenceLogProbEvidenceSHA256") != evidence.get(
            "referenceLogProbEvidenceSHA256"
        ):
            raise RuntimeError("DPO resume reference evidence is not lineage-bound")
        _verify_reference_log_prob_lineage(evidence, expected=static)
        _restore_reference_log_prob_columns(trainer, evidence)
        reconstructed = _build_reference_log_prob_evidence(
            trainer,
            cfg,
            cfg_path=cfg_path,
            parent_sft_adapter_sha256=parent_sft_adapter_sha256,
            source_rows_by_split=source_rows_by_split,
        )
        if reconstructed != evidence:
            raise RuntimeError("Restored DPO reference columns drifted from evidence")
        return {"train": True, "evaluation": True}, evidence, True

    if evidence_path.exists() or evidence_path.is_symlink():
        if checkpoint_lineage_path is None:
            raise RuntimeError(
                "Fresh DPO reference precompute refuses pre-existing evidence"
            )
        record = _read_preference_checkpoint_lineage(checkpoint_lineage_path)
        if (
            evidence_path.is_symlink()
            or not evidence_path.is_file()
            or record.get("referenceLogProbEvidenceSHA256") is not None
            or record.get("checkpoints")
        ):
            raise RuntimeError(
                "DPO reference precompute refuses unsafe or bound pre-existing evidence"
            )
        # A kill may land after the evidence file is atomically installed but
        # before its digest is committed to checkpoint lineage. The unbound
        # scalars are never trusted; discard and recompute them from the frozen
        # reference adapter.
        evidence_path.unlink()
    precomputed = _precompute_reference_log_probs_before_training(
        trainer,
        training_args,
        has_eval_dataset=True,
    )
    evidence = _build_reference_log_prob_evidence(
        trainer,
        cfg,
        cfg_path=cfg_path,
        parent_sft_adapter_sha256=parent_sft_adapter_sha256,
        source_rows_by_split=source_rows_by_split,
    )
    _write_json_atomic(evidence_path, evidence)
    if checkpoint_lineage_path is not None:
        _bind_reference_log_prob_evidence(
            checkpoint_lineage_path,
            evidence_path,
            evidence,
        )
    return precomputed, evidence, False


def _drop_precomputed_reference_adapter(
    trainer: Any,
    *,
    reference_log_prob_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep DPO checkpoints single-adapter after frozen reference precompute.

    Transformers 4.57.6 restores a multi-adapter PEFT checkpoint by loading
    adapter subdirectories when any exist. Keeping the frozen SFT reference in
    those checkpoints can therefore bypass the root policy adapter on resume.
    Once both datasets carry immutable reference log probabilities, the
    reference adapter is no longer needed by the training or evaluation loss.
    """

    if (
        getattr(trainer, "_precomputed_train_ref_log_probs", None) is not True
        or (
            getattr(trainer, "eval_dataset", None) is not None
            and getattr(trainer, "_precomputed_eval_ref_log_probs", None)
            is not True
        )
    ):
        raise RuntimeError(
            "Reference adapter cannot be removed before all reference log probabilities are frozen"
        )
    verified_evidence = verify_reference_log_prob_evidence(
        dict(reference_log_prob_evidence)
    )
    if any(
        verified_evidence["splits"][split]["rowCount"] <= 0
        for split in ("train", "validation")
    ):
        raise RuntimeError("Reference adapter removal requires qualified train and eval evidence")
    model = trainer.model
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, Mapping) or set(peft_config) != {
        POLICY_ADAPTER_NAME,
        REFERENCE_ADAPTER_NAME,
    }:
        raise RuntimeError(
            "DPO model must contain exactly the policy and frozen reference adapters"
        )
    set_adapter = getattr(model, "set_adapter", None)
    delete_adapter = getattr(model, "delete_adapter", None)
    if not callable(set_adapter) or not callable(delete_adapter):
        raise RuntimeError("DPO model lacks controlled adapter lifecycle methods")
    set_adapter(POLICY_ADAPTER_NAME)
    delete_adapter(REFERENCE_ADAPTER_NAME)
    remaining = getattr(model, "peft_config", None)
    if not isinstance(remaining, Mapping) or set(remaining) != {
        POLICY_ADAPTER_NAME
    }:
        raise RuntimeError("Frozen reference adapter deletion was not effective")
    set_adapter(POLICY_ADAPTER_NAME)
    return {
        "referenceAdapterRemovedAfterPrecompute": True,
        "checkpointAdapterNames": [POLICY_ADAPTER_NAME],
    }


def _shared_finalized_variant_validator() -> Any:
    """Load the crawler's canonical variant validator in repo and deployed layouts."""

    try:
        module = importlib.import_module(
            "lumen_manifest_crawler.dataset.adapter_evaluation"
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "lumen_manifest_crawler",
            "lumen_manifest_crawler.dataset",
            "lumen_manifest_crawler.dataset.adapter_evaluation",
        }:
            raise
        repo_root = Path(__file__).resolve().parents[3]
        crawler_root = repo_root / "tools" / "lumen_manifest_crawler"
        if not crawler_root.is_dir():
            raise RuntimeError(
                "The shared finalized-variant verifier is not bundled with preference training"
            ) from exc
        if str(crawler_root) not in sys.path:
            sys.path.insert(0, str(crawler_root))
        module = importlib.import_module(
            "lumen_manifest_crawler.dataset.adapter_evaluation"
        )
    validator = getattr(module, "_valid_variant_manifest", None)
    if not callable(validator):
        raise RuntimeError("The shared finalized-variant verifier is unavailable")
    return validator


def _expected_sft_parent_lineage(cfg: Mapping[str, Any]) -> dict[str, Any]:
    base_model_id = cfg.get("baseModelID", cfg.get("base_model_name"))
    if base_model_id != cfg.get("base_model_name"):
        raise RuntimeError("Preference config base-model identities do not agree")
    environment_lock = cfg.get("trainingEnvironmentLock")
    if not isinstance(environment_lock, Mapping):
        raise RuntimeError("Preference config is missing its training-environment lock")
    environment_lock_sha256 = _canonical_sha256(dict(environment_lock))
    configured_lock_sha256 = cfg.get("trainingEnvironmentLockSHA256")
    if configured_lock_sha256 is not None and configured_lock_sha256 != environment_lock_sha256:
        raise RuntimeError("Preference config training-environment lock digest is invalid")
    attestation = cfg.get("variantAttestation")
    if (
        not isinstance(attestation, Mapping)
        or attestation.get("schema")
        != TRAINING_VARIANT_ATTESTATION_SCHEMA
    ):
        raise RuntimeError("Preference config is missing its variant attestation")
    if (
        attestation.get("trainingEnvironmentLockSHA256")
        != environment_lock_sha256
    ):
        raise RuntimeError("Preference config variant attestation has environment-lock drift")
    phase_digests = cfg.get("trainingCodeSHA256ByPhase")
    if not isinstance(phase_digests, Mapping):
        raise RuntimeError("Preference config is missing phase-specific code lineage")
    return {
        "agent": cfg.get("agent"),
        "variant": cfg.get("variant"),
        "sourceVariantManifestSHA256": cfg.get("variantManifestSHA256"),
        "seed": cfg.get("seed"),
        "baseModelID": base_model_id,
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
        "trainingConfigSHA256": attestation.get(
            "effectiveTrainingConfigSHA256"
        ),
        "trainingConfigInvariantSHA256": attestation.get(
            "trainingConfigInvariantSHA256"
        ),
        "trainingEnvironmentLockSHA256": environment_lock_sha256,
        "trainingDependencyLockSHA256": cfg.get(
            "trainingDependencyLockSHA256"
        ),
        "requirementsSHA256": cfg.get("requirementsSHA256"),
        "resolvedTrainingEnvironmentSHA256": cfg.get(
            "resolvedTrainingEnvironmentSHA256"
        ),
        "spaceConfigurationSHA256": cfg.get("spaceConfigurationSHA256"),
        "runtimeSourceKind": cfg.get("runtimeSourceKind"),
        "trainingCodeSHA256": phase_digests.get("sft"),
    }


def _verified_sft_parent(
    cfg: dict[str, Any],
    *,
    adapter_dir: Path,
    finalized_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved_environment = cfg.get("resolvedTrainingEnvironment")
    if resolved_environment is None:
        resolved_environment = build_resolved_training_environment()
        cfg["resolvedTrainingEnvironment"] = resolved_environment
    if not isinstance(resolved_environment, Mapping):
        raise RuntimeError("Preference config resolved training environment is invalid")
    try:
        resolved_environment_sha256 = verify_resolved_training_environment(
            resolved_environment
        )
    except ValueError as exc:
        raise RuntimeError(
            "Preference config resolved training environment is invalid"
        ) from exc
    configured_resolved_sha256 = cfg.get("resolvedTrainingEnvironmentSHA256")
    if (
        configured_resolved_sha256 is not None
        and configured_resolved_sha256 != resolved_environment_sha256
    ):
        raise RuntimeError("Preference config resolved dependency digest drifted")
    cfg["resolvedTrainingEnvironmentSHA256"] = resolved_environment_sha256
    finalized = json.loads(finalized_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(finalized, dict):
        raise RuntimeError("Finalized SFT variant manifest must be a JSON object")
    unsigned = {
        key: value
        for key, value in finalized.items()
        if key != "variantManifestSHA256"
    }
    if _canonical_sha256(unsigned) != finalized.get("variantManifestSHA256"):
        raise RuntimeError("Finalized SFT variant manifest integrity check failed")
    validator = _shared_finalized_variant_validator()
    if not validator(
        finalized,
        agent=str(cfg.get("agent") or ""),
        expected_variant=str(cfg.get("variant") or ""),
        require_trained_artifact=True,
    ):
        raise RuntimeError(
            "DPO/ORPO input must be a structurally valid finalized SFT variant"
        )

    artifact = finalized.get("artifact")
    if not isinstance(artifact, Mapping):
        raise RuntimeError("Finalized SFT variant is missing its trained artifact")
    expected_lineage = _expected_sft_parent_lineage(cfg)
    for field, expected in expected_lineage.items():
        if finalized.get(field) != expected:
            raise RuntimeError(
                f"Finalized SFT parent {field} does not match preference-training lineage"
            )
    if (
        artifact.get("status") != "trained"
        or artifact.get("trainingPhase") != "sft"
        or artifact.get("effectiveSeed") != cfg.get("seed")
    ):
        raise RuntimeError("DPO/ORPO input must be a finalized SFT artifact")
    parent_sha256 = _require_sha256(
        artifact.get("adapterSHA256"),
        name="finalized SFT adapterSHA256",
    )
    try:
        adapter_manifest = verify_adapter_artifact(
            adapter_dir,
            expected_adapter_sha256=parent_sha256,
            expected_training_phase="sft",
            expected_base_model=str(cfg.get("baseModelID") or ""),
            expected_base_revision=str(cfg.get("baseModelRevision") or ""),
        )
    except ValueError as exc:
        if "base model" in str(exc) or "revision" in str(exc):
            raise RuntimeError(
                "Finalized SFT adapter base model or revision does not match "
                "preference-training configuration"
            ) from exc
        raise
    if artifact.get("adapterManifestSHA256") != adapter_manifest["adapterSHA256"]:
        raise RuntimeError("Finalized SFT manifest does not bind the canonical adapter file manifest")
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if not isinstance(adapter_config, Mapping) or (
        adapter_config.get("base_model_name_or_path") != cfg.get("base_model_name")
        or adapter_config.get("revision") != cfg.get("baseModelRevision")
    ):
        raise RuntimeError(
            "Finalized SFT adapter base model does not match preference-training configuration"
        )
    parent_audit_lineage = {
        **expected_lineage,
        "variantManifestSHA256": finalized["variantManifestSHA256"],
        "adapterSHA256": adapter_manifest["adapterSHA256"],
        "adapterManifestSHA256": artifact["adapterManifestSHA256"],
        "effectiveSeed": artifact["effectiveSeed"],
        "runtimeSourceRevision": finalized["runtimeSourceRevision"],
        "expectedRuntimeSourceRevision": finalized[
            "expectedRuntimeSourceRevision"
        ],
        "observedRepositoryRevision": finalized[
            "observedRepositoryRevision"
        ],
        "observedRuntimeRevision": finalized["observedRuntimeRevision"],
        "runtimeSourceBindingStatus": finalized[
            "runtimeSourceBindingStatus"
        ],
        "runtimeSourceBindingMethod": finalized[
            "runtimeSourceBindingMethod"
        ],
        "trainingEnvironmentSHA256": finalized["trainingEnvironmentSHA256"],
        "resolvedTrainingEnvironmentSHA256": finalized[
            "resolvedTrainingEnvironmentSHA256"
        ],
        "zeroGPUSize": finalized.get("zeroGPUSize"),
        "zeroGPUDurationSeconds": finalized.get("zeroGPUDurationSeconds"),
        "observedAccelerator": finalized.get("observedAccelerator"),
    }
    return finalized, adapter_manifest, parent_audit_lineage


def _finalize_dpo_variant(
    cfg: dict[str, Any],
    *,
    adapter_artifact_manifest: dict[str, Any],
    parent_sft_adapter_sha256: str,
    reference_sft_adapter_sha256: str | None,
    parent_sft_lineage: Mapping[str, Any],
    reference_sft_lineage: Mapping[str, Any] | None,
    preference_trainer: str,
    training_environment: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    source_path = Path(cfg["dataset_dir"]).resolve() / "variant_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        not isinstance(source, dict)
        or source.get("agent") != cfg["agent"]
        or source.get("variant") != cfg["variant"]
        or source.get("variantManifestSHA256") != cfg["variantManifestSHA256"]
    ):
        raise RuntimeError("DPO config is not bound to the selected pending variant manifest")

    import sys

    repo_root = Path(__file__).resolve().parents[3]
    crawler_root = repo_root / "tools" / "lumen_manifest_crawler"
    if crawler_root.is_dir() and str(crawler_root) not in sys.path:
        sys.path.insert(0, str(crawler_root))
    from lumen_manifest_crawler.dataset.adapter_evaluation import (
        finalize_experiment_variant_manifest,
    )

    finalized = finalize_experiment_variant_manifest(
        source,
        adapter_sha256=adapter_artifact_manifest["adapterSHA256"],
        adapter_artifact_manifest=adapter_artifact_manifest,
        training_environment=training_environment,
        training_phase="sft_dpo",
        parent_sft_adapter_sha256=parent_sft_adapter_sha256,
        reference_sft_adapter_sha256=reference_sft_adapter_sha256,
        parent_sft_lineage=parent_sft_lineage,
        reference_sft_lineage=reference_sft_lineage,
        preference_trainer=preference_trainer,
    )
    _write_json_atomic(output_path, finalized)
    return finalized


def _select_preference_runtime_lineage(
    cfg: dict[str, Any],
    *,
    preference_trainer: str,
) -> dict[str, Any]:
    manifests = cfg.get("trainingCodeManifestsByPhase")
    digests = cfg.get("trainingCodeSHA256ByPhase")
    if not isinstance(manifests, dict) or not isinstance(digests, dict):
        raise RuntimeError("Preference training requires phase-specific code manifests")
    manifest = manifests.get(preference_trainer)
    digest = digests.get(preference_trainer)
    if (
        not isinstance(manifest, dict)
        or manifest.get("phase") != preference_trainer
        or manifest.get("trainingCodeSHA256") != digest
    ):
        raise RuntimeError("Preference training-code manifest is invalid")
    cfg["trainingCodeManifest"] = manifest
    cfg["trainingCodeSHA256"] = digest
    environment_payload = {
        "schemaVersion": "lumen.adapter-training-environment/1.0.0",
        "containerImageDigest": cfg["trainingContainerImageDigest"],
        "containerImageDigestSource": cfg[
            "trainingContainerImageDigestSource"
        ],
        "runtimeImageBindingStatus": cfg[
            "trainingRuntimeImageBindingStatus"
        ],
        "runtimeImageBindingVerified": cfg[
            "trainingRuntimeImageBindingVerified"
        ],
        "effectiveSeed": int(cfg["seed"]),
        "environmentLock": cfg["trainingEnvironmentLock"],
        "trainingCodeSHA256": digest,
        "trainingDependencyLockSHA256": cfg[
            "trainingDependencyLockSHA256"
        ],
        "requirementsSHA256": cfg["requirementsSHA256"],
    }
    cfg["trainingEnvironmentSHA256"] = _canonical_sha256(environment_payload)
    return _training_runtime_lineage(cfg, phase=preference_trainer)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    seed, seed_source = _resolve_controlled_seed(cfg)
    sft_adapter_dir = Path(args.sft_adapter_dir).resolve()
    output_dir, dpo_adapter_dir = validate_dpo_artifact_paths(
        cfg,
        sft_adapter_dir=sft_adapter_dir,
    )
    preference_config = _validate_preference_training_config(cfg)
    preference_trainer = preference_config["preferenceTrainer"]
    sft_finalized_variant_manifest = Path(args.sft_finalized_variant_manifest).resolve()
    _, sft_artifact_manifest, parent_sft_lineage = _verified_sft_parent(
        cfg,
        adapter_dir=sft_adapter_dir,
        finalized_manifest_path=sft_finalized_variant_manifest,
    )
    resume_checkpoint, checkpoint_lineage_path, unbound_checkpoints = (
        _bind_and_validate_preference_checkpoint_lineage(
            cfg,
            cfg_path=cfg_path,
            preference_trainer=preference_trainer,
            parent_sft_adapter_sha256=sft_artifact_manifest["adapterSHA256"],
            require_checkpoint=bool(args.resume_from_checkpoint),
        )
    )
    if unbound_checkpoints:
        _prune_unbound_preference_checkpoints(unbound_checkpoints)

    dataset_dir = Path(cfg["dataset_dir"]).resolve()
    train_path = dataset_dir / "train_dpo.jsonl"
    val_path = dataset_dir / "val_dpo.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing DPO dataset split files under {dataset_dir}")

    training_runtime_lineage = _select_preference_runtime_lineage(
        cfg,
        preference_trainer=preference_trainer,
    )
    cfg["resolvedTrainingEnvironment"] = training_runtime_lineage[
        "resolvedTrainingEnvironment"
    ]
    cfg["resolvedTrainingEnvironmentSHA256"] = training_runtime_lineage[
        "resolvedTrainingEnvironmentSHA256"
    ]
    cfg["trainingEnvironmentSHA256"] = None
    training_environment = _training_environment(
        cfg,
        runtime_lineage=training_runtime_lineage,
    )
    cfg["trainingEnvironmentSHA256"] = training_environment[
        "trainingEnvironmentSHA256"
    ]
    _verify_base_model_lineage(cfg)

    cuda_allocator = (
        _latch_expandable_cuda_allocator()
        if preference_trainer == "dpo"
        else None
    )
    _require_unsloth_before_transformers()
    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from peft import PeftModel
        from trl import DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer
        from trl.data_utils import maybe_apply_chat_template
        from transformers import TrainerCallback
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for Unsloth DPO training. Install: unsloth, trl, datasets, transformers, peft, accelerate, bitsandbytes."
        ) from exc

    # Unsloth must patch Transformers before the shared seed helper imports it.
    _seed_everything(seed)

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
        load_in_4bit=bool(cfg["load_in_4bit"]),
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
    verify_chat_template_contract(cfg["chatTemplateContract"], tokenizer=tokenizer)

    train_raw = load_jsonl(train_path)
    val_raw = load_jsonl(val_path)
    train_rows = [row_to_preference(row) for row in train_raw]
    val_rows = [row_to_preference(row) for row in val_raw]
    max_sequence_length = cfg["max_seq_length"]
    max_prompt_length = preference_config["maxPromptLength"]
    token_length_preflight = _preflight_preference_token_lengths(
        {"train": train_rows, "validation": val_rows},
        tokenizer=tokenizer,
        render_preference=maybe_apply_chat_template,
        max_prompt_length=max_prompt_length,
        max_sequence_length=max_sequence_length,
        minimum_prompt_margin_tokens=cfg.get(
            "preference_minimum_prompt_margin_tokens",
            PREFERENCE_MINIMUM_PROMPT_MARGIN_TOKENS,
        ),
        minimum_sequence_margin_tokens=cfg.get(
            "preference_minimum_sequence_margin_tokens",
            PREFERENCE_MINIMUM_SEQUENCE_MARGIN_TOKENS,
        ),
        source_splits={"train": train_raw, "validation": val_raw},
        agent=cfg.get("agent"),
        fleet_loss_share_contract=cfg.get("fleetLossShareContract"),
        public_corpus_loss_share_contract=cfg.get(
            "publicCorpusLossShareContract"
        ),
        fleet_config=cfg,
    )
    token_length_preflight_evidence = (
        _bind_preference_token_length_preflight(
            cfg,
            cfg_path=cfg_path,
            preflight=token_length_preflight,
        )
        if checkpoint_lineage_path is not None
        else token_length_preflight
    )
    if checkpoint_lineage_path is not None:
        _verify_prepared_global_tokenizer_preflight(
            cfg,
            cfg_path=cfg_path,
            phase="preference",
            bound_preflight=token_length_preflight_evidence,
        )
    model = _load_sft_policy(
        model,
        peft_model_class=PeftModel,
        sft_adapter_dir=sft_adapter_dir,
        preference_trainer=preference_trainer,
    )
    peft_base_model_identity = _normalize_peft_base_model_identity(model, cfg)
    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows) if val_rows else None

    output_dir.mkdir(parents=True, exist_ok=True)
    dpo_adapter_dir.mkdir(parents=True, exist_ok=True)

    trainer, training_args = _build_preference_trainer(
        cfg,
        preference_trainer=preference_trainer,
        seed=seed,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        dpo_config_class=DPOConfig,
        dpo_trainer_class=DPOTrainer,
        orpo_config_class=ORPOConfig,
        orpo_trainer_class=ORPOTrainer,
    )
    if checkpoint_lineage_path is not None:
        trainer.add_callback(
            _preference_checkpoint_callback(
                TrainerCallback,
                record_path=checkpoint_lineage_path,
            )
        )

    reference_log_probs_precomputed = None
    reference_log_prob_evidence = None
    reference_log_prob_evidence_reused = False
    checkpoint_adapter_contract = None
    if preference_trainer == "dpo":
        if val_dataset is None:
            raise RuntimeError(
                "DPO requires a non-empty validation split for reference evidence"
            )
        evidence_path = output_dir / DPO_REFERENCE_LOG_PROB_EVIDENCE_FILENAME
        reuse_reference_evidence = False
        if checkpoint_lineage_path is not None and args.resume_from_checkpoint:
            checkpoint_record = _read_preference_checkpoint_lineage(
                checkpoint_lineage_path
            )
            reuse_reference_evidence = (
                checkpoint_record.get("referenceLogProbEvidenceSHA256") is not None
            )
        (
            reference_log_probs_precomputed,
            reference_log_prob_evidence,
            reference_log_prob_evidence_reused,
        ) = _prepare_reference_log_prob_evidence(
            trainer,
            training_args,
            cfg,
            cfg_path=cfg_path,
            parent_sft_adapter_sha256=sft_artifact_manifest["adapterSHA256"],
            source_rows_by_split={
                "train": train_rows,
                "validation": val_rows,
            },
            evidence_path=evidence_path,
            checkpoint_lineage_path=checkpoint_lineage_path,
            reuse_existing=reuse_reference_evidence,
        )
        checkpoint_adapter_contract = _drop_precomputed_reference_adapter(
            trainer,
            reference_log_prob_evidence=reference_log_prob_evidence,
        )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint is not None else None
        )
    )
    evaluation_metrics = trainer.evaluate() if val_dataset is not None else {}
    training_completion = _verified_training_completion_evidence(
        trainer,
        training_args,
        train_result,
        evaluation_metrics,
        has_eval_dataset=val_dataset is not None,
        train_record_count=len(train_rows),
        expected_precision=preference_config["precision"],
    )
    peft_base_model_identity = _save_policy_adapter(
        trainer.model,
        dpo_adapter_dir,
        cfg,
    )
    adapter_tokenizer_binding = _publish_exact_base_tokenizer_subset(
        cfg,
        adapter_output_dir=dpo_adapter_dir,
        snapshot_path=runtime_tokenizer_snapshot_path,
        snapshot_verification=runtime_tokenizer_snapshot_verification,
    )
    dpo_artifact_manifest = write_adapter_artifact_manifest(
        dpo_adapter_dir,
        training_phase="sft_dpo",
        parent_sft_adapter_sha256=sft_artifact_manifest["adapterSHA256"],
        expected_base_model=cfg["baseModelID"],
        expected_base_revision=cfg["baseModelRevision"],
    )
    finalized_manifest_path = output_dir / "finalized_variant_manifest.json"
    finalized_variant = _finalize_dpo_variant(
        cfg,
        adapter_artifact_manifest=dpo_artifact_manifest,
        parent_sft_adapter_sha256=sft_artifact_manifest["adapterSHA256"],
        reference_sft_adapter_sha256=(
            sft_artifact_manifest["adapterSHA256"]
            if preference_trainer == "dpo"
            else None
        ),
        parent_sft_lineage=parent_sft_lineage,
        reference_sft_lineage=(
            parent_sft_lineage if preference_trainer == "dpo" else None
        ),
        preference_trainer=preference_trainer,
        training_environment={
            **training_environment,
            **{
                field: training_runtime_lineage[field]
                for field in (
                    "spaceConfigurationSHA256",
                    "runtimeSourceKind",
                    "runtimeSourceRevision",
                    "expectedRuntimeSourceRevision",
                    "observedRepositoryRevision",
                    "observedRuntimeRevision",
                    "runtimeSourceBindingStatus",
                    "runtimeSourceBindingMethod",
                )
            },
        },
        output_path=finalized_manifest_path,
    )

    report = {
        "schema": "lumen.train_preference.report/1.0.0",
        "agent": cfg["agent"],
        "trainer": "ORPOTrainer" if preference_trainer == "orpo" else "DPOTrainer",
        "base_model_name": cfg["base_model_name"],
        "baseModelRevision": cfg["baseModelRevision"],
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
        "dataset_dir": str(dataset_dir),
        "datasetRepository": cfg.get("datasetRepository"),
        "datasetRevision": cfg.get("datasetRevision"),
        "runResumeLineageSHA256": cfg.get("runResumeLineageSHA256"),
        "resume_from_checkpoint": resume_checkpoint is not None,
        "resume_checkpoint": (
            str(resume_checkpoint) if resume_checkpoint is not None else None
        ),
        "checkpoint_lineage": (
            str(checkpoint_lineage_path)
            if checkpoint_lineage_path is not None
            else None
        ),
        "checkpoint_save_steps": int(training_args.save_steps),
        "checkpoint_save_total_limit": int(training_args.save_total_limit),
        "checkpoint_adapter_contract": checkpoint_adapter_contract,
        "token_length_preflight": token_length_preflight_evidence,
        "token_length_preflight_path": cfg.get(
            "preferenceTokenLengthPreflightPath"
        ),
        "token_length_preflight_sha256": token_length_preflight_evidence.get(
            "preflightSHA256"
        ),
        "checkpoint_recovery_discarded_unbound": [
            str(path) for path in unbound_checkpoints
        ],
        "variantManifestSHA256": cfg["variantManifestSHA256"],
        "config_sha256": _hash_file(cfg_path),
        "preferenceTrainingConfig": preference_config,
        "train_records": len(train_rows),
        "val_records": len(val_rows),
        "output_dir": str(output_dir),
        "adapter_output_dir": str(dpo_adapter_dir),
        "training_phase": "sft_dpo",
        "parent_sft_adapter_sha256": sft_artifact_manifest["adapterSHA256"],
        "reference_sft_adapter_sha256": (
            sft_artifact_manifest["adapterSHA256"]
            if preference_trainer == "dpo"
            else None
        ),
        "reference_log_probs_precomputed": reference_log_probs_precomputed,
        "reference_log_prob_evidence": (
            {
                "path": str(
                    output_dir / DPO_REFERENCE_LOG_PROB_EVIDENCE_FILENAME
                ),
                "referenceLogProbEvidenceSHA256": reference_log_prob_evidence[
                    "referenceLogProbEvidenceSHA256"
                ],
                "fileSHA256": _hash_file(
                    output_dir / DPO_REFERENCE_LOG_PROB_EVIDENCE_FILENAME
                ),
                "reusedFromCheckpointLineage": reference_log_prob_evidence_reused,
                "trainRowCount": reference_log_prob_evidence["splits"]["train"][
                    "rowCount"
                ],
                "validationRowCount": reference_log_prob_evidence["splits"][
                    "validation"
                ]["rowCount"],
            }
            if reference_log_prob_evidence is not None
            else None
        ),
        "cudaAllocator": cuda_allocator,
        "parentSFTLineage": parent_sft_lineage,
        "referenceSFTLineage": (
            parent_sft_lineage if preference_trainer == "dpo" else None
        ),
        "preferenceTrainingRuntime": training_runtime_lineage,
        "seed": seed,
        "seed_source": seed_source,
        "trainingEnvironment": training_environment,
        "trainingEnvironmentSHA256": training_environment[
            "trainingEnvironmentSHA256"
        ],
        "precision": preference_config["precision"],
        **training_runtime_lineage,
        "adapterSHA256": dpo_artifact_manifest["adapterSHA256"],
        "finalized_variant_manifest": str(finalized_manifest_path),
        "finalized_variant_manifest_sha256": finalized_variant["variantManifestSHA256"],
        "trainingCompletion": training_completion,
        "metrics": train_result.metrics,
        "evaluation_metrics": evaluation_metrics,
    }
    _write_json_atomic(output_dir / "dpo_report.json", report)


if __name__ == "__main__":
    main()
