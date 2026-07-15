from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from .adapter_artifact import verify_adapter_artifact, write_adapter_artifact_manifest
    from .training_lineage import (
        build_resolved_training_environment,
        verify_resolved_training_environment,
    )
    from .train_sft import (
        _resolve_controlled_seed,
        _seed_everything,
        _training_environment,
        _training_runtime_lineage,
        _write_json_atomic,
    )
except ImportError:
    from adapter_artifact import verify_adapter_artifact, write_adapter_artifact_manifest
    from training_lineage import (
        build_resolved_training_environment,
        verify_resolved_training_environment,
    )
    from train_sft import (
        _resolve_controlled_seed,
        _seed_everything,
        _training_environment,
        _training_runtime_lineage,
        _write_json_atomic,
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
    "output_dir",
    "adapter_output_dir",
    "dpo_output_dir",
    "dataset_dir",
    "variant",
    "variantManifestSHA256",
    "seed",
}
AGENTS = {"cortex", "executor", "mouth", "mimicry", "rem", "fleet"}
FINETUNE_MARKERS = {"sft", "dpo", "orpo", "lora", "merged", "adapter", "finetune", "finetuned", "training"}
POLICY_ADAPTER_NAME = "default"
REFERENCE_ADAPTER_NAME = "lumen_sft_reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-agent DPO/ORPO adapters with Unsloth.")
    parser.add_argument("--config", required=True, help="Path to agent Unsloth JSON config.")
    parser.add_argument("--sft-adapter-dir", required=True, help="Canonical finalized SFT adapter directory.")
    parser.add_argument(
        "--sft-finalized-variant-manifest",
        required=True,
        help="Finalized SFT variant manifest bound to --sft-adapter-dir.",
    )
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


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def row_to_preference(row: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
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

    if "chosen" not in row or "rejected" not in row:
        raise ValueError("Preference record must include chosen and rejected assistant messages")
    chosen = _completion_messages(row["chosen"], field="chosen")
    rejected = _completion_messages(row["rejected"], field="rejected")
    if _normalized_completion_content(chosen[0]["content"]) == _normalized_completion_content(
        rejected[0]["content"]
    ):
        raise ValueError("Preference chosen and rejected completions must differ")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


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
    common_config = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(cfg["batch_size"]),
        "per_device_eval_batch_size": max(1, int(cfg["batch_size"])),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "learning_rate": float(cfg["learning_rate"]),
        "num_train_epochs": float(cfg["num_train_epochs"]),
        "warmup_steps": int(cfg["warmup_steps"]),
        "logging_steps": int(cfg.get("logging_steps", 10)),
        "eval_strategy": "epoch" if val_dataset is not None else "no",
        "save_strategy": "epoch",
        "save_total_limit": int(cfg.get("save_total_limit", 2)),
        "bf16": bool(cfg.get("bf16", False)),
        "fp16": bool(cfg.get("fp16", True)),
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "max_length": int(cfg["max_seq_length"]),
        "max_prompt_length": int(
            cfg.get("max_prompt_length", int(cfg["max_seq_length"]) // 2)
        ),
    }
    if preference_trainer == "dpo":
        training_args = dpo_config_class(
            **common_config,
            beta=float(cfg.get("dpo_beta", 0.1)),
            model_adapter_name=POLICY_ADAPTER_NAME,
            ref_adapter_name=REFERENCE_ADAPTER_NAME,
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
    if preference_trainer == "orpo":
        training_args = orpo_config_class(
            **common_config,
            beta=float(cfg.get("orpo_beta", 0.1)),
        )
        trainer = orpo_trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )
        return trainer, training_args
    raise ValueError("preference_trainer must be either 'dpo' or 'orpo'")


def _save_policy_adapter(model: Any, output_dir: Path) -> None:
    model.set_adapter(POLICY_ADAPTER_NAME)
    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        selected_adapters=[POLICY_ADAPTER_NAME],
    )


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
    if isinstance(attestation, Mapping) and (
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
    adapter_manifest = verify_adapter_artifact(
        adapter_dir,
        expected_adapter_sha256=parent_sha256,
        expected_training_phase="sft",
    )
    if artifact.get("adapterManifestSHA256") != adapter_manifest["adapterSHA256"]:
        raise RuntimeError("Finalized SFT manifest does not bind the canonical adapter file manifest")
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if not isinstance(adapter_config, Mapping) or (
        adapter_config.get("base_model_name_or_path") != cfg.get("base_model_name")
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
    cfg = load_config(Path(args.config).resolve())
    seed, seed_source = _resolve_controlled_seed(cfg)
    _seed_everything(seed)
    sft_adapter_dir = Path(args.sft_adapter_dir).resolve()
    output_dir, dpo_adapter_dir = validate_dpo_artifact_paths(
        cfg,
        sft_adapter_dir=sft_adapter_dir,
    )
    preference_trainer = str(cfg.get("preference_trainer", "dpo")).lower()
    if preference_trainer not in {"dpo", "orpo"}:
        raise ValueError("preference_trainer must be either 'dpo' or 'orpo'")
    sft_finalized_variant_manifest = Path(args.sft_finalized_variant_manifest).resolve()
    _, sft_artifact_manifest, parent_sft_lineage = _verified_sft_parent(
        cfg,
        adapter_dir=sft_adapter_dir,
        finalized_manifest_path=sft_finalized_variant_manifest,
    )

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

    try:
        from datasets import Dataset
        from unsloth import FastLanguageModel
        from peft import PeftModel
        from trl import DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for Unsloth DPO training. Install: unsloth, trl, datasets, transformers, peft, accelerate, bitsandbytes."
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model_name"],
        revision=cfg["baseModelRevision"],
        max_seq_length=int(cfg["max_seq_length"]),
        load_in_4bit=bool(cfg["load_in_4bit"]),
    )
    model = _load_sft_policy(
        model,
        peft_model_class=PeftModel,
        sft_adapter_dir=sft_adapter_dir,
        preference_trainer=preference_trainer,
    )

    train_raw = load_jsonl(train_path)
    val_raw = load_jsonl(val_path)
    train_rows = [row_to_preference(row) for row in train_raw]
    val_rows = [row_to_preference(row) for row in val_raw]
    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows) if val_rows else None

    output_dir.mkdir(parents=True, exist_ok=True)
    dpo_adapter_dir.mkdir(parents=True, exist_ok=True)

    trainer, _ = _build_preference_trainer(
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

    train_result = trainer.train()
    evaluation_metrics = trainer.evaluate() if val_dataset is not None else {}
    _save_policy_adapter(trainer.model, dpo_adapter_dir)
    tokenizer.save_pretrained(str(dpo_adapter_dir), legacy_format=False)
    dpo_artifact_manifest = write_adapter_artifact_manifest(
        dpo_adapter_dir,
        training_phase="sft_dpo",
        parent_sft_adapter_sha256=sft_artifact_manifest["adapterSHA256"],
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
        "agent": cfg["agent"],
        "trainer": "ORPOTrainer" if preference_trainer == "orpo" else "DPOTrainer",
        "dataset_dir": str(dataset_dir),
        "datasetRepository": cfg.get("datasetRepository"),
        "datasetRevision": cfg.get("datasetRevision"),
        "runResumeLineageSHA256": cfg.get("runResumeLineageSHA256"),
        "variantManifestSHA256": cfg["variantManifestSHA256"],
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
        **training_runtime_lineage,
        "adapterSHA256": dpo_artifact_manifest["adapterSHA256"],
        "finalized_variant_manifest": str(finalized_manifest_path),
        "finalized_variant_manifest_sha256": finalized_variant["variantManifestSHA256"],
        "metrics": train_result.metrics,
        "evaluation_metrics": evaluation_metrics,
    }
    _write_json_atomic(output_dir / "dpo_report.json", report)


if __name__ == "__main__":
    main()
