from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    from .adapter_artifact import verify_adapter_artifact, write_adapter_artifact_manifest
    from .train_sft import _resolve_controlled_seed, _seed_everything, _training_environment
except ImportError:
    from adapter_artifact import verify_adapter_artifact, write_adapter_artifact_manifest
    from train_sft import _resolve_controlled_seed, _seed_everything, _training_environment


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


def render_messages(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages if isinstance(m, dict))


def row_to_preference(tokenizer: Any, row: dict[str, Any]) -> dict[str, str]:
    prompt_messages = row.get("prompt")
    if not isinstance(prompt_messages, list):
        prompt_messages = [{"role": "user", "content": "Follow the manifest."}]
    prompt_text = render_messages(tokenizer, prompt_messages)
    chosen = row.get("chosen", {})
    rejected = row.get("rejected", {})
    chosen_text = chosen.get("content") if isinstance(chosen, dict) else ""
    rejected_text = rejected.get("content") if isinstance(rejected, dict) else ""
    return {
        "prompt": prompt_text,
        "chosen": chosen_text or "",
        "rejected": rejected_text or "",
    }


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
        "eval_strategy": "steps" if val_dataset is not None else "no",
        "eval_steps": int(cfg.get("eval_steps", 50)),
        "save_steps": int(cfg.get("save_steps", 100)),
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


def _verified_sft_parent(
    cfg: dict[str, Any],
    *,
    adapter_dir: Path,
    finalized_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    artifact = finalized.get("artifact")
    training_environment = finalized.get("trainingEnvironment")
    if (
        finalized.get("agent") != cfg["agent"]
        or finalized.get("variant") != cfg["variant"]
        or finalized.get("seed") != cfg["seed"]
        or finalized.get("sourceVariantManifestSHA256")
        != cfg["variantManifestSHA256"]
        or not isinstance(artifact, dict)
        or artifact.get("status") != "trained"
        or artifact.get("trainingPhase") != "sft"
        or artifact.get("effectiveSeed") != cfg["seed"]
        or not isinstance(training_environment, dict)
        or training_environment.get("effectiveSeed") != cfg["seed"]
    ):
        raise RuntimeError("DPO input must be a finalized SFT artifact for the selected variant")
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
    return finalized, adapter_manifest


def _finalize_dpo_variant(
    cfg: dict[str, Any],
    *,
    adapter_artifact_manifest: dict[str, Any],
    parent_sft_adapter_sha256: str,
    reference_sft_adapter_sha256: str | None,
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
        preference_trainer=preference_trainer,
    )
    output_path.write_text(
        json.dumps(finalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return finalized


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
    sft_finalized_variant_manifest = Path(args.sft_finalized_variant_manifest).resolve()
    _, sft_artifact_manifest = _verified_sft_parent(
        cfg,
        adapter_dir=sft_adapter_dir,
        finalized_manifest_path=sft_finalized_variant_manifest,
    )

    dataset_dir = Path(cfg["dataset_dir"]).resolve()
    train_path = dataset_dir / "train_dpo.jsonl"
    val_path = dataset_dir / "val_dpo.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing DPO dataset split files under {dataset_dir}")

    try:
        from datasets import Dataset
        from unsloth import FastLanguageModel
        from peft import PeftModel
        from trl import DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for Unsloth DPO training. Install: unsloth, trl, datasets, transformers, peft, accelerate, bitsandbytes."
        ) from exc

    preference_trainer = str(cfg.get("preference_trainer", "dpo")).lower()
    if preference_trainer not in {"dpo", "orpo"}:
        raise ValueError("preference_trainer must be either 'dpo' or 'orpo'")

    _verify_base_model_lineage(cfg)
    training_environment = _training_environment(cfg)

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
    train_rows = [row_to_preference(tokenizer, row) for row in train_raw]
    val_rows = [row_to_preference(tokenizer, row) for row in val_raw]
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
    _save_policy_adapter(trainer.model, dpo_adapter_dir)
    tokenizer.save_pretrained(str(dpo_adapter_dir))
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
        preference_trainer=preference_trainer,
        training_environment=training_environment,
        output_path=finalized_manifest_path,
    )

    report = {
        "agent": cfg["agent"],
        "trainer": "ORPOTrainer" if preference_trainer == "orpo" else "DPOTrainer",
        "dataset_dir": str(dataset_dir),
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
        "seed": seed,
        "seed_source": seed_source,
        "adapterSHA256": dpo_artifact_manifest["adapterSHA256"],
        "finalized_variant_manifest": str(finalized_manifest_path),
        "finalized_variant_manifest_sha256": finalized_variant["variantManifestSHA256"],
        "metrics": train_result.metrics,
    }
    (output_dir / "dpo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
