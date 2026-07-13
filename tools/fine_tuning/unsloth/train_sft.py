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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from .adapter_artifact import write_adapter_artifact_manifest
except ImportError:
    from adapter_artifact import write_adapter_artifact_manifest


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
    "dataset_dir",
    "variant",
    "variantManifestSHA256",
    "seed",
}
AGENTS = {"cortex", "executor", "mouth", "mimicry", "rem", "fleet"}
FINETUNE_MARKERS = {"sft", "dpo", "orpo", "lora", "merged", "adapter", "finetune", "finetuned", "training"}


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


def _require_sha256(value: Any, *, name: str, prefix: bool = False) -> str:
    text = str(value or "")
    pattern = r"sha256:[0-9a-f]{64}" if prefix else r"[0-9a-f]{64}"
    if re.fullmatch(pattern, text) is None:
        raise RuntimeError(f"{name} must be an immutable lowercase SHA-256 digest")
    return text


def _training_environment(cfg: dict[str, Any]) -> dict[str, Any]:
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
    actual_packages = {name: _package_version(name) for name in sorted(expected_packages)}
    if actual_packages != expected_packages:
        raise RuntimeError(
            "Training package versions drifted from lock: "
            + json.dumps({"expected": expected_packages, "actual": actual_packages}, sort_keys=True)
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
    payload = {
        "schemaVersion": "lumen.adapter-training-environment/1.0.0",
        "containerImageDigest": container_digest,
        "containerImageDigestSource": digest_source,
        "runtimeImageBindingStatus": binding_status,
        "runtimeImageBindingVerified": binding_verified,
        "effectiveSeed": int(cfg["seed"]),
        "environmentLock": lock,
    }
    digest = _canonical_sha256(payload)
    if digest != cfg.get("trainingEnvironmentSHA256"):
        raise RuntimeError("trainingEnvironmentSHA256 is not bound to the recorded environment payload")
    return {**payload, "trainingEnvironmentSHA256": digest}


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
        training_environment=training_environment,
        training_phase="sft",
    )
    destination = output_dir / "finalized_variant_manifest.json"
    destination.write_text(
        json.dumps(finalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return finalized


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    seed, seed_source = _resolve_controlled_seed(cfg, cli_seed=args.seed)
    _seed_everything(seed)

    dataset_dir = Path(cfg["dataset_dir"]).resolve()
    train_path = dataset_dir / "train_sft.jsonl"
    val_path = dataset_dir / "val_sft.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Expected {train_path} and {val_path}")

    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
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

    training_environment = _training_environment(cfg)
    _verify_base_model_lineage(cfg)

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
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(cfg.get("eval_steps", 50)),
        save_steps=int(cfg.get("save_steps", 100)),
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

    resume_checkpoint: bool | str = False
    if args.resume_from_checkpoint:
        # Trainer accepts True to auto-discover the latest checkpoint in output_dir.
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        resume_checkpoint = True if checkpoints else False

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint or None)
    trainer.model.save_pretrained(str(adapter_output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_output_dir))
    adapter_artifact_manifest = write_adapter_artifact_manifest(
        adapter_output_dir,
        training_phase="sft",
    )
    finalized_variant_manifest = _finalize_variant_manifest(
        cfg,
        adapter_artifact_manifest=adapter_artifact_manifest,
        training_environment=training_environment,
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
        "config_path": str(cfg_path),
        "config_sha256": _hash_file(cfg_path),
        "dataset_dir": str(dataset_dir),
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
        "resume_from_checkpoint": bool(resume_checkpoint),
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
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "train_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
