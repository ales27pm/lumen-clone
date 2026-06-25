from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_CONFIG_KEYS = {
    "agent",
    "base_model_name",
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
    "dataset_dir",
}
AGENTS = {"cortex", "executor", "mouth", "mimicry", "rem", "fleet"}
FINETUNE_MARKERS = {"sft", "dpo", "orpo", "lora", "merged", "adapter", "finetune", "finetuned"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-agent SFT adapters with Unsloth.")
    parser.add_argument("--config", required=True, help="Path to agent Unsloth JSON config.")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed (overrides config seed; falls back to LUMEN_TRAIN_SEED env var).")
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
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
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
    try:
        import transformers  # type: ignore

        transformers.set_seed(seed)
    except Exception:
        pass


def _hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    seed_source = "default"
    if args.seed is not None:
        seed = int(args.seed)
        seed_source = "cli"
    elif os.environ.get("LUMEN_TRAIN_SEED"):
        seed = int(os.environ["LUMEN_TRAIN_SEED"])
        seed_source = "env"
    elif "seed" in cfg:
        seed = int(cfg["seed"])
        seed_source = "config"
    else:
        seed = 42
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

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model_name"],
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

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)

    output_dir = Path(cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    repo_root = Path(__file__).resolve().parents[3]
    manifest = {
        "schema": "lumen.train_sft.manifest/1.0.0",
        "agent": cfg["agent"],
        "base_model_name": cfg["base_model_name"],
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
