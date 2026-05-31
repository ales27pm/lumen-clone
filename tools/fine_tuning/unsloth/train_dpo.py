from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Train per-agent DPO/ORPO adapters with Unsloth.")
    parser.add_argument("--config", required=True, help="Path to agent Unsloth JSON config.")
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
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


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


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).resolve())

    dataset_dir = Path(cfg["dataset_dir"]).resolve()
    train_path = dataset_dir / "train_dpo.jsonl"
    val_path = dataset_dir / "val_dpo.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing DPO dataset split files under {dataset_dir}")

    try:
        from datasets import Dataset
        from transformers import TrainingArguments
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for Unsloth DPO training. Install: unsloth, trl, datasets, transformers, peft, accelerate, bitsandbytes."
        ) from exc

    preference_trainer = str(cfg.get("preference_trainer", "dpo")).lower()
    try:
        if preference_trainer == "orpo":
            from trl import ORPOTrainer as PreferenceTrainer
        else:
            from trl import DPOTrainer as PreferenceTrainer
    except ImportError as exc:
        raise RuntimeError("TRL preference trainer import failed. Ensure `trl` is installed and supports DPO/ORPO.") from exc

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
    )

    train_raw = load_jsonl(train_path)
    val_raw = load_jsonl(val_path)
    train_rows = [row_to_preference(tokenizer, row) for row in train_raw]
    val_rows = [row_to_preference(tokenizer, row) for row in val_raw]
    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows) if val_rows else None

    output_dir = Path(cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=max(1, int(cfg["batch_size"])),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        warmup_steps=int(cfg["warmup_steps"]),
        logging_steps=int(cfg.get("logging_steps", 10)),
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=int(cfg.get("eval_steps", 50)),
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        bf16=bool(cfg.get("bf16", False)),
        fp16=bool(cfg.get("fp16", True)),
        report_to=[],
    )

    trainer = PreferenceTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        beta=float(cfg.get("dpo_beta", 0.1)),
        max_length=int(cfg["max_seq_length"]),
        max_prompt_length=int(cfg.get("max_prompt_length", cfg["max_seq_length"] // 2)),
    )

    train_result = trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    report = {
        "agent": cfg["agent"],
        "trainer": "ORPOTrainer" if preference_trainer == "orpo" else "DPOTrainer",
        "dataset_dir": str(dataset_dir),
        "train_records": len(train_rows),
        "val_records": len(val_rows),
        "output_dir": str(output_dir),
        "metrics": train_result.metrics,
    }
    (output_dir / "dpo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
