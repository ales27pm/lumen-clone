import argparse
import json
from pathlib import Path

AGENTS = {"cortex", "executor", "mouth", "mimicry", "rem", "fleet"}
FINETUNE_MARKERS = {"sft", "dpo", "orpo", "lora", "merged", "adapter", "finetune", "finetuned"}


def _tokenize_path(value: str) -> set[str]:
    return set("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def validate_artifact_path_config(cfg: dict) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate merge inputs and scaffold LoRA merge run metadata")
    parser.add_argument("--config", required=True, help="Path to per-agent Unsloth config JSON")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    required_keys = ("agent", "base_model_name", "output_dir")
    missing = [key for key in required_keys if not cfg.get(key)]
    if missing:
        raise ValueError(f"Missing required config fields for merge: {', '.join(missing)}")
    validate_artifact_path_config(cfg)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "agent": cfg["agent"],
        "base_model_name": cfg["base_model_name"],
        "output_dir": str(output_dir),
        "status": "ready_for_merge",
        "next_step": "Run PEFT merge_and_unload() with adapter checkpoint after training",
    }
    (output_dir / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"prepared merge scaffolding for {cfg['agent']} -> {output_dir}")


if __name__ == "__main__":
    main()
