# Lumen Agent Fine-Tuning (Unsloth)

This directory contains per-agent training scripts and configs for **Unsloth**.

## Workflow

1. Generate datasets.
```bash
python -m lumen_manifest_crawler generate \
  --root . \
  --output generated/agent_manifest \
  --generate-agent-fine-tuning \
  --fine-tuning-output generated/fine_tuning \
  --pretty \
  --strict
```

2. Inspect each `generated/fine_tuning/<agent>/dataset_card.json`.

3. Train SFT per agent.
```bash
python tools/fine_tuning/unsloth/train_sft.py --config tools/fine_tuning/unsloth/configs/cortex.json
python tools/fine_tuning/unsloth/train_sft.py --config tools/fine_tuning/unsloth/configs/executor.json
python tools/fine_tuning/unsloth/train_sft.py --config tools/fine_tuning/unsloth/configs/mouth.json
python tools/fine_tuning/unsloth/train_sft.py --config tools/fine_tuning/unsloth/configs/mimicry.json
python tools/fine_tuning/unsloth/train_sft.py --config tools/fine_tuning/unsloth/configs/rem.json
python tools/fine_tuning/unsloth/train_sft.py --config tools/fine_tuning/unsloth/configs/fleet.json
```

4. Optionally train DPO/ORPO per agent.
```bash
python tools/fine_tuning/unsloth/train_dpo.py --config tools/fine_tuning/unsloth/configs/cortex.json
```

5. Merge adapters if needed.
```bash
python tools/fine_tuning/unsloth/merge_lora.py --config tools/fine_tuning/unsloth/configs/cortex.json
```

6. Export merged GGUF per agent.
```bash
.venv-unsloth/bin/python tools/fine_tuning/unsloth/export_gguf.py \
  --config-dir tools/fine_tuning/unsloth/configs \
  --agents cortex,executor,mouth,mimicry,rem,fleet \
  --quantization q4_k_m \
  --output-root models/gguf_merged \
  --manifest-output generated/fine_tuning/merged_gguf_manifest.json
```

7. Optional: upload merged GGUFs to Hugging Face in one pass.
```bash
.venv-unsloth/bin/python tools/fine_tuning/unsloth/export_gguf.py \
  --config-dir tools/fine_tuning/unsloth/configs \
  --agents cortex,executor,mouth,mimicry,rem,fleet \
  --quantization q4_k_m \
  --output-root models/gguf_merged \
  --hf-repo-id ales27pm/lumen-fleet-gguf \
  --manifest-output generated/fine_tuning/merged_gguf_manifest.json
```

8. Evaluate with `generated/fine_tuning/<agent>/eval.jsonl`.

9. Never train on private app exports unless explicitly sanitized.

## Deployment Notes

- The app can use LoRA adapters differently per slot if the runtime supports it.
- If using one small base model on-device, train separate LoRA adapters per slot.
- If runtime cannot hot-swap LoRA, merge strongest common adapters or train a unified fleet adapter.
