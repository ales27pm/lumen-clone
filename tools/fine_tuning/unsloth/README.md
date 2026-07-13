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

3. Train SFT with a run-scoped config that binds the explicit experiment variant, base-model
lineage, and declared runtime-image audit value. The Ubuntu launcher prepares those configs and
trains all selected agents.
```bash
export LUMEN_AIO_EXPERIMENT_VARIANT="internal_plus_public_baseline"
export LUMEN_AIO_CONTAINER_IMAGE_DIGEST="sha256:<64-lowercase-hex-digest>"
export LUMEN_AIO_RUN_ROOT="$PWD/.local/ubuntu_finetune_runs/baseline-run"
bash scripts/ubuntu_train_lumen_adapters_aio.sh
```

4. Optionally train DPO/ORPO per agent from the verified finalized SFT artifact. DPO writes a
separate `sft_dpo` adapter and records the parent SFT digest.
```bash
python tools/fine_tuning/unsloth/train_dpo.py \
  --config "$LUMEN_AIO_RUN_ROOT/configs/cortex.json" \
  --sft-adapter-dir "$LUMEN_AIO_RUN_ROOT/models/lora_qwen3_bootstrap/cortex" \
  --sft-finalized-variant-manifest "$LUMEN_AIO_RUN_ROOT/training/cortex/finalized_variant_manifest.json"
```

5. Merge adapters if needed.
```bash
python tools/fine_tuning/unsloth/merge_lora.py --config tools/fine_tuning/unsloth/configs/cortex.json
```

6. Export merged GGUF per agent.
```bash
.venv-unsloth/bin/python tools/fine_tuning/unsloth/export_gguf.py \
  --config-dir "$LUMEN_AIO_RUN_ROOT/configs" \
  --agents cortex,executor,mouth,mimicry,rem,fleet \
  --quantization q4_k_m \
  --output-root models/gguf_merged \
  --manifest-output generated/fine_tuning/merged_gguf_manifest.json
```

7. Optional: upload merged GGUFs to Hugging Face in one pass.
```bash
.venv-unsloth/bin/python tools/fine_tuning/unsloth/export_gguf.py \
  --config-dir "$LUMEN_AIO_RUN_ROOT/configs" \
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
