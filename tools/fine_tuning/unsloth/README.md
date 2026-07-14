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
lineage, phase-specific training-code digest, dependency lock, source Git commit, and declared
runtime-image audit value. The Ubuntu launcher prepares those configs and trains all selected
agents. Local AIO resume remains disabled until it emits the same run/checkpoint contract as the
ZeroGPU path; direct `--resume-from-checkpoint` calls without that contract fail closed.
```bash
export LUMEN_AIO_EXPERIMENT_VARIANT="internal_plus_public_baseline"
export LUMEN_AIO_CONTAINER_IMAGE_DIGEST="sha256:<64-lowercase-hex-digest>"
export LUMEN_AIO_RUN_ROOT="$PWD/.local/ubuntu_finetune_runs/baseline-run"
bash scripts/ubuntu_train_lumen_adapters_aio.sh
```

4. Optionally train DPO/ORPO per agent from the verified finalized SFT artifact. DPO writes a
separate `sft_dpo` adapter and records the parent SFT digest. The parent boundary verifies the
self-hashed finalized SFT manifest, canonical adapter bytes and base-model declaration, effective
seed, experiment/source manifest, full base-model index/shard/tokenizer contract, environment and
dependency locks, requirements digest, runtime kind, and SFT phase code digest. Preference records
remain structured message lists so pinned TRL 0.24 applies the Qwen chat template and assistant-turn
boundaries.
```bash
python tools/fine_tuning/unsloth/train_dpo.py \
  --config "$LUMEN_AIO_RUN_ROOT/configs/cortex.json" \
  --sft-adapter-dir "$LUMEN_AIO_RUN_ROOT/models/lora_qwen3_bootstrap/cortex" \
  --sft-finalized-variant-manifest "$LUMEN_AIO_RUN_ROOT/training/cortex/finalized_variant_manifest.json"
```

5. Use the legacy merge helper only when a non-GGUF merged artifact is specifically needed.
```bash
python tools/fine_tuning/unsloth/merge_lora.py --config tools/fine_tuning/unsloth/configs/cortex.json
```

6. Export merged GGUF per agent.
```bash
.venv-unsloth/bin/python tools/fine_tuning/unsloth/export_gguf.py \
  --release-bake \
  --config-dir "$LUMEN_AIO_RUN_ROOT/configs" \
  --agents cortex,executor,mouth,mimicry,rem,fleet \
  --quantization q4_k_m \
  --output-root models/gguf_merged \
  --manifest-output generated/fine_tuning/merged_gguf_manifest.json
```

7. Optional: upload merged GGUFs to Hugging Face in one pass.
```bash
.venv-unsloth/bin/python tools/fine_tuning/unsloth/export_gguf.py \
  --release-bake \
  --config-dir "$LUMEN_AIO_RUN_ROOT/configs" \
  --agents cortex,executor,mouth,mimicry,rem,fleet \
  --quantization q4_k_m \
  --output-root models/gguf_merged \
  --hf-repo-id ales27pm/lumen-fleet-gguf \
  --manifest-output generated/fine_tuning/merged_gguf_manifest.json
```

8. Evaluate with `generated/fine_tuning/<agent>/eval.jsonl`.

9. Never train on private app exports unless explicitly sanitized.

## ZeroGPU authorization and resume

Use `scripts/hf_zerogpu_train_lumen_adapters_aio.sh` with separate
`LUMEN_ZERO_GPU_ADMIN_TOKEN` and fine-grained `LUMEN_ZERO_GPU_HUB_TOKEN` credentials. The Space,
dataset repository, and adapter/model repository are private by default. Their only public
overrides are `LUMEN_ZERO_GPU_PUBLIC_SPACE=1`, `LUMEN_ZERO_GPU_PUBLIC_DATASET=1`, and
`LUMEN_ZERO_GPU_PUBLIC_ADAPTERS=1`, respectively. Each changes only its named repository, and a
public Space still requires the admin header. Dataset uploads are pinned by their returned Hub
commit SHA.

`LUMEN_ZERO_GPU_RESUME=1` is accepted only when the existing self-hashed run manifest, original
local dataset snapshot, prepared configs, checkpoint-lineage records, and at least one checkpoint
all match the requested lineage. Fresh runs reject existing workspaces unless
`LUMEN_ZERO_GPU_DESTRUCTIVE_RESET=1` is explicit. Resume and destructive reset are mutually
exclusive.

The built Space deploys a stable package and invokes:

```bash
python -m lumen_training.train_sft --help
python -m lumen_training.train_dpo --help
```

`lumen_training.train_dpo` selects DPO or ORPO from the config and imports shared SFT helpers through
the same package. The deployed process does not rely on the repository source tree being on
`PYTHONPATH`.

Before model loading, the runtime verifies the full deployed code closure: `app.py`, requirements,
the entire `lumen_training` package, and all covered Python and runtime-loaded JSON/text/config
resources in `lumen_manifest_crawler`. The manifest's closure policy is checked bidirectionally,
so both declared-file drift and unexpected behavior files fail. Explicit volatile run files do not
change the controlled digest. Separate SFT, DPO, and ORPO digests are retained under one bundle
digest, alongside the direct-dependency lock and requirements hash.

Runtime-source lineage keeps the expected uploaded Space revision, authenticated repository-head
observation, platform runtime observation, binding status, and method as separate fields. A
repository-head match does not prove what the container executes; when no trusted platform runtime
metadata exists the source binding remains operator-declared and unverified. Parent SFT audit
evidence, DPO frozen-reference evidence, and the new preference-training runtime are likewise kept
separate. These fields propagate into reports and finalized manifests, but promotion is still
unsupported until a trusted runtime-image attestation exists.

## Deployment Notes

- The app can use LoRA adapters differently per slot if the runtime supports it.
- If using one small base model on-device, train separate LoRA adapters per slot.
- If runtime cannot hot-swap LoRA, merge strongest common adapters or train a unified fleet adapter.
