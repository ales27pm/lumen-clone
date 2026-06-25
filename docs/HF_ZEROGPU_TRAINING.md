# Hugging Face ZeroGPU Adapter Training

Lumen can publish a self-contained Gradio Space that trains the role adapters on Hugging Face ZeroGPU, using the current generated fine-tuning datasets as a fresh run snapshot.

## One-click command

```bash
export HF_TOKEN="hf_..."
bash scripts/hf_zerogpu_train_lumen_adapters_aio.sh
```

The script:

1. Creates an ignored local run directory under `.local/hf_zerogpu_runs/<run-id>`.
2. Copies `generated/agent_manifest/fine_tuning` into an immutable dataset snapshot for that run.
3. Creates or updates:
   - a dataset repo for the run snapshot,
   - a model repo for adapters,
   - a Gradio Space for training.
4. Uploads the dataset snapshot and Space bundle.
5. Adds Space variables/secrets for the dataset and adapter repos.
6. Requests ZeroGPU hardware through the Hugging Face Hub API when supported by the installed `huggingface_hub`.
7. Triggers the Space training API by default.

## Important variables

```bash
export LUMEN_ZERO_GPU_SPACE_REPO="ales27pm/lumen-zerogpu-adapter-trainer"
export LUMEN_ZERO_GPU_DATASET_REPO="ales27pm/lumen-zerogpu-training-datasets"
export LUMEN_ZERO_GPU_ADAPTER_REPO="ales27pm/lumen-qwen3-bootstrap-adapters-gguf"
export LUMEN_ZERO_GPU_AGENTS="cortex,executor,mouth,mimicry,rem,fleet"
export LUMEN_ZERO_GPU_SIZE="large"                 # or xlarge
export LUMEN_ZERO_GPU_DURATION_SECONDS="3600"
export LUMEN_ZERO_GPU_TRIGGER="1"                  # set 0 to only deploy
export LUMEN_ZERO_GPU_DRY_RUN="1"                  # local validation only
```

If the Hub API rejects the hardware id, select ZeroGPU manually in the Space settings and rerun with `LUMEN_ZERO_GPU_TRIGGER=1`.

## Freshness policy

The default path is intentionally fresh:

- each run gets a new run id and dataset snapshot path,
- the Space deletes the run workdir before training unless `resume` is explicitly selected,
- LoRA adapters are written under `runs/<run-id>/adapters/<agent>` in the adapter repo,
- merge/release-bake remains disabled by default.

This prevents accidental continuation from old checkpoints. It does not prove dataset quality; if the generated dataset contains contaminated examples, the training run will still learn them. The Space performs the same basic null-output guard as the local Ubuntu AIO runner before launching GPU work.
