# Hugging Face ZeroGPU Adapter Training

Lumen can publish a self-contained Gradio Space that trains the role adapters on Hugging Face ZeroGPU, using the current generated fine-tuning datasets as a fresh run snapshot.

## One-click command

```bash
export HF_TOKEN="hf_..."
bash scripts/hf_zerogpu_train_lumen_adapters_aio.sh
```

The script:

1. Creates an ignored local run directory under `.local/hf_zerogpu_runs/<run-id>`.
2. Copies the configured fine-tuning source into an immutable dataset snapshot for that run. The
   launcher uses `generated/agent_manifest/fine_tuning` when present and otherwise falls back to
   `generated/fine_tuning`.
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
export LUMEN_ZERO_GPU_DURATION_SECONDS="1200"
export LUMEN_ZERO_GPU_TRIGGER="1"                  # set 0 to only deploy
export LUMEN_ZERO_GPU_DRY_RUN="1"                  # local validation only
```

If the Hub API rejects the hardware id, select ZeroGPU manually in the Space settings and rerun with `LUMEN_ZERO_GPU_TRIGGER=1`.

The launcher always overwrites optional record, sequence-length, and epoch Space variables. When
the corresponding local environment variable is unset it writes `0`, which means “use the
generated per-adapter config.” This prevents smoke-test caps from a previous Space run from
silently limiting a full training run.

ZeroGPU duration limits are account- and GPU-size-dependent. The Hub may weight a requested wall
duration by the selected GPU size, so a nominal duration can exceed the scheduler maximum. Treat
`ZeroGPU illegal duration`, quota errors, and a disconnected long-lived event stream as terminal
for that attempt; do not automatically restart an uncapped training job.

The current uncapped Cortex dataset is substantially larger than the other adapter datasets and
may not finish inside one ZeroGPU lease. Use a dedicated GPU backend or implement durable,
Hub-backed checkpoint sharding before claiming full Cortex training. Reducing record or epoch
counts is a sampled training run, not a full run.

## Freshness policy

The default path is intentionally fresh:

- each run gets a new run id and dataset snapshot path,
- the Space deletes the run workdir before training unless `resume` is explicitly selected,
- LoRA adapters are written under `runs/<run-id>/adapters/<agent>` in the adapter repo,
- merge/release-bake remains disabled by default.

This prevents accidental continuation from old checkpoints. It does not prove dataset quality; if the generated dataset contains contaminated examples, the training run will still learn them. The Space performs the same basic null-output guard as the local Ubuntu AIO runner before launching GPU work.
