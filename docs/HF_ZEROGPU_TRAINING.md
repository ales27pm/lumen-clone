# Hugging Face ZeroGPU Adapter Training

Lumen can publish a self-contained Gradio Space that trains the role adapters on Hugging Face ZeroGPU, using the current generated fine-tuning datasets as a fresh run snapshot.

## One-click command

```bash
export LUMEN_ZERO_GPU_HUB_TOKEN="hf_fine_grained_repository_token"
export LUMEN_ZERO_GPU_ADMIN_TOKEN="<random-secret-of-at-least-32-characters>"
export LUMEN_ZERO_GPU_EXPERIMENT_VARIANT="internal_plus_public_baseline"
export LUMEN_ZERO_GPU_CONTAINER_IMAGE_DIGEST="sha256:<64-lowercase-hex-digest>"
bash scripts/hf_zerogpu_train_lumen_adapters_aio.sh
```

`LUMEN_ZERO_GPU_HUB_TOKEN` and `LUMEN_ZERO_GPU_ADMIN_TOKEN` are separate security
boundaries. The Hub token must be a fine-grained token limited to the selected Space, dataset,
and adapter repositories. The admin token authorizes the application endpoint and is sent only in
`X-Lumen-Admin-Token`; it is never written to defaults, run manifests, summaries, or errors.

There is deliberately no default experiment variant or container digest. Choose the baseline,
optimized, or internal-only corpus explicitly for every production run. The digest is an
operator-declared audit value for the intended image; Gradio ZeroGPU does not expose trusted
runtime-image provenance that Lumen can compare with it. Runs therefore record
`manual_validation_required` and cannot use the declaration alone as promotion evidence. The
selected variant is included in the run ID. Automated promotion is intentionally unsupported
until Lumen has an independently verifiable runtime-image attestation; manifest JSON cannot
self-assert a trusted binding.

The script:

1. Creates an ignored local run directory under `.local/hf_zerogpu_runs/<run-id>`.
2. Copies the configured fine-tuning source into an immutable dataset snapshot for that run. The
   launcher uses `generated/agent_manifest/fine_tuning` when present and otherwise falls back to
   `generated/fine_tuning`.
3. Uploads that snapshot first and captures the full Hugging Face dataset commit SHA. The Space is
   configured with that exact commit; `main` is rejected for real training.
4. Creates or updates three private repositories by default:
   - a private dataset repo for the run snapshot,
   - a private model repo for adapters,
   - a private Gradio Space for training.
5. Uploads a self-verifying Space bundle, captures the Space commit SHA, and configures the dataset
   revision, runtime source revision, admin secret, and repository-scoped Hub token.
6. Requests ZeroGPU hardware through the Hugging Face Hub API when supported by the installed `huggingface_hub`.
7. Triggers the authenticated Space training API by default. Only one training operation may hold
   the process lock at a time.

## Important variables

```bash
export LUMEN_ZERO_GPU_SPACE_REPO="ales27pm/lumen-zerogpu-adapter-trainer"
export LUMEN_ZERO_GPU_DATASET_REPO="ales27pm/lumen-zerogpu-training-datasets"
export LUMEN_ZERO_GPU_ADAPTER_REPO="ales27pm/lumen-qwen3-bootstrap-adapters-gguf"
export LUMEN_ZERO_GPU_AGENTS="cortex,executor,mouth,mimicry,rem,fleet"
export LUMEN_ZERO_GPU_EXPERIMENT_VARIANT="internal_plus_public_baseline"
export LUMEN_ZERO_GPU_CONTAINER_IMAGE_DIGEST="sha256:<64-lowercase-hex-digest>"
export LUMEN_ZERO_GPU_SIZE="large"                 # or xlarge
export LUMEN_ZERO_GPU_DURATION_SECONDS="1200"
export LUMEN_ZERO_GPU_TRIGGER="1"                  # set 0 to only deploy
export LUMEN_ZERO_GPU_DRY_RUN="1"                  # local validation only
export LUMEN_ZERO_GPU_RESUME="0"                   # set 1 only for an unchanged existing run
export LUMEN_ZERO_GPU_DESTRUCTIVE_RESET="0"        # explicit replacement of an existing fresh-run workspace
export LUMEN_ZERO_GPU_PUBLIC_SPACE="0"             # private by default; public requires explicit 1
export LUMEN_ZERO_GPU_PUBLIC_DATASET="0"           # private by default; public requires explicit 1
export LUMEN_ZERO_GPU_PUBLIC_ADAPTERS="0"          # private by default; public requires explicit 1
```

Omitting all three public variables keeps the Space, dataset, and adapter/model repositories
private. Set only the repository-specific override whose visibility is intentionally public. The
standalone builder exposes the equivalent `--public-space`, `--public-dataset`, and
`--public-adapters` flags; its deprecated hidden `--private-*` aliases do not change the
private-by-default behavior. Repository visibility is printed before upload, without credentials.

Public Space deployment never disables application authorization. A missing or invalid admin header is
rejected before GPU allocation, filesystem changes, snapshot access, Hub-token access, or Hub API
construction. Conflicting requests return a stable `training_already_active` error. External
failures expose only a safe code, correlation ID, and concise message; the traceback remains in
server logs under that correlation ID.

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

## Fresh runs and resume

The default path is intentionally fresh and fail-closed:

- each run gets a new run ID, immutable dataset commit, and local dataset snapshot path;
- an existing run directory is rejected unless `LUMEN_ZERO_GPU_DESTRUCTIVE_RESET=1` is explicitly
  selected; fresh training never silently deletes a resumable run;
- LoRA adapters are written under `runs/<run-id>/adapters/<agent>` in the adapter repo,
- each adapter upload contains a canonical per-file digest manifest and a finalized experiment
  manifest bound to that adapter digest,
- merge/release-bake remains disabled by default.

Resume reuses the original local snapshot and reads the existing self-hashed run manifest before
any mutation. It rejects dataset-commit, dataset-file, lane, variant, seed, base-model,
environment, training-code, dependency, runtime-source, assistant-loss, and path drift. Every
checkpoint is associated with the run through a self-hashed checkpoint-lineage record and a
canonical recursive file digest. Resume requires at least one recorded checkpoint and never
redownloads, recopies, or replaces the snapshot or configs.

The trainer enforces the same checkpoint record when invoked directly with
`--resume-from-checkpoint`; Trainer auto-discovery alone is not accepted.

## Training code and dependency evidence

The builder assembles the executable trainers as a real package:

```text
lumen_training/
├── __init__.py
├── train_sft.py
├── train_dpo.py
├── adapter_artifact.py
└── training_lineage.py
```

The deployed entrypoints are `python -m lumen_training.train_sft` and
`python -m lumen_training.train_dpo`; the latter selects the pinned DPO or ORPO implementation from
the prepared config. This layout keeps relative imports identical in clean Space subprocesses and
does not depend on a repository checkout being present on `PYTHONPATH`.

The training-code bundle covers `app.py`, `requirements.txt`, the entire deployed
`lumen_training` package, and every behavior-affecting `.py`, `.json`, `.jsonl`, `.txt`, config,
shell, and YAML/TOML resource under the deployed `lumen_manifest_crawler` package. Its explicit
`closurePolicy` records the covered roots, included extensions, volatile directory names, and the
run-specific exclusions (`lumen_zero_gpu_defaults.json` and
`lumen_zero_gpu_run_manifest.json`). Verification is bidirectional: a
missing or changed declared file fails, and so does any unexpected behavior-affecting file under a
covered tree. `.git`, logs, checkpoints, outputs, uploads, `__pycache__`, and bytecode are excluded
as volatile runtime state; credentials remain environment secrets and are not deployed files.

The bundle records `trainingCodeBundleSHA256` plus separate
`trainingCodeSHA256ByPhase.{sft,dpo,orpo}` identities. The Space reconstructs the deployed tree and
verifies the requested phase before model loading. This means a package initializer, transitively
imported crawler helper, evaluation registry/fingerprint JSON resource, trainer, Space app, or
requirements change changes the controlled identity. Requirements and installed direct
dependencies, the Unsloth VCS commit, Python/CUDA versions, and the optional llama.cpp converter
revision must also match the generated dependency lock.

Runtime-source evidence deliberately separates:

- `expectedRuntimeSourceRevision`: the immutable Space commit configured by the builder;
- `observedRepositoryRevision`: an authenticated observation of the Space repository head;
- `observedRuntimeRevision`: a platform-provided executing revision, when such trusted metadata is
  actually available;
- `runtimeSourceBindingStatus` and `runtimeSourceBindingMethod`: the confidence and source of the
  observation.

A well-formed environment variable or a repository-head match is not proof of which Space commit
the running container executes. When Hugging Face supplies no independently attested runtime
revision, the binding remains `operator_declared_unverified`; repository head is supplemental audit
evidence only. `trainingCodeSHA256` remains the controlled comparison identity, so distinct
run-specific Space commits can be compared only when their verified training-code and dependency
digests match. Expected and observed source evidence is retained in run, checkpoint, training, and
finalized lineage rather than being collapsed into one asserted revision.

## Preference-training parent boundary

DPO and ORPO accept only a self-hash-valid, trained, finalized SFT variant whose canonical adapter
directory verifies against its declared digest. Parent validation matches the active preference
config across agent, experiment variant, source variant manifest, effective seed, complete base
model/index/shard/tokenizer contract, environment and dependency locks, requirements digest,
runtime source kind, and the SFT phase code digest. `adapter_config.json` must name the same base
model, and any missing, extra, or modified adapter file fails verification.

Preference reports keep three concepts distinct: `parentSFTLineage` records the immutable SFT
parent and its original runtime audit evidence; DPO additionally records the same verified frozen
policy as `referenceSFTLineage`; and `preferenceTrainingRuntime` records the DPO/ORPO execution.
ORPO has no separate reference policy but still enforces the complete parent contract.

These controls do not make the operator-declared container digest trustworthy. Promotion remains
unsupported until the platform supplies an independently verifiable runtime-image attestation.
