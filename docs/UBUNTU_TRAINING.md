# Ubuntu Lumen Training

This is the canonical runbook for training Lumen's six role adapters on an
Ubuntu machine with an NVIDIA GPU. The host launcher builds a pinned CUDA
container and runs the repository's lineage checks inside it, so the host does
not need a hand-built Python or CUDA environment.

The one-click command is:

```bash
bash scripts/ubuntu_train_lumen_full_pipeline.sh
```

By default it trains the `internal_plus_public_optimized` variant in fail-fast
risk order: Fleet, Executor, Mouth, REM, Mimicry, then Cortex. Each role runs
SFT and then preference
training (DPO under the current controlled manifests). It writes LoRA adapters
and adapter-only GGUF files, then runs the frozen per-role evaluation suites.
It does not upload artifacts, publish a repository, merge adapters into full
base-model GGUFs, or promote an artifact into the iOS runtime.

## Host Prerequisites

The launcher checks prerequisites and fails before starting a training run. It
does not install host drivers, Docker, or privileged system packages.

Install and verify:

1. A supported NVIDIA driver. `nvidia-smi` must work on the host.
2. [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/).
3. The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
   configured for Docker.
4. Git, Python 3, and enough free local storage for the repository, container layers,
   Hugging Face cache, checkpoints, and run artifacts.
5. Outbound HTTPS access to GitHub, Hugging Face, the PyTorch wheel index, and
   Python package indexes while the image and model cache are populated.

Verify GPU access from both the host and a container:

```bash
nvidia-smi
docker info
docker run --rm --gpus all \
  nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97 \
  nvidia-smi
```

Run Docker as a regular non-root user permitted to access the Docker daemon; do
not invoke the launcher with `sudo`. The image build creates a passwd/group
entry and writable home for that user's numeric UID/GID, and the launcher
verifies the mapping before checking GPU access. This is required by Torch and
other Python packages that resolve the current account through the system user
database. The controlled base model is public, so training disables implicit
Hub tokens and mounts only
credential-free cache subdirectories. A token is needed only for `--upload`;
supply it with `--token-file`, or keep an owner-only token at `$HF_HOME/token`.
The token is mounted only into the short-lived upload container and is never
forwarded through Docker metadata.

## Controlled Container Stack

The training image is based on
`nvidia/cuda:12.8.1-devel-ubuntu22.04` at immutable OCI manifest
`sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97`
and uses Python 3.10. The controlled
training lock requires CUDA 12.8 and these direct dependency versions:

| Package | Version |
|---|---:|
| PyTorch | 2.9.1 |
| torchvision | 0.24.1 |
| torchaudio | 2.9.1 |
| datasets | 4.3.0 |
| transformers | 4.57.6 |
| TRL | 0.24.0 |
| PEFT | 0.19.1 |
| accelerate | 1.14.0 |
| bitsandbytes | 0.49.2 |
| sentencepiece | 0.2.2 |
| protobuf | 7.35.1 |
| huggingface_hub | 0.36.2 |
| hf_transfer | 0.1.9 |
| trackio | 0.20.2 |
| gradio | 6.17.3 |
| spaces | 0.51.0 |
| unsloth_zoo | 2026.7.2 |

Unsloth is pinned to Git commit
`935474c20aabc2aadb1da17338959c7c6f9bdafe`. Adapter GGUF conversion uses
llama.cpp revision `34558825a27f4d74dcfd7a91bfde4464baa2a30a`.
The launcher records the locally built image ID and the resolved training
environment in the run lineage. That record is audit evidence; it is not a
trusted production runtime-image attestation.

The image is specific to the invoking host UID/GID. Reusing it with
`--no-build` under another account fails the runtime-identity preflight instead
of allowing a later `getpwuid` crash. Runtime compiler caches live below the
owned container home and remain isolated from the repository and output tree.

The host launcher also requires an exact clean Git worktree before image build,
fresh training, resume, or upload. Staged, unstaged, and untracked files, dirty
or mismatched submodules, and ignored files inside the Ubuntu execution closure
fail closed. The recorded source contract includes the base commit, canonical
working-tree digest, `dirtyState=false`, the complete Ubuntu orchestration-file
manifest, and `ubuntuOrchestrationCodeSHA256`. That closure covers both
launchers, the container recipe and Docker ignore policy, trainers, evaluator,
GGUF helper, uploader, imported crawler package, and ZeroGPU runtime sources.

Docker bakes the verified closure and frozen generated inputs under
`/opt/lumen/source`; the host checkout is not mounted into the GPU or upload
container. The built image reconstructs the orchestration digest before it is
accepted, and `--no-build` still requires its source record to equal the current
clean checkout. Run manifests, prepared configs, resume verification,
evaluation evidence, summaries, and upload receipts retain that source record.

The CUDA base is OCI-digest-pinned. Python wheel artifacts are version-pinned
but are not installed from a hash-locked wheelhouse. For that reason,
runtime-image promotion remains prohibited. Credential-scoped upload requires
an image built by the same launcher invocation; the token is not placed in the
Python process environment and is read only after local evidence and uploader
imports have passed. A future release-grade publisher should additionally use
a minimal image and hashed package artifacts.

## Capacity Planning

These are engineering estimates, not enforced minimums. Actual use varies with
the GPU, Docker storage driver, model cache state, checkpoint policy, selected
roles, and variant count.

| Resource | Practical estimate |
|---|---|
| GPU | Modern NVIDIA CUDA GPU |
| VRAM | The controlled low-memory path has passed Cortex training on an 8 GB RTX 2070; 12 GB or more remains recommended until a fresh six-role run completes |
| System RAM | 32 GB; 64 GB for release-bake experiments |
| Free disk | 50 GB for adapter-only work; 80--100 GB when retaining caches, checkpoints, or merged GGUF experiments |

The shared base is `Qwen/Qwen3-1.7B`, pinned to revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Its two controlled weight shards
total about 3.78 GiB. Docker image layers, package wheels, optimizer state, and
per-variant outputs require substantially more space than the model weights.

## Run The Pipeline

From a clean clone or the repository checkout whose commit should be recorded:

```bash
cd /absolute/path/to/lumen-clone
bash scripts/ubuntu_train_lumen_full_pipeline.sh
```

Keep that checkout immutable until the launcher has completed its independent
postcondition. A dedicated clean worktree is recommended for long runs; host
source drift deliberately blocks later resume, postcondition, and upload trust
boundaries even though the GPU container executes only its baked source copy.

The launcher builds the pinned image, checks Docker GPU access, derives the
actual local image ID, verifies the image-baked source closure, and mounts only
the exact per-run root, a read-only cross-container lock identity, and the three
credential-free persistent Hugging Face cache subdirectories. It first runs one
exact-tokenizer preflight for every requested role before loading model weights
or creating an optimizer. That gate checks SFT sequence margins, DPO prompt and
sequence margins, and Fleet's exact assistant-target loss-share limits. The
tokenizer identity covers the pinned `config.json`, `merges.txt`,
`tokenizer.json`, `tokenizer_config.json`, and `vocab.json` bytes rather than
only `tokenizer.json`; the trainers must use that verified snapshot and bind any
derived tokenizer files saved with an adapter. The independent `--prepare-only`
postcondition reads the run from an exact read-only mount, rejects nested
mounts or path substitution, and checks resource headroom before repeating the
token-ID evidence. On an execution that continues past the prepare-only
early-exit boundary, a second gate uses `train_sft --runtime-binding-smoke` to
load the private model and tokenizer through the real pinned Unsloth path
before converter setup, PEFT, or trainer construction. Prepared configs are
grouped only when their complete runtime-load contracts match; the current six
roles therefore require one load, while any future sequence-length or loader
difference requires its own load. The gate atomically persists and re-verifies
the self-hashed `training/runtime_binding_smoke.json` report on resume. Each
load must prove the explicit FP16/BF16 choice, requested sequence length, exact
196-projection CUDA NF4 materialization, the complete Qwen parameter inventory
on CUDA, and a finite four-token CUDA forward with the expected logits shape
and dtype. The offline report verifier independently reconstructs those facts
from the pinned `config.json`; a merely self-consistent runtime report is not
accepted. Each
selected role then completes this order before the next role starts:

```text
prepare and verify isolated dataset/config snapshot
SFT train -> verify finalized SFT manifest and adapter bytes
DPO train -> verify parent SFT lineage and finalized DPO manifest
frozen deterministic inference -> score and enforce the quality gate
adapter-only GGUF conversion -> verify output digest
write run summary
```

An intentionally prepared SFT-only diagnostic run stops after the verified SFT phase. Its
summary has `status=sft_only_diagnostic_complete` and `trainingScope=sft_only`; evaluation and
GGUF are not applicable. Summary reconstruction and upload for that state read only the SFT
artifact and report and must not probe preference, evaluation, or GGUF paths.

The default order minimizes wasted GPU time if an unpiloted role fails. It does
not change any role's dataset, seed, hyperparameters, evaluation, or output.
An explicit `--agents` list is honored in the exact order supplied.

The current preference trainer is DPO. ORPO is an alternative controlled
trainer, not an additional phase; selecting it requires regenerated controlled
variant manifests.

Useful selections:

```bash
# Default result, written below an explicit absolute host path.
bash scripts/ubuntu_train_lumen_full_pipeline.sh \
  --output-dir /srv/lumen-training

# Train only selected roles.
bash scripts/ubuntu_train_lumen_full_pipeline.sh \
  --agents cortex,executor,mouth

# Run baseline and optimized as a fail-fast sequential batch.
bash scripts/ubuntu_train_lumen_full_pipeline.sh \
  --variant baseline-and-optimized \
  --output-dir /srv/lumen-training/baseline-and-optimized

# Add the internal-only ablation to baseline and optimized.
bash scripts/ubuntu_train_lumen_full_pipeline.sh \
  --variant all \
  --output-dir /srv/lumen-training/all-variants

# Keep adapter directories but skip adapter-GGUF conversion.
# Full evaluation still runs and records status=complete_without_gguf.
bash scripts/ubuntu_train_lumen_full_pipeline.sh --no-gguf

# Run only a bounded evaluation smoke pass while checking the host setup.
bash scripts/ubuntu_train_lumen_full_pipeline.sh --eval-smoke 2

# Skip evaluation only for an intentional training-only diagnostic run.
bash scripts/ubuntu_train_lumen_full_pipeline.sh --no-evaluate
```

Supported `--variant` values are:

| Value | Dataset variants | Jobs for all six roles |
|---|---|---:|
| `optimized` or `internal_plus_public_optimized` | optimized | 12 |
| `baseline` or `internal_plus_public_baseline` | baseline | 12 |
| `internal` or `internal_only` | internal-only ablation | 12 |
| `baseline-and-optimized` | baseline and optimized sequential batch | 24 |
| `all` | internal-only, baseline, and optimized | 36 |

One job means one role and one training phase. The counts therefore include
six roles times SFT plus DPO for each selected variant.
The multi-variant selectors are fail-fast batch conveniences. They do not emit
a baseline-versus-optimized promotion decision, and a quality-gate failure in
an earlier variant stops the batch. Use the separately validated comparison
workflow before making a promotion claim.

Additional launcher controls:

| Option | Meaning |
|---|---|
| `--output-dir <dir>` | Select the host parent directory for per-variant run directories. |
| `--hf-cache <dir>` | Select the persistent host Hugging Face cache. |
| `--run-id <id>` | Give the run a stable operator-selected identifier. A variant suffix is added. |
| `--image-tag <tag>` | Override the local Docker image tag used for build/run. |
| `--no-build` | Reuse an existing image. It cannot be combined with credential-scoped upload. |
| `--no-pull` | Build without re-fetching the immutable pinned CUDA base digest. |
| `--prepare-only` | Run controlled input, environment, and config preparation without model loading or training. The real Unsloth binding smoke runs only on executions that continue beyond this early exit. |
| `--overwrite` | Destructively replace the selected run directory after path-safety checks. |
| `--resume` | Reuse an existing run, skip phases whose complete artifacts re-verify, resume incomplete phases from a verified checkpoint when available, and otherwise restart the phase. |
| `--no-evaluate` | Skip frozen inference/scoring. Full evaluation is enabled by default. |
| `--eval-smoke <n>` | Evaluate a deterministic semantic cohort of `n` cases per role; `n` must be smaller than every selected role's frozen suite, and this is smoke evidence, not a quality pass. |
| `--token-file <file>` | Mount an owner-only, mode-600 token only into the upload container. |
| `--upload` | Upload a full quality-passed run after training. Upload is off by default and the destination is private by default. |
| `--allow-diagnostic-upload` | With `--upload`, explicitly permit smoke or unevaluated artifacts under `diagnostic-runs/`, or a verified SFT-only diagnostic under `diagnostic-sft-runs/`. |
| `--public` | With `--upload`, explicitly request public visibility. Public publication is never the default. |

Ubuntu resume accepts only cryptographically bound pipeline state. With
`--resume`, the image ID, source revision, environment, agents, variant,
prepared-config digest, exact run-scoped paths, self-hashed run manifest, and
the immutable execution plan must still match. The execution plan binds
`evaluationScope`, `evaluationMaxExamples`, and `ggufRequested` before training;
its digest is retained by each config's variant attestation, reconstructed from
the hash-bound configs, and repeated in evaluation evidence, summaries, and
upload receipts. Resume or rehashed evidence therefore cannot relabel a
different cohort or missing conversion as an operator skip. A fully verified
SFT or DPO phase is kept. An incomplete phase resumes only from the newest
complete checkpoint whose dataset, config, code, parent adapter, optimizer
progress, and checkpoint bytes all re-verify; an incomplete newest checkpoint
or any lineage drift fails closed. If there is no valid checkpoint, only that
agent's owned incomplete phase is restarted. Frozen inference has its own
private, self-hashed `evaluation_checkpoint.json`: every completed selected
case and its raw attempts are fsync-committed before the next case. Creation of
every missing private evaluation-directory component fsyncs the new inode and
its parent entry before the first journal write, so a later checkpoint fsync
does not leave the directory name outside the power-loss durability boundary.
The journal
binds the adapter, config, evaluator code, frozen evaluation file, behavior/tool
manifest, execution plan, generation settings, and selected-record order.
`--resume` preserves an incomplete journal only when it is the directory's sole
entry. A complete journal may coexist with only the known private subset of the
three canonical final files, covering interruption between their atomic
publication boundaries; those files are deterministically reconstructed and
overwritten. Only after the existing journal verifies, the evaluator removes
owned mode-0600 regular orphan temps whose basenames exactly match its atomic
writer for the journal or one of the three final files. A symlink, unsafe
lookalike, temp without a valid journal, tamper, duplicate, non-prefix, unknown
entry, incomplete journal mixed with final files, binding drift, or operational
verifier error fails closed while preserving the directory. The launcher also
independently reconstructs a terminal candidate/report/run-manifest trio: a
verified quality failure is retained and stops the run instead of being
misclassified as disposable partial state. Automatic cleanup is never inferred
from an untyped verifier exit.
When the exact prefix is complete, canonical final evidence is reconstructed
without another model load. The final evidence directory contains only the
candidate/report/run-manifest trio; a leftover journal prevents final
verification. The outer launcher also retains
the exact Docker container across terminal or launcher disconnects. A
subsequent explicit `--resume` authenticates its immutable launch environment
and either reattaches to the running container, starts a never-started fresh
container, or verifies the postcondition of an exited container before any
replacement. A never-started overwrite container still requires `--overwrite`.
For a multi-variant batch, existing variant directories resume and a variant
that had not started yet is prepared fresh. Direct unbound checkpoint selection
remains disabled.
`--resume` and `--overwrite` are mutually exclusive. Use `--overwrite` only
when the existing run is intentionally disposable; the host wrapper deletes
only the constructed `<output-dir>/<run-id>-<variant>` child after validating
the run ID, agents, and variant selector.

Use the original run ID and image when resuming. `--no-build` avoids replacing
the selected local image before its recorded image ID is checked:

```bash
bash scripts/ubuntu_train_lumen_full_pipeline.sh \
  --resume \
  --no-build \
  --run-id 20260714T120000Z
```

`--no-build` is an optimization for an image already built from the same clean
checkout and the same host UID/GID. A matching tag alone is rejected unless the
image-baked source record, working-tree digest, and orchestration digest match
the current checkout; the identity and environment checks must also pass.

## Outputs And Retention

The output parent defaults to `.local/ubuntu_finetune_runs`. The run ID defaults
to a UTC timestamp plus a random collision-resistant suffix. Each exact variant
therefore uses:

```text
<output-dir>/<run-id>-<exact-variant>/
```

For a single variant, that directory contains:

```text
aio_run_manifest.json
aio_summary.json
training_environment.json
global_tokenizer_preflight.json
upload_receipts.json                 # only after an explicit successful upload
configs/<agent>.json
configs/<agent>.final.json
generated/fine_tuning/...
generated/agent_manifest/AgentBehaviorManifest.json
logs/...
models/lora_qwen3_bootstrap/<agent>/...
models/lora_qwen3_dpo/<agent>/...
models/lora_qwen3_gguf/lumen-<agent>-lora.gguf
models/lora_qwen3_gguf_receipts/lumen-<agent>-lora.conversion.json
training/<agent>/finalized_variant_manifest.json
training/<agent>/training_report.json
training/<agent>/sft_token_length_preflight.json
training/<agent>/dpo/finalized_variant_manifest.json
training/<agent>/dpo/dpo_report.json
training/<agent>/dpo/token_length_preflight.json
evaluation/<agent>/candidate_outputs.jsonl
evaluation/<agent>/evaluation_report.json
evaluation/<agent>/evaluation_run_manifest.json
# evaluation/<agent>/evaluation_checkpoint.json exists only while interrupted
```

The host wrapper gives each member of `baseline-and-optimized` or `all` a separate variant
subdirectory so adapters, reports, configs, and lineage records cannot overwrite
one another. Treat finalized manifests and summary JSON as part of the model
artifact: they bind the dataset, base-model revision, training code,
dependencies, source commit, environment observation, and canonical adapter
bytes.

The default export is adapter-first. `lora_qwen3_dpo` is the final preference-
trained adapter, while `lora_qwen3_bootstrap` is its verified SFT parent.
Adapter-only GGUF files do not contain the full Qwen base. Keep the pinned base
model and its lineage available when consuming them.

Do not commit run directories, model caches, checkpoints, adapters, or GGUF
files to Git.

## Upload Safety

Upload is a separate, explicit action after local training. Create a regular
token file outside every mounted repository, output, and cache tree; keep it
owned by the current user, limit its permissions, and pass only its path:

```bash
chmod 600 /secure/path/lumen-hf-token
bash scripts/ubuntu_train_lumen_full_pipeline.sh \
  --upload \
  --token-file /secure/path/lumen-hf-token
```

The verified image copy bound into the immutable run manifest is used.
Immediately before remote mutation, the upload helper re-verifies adapter bytes, finalized
manifests, evaluation file hashes, GGUF digests, run/summary evidence, and the
exact allowlist. The upload process runs with isolated Python, no host source
mount or repository `PYTHONPATH`, a read-only root filesystem, read-only run
artifacts, and a separate receipt-only writable mount. The HF token is its only
credential mount. It durably records a local intent and parent-bound attempt,
places a self-hashed intent marker in one atomic commit, and records the
immutable commit OID before writing the final receipt. After a crash between
remote commit and local receipt, it adopts only the unique journaled commit
whose parent, title, exact prefix path set, and bytes all re-verify. An unrelated
later repository head is allowed only when that head still contains the exact
immutable run prefix; an unjournaled existing prefix is always rejected. The
default is private. Public
visibility requires both `--upload` and `--public`:

```bash
bash scripts/ubuntu_train_lumen_full_pipeline.sh \
  --upload \
  --public \
  --token-file /secure/path/lumen-hf-token
```

Use a fine-grained token with only the required repository permission. Do not
put tokens in commands, checked-in environment files, logs, configs, or run
manifests.

Ordinary `--upload` requires `evaluationStatus=quality_gate_passed`,
`evaluationScope=full`, and `promotionEligible=true`. A full quality pass is
uploadable whether `ggufStatus` is `verified` or `skipped_by_operator`; the
latter is the intentional `--no-gguf` case. Smoke and unevaluated artifacts
require a second, explicit acknowledgement and never use the qualified run
namespace:

In this upload record, `promotionEligible` means eligible for ordinary artifact
publication after the frozen local gate. It is not a shipped-runtime or device
promotion claim. Fresh TestFlight/device evidence must still show
`runtimePath=sharedAdapter`, the expected adapter slot, and
`adapterApplied=true` without fallback before release promotion.

```bash
bash scripts/ubuntu_train_lumen_full_pipeline.sh \
  --eval-smoke 2 \
  --upload \
  --allow-diagnostic-upload \
  --token-file /secure/path/lumen-hf-token
```

Qualified uploads use `runs/<run-id>/`; preference-trained diagnostic uploads use
`diagnostic-runs/<run-id>/`; SFT-only diagnostic uploads use
`diagnostic-sft-runs/<run-id>/`. The upload receipt binds the namespace, exact
remote prefix, qualification, promotion eligibility, evaluation status/scope,
exact execution-plan digest, GGUF status, whether GGUF was included, and whether
the diagnostic override was applied. It also binds the exact pre-training
runtime-binding smoke report and included SFT/DPO training-report file digests,
plus their compact runtime-model, runtime-tokenizer, PEFT-base,
adapter-tokenizer, and private-snapshot identities. SFT-only publication includes
only the SFT report and adapter. `--no-evaluate` and `--eval-smoke` are mutually
exclusive regardless of argument order.

## Evidence And Promotion Boundaries

A successful default Ubuntu run proves only the gates recorded by that run:
controlled dataset/config preparation, local container environment checks,
SFT/DPO training completion, adapter-lineage verification, the frozen
evaluation quality gate, and any requested adapter conversion or upload. A
bounded `--eval-smoke` run records smoke evidence and cannot become a full
quality pass. Its top-level summary status is `smoke_complete`; a run without
evaluation is `training_complete_without_full_evaluation`. A complete all-agent
frozen quality pass records `status=complete` when all requested GGUFs verify,
or `status=complete_without_gguf` when conversion was disabled before execution.
An intentionally SFT-only diagnostic records `status=sft_only_diagnostic_complete`,
`trainingScope=sft_only`, `qualification=diagnostic_only`, and
`promotionEligible=false`; it is never relabeled as a preference-trained,
evaluated, or GGUF-complete run.
The summary records evaluation and conversion independently through
`evaluationStatus`, `evaluationScope`, and `ggufStatus`; partial or mixed agent
evidence is rejected.

The execution-plan and summary schemas are fail-closed. Runs prepared with the
older schema, which did not bind these independent state dimensions, are
intentionally non-resumable and non-uploadable. Start a fresh run rather than
rewriting historical evidence.

It does not by itself prove:

- frozen-suite model quality when `--no-evaluate` or `--eval-smoke` was used;
- compatibility with the shipped iOS adapter loader;
- a trusted production runtime-image binding;
- TestFlight or physical-device behavior;
- live tool calls, RAG, memory, voice, or AppIntent flows;
- signed release packaging or runtime promotion.

GGUF qualification proves converter and reader provenance,
adapter/base/tokenizer lineage, the final file digest, and structural
readability. It does not claim an independent tensor-by-tensor equivalence
proof between the PEFT adapter and the GGUF representation; the summary records
that residual explicitly.

Runtime-image promotion remains intentionally unsupported. Do not change the
iOS runtime artifact pointer merely because local training completed. Promotion
requires artifact compatibility, trusted attestation, and device/TestFlight
evidence owned by the release workflow even when the frozen evaluation passes.

## Troubleshooting

- **`nvidia-smi` fails on the host:** fix the host driver before using Docker.
- **The CUDA container cannot see a GPU:** configure the NVIDIA Container
  Toolkit for Docker, restart Docker, and repeat the container verification
  command above.
- **`Failed to create stream fd` appears:** treat this as a launch blocker until
  its cause is known. First check `/tmp` free space, inode use, and any per-user
  quota, then prove that a small temporary file descriptor can be created,
  written, closed, and removed. Review stale temporary artifacts before
  removing only the paths known to be unused, or select a controlled writable
  `TMPDIR`; do not delete repository, model-cache, run-output, or evidence
  paths. A distinct Ubuntu desktop case can print `Operation not permitted`
  when a login-shell input-method initializer attempts optional journal
  logging. Only after the temporary-file probe succeeds, use an ordinary
  non-login shell and avoid `bash -l` or `bash -lc`. Do not dismiss the message
  as harmless unless the intended command actually starts and remains healthy.
- **Docker remains permission-denied after adding the account to the `docker`
  group:** start a new non-login user session (or restart the terminal/Codex
  host) so supplementary groups refresh, confirm `id` lists `docker`, and rerun
  `docker info`. Do not work around it by running the training launcher with
  `sudo`.
- **Python or CUDA lock mismatch:** rebuild without `--no-build`; do not bypass
  the environment gate.
- **Source-integrity or clean-checkout failure:** preserve or commit intended
  source changes and remove unintended untracked execution files. Do not bypass
  the gate or upload from an image built from another source digest.
- **`KeyError: 'getpwuid(): uid not found'`:** pull the current launcher and
  rebuild without `--no-build`. Do not use `sudo`, bind-mount the host
  `/etc/passwd`, or set a fake `USER` value. Because the repaired image and
  training source have new lineage, start a new run ID (or deliberately use
  `--overwrite` after preserving the failed run) rather than resuming the old
  prepared run.
- **A run directory already exists:** use a new `--run-id` or `--output-dir`,
  use `--resume` when its environment and phase lineage still match, or choose
  `--overwrite` deliberately after reviewing the existing artifacts.
- **The terminal or Codex task disconnects while training:** do not delete the
  retained container or run root. Rerun the same clean source and run ID with
  `--resume --no-build`; the launcher authenticates and reattaches to the exact
  container or checkpoint state.
- **The process stops during frozen evaluation:** keep the private evaluation
  directory unchanged and resume the same run. The launcher preserves an exact
  journal-only prefix, or a complete journal plus only a known partial-final
  file subset. It never trusts edited, reordered, duplicated, incomplete mixed,
  or unknown evidence. A complete verified journal can finalize without
  reloading the model.
- **Frozen evaluation exits 2 or 3:** keep the generated candidate outputs and
  reports. Exit 2 means malformed/empty role output; exit 3 means the complete
  suite failed its quality gate. Neither is a successful full pipeline.
- **Lineage or finalized-manifest mismatch:** preserve the failed run for
  diagnosis. Do not copy adapters across variant directories or edit generated
  configs to make the gate pass.
- **Disk pressure:** remove only old, reviewed run directories or Docker/cache
  data whose lineage is no longer needed. Do not run broad cleanup while a
  training process is active.
