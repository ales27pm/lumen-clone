# Ubuntu Lumen Training

This is the canonical runbook for training Lumen's six role adapters on an
Ubuntu machine with an NVIDIA GPU. The host launcher builds a pinned CUDA
container and runs the repository's lineage checks inside it, so the host does
not need a hand-built Python or CUDA environment.

The one-click command is:

```bash
bash scripts/ubuntu_train_lumen_full_pipeline.sh
```

By default it trains the `internal_plus_public_optimized` variant for Cortex,
Executor, Mouth, Mimicry, REM, and Fleet. Each role runs SFT and then preference
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
4. Git and enough free local storage for the repository, container layers,
   Hugging Face cache, checkpoints, and run artifacts.
5. Outbound HTTPS access to GitHub, Hugging Face, the PyTorch wheel index, and
   Python package indexes while the image and model cache are populated.

Verify GPU access from both the host and a container:

```bash
nvidia-smi
docker info
docker run --rm --gpus all nvidia/cuda:12.8.1-devel-ubuntu22.04 nvidia-smi
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
`nvidia/cuda:12.8.1-devel-ubuntu22.04` and uses Python 3.10. The controlled
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

The CUDA base is version-tagged rather than OCI-digest-pinned, and Python wheel
artifacts are version-pinned but not installed from a hash-locked wheelhouse.
For that reason, runtime-image promotion remains prohibited. Credential-scoped
upload requires an image built by the same launcher invocation; the token is
not placed in the Python process environment and is read only after local
evidence and uploader imports have passed. A future release-grade publisher
should additionally use an OCI-digest-pinned minimal image and hashed package
artifacts.

## Capacity Planning

These are engineering estimates, not enforced minimums. Actual use varies with
the GPU, Docker storage driver, model cache state, checkpoint policy, selected
roles, and variant count.

| Resource | Practical estimate |
|---|---|
| GPU | Modern NVIDIA CUDA GPU |
| VRAM | 12 GB floor; 16 GB recommended; 24 GB comfortable |
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

The launcher builds the pinned image, checks Docker GPU access, derives the
actual local image ID, mounts the checkout and a persistent Hugging Face cache,
and creates a fresh run directory. Each selected role completes this order:

```text
prepare and verify isolated dataset/config snapshot
SFT train -> verify finalized SFT manifest and adapter bytes
DPO train -> verify parent SFT lineage and finalized DPO manifest
frozen deterministic inference -> score and enforce the quality gate
adapter-only GGUF conversion -> verify output digest
write run summary
```

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
| `--no-pull` | Build without refreshing the pinned CUDA base tag. |
| `--prepare-only` | Run controlled input, environment, and config preparation without model training. |
| `--overwrite` | Destructively replace the selected run directory after path-safety checks. |
| `--resume` | Reuse an existing run, skip phases whose complete artifacts re-verify, and restart incomplete phases. |
| `--no-evaluate` | Skip frozen inference/scoring. Full evaluation is enabled by default. |
| `--eval-smoke <n>` | Evaluate a deterministic semantic cohort of `n` cases per role; this is smoke evidence, not a quality pass. |
| `--token-file <file>` | Mount an owner-only, mode-600 token only into the upload container. |
| `--upload` | Upload a full quality-passed run after training. Upload is off by default and the destination is private by default. |
| `--allow-diagnostic-upload` | With `--upload`, explicitly permit smoke or unevaluated artifacts under the separate `diagnostic-runs/` namespace. |
| `--public` | With `--upload`, explicitly request public visibility. Public publication is never the default. |

Ubuntu resume is phase-boundary resume, not arbitrary checkpoint resume. With
`--resume`, the image ID, source revision, environment, agents, variant,
prepared-config digest, exact run-scoped paths, self-hashed run manifest, and
the immutable execution plan must still match. The execution plan binds
`evaluationScope`, `evaluationMaxExamples`, and `ggufRequested` before training;
resume cannot relabel missing evidence as an operator skip. A fully verified SFT or DPO phase is kept; an incomplete
phase is removed only inside that agent's owned run subtree and restarted. For
a multi-variant batch, existing variant directories resume and a variant that
had not started yet is prepared fresh. Direct unbound
`--resume-from-checkpoint` remains disabled.
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

`--no-build` is an optimization for an image already built from the same
checkout and the same host UID/GID. A matching tag alone is not proof that its
contents match the current source; the pipeline's identity, environment, and
source-lineage checks remain authoritative.

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
upload_receipts.json                 # only after an explicit successful upload
configs/<agent>.json
configs/<agent>.final.json
generated/fine_tuning/...
generated/agent_manifest/AgentBehaviorManifest.json
logs/...
models/lora_qwen3_bootstrap/<agent>/...
models/lora_qwen3_dpo/<agent>/...
models/lora_qwen3_gguf/lumen-<agent>-lora.gguf
training/<agent>/finalized_variant_manifest.json
training/<agent>/training_report.json
training/<agent>/dpo/finalized_variant_manifest.json
training/<agent>/dpo/dpo_report.json
evaluation/<agent>/candidate_outputs.jsonl
evaluation/<agent>/evaluation_report.json
evaluation/<agent>/evaluation_run_manifest.json
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

The repository bound into the immutable run manifest is used. Immediately
before remote mutation, the upload helper re-verifies adapter bytes, finalized
manifests, evaluation file hashes, GGUF digests, run/summary evidence, and the
exact allowlist. It creates one atomic commit guarded by the observed remote
parent and refuses an existing run prefix. The default is private. Public
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

```bash
bash scripts/ubuntu_train_lumen_full_pipeline.sh \
  --eval-smoke 2 \
  --upload \
  --allow-diagnostic-upload \
  --token-file /secure/path/lumen-hf-token
```

Qualified uploads use `runs/<run-id>/`; diagnostic uploads use
`diagnostic-runs/<run-id>/`. The upload receipt binds the namespace, exact
remote prefix, qualification, promotion eligibility, evaluation status/scope,
GGUF status, whether GGUF was included, and whether the diagnostic override was
applied. `--no-evaluate` and `--eval-smoke` are mutually exclusive regardless
of argument order.

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

Runtime-image promotion remains intentionally unsupported. Do not change the
iOS runtime artifact pointer merely because local training completed. Promotion
requires artifact compatibility, trusted attestation, and device/TestFlight
evidence owned by the release workflow even when the frozen evaluation passes.

## Troubleshooting

- **`nvidia-smi` fails on the host:** fix the host driver before using Docker.
- **The CUDA container cannot see a GPU:** configure the NVIDIA Container
  Toolkit for Docker, restart Docker, and repeat the container verification
  command above.
- **Python or CUDA lock mismatch:** rebuild without `--no-build`; do not bypass
  the environment gate.
- **`KeyError: 'getpwuid(): uid not found'`:** pull the current launcher and
  rebuild without `--no-build`. Do not use `sudo`, bind-mount the host
  `/etc/passwd`, or set a fake `USER` value. Because the repaired image and
  training source have new lineage, start a new run ID (or deliberately use
  `--overwrite` after preserving the failed run) rather than resuming the old
  prepared run.
- **A run directory already exists:** use a new `--run-id` or `--output-dir`,
  use `--resume` when its environment and phase lineage still match, or choose
  `--overwrite` deliberately after reviewing the existing artifacts.
- **Frozen evaluation exits 2 or 3:** keep the generated candidate outputs and
  reports. Exit 2 means malformed/empty role output; exit 3 means the complete
  suite failed its quality gate. Neither is a successful full pipeline.
- **Lineage or finalized-manifest mismatch:** preserve the failed run for
  diagnosis. Do not copy adapters across variant directories or edit generated
  configs to make the gate pass.
- **Disk pressure:** remove only old, reviewed run directories or Docker/cache
  data whose lineage is no longer needed. Do not run broad cleanup while a
  training process is active.
