# Adapter-first fine-tuning loop

Lumen should keep one shared agent base model plus role-specific adapters during normal improvement cycles.

## Default training artifact

The default output of a role fine-tuning round is a LoRA/adapter artifact, not a merged full model.
Its identity is the canonical digest of the sorted, allowlisted PEFT/LoRA file manifest; missing,
extra, or modified files invalidate the artifact.

```text
Qwen/Qwen3-1.7B base model
├── cortex adapter
├── executor adapter
├── rem adapter
├── mouth adapter
├── mimicry adapter
└── fleet adapter
```

The base model stays shared. Each role carries its own adapter and role-specific system prompt/config.

## Runtime strategy

Default runtime binding:

```text
load shared base model once
select adapter by agent slot
bind the role system prompt
run inference
```

Expected generated manifests:

```text
generated/.../fine_tuning/adapter_runtime_manifest.json
fine_tuning/<agent>/adapter_export_plan.json
fine_tuning/<agent>/unsloth_config.json
```

`adapter_runtime_manifest.json` is the high-level runtime contract. Each `adapter_export_plan.json` describes the per-agent adapter binding and optional release-bake policy.

## Merge policy

Merging adapters into full GGUF artifacts is no longer part of the default training loop.

Default:

```text
train/evaluate adapter
keep adapter separate
promote or roll back the adapter
```

Optional release bake:

```text
train/evaluate adapter
adapter passes gates
runtime cannot load adapters dynamically, or release build needs a baked artifact
run explicit release bake
export merged GGUF
```

The release bake is manual and explicit:

```bash
python tools/fine_tuning/unsloth/export_gguf.py --release-bake --agents cortex,executor
```

Running the exporter without `--release-bake` must not merge anything. It writes a skipped adapter-first manifest and exits.

## Why this matters

Adapter-first training gives Lumen:

- one shared base model to cache/load;
- smaller per-role artifacts;
- faster role rollback;
- cleaner A/B testing;
- no duplicated full-model GGUF per agent during every loop;
- less storage churn across repeated improvement cycles.

## Promotion unit

The promotion/rollback unit is the adapter, not the base model.

Promote an adapter only if it passes the role-specific gates in `docs/APP_PLAN.md`.
Promotion is currently fail-closed as unsupported because the available Ubuntu and ZeroGPU
launchers can record only an operator-declared runtime-image digest. A signed or independently
verifiable runtime-image attestation must exist before the promotion gate can be enabled.

When preference training is enabled, DPO and ORPO must start from a verified finalized SFT adapter,
write to a separate adapter directory, and identify the SFT parent digest. The parent verifier
checks the finalized-manifest self-hash and status, canonical adapter directory, effective seed,
experiment/source manifest, complete base-model index/shard/tokenizer contract, environment and
dependency locks, requirements digest, runtime-source kind, and SFT code digest.
`adapter_config.json` must name the configured base model. Preference training must never overwrite
or replace the SFT artifact in place.

DPO lineage records the immutable SFT input in `parentSFTLineage`, the frozen SFT policy used by the
objective in `referenceSFTLineage`, and the new execution separately in
`preferenceTrainingRuntime`. ORPO retains the same complete parent validation even though it has no
separate frozen-reference policy. The parent's runtime revision remains parent audit evidence; it
is never overwritten by the DPO/ORPO runtime revision.

DPO and ORPO inputs stay conversational through TRL preprocessing: `prompt` is a validated
system/user conversation ending at an assistant generation boundary, while `chosen` and
`rejected` are assistant-message lists. Missing roles, empty completions, identical preference
pairs, and generic synthesized fallbacks are rejected. The pinned Qwen tokenizer and TRL 0.24
apply the chat template; the dataset compiler does not flatten preference turns into strings.

## Reproducible run identity

Every real ZeroGPU run binds the uploaded dataset repository to its full immutable commit SHA.
The run/resume lineage also binds each agent's variant manifest, lane and corpus hashes,
controlled training config, base-model shard contract, seed, environment lock, phase-specific
training code, dependency lock, runtime source revision, and checkpoint/output paths. A resume
must match the entire lineage and reuse the original local snapshot and recorded checkpoints.

The canonical training-code bundle hashes the complete deployed executable/data closure, not a
curated module list. It includes `app.py`, `requirements.txt`, the complete `lumen_training`
package, and all covered Python and runtime-loaded JSON/text/config resources in the deployed
`lumen_manifest_crawler` tree. The closure policy rejects missing or changed declared files and
unexpected behavior-affecting files in either package. Only explicitly enumerated volatile run
state is excluded. The bundle exposes one overall digest and SFT/DPO/ORPO phase digests.

The built Space executes `python -m lumen_training.train_sft` and
`python -m lumen_training.train_dpo`, so module imports do not depend on the source checkout. The
dependency lock covers all direct runtime packages plus Python, CUDA, Unsloth, and llama.cpp
revisions. Local Ubuntu runs record the source Git commit. ZeroGPU keeps the expected uploaded
Space revision separate from observed repository head and observed runtime revision. Repository
head equality is supplemental evidence, not proof of the executing container; absent trusted
platform metadata, the runtime-source binding remains explicitly unverified. Controlled
comparisons use the verified code and dependency digests.

At runtime, a second environment identity enumerates every installed distribution and binds its
version, safe direct/VCS provenance, and behavior-bearing `RECORD` content. The resulting
`resolvedTrainingEnvironmentSHA256` must agree across controlled comparisons. ZeroGPU also binds a
canonical README front-matter contract through `spaceConfigurationSHA256`, preventing the SDK,
entrypoint, or Python runtime from drifting independently of the training lineage.

ZeroGPU creates the Space, immutable dataset repository, and adapter/model repository as private
unless the operator explicitly selects the corresponding `--public-space`, `--public-dataset`, or
`--public-adapters` override. Making the Space public never bypasses application-level admin-token
authorization. The builder updates and reads back the actual visibility of all three repositories
before uploading, including repositories that already existed. The browser page is status-only;
training is invoked through the authenticated machine endpoint.

The one-click launcher requires `LUMEN_ZERO_GPU_RESUME_BATCH` when a resumable run contains more
than one agent batch. It invokes only that explicit batch and rejects an ambiguous general resume.
Because ordinary Space-local disk is ephemeral across restart or redeployment, checkpoint resume is
limited to an intact deployment until external checkpoint persistence is implemented.

Rollback immediately if an adapter causes:

- sentinel leakage;
- manifest-tool hallucination;
- strict JSON regression for Executor;
- TestFlight/E2E pass-rate regression;
- latency or memory regression outside the allowed budget.

## Relationship with Qwen3 migration

The intended Qwen3 agent migration uses:

```text
base model: Qwen/Qwen3-1.7B
role artifacts: LoRA/adapters
embedding: Qwen/Qwen3-Embedding-0.6B
optional reranker: Qwen/Qwen3-Reranker-0.6B
```

Do not ship six independent full 1.7B models by default. Use one base plus role adapters unless the selected runtime backend cannot load adapters dynamically.
