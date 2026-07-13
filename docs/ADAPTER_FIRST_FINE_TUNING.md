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

When preference training is enabled, DPO must start from a verified finalized SFT adapter, write
to a separate adapter directory, and identify the SFT parent digest. It must never overwrite or
replace the SFT artifact in place.

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
