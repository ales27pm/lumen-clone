# Lumen audit-to-adapter pipeline

## Evidence status

- **Label:** `planned_contract`
- **What this document proves:** the intended audit-to-adapter contract, artifact naming, stage ownership, and validation boundaries for the adapter-first loop.
- **What this document does not prove:** that a specific audit export was captured on device, that adapters were trained or deployed, or that live E2E scenarios passed.

This document is the canonical end-to-end contract for the Lumen Qwen3 adapter-first loop. It ties together the app runtime audit JSONs, code crawl, dataset generation, adapter training, GGUF adapter conversion, Hugging Face publication, and iOS installation/runtime selection.

The machine-readable version lives in `tools/pipeline/audit_to_adapter_contract.py`. The validator is `tools/pipeline/validate_audit_to_adapter_pipeline.py`.

## Default runtime shape

The default iOS Qwen3 runtime is:

```text
one shared Qwen3 chat base GGUF
+ one Qwen3 embedding GGUF
+ one active role LoRA GGUF adapter at a time
+ role-specific system prompts/contracts
```

It must not become:

```text
cortex full GGUF
executor full GGUF
mouth full GGUF
mimicry full GGUF
rem full GGUF
```

Release-baked full GGUFs are manual fallback artifacts only. They must not appear in the default Qwen3 model catalog.

See `docs/RUNTIME_STATUS_MATRIX.md` before interpreting the default runtime shape as shipped behavior: the adapter-first shape is the product target, while each app surface remains labeled as live, partial, compatibility bridge, or planned in the runtime matrix.

## Artifact contract

### Shared chat base

```text
Repo: ales27pm/lumen-qwen3-bootstrap-gguf
File: lumen-qwen3-fast-shared-q4_k_m.gguf
Base: Qwen/Qwen3-1.7B
Role: chat/shared-base
Local default: models/base_qwen3_fast/lumen-qwen3-fast-shared-q4_k_m.gguf
```

### Embedding model

```text
Repo: Qwen/Qwen3-Embedding-0.6B-GGUF
File: Qwen3-Embedding-0.6B-Q8_0.gguf
Role: embedding
Runtime use: memory, RAG, source-map, tool schema retrieval, repair retrieval
```

### Role adapters

```text
Repo: ales27pm/lumen-qwen3-bootstrap-adapters-gguf
Files:
  lumen-cortex-lora.gguf
  lumen-executor-lora.gguf
  lumen-mouth-lora.gguf
  lumen-mimicry-lora.gguf
  lumen-rem-lora.gguf
  lumen-fleet-lora.gguf
```

Live runtime adapter slots are `cortex`, `executor`, `mouth`, `mimicry`, and `rem`. `fleet` is trained and downloadable, but it is not a live slot until a dedicated runtime contract adds one.

## Pipeline stages

### 1. Runtime audit ingestion

Input audit JSONs come from:

```text
exports/*.json
generated/runtime_audits/*.json
generated/runtime_audit/*.json
generated/testflight_exports/*.json
generated/agent_improvement_loop/runtime_audits/*.json
```

Runtime audit JSONs are not optional decoration. They are training and validation evidence. Useful fields include:

```text
adapterApplied
adapterSlot
activeAdapterSlot
adapterFailureReason
runtimePath
modelFamily
baseModelPath
adapterPath
```

Command shape:

```bash
python -m lumen_manifest_crawler improve-loop \
  --root "$PWD" \
  --output generated/agent_manifest \
  --loop-output generated/agent_improvement_loop \
  --fine-tuning-output generated/fine_tuning \
  --strict \
  --deterministic \
  --pretty \
  --generate-system-prompts \
  --generate-agent-fine-tuning \
  --runtime-audit <audit.json>
```

The terminal launcher wraps this as `python tools/lumen_terminal_improve_loop.py --mode generate`.

### 2. Dataset generation

Generated per-role SFT datasets live under:

```text
generated/fine_tuning/cortex
generated/fine_tuning/executor
generated/fine_tuning/mouth
generated/fine_tuning/mimicry
generated/fine_tuning/rem
generated/fine_tuning/fleet
```

Every role dataset should include `train_sft.jsonl` and `val_sft.jsonl`.

### 3. Adapter-only training

Full Ubuntu training is owned by the pinned one-click launcher. Do not train
from the retired static bootstrap configs: the launcher snapshots the selected
controlled variant and binds its dataset, config, base-model, source, code,
dependency, and runtime lineage before any optimizer work.

```bash
bash scripts/ubuntu_train_lumen_full_pipeline.sh
```

The default run performs SFT followed by DPO for all six roles and preserves
the verified SFT parent separately from the final preference adapter.

### 4. Adapter-only GGUF conversion

Conversion is part of the canonical launcher and uses the finalized preference
adapter plus the pinned base-model artifacts. It fetches the converter from the
pinned llama.cpp revision and verifies that checkout before execution. Do not
run an unbound converter against `models/lora_qwen3_bootstrap`.

Expected per-role outputs include:

```text
<run-root>/models/lora_qwen3_dpo/<role>/
<run-root>/models/lora_qwen3_gguf/lumen-<role>-lora.gguf
```

### 5. Hugging Face upload

Upload is off by default and occurs only after local evidence re-verifies. The
launcher starts a separate credential-scoped container, performs one atomic
allowlisted commit, and keeps the repository private unless `--public` is
explicitly requested.

```bash
bash scripts/ubuntu_train_lumen_full_pipeline.sh \
  --upload \
  --token-file /secure/path/lumen-hf-token
```

### 6. iOS installation and runtime resolution

The iOS app installs the selected family through `ModelLaunchBootstrap` and `ModelDownloader`. For Qwen3, the catalog is generated from `LumenTrainedModelRuntimeRegistry.qwen3AdapterBootstrapContract`.

The app stores downloaded artifacts as:

```text
ModelRole.chat         -> shared base
ModelRole.embedding    -> Qwen3 embedding model
ModelRole.roleAdapter  -> role LoRA GGUF adapters
```

Role adapters must not be directly activatable in the UI. They become active only when `SlotModelRuntimeCoordinator.ensureReadyWithMetrics(slot:)` loads the shared base and applies the requested slot adapter.

## Required validation

Static validation:

```bash
python tools/pipeline/validate_audit_to_adapter_pipeline.py
python tools/check_adapter_runtime_invariants.py
bash scripts/check-lumen-integration-gate.sh
```

Require local generated artifacts too:

```bash
python tools/pipeline/validate_audit_to_adapter_pipeline.py --require-generated-artifacts
```

Require at least one runtime audit JSON:

```bash
python tools/pipeline/validate_audit_to_adapter_pipeline.py --require-runtime-audit
```

Real-device validation is still mandatory. Static checks cannot prove Metal memory behavior or adapter application on an iPhone. A valid device smoke test must show `runtimePath=sharedAdapter`, the expected `adapterSlot`, and `adapterApplied=true` for role turns.
