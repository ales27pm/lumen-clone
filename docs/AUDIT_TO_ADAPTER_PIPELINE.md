# Lumen audit-to-adapter pipeline

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

Adapter training is intentionally separate from GGUF conversion.

```bash
for role in cortex executor mouth mimicry rem fleet; do
  python tools/fine_tuning/unsloth/train_sft.py \
    --config "tools/fine_tuning/unsloth/configs_qwen3_bootstrap/$role.json" \
    --seed 42 \
    --assistant-only-loss
done
```

Expected outputs:

```text
models/lora_qwen3_bootstrap/cortex
models/lora_qwen3_bootstrap/executor
models/lora_qwen3_bootstrap/mouth
models/lora_qwen3_bootstrap/mimicry
models/lora_qwen3_bootstrap/rem
models/lora_qwen3_bootstrap/fleet
```

### 4. Explicit LoRA-to-GGUF adapter conversion

Conversion is a separate stage. It must pass the base model explicitly.

```bash
for role in cortex executor mouth mimicry rem fleet; do
  python ~/.unsloth/llama.cpp/convert_lora_to_gguf.py \
    "models/lora_qwen3_bootstrap/$role" \
    --outfile "models/lora_qwen3_gguf/lumen-$role-lora.gguf" \
    --base-model-id Qwen/Qwen3-1.7B
done
```

### 5. Hugging Face upload

Adapters:

```bash
hf repos create ales27pm/lumen-qwen3-bootstrap-adapters-gguf \
  --type model \
  --exist-ok

hf upload ales27pm/lumen-qwen3-bootstrap-adapters-gguf \
  models/lora_qwen3_gguf \
  . \
  --repo-type model
```

Shared base:

```bash
hf upload-large-folder ales27pm/lumen-qwen3-bootstrap-gguf \
  models/base_qwen3_fast \
  --repo-type model
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
python tools/lumen_terminal_improve_loop.py --mode preflight --dry-run --skip-pytest
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
