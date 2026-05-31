# Lumen Qwen3 Adapter Runtime and Improve-Loop Doctrine

This document is a drift guard. It describes the runtime architecture, artifact layout, training loop, release-bake policy, and runtime-audit fields that must remain aligned after the Qwen3 adapter-runtime migration.

The goal is speed and stability on iPhone. The default Qwen3 runtime must never regress to loading multiple full chat GGUFs per role.

## Non-negotiable runtime invariant

Default Qwen3 runtime shape:

```text
one resident Qwen3 chat base GGUF
+ one dedicated Qwen3 embedding GGUF
+ role-specific LoRA GGUF adapters
+ role-specific system prompts
```

Default Qwen3 runtime must not be:

```text
cortex full GGUF
executor full GGUF
mouth full GGUF
mimicry full GGUF
rem full GGUF
```

The default Qwen3 path must load the shared chat base once and switch adapters per role. Slot changes must not unload/reload the full chat model.

## Default artifact contract

### Shared chat base

```text
Repo: ales27pm/lumen-qwen3-bootstrap-gguf
File: lumen-qwen3-fast-shared-q4_k_m.gguf
Role: chat
Runtime meaning: shared Qwen3 base for every agent role
```

### Embedding model

```text
Repo: Qwen/Qwen3-Embedding-0.6B-GGUF
File: Qwen3-Embedding-0.6B-Q8_0.gguf
Role: embedding
Runtime meaning: retrieval, memory, RAG, source-map, repair retrieval
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
Role: roleAdapter
Runtime meaning: per-slot behavior adapters applied on top of the shared chat base
```

The `fleet` adapter is cataloged and downloadable as a role adapter. It is not an embedding model. It should remain downloaded-only until a deliberate live Fleet runtime slot is introduced.

## App runtime rules

### Shared-base loading

`AppLlamaService` owns one adapter runtime for Qwen3:

```text
sharedChatRuntime
sharedChatBasePath
roleAdapters[slot]
activeAdapterSlot
```

Qwen3 slot readiness means:

1. shared base exists locally;
2. shared base is loaded if needed;
3. role adapter exists locally if configured;
4. role adapter is loaded if needed;
5. selected adapter is activated for the current slot.

Slot readiness must not call `unloadAllChat()` for Qwen3 role switches.

### Adapter activation

Before applying any role adapter, the runtime must clear all currently active adapters.

Required behavior:

```text
activate cortex:
  clear active LoRA adapters
  apply cortex adapter only

activate executor:
  clear active LoRA adapters
  apply executor adapter only

activate mouth:
  clear active LoRA adapters
  apply mouth adapter only
```

Adapter stacking is a bug. Do not allow accidental `cortex + executor + mouth` adapter accumulation.

If adapter activation fails:

```text
clear active adapters
unload failed adapter handle for that slot
set activeAdapterSlot = nil
record adapterFailureReason
continue with shared base + role prompt only
```

This emergency fallback is allowed so the app remains usable, but the runtime trace must make the fallback visible to the improve-loop.

### Fast call policy

Do not run every role synchronously for every message.

Default orchestration:

```text
simple chat:
  mouth only

explicit style rewrite request:
  mouth -> mimicry

tool/action request:
  cortex -> executor -> mouth

post-turn audit:
  rem background only
```

REM must not block first-token or final-answer latency.

Mimicry must not run for normal chat unless the user explicitly asks for style/tone rewriting.

## Improve-loop alignment

The improve-loop must train and validate an adapter-first system. It must not silently convert the default runtime back into role-baked full GGUFs.

### Dataset generation

The agent datasets still remain role-specific:

```text
cortex
executor
mouth
mimicry
rem
fleet
```

But the trained outputs should be role adapters by default, not six standalone runtime chat models.

Expected training shape:

```text
base: Qwen/Qwen3-1.7B or configured Qwen3 base
outputs:
  models/lora_qwen3_bootstrap/cortex
  models/lora_qwen3_bootstrap/executor
  models/lora_qwen3_bootstrap/mouth
  models/lora_qwen3_bootstrap/mimicry
  models/lora_qwen3_bootstrap/rem
  models/lora_qwen3_bootstrap/fleet
```

Expected adapter conversion (base model is mandatory — either `--base-model-id`
or `--base` pointing at a local config dir):

```bash
mkdir -p models/lora_qwen3_gguf

for agent in cortex executor mouth mimicry rem fleet; do
  python ~/.unsloth/llama.cpp/convert_lora_to_gguf.py \
    "models/lora_qwen3_bootstrap/$agent" \
    --outfile "models/lora_qwen3_gguf/lumen-$agent-lora.gguf" \
    --base-model-id Qwen/Qwen3-1.7B
done
```

Expected adapter upload (current Hugging Face CLI uses `hf repos create`, not
the legacy `hf repo create`):

```bash
hf repos create ales27pm/lumen-qwen3-bootstrap-adapters-gguf \
  --type model \
  --private \
  --exist-ok \
  --yes

hf upload ales27pm/lumen-qwen3-bootstrap-adapters-gguf \
  models/lora_qwen3_gguf \
  . \
  --repo-type model
```

For the shared base GGUF (large file, resumable), prefer:

```bash
hf upload-large-folder ales27pm/lumen-qwen3-bootstrap-gguf \
  models/base_qwen3_fast \
  --repo-type model
```

### Resumable terminal AIO loop

The terminal launcher `tools/lumen_terminal_improve_loop.py` is the single
entrypoint for running the full local cycle. It records each stage's argv,
input hash, output paths and status to `pipeline_state.json`, so reruns can
skip stages whose inputs are unchanged.

```bash
python tools/lumen_terminal_improve_loop.py \
  --mode full \
  --resume \
  --state-file generated/agent_improvement_loop/pipeline_state.json \
  --config-dir tools/fine_tuning/unsloth/configs_qwen3_bootstrap \
  --agents cortex,executor,mouth,mimicry,rem,fleet \
  --base-model-id Qwen/Qwen3-1.7B \
  --seed 42 \
  --assistant-only-loss \
  --hf-private \
  --fail-if-missing-qwen3-config \
  --stop-on-error
```

Preflight strictness:

- Fails if `tools/fine_tuning/unsloth/configs_qwen3_bootstrap/` is missing.
- Fails if any agent config in that dir still references a Qwen2.x base.
- Fails if `merge_adapters_by_default` or `release_bake_enabled_by_default`
  is true in any Qwen3 bootstrap config.

### Release-bake policy

Release-baking adapters into per-role full GGUFs is optional and manual only.

The default improve-loop must keep:

```json
{
  "merge_adapters_by_default": false,
  "release_bake_enabled_by_default": false
}
```

`tools/fine_tuning/unsloth/export_gguf.py` must continue to skip merging unless `--release-bake` is explicitly passed.

Allowed:

```bash
python tools/fine_tuning/unsloth/export_gguf.py \
  --config-dir tools/fine_tuning/unsloth/configs_qwen3_bootstrap
```

This should write an adapter-first manifest and skip merged GGUF export.

Manual fallback only:

```bash
python tools/fine_tuning/unsloth/export_gguf.py \
  --release-bake \
  --config-dir tools/fine_tuning/unsloth/configs_qwen3_bootstrap
```

This can produce `lumen-<role>-release-bake-*.gguf` artifacts, but those artifacts must not become Qwen3 default first-launch downloads.

## Runtime audit export contract

The in-app audit JSON must expose enough evidence for the improve-loop to confirm that the app is using the adapter runtime.

For Qwen3 model turns, traces should include:

```json
{
  "modelFamily": "qwen3",
  "baseModelPath": ".../lumen-qwen3-fast-shared-q4_k_m.gguf",
  "adapterID": "...",
  "adapterSlot": "mouth",
  "adapterPath": ".../lumen-mouth-lora.gguf",
  "adapterApplied": true,
  "adapterScale": 1.0,
  "adapterFailureReason": null,
  "generationElapsedMs": 0,
  "firstTokenLatencyMs": null,
  "outputTokenCount": null
}
```

`outputTokenCount` must remain `null` unless it is a real tokenizer/runtime token count. A whitespace word count must never be reported as token count.

`allowedToolIDs` rules:

- Cortex and Executor tool-selection turns should export the actual allowed tool set when available.
- Mouth, Mimicry, and REM should keep `allowedToolIDs` empty unless the prompt explicitly contains an `Available tools:` block.

## Improve-loop drift checks

A future improve-loop or Codex change should be rejected if any of these become true:

1. Qwen3 default catalog contains more than one `.chat` artifact.
2. Qwen3 default catalog contains role-baked `lumen-*-release-bake-*.gguf` files.
3. `lumen-fleet-lora.gguf` is treated as `.embedding`.
4. `SlotModelRuntimeCoordinator` calls `unloadAllChat()` on normal Qwen3 slot switches.
5. `activateRoleAdapter` applies a new adapter without clearing existing active adapters first.
6. Simple chat synchronously runs Cortex, Executor, Mimicry, and REM.
7. REM blocks the user-facing answer path.
8. `outputTokenCount` is populated from whitespace word count.
9. Runtime traces omit `adapterApplied` or `adapterSlot` for Qwen3 model turns.
10. `export_gguf.py` merges adapters by default without `--release-bake`.
11. `tools/lumen_terminal_improve_loop.py` calls the legacy `hf repo create`
    or invokes `convert_lora_to_gguf.py` without `--base` / `--base-model-id`.
12. Any Qwen3 bootstrap config references a non-Qwen3 base model.

## Required test coverage

The repository should keep deterministic tests or review checks for:

- Qwen3 bootstrap catalog has exactly one chat base.
- Qwen3 bootstrap catalog has exactly one embedding model.
- Qwen3 bootstrap catalog has exactly six role adapters.
- Qwen3 bootstrap catalog has no release-bake files by default.
- Fleet adapter is `.roleAdapter`, not `.embedding`.
- Trace initializer remains backward-compatible when adapter fields are absent.
- Adapter trace metadata defaults do not fabricate adapter success.

## Human local validation before merge/release

Codex/cloud environments may not have Xcode. A human/local macOS environment must run:

```bash
xcodebuild -project ios/Lumen.xcodeproj \
  -scheme Lumen \
  -destination 'platform=iOS Simulator,name=iPhone 16 Pro' \
  build
```

If the simulator name differs:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -showdestinations
```

Then run a device smoke test:

1. Install the build on device.
2. Select `Qwen3 fast adapter bootstrap`.
3. Download / repair selected family.
4. Confirm one shared base, one embedding model, and role adapters are downloaded.
5. Run a normal chat prompt.
6. Confirm first response is Mouth-only.
7. Run a tool/action prompt.
8. Confirm Cortex -> Executor -> Mouth path.
9. Export runtime audit.
10. Confirm traces show `adapterApplied=true` and the expected `adapterSlot` for role turns.

## Decision record

Default Lumen Qwen3 runtime is adapter-first.

Role-baked full GGUFs are emergency/manual fallback artifacts, not the product runtime target.
