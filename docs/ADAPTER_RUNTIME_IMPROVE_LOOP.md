# Lumen Qwen3 Adapter Runtime and Improve-Loop Doctrine

## Evidence status

- **Label:** `planned_contract`
- **What this document proves:** the intended Qwen3 adapter-runtime doctrine, artifact layout, runtime invariants, and improve-loop responsibilities.
- **What this document does not prove:** that the current app build loaded adapters on device, avoided fallback behavior, or passed live E2E validation.

This document is a drift guard. It describes the runtime architecture, artifact layout, training loop, release-bake policy, and runtime-audit fields that must remain aligned after the Qwen3 adapter-runtime migration.

The goal is speed and stability on iPhone. The default Qwen3 runtime must never regress to loading multiple full chat GGUFs per role.

For the complete developer testing, debugging, diagnostics, TestFlight, and
learning workflow, use `docs/DEVELOPER_IMPROVE_FRAMEWORK.md` as the canonical
framework. This document owns the Qwen3 adapter-runtime invariants inside that
framework.

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
base: pinned Qwen/Qwen3-1.7B
per role:
  <run-root>/models/lora_qwen3_bootstrap/<role>   # verified SFT parent
  <run-root>/models/lora_qwen3_dpo/<role>         # final preference adapter
  <run-root>/models/lora_qwen3_gguf/lumen-<role>-lora.gguf
  <run-root>/evaluation/<role>/...
```

The canonical launcher owns adapter conversion and optional upload. It verifies
the pinned llama.cpp checkout and base artifacts before conversion. Upload is
off by default; `--upload` scopes an owner-only token to a separate container,
re-verifies exact evidence, and makes one private atomic commit unless
`--public` is explicit. Ordinary upload requires a full frozen quality pass,
including a `complete_without_gguf` run when conversion was disabled. Smoke or
unevaluated publication additionally requires `--allow-diagnostic-upload` and
uses the separate `diagnostic-runs/` namespace with `promotionEligible=false`.
Manual converter and broad `hf upload` shortcuts are not part of this contract.

### Canonical Ubuntu training loop

`scripts/ubuntu_train_lumen_full_pipeline.sh` is the only supported full
Ubuntu training entrypoint. It consumes the controlled generated variants,
builds the pinned container, trains SFT and DPO adapters, runs frozen
evaluation, converts the final preference adapters to adapter-only GGUF, and
writes self-verified run evidence. The older terminal improve-loop remains a
dataset/developer utility and must not be used as the training launcher.

```bash
bash scripts/ubuntu_train_lumen_full_pipeline.sh
```

Use `--resume --run-id <original-id> --no-build` to resume verified phase
boundaries. See `docs/UBUNTU_TRAINING.md` for host prerequisites, selectors,
credential isolation, outputs, and evidence limits.

### Release-bake policy

Release-baking adapters into per-role full GGUFs is outside the canonical
Ubuntu training contract.

The default improve-loop must keep:

```json
{
  "merge_adapters_by_default": false,
  "release_bake_enabled_by_default": false
}
```

The directory exporter must continue to skip merging unless `--release-bake` is
explicit. In release-bake mode it consumes only run-scoped
`<agent>.final.json` configs inside the pinned container and never falls back to
pending/SFT `<agent>.json` configs. The prepared paths and snapshot verification
objects must be used with the original `/outputs` mount; rebasing them would
invalidate the attestation. A merged artifact receives new lineage and must
rerun evaluation and release approval. It must not become a Qwen3 default
download merely because adapter training completed.

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
11. `tools/lumen_terminal_improve_loop.py` permits its retired
    training/conversion/upload modes instead of routing operators to the
    canonical Ubuntu launcher.
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
