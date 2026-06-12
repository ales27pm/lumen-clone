# Technical Hardening Guide: iOS LoRA Adapter Runtime and Swift Validation

This guide documents the physical failure modes behind Lumen's on-device LoRA adapter runtime. It deliberately uses a **Poison and Antidote** format: every safeguard is paired with the failure state that made it necessary. Future changes must preserve both parts. A mechanical fix without its failure mode is easy to misread as style and optimize away.

## Strategic context: unified-memory inference is a physical constraint

iOS devices use a unified memory architecture. The app heap, Metal buffers, llama.cpp runtime state, KV cache, and adapter weights all compete for the same physical memory budget. The Apple Neural Engine (ANE) is not used by the current GGUF / llama.cpp path; the current acceleration path is Metal. The practical hazard is still severe: invalid adapter state, excess working set growth, logit corruption, and possible Jetsam termination.

Documentation is therefore part of the runtime safety system. It explains why the code must prefer rigid adapter sequencing over clever shortcuts.

---

## Resolution 1: single-adapter stacking guard

### Poison: PR 171 adapter overlap failure

The historical failure mode was role switching without a hard adapter clear. Switching Cortex, Executor, Mouth, Mimicry, or REM while a previous LoRA remained active could leave overlapping matrices in the shared model context. The symptoms were:

- memory growth after repeated role transitions;
- silent output corruption from summed or stale adapter weights;
- garbage tokens or malformed JSON;
- apparent personality/slot drift because the wrong adapter influence remained active;
- process termination under iOS memory pressure, including Jetsam kills.

The user-visible symptom can look like a model quality problem, but the root cause is mechanical: the base model plus adapter stack is no longer in the intended state.

### Antidote: mandatory three-step activation sequence

`AppLlamaService.activateRoleAdapter(slot:)` and `AdapterChatRuntime.activateRoleAdapter(slot:scale:)` must preserve a single-adapter flow:

1. **Memory purge**: clear all active LoRA adapters from the context via `LlamaContext.removeAllLoraAdapters()` before applying another role adapter.
2. **Adapter application**: apply only the requested role adapter via `LlamaContext.apply(loraAdapter:scale:)`.
3. **State finalization**: set `activeAdapterSlot` only after the adapter application succeeds.

On failure, the runtime must clear adapters again, reset `activeAdapterSlot` to `nil`, store `lastAdapterFailureReason`, and fall back to the shared base model state rather than leaving a half-applied adapter.

### Runtime invariant

At any generation boundary, Lumen must have either:

- zero active LoRA adapters; or
- exactly one active role adapter matching the current live slot.

The runtime must never depend on additive adapter stacking for live user generation.

---

## Resolution 2: architectural isolation and dependency pinning

### Poison: unstable bindings and experimental artifacts

Swift code crosses into C/C++ through `swift-llama-cpp`. Swift ARC does not make native llama.cpp structs safe. If a C++ binding changes layout, ownership, or symbol behavior, Swift may continue compiling while reading or writing incompatible native memory. Even a small struct offset change can produce undefined behavior, segmentation faults, silent corruption, or impossible-to-debug decode failures.

The Fleet adapter is also an experimental artifact. It is useful for training, planning, and whole-system knowledge, but it is not a live generation slot until deliberately introduced and validated. Allowing it to behave as an embedding model or arbitrary chat slot would introduce another unknown variable into the adapter runtime.

### Antidote: pinned foundation and quarantined Fleet role

The Xcode package dependency for `swift-llama-cpp` must remain pinned to **exact version `1.2.0`**. Do not move to branch-based or range-based resolution without a dedicated native-runtime review.

The Fleet adapter must remain a downloadable role adapter artifact and must not be treated as:

- an embedding model;
- a default live chat slot;
- a replacement for Cortex, Executor, Mouth, Mimicry, or REM;
- a release-baked merged GGUF unless a manual release-bake path explicitly requests it.

### Native symbol audit

The following symbols are the manual verification list for any dependency or binding upgrade:

```text
llama_adapter_lora_init(model, path)
llama_adapter_lora_free(adapter)
llama_set_adapter_lora(ctx, adapter, scale)
llama_rm_adapter_lora(ctx, adapter)
llama_clear_adapter_lora(ctx)
LlamaLoraAdapter(model:path:)
LlamaContext.apply(loraAdapter:scale:)
LlamaContext.removeAllLoraAdapters()
```

A dependency change that affects any of these symbols requires physical-device validation and runtime audit export review.

---

## Resolution 3: diagnostic-first smoke testing and fallback auditing

### Poison: graceful fallback can hide adapter failure

Lumen intentionally falls back to a base model or deterministic path when role adapters fail. That preserves responsiveness, but it can mask a serious regression: specialized role behavior disappears while the app still appears to answer.

A CI simulator cannot fully validate the physical runtime state of a real iPhone under Metal, memory pressure, thermal pressure, and local model file conditions. Real-device smoke testing is mandatory before trusting adapter runtime changes.

### Antidote: 12-step real-device smoke test

1. Pull the target branch.
2. Build locally with Xcode.
3. Install on a physical iOS device.
4. Open Settings -> Fleet -> Qwen3 fast adapter bootstrap.
5. Download or repair the selected model family.
6. Confirm one shared base, embedding model, and all role adapters are present.
7. Run a simple chat prompt.
8. Confirm the first response uses the Mouth-only path and does not unexpectedly run Cortex/Executor.
9. Run a tool-backed request.
10. Confirm the expected Cortex -> Executor -> Mouth path.
11. Export the runtime audit and verify adapter traces.
12. Confirm `adapterApplied=true`, expected `adapterSlot`, no unexpected Fleet live slot, and no adapter stacking evidence.

### Telemetry fields to inspect

Runtime audit traces must expose enough state to detect silent fallback:

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

If the UI response looks correct but telemetry shows `adapterApplied=false` for a role turn, treat the result as a failed adapter runtime validation.

---

## Local validation requirements

Do not rely on remote automation for physical runtime validation. Run these commands locally on a Mac with Xcode. A real iOS device is required for the smoke test portion.

```bash
python tools/check_adapter_runtime_invariants.py
python -m pytest tools/lumen_manifest_crawler/tests
python tools/lumen_terminal_improve_loop.py --mode preflight --dry-run --skip-pytest
```

These commands are static and preflight guards. They do not replace physical-device smoke testing.

---

## Change-control rule

Any PR that touches these areas must update this guide if it changes the poison or the antidote:

- `AppLlamaService.activateRoleAdapter(slot:)`
- `AdapterChatRuntime.activateRoleAdapter(slot:scale:)`
- `LlamaContext.apply(loraAdapter:scale:)`
- `LlamaContext.removeAllLoraAdapters()`
- `ModelAdapterRuntimeContract`
- Qwen3 model catalog or adapter role contract
- `swift-llama-cpp` package pin
- runtime audit adapter trace schema
- Fleet adapter activation semantics

The default posture is conservative: preserve single-adapter flow, preserve exact dependency pins, preserve real-device smoke testing, and preserve telemetry that proves adapter application rather than assuming it.
