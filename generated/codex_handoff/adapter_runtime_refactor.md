# Lumen Qwen3 Adapter Runtime Refactor Handoff

## Changed files

- `ios/Lumen/Models/StoredModel.swift`
- `ios/Lumen/Services/ModelFamilySelection.swift`
- `ios/Lumen/Services/ModelFleet.swift`
- `ios/Lumen/Services/ModelLaunchBootstrap.swift`
- `ios/Lumen/Services/ModelLoader.swift`
- `ios/Lumen/Services/SlotModelRuntimeCoordinator.swift`
- `ios/Lumen/Services/LlamaService.swift`
- `ios/Lumen/Services/RolePipelineAgentService.swift`
- `ios/Lumen/Services/AgentGrounding/AgentBehaviorTrace.swift`
- `ios/Lumen/Views/ModelsView.swift`
- `ios/LumenTests/LumenFleetTests.swift`

## Runtime flow before

- Qwen3 first-launch catalog pointed at five role-baked full chat GGUFs.
- `LumenModelFleetResolver` assigned full chat models per role.
- `SlotModelRuntimeCoordinator.ensureReady(slot:)` unloaded the current chat runtime and loaded the slot-specific full GGUF.
- `AppLlamaService` stored `chatRuntimes: [LumenModelSlot: ChatRuntime]`, so role switches caused model churn.
- Simple chat still flowed through Mouth and then Mimicry synchronously.

## Runtime flow after

- Qwen3 default catalog downloads one shared chat base, one embedding GGUF, and six role LoRA adapter GGUFs.
- `LumenModelFleetResolver` recognizes the Qwen3 shared base and attaches optional role adapter metadata to each slot assignment.
- `SlotModelRuntimeCoordinator.ensureReady(slot:)` loads the shared Qwen3 base once, lazily loads the slot adapter, activates it, and does not call `unloadAllChat()` for Qwen3 slot switches.
- `AppLlamaService` has a shared adapter runtime (`sharedChatRuntime`, `sharedChatBasePath`) plus `roleAdapters` and `activeAdapterSlot`.
- Role-specific grounding/system prompts are still applied via `req.groundingSystemPrompt(for:)` before generation.
- If a role adapter is missing or fails to activate, the coordinator logs the structured failure, clears active adapters, and continues on the shared base with the role prompt.
- Simple chat uses Mouth only. Mimicry is only invoked for explicit style-rewrite requests. REM remains post-turn/background and non-blocking.

## New Hugging Face artifact layout

Shared base repo:

- `ales27pm/lumen-qwen3-bootstrap-gguf`
- `lumen-qwen3-fast-shared-q4_k_m.gguf`

Adapter repo:

- `ales27pm/lumen-qwen3-bootstrap-adapters-gguf`
- `lumen-cortex-lora.gguf`
- `lumen-executor-lora.gguf`
- `lumen-mouth-lora.gguf`
- `lumen-mimicry-lora.gguf`
- `lumen-rem-lora.gguf`
- `lumen-fleet-lora.gguf`

Embedding remains separate:

- `Qwen/Qwen3-Embedding-0.6B-GGUF`
- `Qwen3-Embedding-0.6B-Q8_0.gguf`

## Adapter conversion commands

```bash
cd ~/Lumen
mkdir -p models/lora_qwen3_gguf

for agent in cortex executor mouth mimicry rem fleet; do
  python ~/.unsloth/llama.cpp/convert_lora_to_gguf.py \
    "models/lora_qwen3_bootstrap/$agent" \
    --outfile "models/lora_qwen3_gguf/lumen-$agent-lora.gguf"
done

find models/lora_qwen3_gguf -name "*.gguf" -type f -exec ls -lh {} +
```

## Upload commands

Adapters:

```bash
hf repo create ales27pm/lumen-qwen3-bootstrap-adapters-gguf \
  --type model \
  --private \
  --yes

hf upload ales27pm/lumen-qwen3-bootstrap-adapters-gguf \
  models/lora_qwen3_gguf \
  . \
  --repo-type model
```

Base model:

```bash
hf upload ales27pm/lumen-qwen3-bootstrap-gguf \
  models/base_qwen3_fast/lumen-qwen3-fast-shared-q4_k_m.gguf \
  lumen-qwen3-fast-shared-q4_k_m.gguf \
  --repo-type model
```

## Local validation commands for human Xcode environment

Codex must not run this because the environment does not have Xcode/xcodebuild:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16 Pro' build
```

Codex-available Python check:

```bash
python -m pytest tools/lumen_manifest_crawler/tests
```

## llama.cpp / SwiftLlama adapter bridge verification

The Xcode project pins `swift-llama-cpp` exact version `1.2.0`. Inspection of that package version found these real adapter APIs:

- `llama_adapter_lora_init(model, path)`
- `llama_adapter_lora_free(adapter)`
- `llama_set_adapter_lora(ctx, adapter, scale)`
- `llama_rm_adapter_lora(ctx, adapter)`
- `llama_clear_adapter_lora(ctx)`
- Swift wrappers: `LlamaLoraAdapter`, `LlamaContext.apply(loraAdapter:scale:)`, and `LlamaContext.removeAllLoraAdapters()`.

Because `SwiftLlama.LlamaService` hides its internal `Llama` actor/context, the app-side bridge uses public `SwiftLlama` wrappers (`LlamaModel`, `LlamaContext`, `LlamaBatch`, `LlamaSampler`, `LlamaLoraAdapter`) to run the shared-base adapter path directly. The legacy `SwiftLlama.LlamaService` path remains for Qwen2.5/full-model fallback.

## Swift symbols requiring local Xcode validation

- `AdapterChatRuntime.streamCompletion(...)`
- `LlamaLoraAdapter(model:path:)`
- `LlamaContext.apply(loraAdapter:scale:)`
- `LlamaContext.removeAllLoraAdapters()`
- `LlamaSampler(config:model:)`
- `LlamaBatch(initialSize:)`
- `AgentBehaviorTrace` optional adapter metadata Codable compatibility
- `ModelRole.roleAdapter` SwiftData persisted string compatibility

## Unresolved native bridge risks

- Codex could inspect the pinned Swift package from GitHub but could not compile the iOS target locally because `xcodebuild` is unavailable.
- If the locally resolved Swift package differs from exact version `1.2.0`, adapter method spellings may differ. Re-resolve packages in Xcode and verify the methods above.
- The direct app-side adapter runtime intentionally avoids fake APIs, but its generation loop should be validated with a real GGUF base and LoRA adapter on device/simulator.

## Follow-up hardening pass

This follow-up was added after PR #171 review to harden the adapter runtime without re-architecting the already-open PR.

### Adapter stacking fix

`AppLlamaService.activateRoleAdapter(slot:)` now clears all active LoRA adapters immediately before applying the selected role adapter. This makes each role switch explicitly single-adapter:

1. clear all active adapters
2. apply selected role adapter
3. set `activeAdapterSlot` only after successful apply

If apply fails, the runtime clears adapters again, sets `activeAdapterSlot = nil`, stores `lastAdapterFailureReason = error.localizedDescription`, and lets the coordinator continue with shared-base + role prompt fallback.

### Fleet adapter behavior

`lumen-fleet-lora.gguf` is cataloged and downloaded as a `.roleAdapter` artifact with role tag `fleet`. It is not mapped through `.embedding`, is not selected as the active embedding model, and is not used by any live generation slot yet. The live runtime currently activates only cortex, executor, mouth, mimicry, and rem adapters. The fleet adapter is downloaded-only until a safe live Fleet runtime role is introduced deliberately.

### SwiftLlama symbol validation status

The pinned `swift-llama-cpp` package version remains `1.2.0`. The follow-up rechecked these public symbols in that package:

- `LlamaModel(path:parameters:)`
- `LlamaContext(model:parameters:)`
- `LlamaBatch(initialSize:)`
- `LlamaBatch.size`
- `LlamaBatch.setLastTokenLogits(_:)`
- `LlamaSampler(config:model:)`
- `LlamaSampler.sample(context:)`
- `LlamaModel.applyChatTemplate(to:addAssistant:)`
- `LlamaModel.tokenize(text:addBos:special:)`
- `LlamaModel.shouldAddBos()`
- `LlamaModel.isEogToken(_:)`
- `LlamaModel.piece(from:)`
- `LlamaLoraAdapter(model:path:)`
- `LlamaContext.apply(loraAdapter:scale:)`
- `LlamaContext.removeAllLoraAdapters()`
- `LlamaContext.clearKVCache()`
- `LlamaContext.decode(batch:)`

`LlamaSampler.sample(context:)` is documented by the wrapper as using `llama_sampler_sample(..., idx: -1)` and accepting/updating sampler state internally. No extra manual `accept(token:)` call was added.

### AdapterChatRuntime serialization fix

`AdapterChatRuntime` is now an actor. Its mutable llama.cpp state (`LlamaContext`, `LlamaBatch`, processed token cache, current token position, and loaded adapter handles) is actor-isolated, and stream generation enters an actor-isolated generation method instead of mutating runtime state directly from the `AsyncThrowingStream` closure.

### Final prompt-token logits fix

Prompt ingestion now marks the final prompt token with `logits=true` during the main token loop and decodes whenever the batch is full or the final token is reached. This avoids the prior boundary bug where prompts with token counts exactly divisible by `batchSize` could finish without computing logits for the final prompt token.

### outputTokenCount decision

`outputTokenCount` is left as `nil` for now. The previous value was a whitespace-delimited word count, not a model token count. Exact token counts should be threaded through both the adapter runtime and legacy SwiftLlama runtime before this field is populated again.

### Direct generation loop risks

The direct generation loop still needs local Xcode validation because Codex cannot run `xcodebuild`. It deliberately clears KV cache for each completion for correctness and to avoid prompt residue between roles. This is not optimal for latency, but it is the safer behavior until a local Xcode/device smoke test validates adapter switching and output quality.

### Local validation command

Do not run this in Codex. Run locally on a Mac with Xcode:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16 Pro' build
```

### Real-device smoke test plan

1. Pull branch `codex/refactor-lumen-ios-model-runtime-architecture`.
2. Build locally with Xcode.
3. Install on device.
4. Settings → Fleet → Qwen3 fast adapter bootstrap.
5. Download / repair selected family.
6. Confirm one shared base + embedding + adapters downloaded.
7. Run a simple chat.
8. Confirm first response is Mouth-only and no Cortex/Executor/Mimicry synchronous path ran.
9. Run a tool request.
10. Confirm Cortex → Executor → Mouth path.
11. Export runtime audit.
12. Check traces show `adapterApplied=true` for relevant slots and no adapter stacking.
