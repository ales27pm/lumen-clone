# AGENTS.md

## Scope

Governs `ios/Lumen/Services/LLM/`: backend protocols/events/errors, GGUF bridge abstractions, model catalog/storage/selection, device/runtime policy, budgets, reasoning stream parsing, and the tiny intent engine. Native SwiftLlama ownership in `Services/LlamaService.swift` remains governed by the parent. Parent rules: [`../AGENTS.md`](../AGENTS.md).

## Role In The System

This subtree defines what an LLM backend must do and whether a model/backend is compatible with the current device. It separates model files and policy from the production kernel and native service implementation.

## Key Files And Entry Points

- `LLMEngine.swift`, `LLMRequest.swift`, `LLMTokenEvent.swift`, `LLMEngineError.swift`: backend contract.
- `LLMEngineFactory.swift` and `LLMEngineRouter.swift`: construction/selection boundary.
- `GGUF/GGUFEngine.swift`, `GGUFNativeBridge.swift`, and `Native/LumenGGUFBridge.h`: GGUF abstraction/header.
- `GGUF/UnavailableGGUFNativeBridge.swift`: DEBUG-only unavailable bridge path.
- `Models/LLMModelStorage.swift`, `ModelFileValidator.swift`, `ModelSelectionService.swift`, `BuiltInModelCatalog.swift`: model artifact ownership.
- `Policy/DeviceModelPolicy.swift` and related snapshots/estimates: fit/thermal/power decision.
- `ReasoningAwareStreamParser.swift`: separates reasoning/control from user-visible text.

## Public Interfaces

Backend capability flags, request/stream/error types, model catalog IDs/URLs/hashes, storage records, fit decisions, runtime power state, and GGUF bridge protocol are consumed by the runtime router, model loader, views, and tests.

## Internal Structure

Catalog/installed record -> file hash/format validation -> device fit decision -> factory/router -> engine stream. The GGUF Swift engine delegates to a native bridge protocol; the tracked tree contains a header and DEBUG unavailable implementation, not a verified Release native implementation.

## Incoming Dependencies

`AssistantRuntimeRouter`, `ModelLoader`, `LlamaService`, model/settings views, and tests call this layer.

## Outgoing Dependencies

The subtree uses filesystem/CryptoKit-style hashing, device/process state, SwiftLlama-facing types through parent services, and optional GGUF native linkage. It does not execute tools or render UI.

## Data And Control Flow

Selected model -> validated file/hash -> capability/fit preflight -> backend creation -> typed stream -> reasoning-aware parse -> kernel adapter. Cancellation/errors propagate without fallback substitution.

## Local Invariants

- Release cannot select the DEBUG unavailable GGUF bridge or an uncompiled/staged backend.
- A model is usable only after path, format, integrity, memory-fit, and backend-capability validation.
- Preserve exact model/revision/hash identity; do not infer compatibility from filename.
- Thermal/power/memory degradation is explicit and can prevent generation.
- Reasoning/control tokens must not leak into final user-visible text.
- Keep backend errors typed; do not return an empty stream as success.

## Coordinated Changes

Backend capability changes require Assistant routing/adapters, `LlamaService`, `ModelLoader`, tests, hardening scripts, and runtime docs. Model catalog/storage changes require UI, stored model compatibility, artifact lineage, and URL/hash tests. Native bridge changes require Xcode linkage/build settings and Release validation.

## Safe Editing Rules

Do not advertise a backend as available until a real linked implementation and Release evidence exist. Keep unavailable/experimental implementations compile-time excluded from Release selection. Validate large model files without loading them wholly into main-actor memory.

## Validation

From the repository root:

```bash
python3 tools/check_release_hardening.py
python3 tools/check_adapter_runtime_invariants.py
python3 tools/check_ios_lora_hardening_invariants.py
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/LLMEngineFoundationTests -only-testing:LumenTests/LLMDevicePolicyTests -only-testing:LumenTests/LLMModelStorageTests -only-testing:LumenTests/GGUFEngineScaffoldTests
```

Also inspect `ReasoningAwareStreamParserTests.swift`, `CatalogModelURLTests.swift`, and `RuntimeRouterTests.swift`.

## Common Failure Modes

- DEBUG bridge availability leaks into Release routing.
- Catalog metadata and artifact hash/revision diverge.
- A fit calculation ignores adapter/KV/cache overhead.
- An empty/cancelled stream is treated as a valid completion.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). Native SwiftLlama/adapter ownership is documented there; kernel routing is in [`../../Assistant/AGENTS.md`](../../Assistant/AGENTS.md).
