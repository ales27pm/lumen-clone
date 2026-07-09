# Runtime Status Matrix

This matrix describes the Release product surface. A feature is either shipped, excluded from Release, or available only in DEBUG diagnostics. Historical PR notes may describe older migration states; this file is the current shipping reference.

| Surface | Release status | Runtime owner | User-visible behavior |
| --- | --- | --- | --- |
| Foreground chat, text turns | Shipped | `AssistantKernel.run(...)` with `LlamaRuntimeAdapter.live(...)` backed by `AppLlamaService` | Uses the local SwiftLlama/AppLlamaService path when a chat model is loaded. Missing or failed models surface typed runtime errors instead of deterministic assistant text. |
| Foreground chat, tool-required turns | Shipped for validated intent-routed tool actions | `AssistantKernel.run(...)` -> `StructuredToolCallValidator` -> `SecureToolRegistry` | Tool actions are selected deterministically from routed intent, schema-validated, emitted as `toolInvocation`/`toolResult` events, and surfaced as typed results. Live model-backed post-tool synthesis still needs device/runtime evidence before broader parity claims. |
| Voice text-only turns | Shipped | `VoiceCommandRouter` -> `AssistantKernel.run(...)` | Foreground, user-initiated voice turns route through the kernel and cancel on scene transitions. |
| Voice tool-capable turns | Shipped for validated intent-routed tool actions | `VoiceCommandRouter` -> `AssistantKernel.run(...)` -> `SecureToolRegistry` | Voice reuses the native kernel tool execution path. Tool calls require schema validation, registry policy, and typed tool results; missing context or permission remains an explicit failure/degraded state. |
| AppIntents | Shipped only for guarded local actions | `LumenAskIntent`, `LumenAddMemoryIntent`, `LumenMemorySearchIntent`, `LumenRunTriggerIntent` | Intents return degraded or open-app responses when model, memory, approval, or store context is unavailable. They must not claim tool execution without the required context. |
| Trigger and headless execution | Shipped only for background-safe coordination | `HeadlessAgentKernelRunner`, `BackgroundToolExecutionPolicy`, `TriggerScheduler` | Background work cannot prompt for permissions or load model assets. Background-safe tool-only execution is assessed by policy, routed through secure tool definitions, and skipped with explicit diagnostics when required context is missing. |
| Role adapter runtime | Shipped for adapter lifecycle and diagnostics | `ModelFleet`, `SlotModelRuntimeCoordinator`, `AppLlamaService` | Qwen3 shared-base plus role-adapter artifacts are verified before use; missing role adapters fail with typed diagnostics. |
| REM/runtime repair workflows | DEBUG diagnostics only | `RemCycleService`, generated improve-loop artifacts | Not presented as a shipped autonomous assistant capability. |
| RAG search | Shipped | `RAGStore`, `RAGEngine`, `RAGSearchTool` | Empty results, embedding failures, fetch failures, and lexical fallback modes are surfaced through diagnostics. |
| Memory recall/save | Shipped | `MemoryStore`, `MemoryEngine`, `MemorySearchTool`, `MemoryCaptureQueue` | Empty store, embedding failures, save failures, and pending capture queues are distinct states. |
| FoundationModels text runtime | Excluded from Release routing | `FoundationModelsRuntimeAdapter` | Reported as experimental and non-selectable until a real generation implementation exists. |
| CoreML embedding runtime | Excluded from Release routing | `CoreMLRuntimeAdapter` | Reported as experimental and non-selectable until real embedding extraction exists. |
| Deterministic runtime | DEBUG diagnostics only | `DeterministicFallbackRuntime` | Release routing cannot select it. DEBUG tests may select it explicitly. |
| GGUF native engine | Excluded unless a compiled native bridge is supplied | `GGUFEngine`, `GGUFNativeBridge` | The unavailable bridge is DEBUG-only. Release factory registration cannot install an unavailable GGUF backend. |
| Persistent diagnostics and E2E probes | Shipped for static/local diagnostics; live legacy probes DEBUG-only | `DiagnosticsProvider`, `PersistentRuntimeDiagnosticsRunner`, `E2ETestRunner` | Release diagnostics can export structured state. Legacy live probes are skipped in Release. |

## Release Readiness Vocabulary

- **Shipped** means the Release app exposes the feature and has production code paths plus tests or diagnostics for failure modes.
- **Excluded from Release** means the Release app does not route user requests through that path.
- **DEBUG diagnostics only** means the path may exist for migration evidence, tests, or developer tools, but must not be user-selectable in Release.
