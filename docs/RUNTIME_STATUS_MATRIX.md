# Runtime Status Matrix

This matrix describes the Release product surface. A feature is either shipped, excluded from Release, or available only in DEBUG diagnostics. Historical PR notes may describe older migration states; this file is the current shipping reference.

| Surface | Release status | Runtime owner | User-visible behavior |
| --- | --- | --- | --- |
| Foreground chat, text turns | Shipped | `AssistantKernel.run(...)` with `LlamaRuntimeAdapter.live(...)` backed by `AppLlamaService` | Uses the local SwiftLlama/AppLlamaService path when a chat model is loaded. Missing or failed models surface typed runtime errors instead of deterministic assistant text. |
| Foreground chat, tool-required turns | Excluded from Release until native kernel tool execution is enabled | `AssistantKernel+Streaming` | Release returns an explicit message that tool-capable agent turns are excluded. DEBUG builds may exercise the legacy migration probe. |
| Voice text-only turns | Shipped | `VoiceCommandRouter` -> `AssistantKernel.run(...)` | Foreground, user-initiated voice turns route through the kernel and cancel on scene transitions. |
| Voice tool-capable turns | Excluded from Release | `VoiceCommandRouter` | Release returns an explicit unavailable message. DEBUG builds may exercise legacy migration probes. |
| AppIntents | Shipped only for guarded local actions | `LumenAskIntent`, `LumenAddMemoryIntent`, `LumenMemorySearchIntent`, `LumenRunTriggerIntent` | Intents return degraded or open-app responses when model, memory, approval, or store context is unavailable. They must not claim tool execution without the required context. |
| Trigger and headless execution | Shipped only for background-safe coordination | `HeadlessAgentKernelRunner`, `BackgroundToolBridgePolicy`, `TriggerScheduler` | Background work cannot prompt for permissions or load unavailable model assets. Tool-capable legacy execution is excluded from Release. |
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
