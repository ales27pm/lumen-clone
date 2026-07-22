# Runtime Status Matrix

This matrix describes the Release product surface. A feature is either shipped, excluded from Release, or available only in DEBUG diagnostics. Historical PR notes may describe older migration states; this file is the current shipping reference.

Offline adapter qualification does not redefine this Release surface. Cortex's five-field route and `actionStep` JSON are training/evaluation shapes, whereas shipped structured turns use the current constrained `action` or `final` JSON wire contract and leave routing, clarification, manifest validation, approval, and persistence to the native runtime. Offline adapter/GGUF results, including any future verified `603/603` aggregate, do not establish that newly trained artifacts are installed or selected in iOS and do not substitute for device or TestFlight evidence.

| Surface | Release status | Runtime owner | User-visible behavior |
| --- | --- | --- | --- |
| Foreground chat, text turns | Shipped | `AssistantKernel.run(...)` with `LlamaRuntimeAdapter.live(...)` backed by `AppLlamaService` | Uses the local SwiftLlama/AppLlamaService path when a chat model is loaded. Missing or failed models surface typed runtime errors instead of deterministic assistant text. |
| Foreground chat, tool-required turns | Shipped for validated structured and policy-first actions | `AssistantKernel.run(...)` -> `StructuredAgentKernelExecutor` -> `StructuredToolCallValidator` -> `SecureToolRegistry` | Release structured turns use injected runtime readiness, constrained JSON, fail-closed tool visibility, schema validation, approval policy, trusted observations, and bounded finalization. Eligible policy-first turns use the same secure execution boundary. Fresh device evidence is still required for live model-backed parity claims. |
| Voice text-only turns | Shipped | `VoiceCommandRouter` -> `AssistantKernel.run(...)` | Foreground, user-initiated voice turns route through the kernel and cancel on scene transitions. |
| Voice tool-capable turns | Shipped for validated structured and policy-first actions | `VoiceCommandRouter` -> `AssistantKernel.run(...)` -> `StructuredAgentKernelExecutor` / `SecureToolRegistry` | Voice reuses the native kernel tool boundary. Tool calls require fail-closed visibility, schema validation, registry policy, and typed results; missing context or permission remains an explicit failure/degraded state. |
| AppIntents | Shipped only for guarded local actions | `LumenAskIntent`, `LumenAddMemoryIntent`, `LumenMemorySearchIntent`, `LumenRunTriggerIntent` | Intents return degraded or open-app responses when model, memory, approval, or store context is unavailable. They must not claim tool execution without the required context. |
| Trigger and headless execution | Shipped only for background-safe coordination | `HeadlessAgentKernelRunner`, `BackgroundToolExecutionPolicy`, `TriggerScheduler` | Background work cannot prompt for permissions or load model assets. Background-safe tool-only execution is assessed by policy, routed through secure tool definitions, and skipped with explicit diagnostics when required context is missing. |
| Role adapter runtime | Shipped for adapter lifecycle and diagnostics | `ModelFleet`, `SlotModelRuntimeCoordinator`, `AppLlamaService` | Qwen3 shared-base plus role-adapter artifacts are verified before use; missing role adapters fail with typed diagnostics. |
| REM/runtime repair workflows | DEBUG diagnostics only | `RemCycleService`, generated improve-loop artifacts | Not presented as a shipped autonomous assistant capability. |
| RAG search | Shipped | `RAGStore`, `RAGEngine`, `RAGSearchTool` | Empty results, embedding failures, fetch failures, lexical degradation, and stale embedding metadata are surfaced. Vectors carry an atomic content-digest model identity plus format and dimension; incompatible chunks require explicit reindexing. |
| Memory recall/save | Shipped | `MemoryStore`, `MemoryEngine`, `MemorySearchTool`, `MemoryCaptureQueue` | Empty store, embedding failures, save failures, and pending capture queues are distinct states. |
| FoundationModels text runtime | Excluded from Release routing | `FoundationModelsRuntimeAdapter` | Reported as experimental and non-selectable until a real generation implementation exists. |
| CoreML embedding runtime | Excluded from Release routing | `CoreMLRuntimeAdapter` | Reported as experimental and non-selectable until real embedding extraction exists. |
| Deterministic runtime | DEBUG diagnostics only | `DeterministicFallbackRuntime` | Release routing cannot select it. DEBUG tests may select it explicitly. |
| GGUF native engine | Excluded unless a compiled native bridge is supplied | `GGUFEngine`, `GGUFNativeBridge` | The unavailable bridge is DEBUG-only. Release factory registration cannot install an unavailable GGUF backend. |
| Persistent diagnostics and E2E probes | Shipped for native structured/local diagnostics; live legacy probes DEBUG-only | `DiagnosticsProvider`, `PersistentRuntimeDiagnosticsRunner`, `E2ETestRunner` | Release live E2E scenarios enter `AssistantKernel` and the structured executor, preserve strict runtime evidence, and export redacted sidecars joined by opaque correlation tokens. Non-actionable degradation is quarantined; remaining legacy probes are skipped outside DEBUG. |

## Release Readiness Vocabulary

- **Shipped** means the Release app exposes the feature and has production code paths plus tests or diagnostics for failure modes.
- **Excluded from Release** means the Release app does not route user requests through that path.
- **DEBUG diagnostics only** means the path may exist for migration evidence, tests, or developer tools, but must not be user-selectable in Release.
