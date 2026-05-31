# ModelContext Injection Audit

## Interactive generation call sites

| File / function | Has `@Environment(\.modelContext)` | Explicitly passed to service | SharedContainer fallback used | Degraded fallback possible | Status |
|---|---:|---:|---:|---:|---|
| `ChatView.runAgent(...)` | yes | yes (`LegacyAgentRunOptions.modelContext`) | yes (provider fallback if nil) | yes | migrated |
| `VoiceModeView.runAgent(...)` | yes | yes (`LegacyAgentRunOptions.modelContext`) | yes | yes | migrated |
| `TriggersView.runNow(...)` manual trigger | yes | yes (direct `context:` to scheduler/runner path) | n/a | no (context is explicit) | migrated |
| `AgentRunner.runHeadless(...)` | caller-provided | yes (`context` param) | no SharedContainer fallback; caller must provide a live context | no (context is required) | migrated |
| `RootView` direct generation launch | n/a | n/a | n/a | n/a | no direct launch path |

## Notes
- Interactive legacy services retain safe degraded behavior when direct UI `ModelContext` is unavailable by resolving through `LegacyGroundingContextProvider` / `SharedContainer` where available.
- `AgentRunner.runHeadless(...)` does not resolve `SharedContainer`; it requires an explicit caller-provided `ModelContext` and now routes legacy grounding through `LegacyTurnGroundingCoordinator`.
- Primary UI paths prefer direct context from SwiftUI where available.
