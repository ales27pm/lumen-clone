# Background Processing

Lumen retains existing `TriggerScheduler` identifiers:
- `com.27pm.lumen.agent.refresh`
- `com.27pm.lumen.agent.process`

`BackgroundOrchestrator` is additive and wraps scheduling/handling:
- `register()` and `schedule()` delegate to `TriggerScheduler`.
- `runTriggerScan()` delegates to `TriggerScheduler.fireDueTriggers`.
- Non-existent workloads (memory consolidation/RAG maintenance/model housekeeping) record explicit `not_available` metrics instead of fake success.

`BackgroundExecutionLease` prevents concurrent workloads by category and auto-expires stale leases.


## Launch wiring status
`BackgroundOrchestrator` is intentionally staged and is not launch-wired as a second background registration path in this PR. Existing app startup continues to register `TriggerScheduler` directly. Wiring both without an Xcode/device validation pass risks duplicate `BGTaskScheduler` registration for the same identifiers, so the orchestrator remains a documented wrapper until the macOS build and on-device background-task validation can confirm a safe handoff.
