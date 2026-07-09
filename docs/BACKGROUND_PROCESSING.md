# Background Processing

Lumen retains existing `TriggerScheduler` identifiers:
- `com.27pm.lumenclone.agent.refresh`
- `com.27pm.lumenclone.agent.process`

`BackgroundOrchestrator` is additive and wraps scheduling/handling:
- `register()` and `schedule()` delegate to `TriggerScheduler`.
- launch startup calls the orchestrator facade, while `TriggerScheduler` remains the single owner of BGTaskScheduler identifiers and registration.
- BG refresh and processing callbacks delegate to `BackgroundOrchestrator` so leases, metrics, and policy checks are applied in one place.
- `runTriggerScan()` delegates to `TriggerScheduler.fireDueTriggers` without requiring the chat runtime to be loaded.
- BG processing runs trigger scan first, then attempts runtime self-improvement snapshot maintenance, local memory dedupe, current RAG maintenance, and model housekeeping while the short background deadline remains.
- BG task expiration cancels the in-flight trigger/maintenance task and completes the BG task exactly once. Cancellation or expiration is recorded as failed completion; policy skips and local-only no-op outcomes complete successfully.
- background trigger scans may run background-safe local tool-only prompts, but they do not authorize loading model assets.
- `BackgroundToolExecutionPolicy` produces a structured assessment before tool-only execution. Requests that need clarification, external/web tools, foreground approval, missing permissions, or unsupported tool mappings are skipped with an explicit reason instead of falling through to background model work.
- background tool execution denies/degrades when permission is missing or not yet determined; it never initiates foreground permission prompts from a background task.
- deterministic local maintenance, such as memory dedupe and current RAG maintenance, is governed by `BackgroundTaskPolicy` instead of the heavy-model gate and records skip metrics when the budget or container is unavailable.
- runtime self-improvement maintenance is local-only: it builds a bounded `SelfModelSnapshot`, runs already-safe maintenance/compaction work off the UI actor, records redacted metrics, and never uploads, trains, writes generated datasets, mutates model weights, or authorizes model loading.
- `BackgroundTaskPolicy` treats runtime self-improvement as zero-token work with `allowModelLoading=false`; the policy uses the persisted agent-mode setting as the background-agent enablement gate instead of silently forcing background agents on.
- model housekeeping is a background-safe cleanup pass that unloads optional chat slots and records which slots were released; it never authorizes model loading.
- app refresh and processing entrypoints record `shared_container_unavailable` metrics when SwiftData is unavailable instead of returning silently.

`BackgroundExecutionLease` prevents concurrent workloads by category and auto-expires stale leases.


## Launch wiring status
`BackgroundOrchestrator` is launch-wired through a single delegation path: startup calls the orchestrator facade, the facade calls `TriggerScheduler.registerTasks()`, and the registered BGTaskScheduler callbacks call back into the orchestrator handlers. This avoids duplicate BGTaskScheduler registration while making the orchestrator the policy/lease/metrics boundary for background task handling.

Device validation is still required for actual iOS background scheduling behavior. Generic simulator builds prove compilation only; they do not prove that iOS will grant, launch, or keep alive a background task on a physical device.
