# AGENTS.md

## Scope

Governs `ios/Lumen/System/`: cancellation, compute/resource policy, pressure/thermal/power monitoring, runtime metrics, disk-write budgets, deferred maintenance, capability snapshots, entitlements, and scene-transition coordination. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

This subtree supplies cross-cutting runtime safety signals and lifecycle coordination. It can defer/cancel work but does not own model inference, persistence, background tasks, or UI.

## Key Files And Entry Points

- `AppCancellationBus.swift`: app-wide cancellation propagation.
- `ResourceBudgetGate.swift`, `ComputePolicy.swift`, `CPUWatchdogGuard.swift`: admission/degradation policy.
- `SceneTransitionCoordinator.swift`: bounded nonblocking scene handling and cancellation.
- `MemoryPressureMonitor.swift`, `ThermalStateMonitor.swift`, `PowerModeMonitor.swift`: operating conditions.
- `DiskWriteBudget.swift`, `DeferredMaintenanceQueue.swift`: bounded deferred writes/work.
- `RuntimeMetricsStore.swift`, `RuntimeMetric.swift`: privacy-safe runtime metrics.
- `RuntimeEntitlements.swift`: capability diagnostics.

## Public Interfaces

Cancellation events, resource decisions, capability/pressure snapshots, metric categories, scene transition callbacks, and disk-write admission are consumed by startup, kernel, model services, background, voice, diagnostics, and tests.

## Internal Structure

Platform monitors produce snapshots -> compute/resource gates classify work -> callers admit/defer/reject -> cancellation bus and scene coordinator terminate incompatible work -> metrics record metadata. Ownership remains with the subsystem performing the work.

## Incoming Dependencies

App lifecycle, model/runtime services, background orchestration, voice/CarPlay, stores, and diagnostics query or subscribe to this layer.

## Outgoing Dependencies

Foundation/ProcessInfo, UIKit scene state, metric/notification APIs, and lightweight persistence/queues. This layer must not load models or read private content.

## Data And Control Flow

OS state/scene event -> normalized snapshot -> policy decision/cancellation -> caller cleanup/defer -> redacted metric. Scene callbacks return quickly; expensive cleanup is cancellable/deferred.

## Local Invariants

- Scene transitions are nonblocking; debug watchdog checks must remain meaningful.
- Cancellation is broadcast once per event and callers treat it as terminal for active work.
- Resource degradation is explicit; do not hide serious thermal/memory pressure behind success.
- Metrics contain no raw prompt, memory, document, tool argument, or token content.
- Budget checks do not create alternate ownership or silently drop required persistence.

## Coordinated Changes

Resource policy changes require LLM/model loader, background, memory-pressure, and focused tests. Cancellation changes require kernel, voice, CarPlay, scene, generation, and background tests. Entitlement metrics require project entitlements and docs review.

## Safe Editing Rules

Keep monitor callbacks lightweight and race-safe. Use immutable snapshots and typed decisions. Do not block main-actor lifecycle callbacks or add detached tasks without cancellation/ownership.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/ResourceBudgetGateTests -only-testing:LumenTests/ComputePolicyTests -only-testing:LumenTests/SceneWatchdogHardeningTests -only-testing:LumenTests/ScenePhaseCancellationTests
```

Also inspect `DeviceCapabilityProfilerTests.swift`, `MemoryPressurePolicyTests.swift`, `DiskWriteGenerationGateTests.swift`, and `RuntimeMetricsStoreTests.swift`.

## Common Failure Modes

- Main-actor scene handling waits for model shutdown.
- Cancellation is observed after a late final/tool event.
- Resource policy uses stale state or ignores serious thermal pressure.
- Metrics accidentally contain payload content.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). Background lease policy is in [`../Background/AGENTS.md`](../Background/AGENTS.md); model policy is in [`../Services/LLM/AGENTS.md`](../Services/LLM/AGENTS.md).
