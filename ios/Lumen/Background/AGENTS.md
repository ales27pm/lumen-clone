# AGENTS.md

## Scope

Governs `ios/Lumen/Background/`. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

This subtree owns bounded background execution policy, leases, entitlement checks, continuation, and orchestration. It performs maintenance and trigger scanning under iOS lifecycle limits; it does not own model inference.

## Key Files And Entry Points

- `BackgroundOrchestrator.swift`: main-actor coordinator using the shared `ModelContext`.
- `BackgroundTaskPolicy.swift`: permitted background operations.
- `BackgroundExecutionLease.swift`: lifetime/expiration contract.
- `BackgroundContinuedProcessingCoordinator.swift` and `BackgroundRuntimeContinuation.swift`: continuation handling.
- `BackgroundEntitlementValidator.swift`: capability diagnostics.
- `ios/Lumen/Services/TriggerScheduler.swift`: BGTask identifier registration/submission and callbacks into this subtree.

## Public Interfaces

Registration callbacks, operation classifications, lease/expiration behavior, and background diagnostic outcomes are consumed by app startup, scheduler code, App Intents, and tests.

## Internal Structure

`TriggerScheduler` registers exact identifiers with `BGTaskScheduler`. A callback enters `BackgroundOrchestrator`, which checks policy/resources, acquires a lease, performs bounded maintenance/trigger work, and completes or cancels on expiration.

## Incoming Dependencies

`LumenApp.swift`, `TriggerScheduler`, scene/lifecycle callbacks, and `LumenRunTriggerIntent` initiate work.

## Outgoing Dependencies

The orchestrator uses SwiftData through `ModelContext`, system resource/cancellation policy, notifications/trigger stores, and bounded maintenance services. It must not call model generation or permission UI.

## Data And Control Flow

BGTask/intent trigger -> entitlement/policy/resource check -> lease -> maintenance or trigger scan -> completion. Expiration -> cancellation -> lease completion; no late persistence or user-visible success after cancellation.

## Local Invariants

- Background startup and execution never load or prompt the language model.
- Work is finite, cancellable, idempotent where retried, and reports completion exactly once.
- Do not request new permission or approval from a background callback.
- Keep BGTask identifiers synchronized with registration/configuration and tests.
- Resource preflight degradation is explicit and non-actionable; do not fabricate training/runtime evidence.

## Coordinated Changes

Identifier or scheduling changes require `Services/TriggerScheduler.swift`, `LumenApp.swift`, entitlements/Info configuration, App Intent policy, and registration tests. Store work requires the owning model/store and persistence audit. Resource rules require `System/` tests.

## Safe Editing Rules

Keep heavy work outside the main actor while preserving main-actor `ModelContext` ownership. Use leases and cancellation rather than detached untracked tasks. Do not turn background processing into an alternate assistant path.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/BackgroundTaskNoModelLoadTests -only-testing:LumenTests/BackgroundTaskRegistrationTests -only-testing:LumenTests/BackgroundTaskPolicyTests -only-testing:LumenTests/BackgroundExecutionLeaseTests
```

Also inspect `BackgroundEntitlementValidatorTests.swift`, `BackgroundMaintenancePolicyTests.swift`, and `ResourceBudgetGateTests.swift`.

## Common Failure Modes

- Registration succeeds in a test but the identifier/configuration differs in the app target.
- Expired work reports success or writes late state.
- Maintenance accidentally constructs/loads a model service.
- A retry duplicates a notification or trigger side effect.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). `Services/TriggerScheduler.swift` remains governed by [`../Services/AGENTS.md`](../Services/AGENTS.md); inspect both files for scheduling changes.
