# AGENTS.md

## Scope

Governs `ios/LumenTests/`, the XCTest unit/integration target for app contracts. Parent rules: [`../AGENTS.md`](../AGENTS.md).

## Role In The System

These tests protect runtime routing, kernel/tool contracts, permissions, persistence, memory/RAG, background, voice, diagnostics, and hardening behavior. They may exercise app-internal symbols; production code must never depend on them.

## Key Files And Entry Points

- `AssistantKernelRunContractTests.swift`, `AssistantKernelStructuredAgentTests.swift`, `AgentKernelBoundaryGuardTests.swift`: kernel ownership/behavior.
- `ToolSchemaBridgeTests.swift`, `SecureToolRegistryTests.swift`, `ToolApprovalBoundaryTests.swift`, `ToolRegistryCoverageTests.swift`: tool safety.
- `PersistenceAuditTests.swift`, memory and RAG tests: storage/retrieval contracts.
- `BackgroundTaskNoModelLoadTests.swift`, `BackgroundTaskRegistrationTests.swift`: background invariants.
- Voice lifecycle/audio/reducer tests: voice ownership and cancellation.
- `E2ETestRunnerHygieneTests.swift` and grounding/diagnostic tests: evidence/privacy semantics.
- `ios/Lumen.xcodeproj/xcshareddata/xcschemes/Lumen.xcscheme`: target execution configuration.

## Public Interfaces

Test class/method names are focused-run selectors and regression documentation. Fixtures/helpers encode expected contract shape but are not production defaults.

## Internal Structure

Tests instantiate focused owners or controlled doubles, drive explicit inputs/events, and assert typed state/results. Static guard tests inspect source/configuration where runtime execution is not appropriate.

## Incoming Dependencies

Xcode/xcodebuild and focused runner scripts discover this target through the project/scheme and synchronized groups.

## Outgoing Dependencies

The tests depend on the `Lumen` target and Apple simulator frameworks. Static tests may read repository files relative to a known root.

## Data And Control Flow

Build-for-testing compiles app/tests -> `.xctestrun` identifies test bundle -> focused/full execution launches simulator host as needed -> XCTest records result. Collection/build success and executed success are separate.

## Local Invariants

- Do not delete, skip, weaken, broaden tolerances, or replace assertions to hide a production failure.
- Test-only deterministic doubles never become Release routing options.
- Keep regression names and assertions tied to the contract/failure they prove.
- Avoid external network, account, real model, or timing dependence in ordinary unit tests.
- Preserve privacy in fixtures and failure messages; use synthetic content.
- A static source scan is not runtime proof, and a simulator double is not model-backed proof.

## Coordinated Changes

Update tests with the owning production contract, not before it. New tools need registry/schema/approval/permission coverage. New persistent fields need existing-store/audit coverage. Kernel events need chat and voice reducer tests. Runtime backends need DEBUG-vs-Release guard tests.

## Safe Editing Rules

Prefer focused deterministic tests and existing helpers. Add a regression for the real invariant and failure path, including cancellation/error branches. Do not create production hooks solely to bypass ownership unless the hook is a narrow, safe dependency injection boundary.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj \
  -scheme Lumen \
  -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' \
  -derivedDataPath build/DerivedData-FocusedSimulatorTests \
  build-for-testing \
  CODE_SIGNING_ALLOWED=NO
```

For bounded focused execution, invoke the runner directly:

```bash
bash scripts/run_focused_simulator_tests.sh
```

The runner defaults to the same DerivedData path, but it still performs its own incremental `build-for-testing` before `test-without-building`; the standalone command is a compile checkpoint, not a no-build prerequisite.

For a single class, use the verified Xcode selector form:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/AssistantKernelRunContractTests
```

## Common Failure Modes

- Discarding the focused DerivedData cache between runner invocations and forcing avoidable full rebuilds.
- Simulator install/test-manager stall is reported as a test failure or, worse, success.
- A DEBUG fallback makes a Release invariant test meaningless.
- A test validates fixture text instead of actual runtime correlation.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). Production subsystem files provide the local contract for each test. `ios/LumenUITests/` inherits the parent `ios/AGENTS.md` because its setup is conventional and no separate invariant was found.
