# AGENTS.md

## Scope

This file governs `ios/`: the Xcode project, app target, unit-test target, UI-test target, scheme, entitlements, bundled resources, and iOS-specific scripts. It refines [`../AGENTS.md`](../AGENTS.md); child files provide subsystem rules.

## Role In The System

`ios/Lumen/` is the shipping native application. `ios/LumenTests/` and `ios/LumenUITests/` validate it. `ios/Lumen.xcodeproj/` defines one app target (`Lumen`), one unit target (`LumenTests`), and one UI target (`LumenUITests`) with an iOS 18.0 deployment target and Xcode-managed Swift package dependencies.

## Key Files And Entry Points

- `ios/Lumen/LumenApp.swift`: `@main`, SwiftData schema/container, bootstrap, background registration, scene handling, and URL handling.
- `ios/Lumen/LumenAppDelegate.swift`: MetricKit, lifecycle callbacks, and Microsoft authentication callback routing.
- `ios/Lumen.xcodeproj/project.pbxproj`: targets, packages, build settings, entitlements, generated Info.plist keys, and synchronized groups.
- `ios/Lumen.xcodeproj/xcshareddata/xcschemes/Lumen.xcscheme`: shared build/test actions and post-build grounding-resource copy.
- `ios/Lumen/Lumen.entitlements` and `ios/Lumen/LumenAppStore.entitlements`: capability declarations for their configurations.
- `ios/Lumen/Scripts/copy_agent_grounding_resources.sh`: build-time verification/copy of generated grounding resources.

## Public Interfaces

The app exposes SwiftUI scenes, URL callbacks, App Intents/Shortcuts, optional CarPlay integration, background task registrations, local notifications, and native tool behavior. Kernel contracts, tool IDs/schemas, persistent model shapes, App Intent identifiers, BGTask identifiers, URL schemes, and entitlements are compatibility surfaces outside their declaring files.

## Internal Structure And Dependency Direction

- Views, voice, App Intents, CarPlay, and developer surfaces call `AssistantKernel` contracts.
- `Assistant/` calls services, grounding, memory/RAG, and secure tools; it must not import presentation ownership into the kernel.
- `Tools/` calls permission/platform adapters through typed execution context.
- `Models/` supplies value and SwiftData types; stores own persistence operations.
- `System/` and `Background/` coordinate cancellation/resource/lifecycle policy without owning inference.
- Direct files under `Services/` own shared runtime/platform/store instances; child service directories refine specialized contracts.

## Incoming And Outgoing Dependencies

Incoming callers are iOS/Shortcuts/CarPlay lifecycle events and test targets. Outgoing dependencies include SwiftUI, SwiftData, BackgroundTasks, AVFoundation, Speech, Contacts, EventKit, Photos, PDFKit, Security, AuthenticationServices, MetricKit, URLSession, SwiftLlama/llama.cpp, Metal, and MSAL when linked. Optional frameworks remain guarded where source uses `canImport`.

## Data And Control Flow

App launch creates persistence and lightweight coordinators without loading the model. User or headless requests enter the kernel, which selects a Release-safe runtime, builds bounded grounding, validates structured actions, executes approved tools, and emits events to surface-specific reducers. Stores persist state through their `ModelContext`; scene/background cancellation propagates through coordinators and `AppCancellationBus`.

## Local Invariants

- Preserve one production kernel path. Do not reintroduce a view-owned or service-owned alternate agent loop.
- Keep Release routing fail closed and DEBUG-only backends behind compile-time guards.
- Do not model-load during launch, background maintenance, scene transition, or passive diagnostics.
- Main-actor and actor annotations express ownership. Do not silence isolation errors with unchecked shared mutable state.
- Every tool action is schema-validated and permission/approval-gated before side effects.
- Keep raw user/model/tool content out of normal logs and diagnostics.
- Treat persistence, audio, BGTask, and URL callback lifecycles as explicit resources with cancellation/cleanup.
- A successful build is not evidence of executed tests, live model use, real-device behavior, signing, or upload.

## Coordinated Changes

- New Swift files are generally discovered by synchronized groups, but resources, build phases, packages, entitlements, URL/Info settings, and target membership still require project/scheme review.
- Changing an `@Model` requires reviewing `LumenApp.swift`, owning stores, diagnostics/export, and `PersistenceAuditTests.swift`. No tracked versioned migration plan was found.
- Changing a tool requires the model catalog, secure registry, schema bridge, route guard, manifest generation/resource mirror, and tests.
- Changing model/runtime code requires routing, hardening scripts, package pin/linkage, memory budgets, cancellation tests, and status docs.
- Changing background IDs, App Intent IDs, URL schemes, or entitlements requires checking operating-system registration/configuration as well as source.

## Safe Editing Rules

Prefer typed errors, explicit state machines, small actor-owned services, and existing reducer patterns. Do not add broad catches returning empty success. Do not directly edit generated `ios/Lumen/AgentBehaviorManifest.json`. Preserve the exact SwiftLlama package pin unless a deliberate runtime migration updates checks and evidence together. Do not add Release code that depends on the tracked GGUF header alone; no Release native implementation is present in this tree.

## Validation

From the repository root, compile first:

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

Use the full `xcodebuild ... test` command from the root file only when needed and CoreSimulator is healthy. Runtime/backend changes also require:

```bash
python3 tools/check_release_hardening.py
python3 tools/check_agent_kernel_boundary.py --strict
python3 tools/check_adapter_runtime_invariants.py
python3 tools/check_ios_lora_hardening_invariants.py
bash scripts/check-lumen-integration-gate.sh
```

## Common Failure Modes

- Tests pass through a DEBUG fallback that cannot exist in Release.
- A synchronized source file compiles, but a required package/resource/build phase is not linked.
- `build-for-testing` is reported as executed XCTest.
- Main-thread model work causes launch or scene watchdog failures.
- A new persistent property launches on a clean simulator but has no compatibility plan for existing stores.
- A sensitive tool asks for permission or approval from a background/headless context.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md).

Child guidance: [`Lumen/Assistant/AGENTS.md`](Lumen/Assistant/AGENTS.md), [`Lumen/AppIntents/AGENTS.md`](Lumen/AppIntents/AGENTS.md), [`Lumen/Background/AGENTS.md`](Lumen/Background/AGENTS.md), [`Lumen/CarPlay/AGENTS.md`](Lumen/CarPlay/AGENTS.md), [`Lumen/Developer/AGENTS.md`](Lumen/Developer/AGENTS.md), [`Lumen/Memory/AGENTS.md`](Lumen/Memory/AGENTS.md), [`Lumen/Models/AGENTS.md`](Lumen/Models/AGENTS.md), [`Lumen/Permissions/AGENTS.md`](Lumen/Permissions/AGENTS.md), [`Lumen/RAG/AGENTS.md`](Lumen/RAG/AGENTS.md), [`Lumen/Services/AGENTS.md`](Lumen/Services/AGENTS.md), [`Lumen/System/AGENTS.md`](Lumen/System/AGENTS.md), [`Lumen/Tools/AGENTS.md`](Lumen/Tools/AGENTS.md), [`Lumen/Views/AGENTS.md`](Lumen/Views/AGENTS.md), [`Lumen/Voice/AGENTS.md`](Lumen/Voice/AGENTS.md), and [`LumenTests/AGENTS.md`](LumenTests/AGENTS.md). `LumenUITests/` inherits this file because no distinct setup or invariant was found.
