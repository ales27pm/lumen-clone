# AGENTS.md

## Scope

Governs `ios/Lumen/Views/`, SwiftUI screens/components and chat event reduction. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

Views present state and user actions for chat, memory, sources, models, tools, permissions, diagnostics, Outlook, triggers, settings, onboarding, and voice. They adapt owning services/controllers; they do not own model/tool/persistence policy.

## Key Files And Entry Points

- `RootView.swift` and `ChatHomeView.swift`: primary navigation/root presentation.
- `ChatView.swift`: kernel request lifecycle, SwiftData conversation/message interaction, cancellation, and deferred save.
- `ChatKernelEventReducer.swift`: event-to-chat-state contract.
- `VoiceModeView.swift`: voice controller presentation.
- `SettingsView.swift`, `ModelsView.swift`, `PermissionsView.swift`, `SourcesView.swift`, `ToolsView.swift`: configuration/status surfaces.
- `DiagnosticsView.swift`, `GroundingDiagnosticsView.swift`, `RuntimeDashboardView.swift`: evidence/diagnostic presentation.

## Public Interfaces

User-visible navigation, labels, accessibility semantics already present, reducer state, bindings, and action callbacks are consumed by users, UI tests, previews, and controllers. `ChatKernelEventReducer` behavior is also a kernel contract consumer.

## Internal Structure

Views observe/model local presentation state, submit operations to owning kernel/controller/service, reduce typed events, and render explicit readiness/failure/recovery states. SwiftData access remains on the appropriate main-actor `ModelContext`.

## Incoming Dependencies

SwiftUI scenes and users drive these views. Kernel/voice/service events update them.

## Outgoing Dependencies

Views call Assistant, Voice, Developer, Permissions, stores/services, and SwiftData. They may access environment values but must not instantiate duplicate production owners.

## Data And Control Flow

User action -> view validation -> owning API -> async events/results -> reducer/main-actor state -> rendered state -> bounded persistence. Cancellation leaves a terminal UI state and prevents late overwrite.

## Local Invariants

- Keep business/tool/runtime policy outside views.
- `ChatView` routes assistant work through `AssistantKernel`; no direct model/tool fallback.
- Reducers handle cancellation/failure/finalization consistently and ignore invalid late events.
- Show explicit unavailable/denied/loading/recovery states rather than vague fallback success.
- Do not surface raw diagnostic secrets/private content.
- Preserve existing design language and mobile layout behavior when adding UI.

## Coordinated Changes

Kernel event changes require both chat and voice reducers/tests. Model/store changes require owning subsystem and persistence review. Permission/tool/model settings changes require their typed APIs and tests, not view-only workarounds.

## Safe Editing Rules

Use small composable views and existing state ownership. Do not perform blocking I/O/model work in `body` or lifecycle callbacks. Avoid new singleton construction in views. Keep privacy/accessibility behavior at least as strong as adjacent components.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/ChatKernelEventReducerTests -only-testing:LumenTests/ToolExecutionPresentationTests -only-testing:LumenTests/PrivacyReportTests
```

Use `ios/LumenUITests/` for an explicitly requested end-to-end UI flow. Unit/reducer success does not prove a visual walkthrough.

## Common Failure Modes

- A view creates a second service/kernel instance.
- Late events overwrite cancelled state.
- A broad fallback label hides permission/runtime failure.
- SwiftData or model work blocks the main actor during rendering/scene change.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). Developer-only views under `ios/Lumen/Developer/` follow [`../Developer/AGENTS.md`](../Developer/AGENTS.md).
