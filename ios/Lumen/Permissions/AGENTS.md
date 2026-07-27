# AGENTS.md

## Scope

Governs `ios/Lumen/Permissions/`, permission domains, state, gates, requests, and privacy-safe diagnostics. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

This subtree is the centralized boundary between Lumen and protected iOS capabilities. Tools, voice, photos, contacts, calendars, location, notifications, and related surfaces must use its typed policy rather than infer authorization independently.

## Key Files And Entry Points

- `PermissionDomain.swift`: canonical protected capability categories.
- `PermissionRegistry.swift`: framework-specific status and explicit request integration.
- `PermissionGate.swift`: execution-time allow/deny boundary.
- `PermissionState.swift`, `PermissionRequestResult.swift`: typed outcomes.
- `PermissionDiagnostics.swift`: redacted status snapshots.

## Public Interfaces

Permission-domain identifiers, state transitions, request results, and diagnostics are consumed by tools, voice, views, App Intents, and tests. Callers rely on not-determined, denied, restricted, unavailable, and allowed remaining distinct.

## Internal Structure

A caller identifies a domain -> registry reads current OS state -> an explicitly user-initiated flow may request -> gate returns typed execution permission -> diagnostics expose only status metadata.

## Incoming Dependencies

Secure tools, voice services/controllers, permissions/privacy views, diagnostics, and policy tests call this subtree.

## Outgoing Dependencies

`PermissionRegistry` integrates with AVFoundation, Speech, Contacts, EventKit, Photos, UserNotifications, and other relevant Apple authorization APIs. Usage descriptions/entitlements remain in Xcode configuration, not this directory.

## Data And Control Flow

Action intent -> permission gate -> current state -> optional explicit request -> typed result -> caller either proceeds, clarifies, or reports denial. Background/headless code may inspect but must not surprise-request.

## Local Invariants

- Permission prompts require an explicit user action and appropriate foreground context.
- Never treat unknown/not-determined as granted.
- Preserve each OS state instead of collapsing to Boolean success/failure.
- Diagnostics expose domains/states, not protected content.
- Permission is necessary but not sufficient: tool approval and schema policy still apply.

## Coordinated Changes

A new domain requires registry, gate/state, relevant Info usage description or entitlement, calling tool/service, privacy/permissions UI, diagnostics, and tests. Removing/renaming a domain requires migration review for settings/diagnostic consumers.

## Safe Editing Rules

Keep OS API access centralized. Do not request permission from model-driven, background, or passive diagnostic code. Return typed denial/unavailability instead of empty tool results.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/PermissionGateTests -only-testing:LumenTests/PermissionDiagnosticsTests -only-testing:LumenTests/PermissionDiagnosticsSnapshotTests -only-testing:LumenTests/AssistantPermissionStateTests
```

Also inspect feature-specific policy tests such as `VoicePermissionPolicyTests.swift` and contacts/calendar/location tool tests.

## Common Failure Modes

- A tool asks for authorization during autonomous execution.
- A simulator-only authorization state masks missing usage-description configuration.
- Denied and unavailable are rendered as no data.
- A new protected API bypasses the registry.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). Tool-side enforcement is refined by [`../Tools/AGENTS.md`](../Tools/AGENTS.md).
