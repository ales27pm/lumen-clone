# AGENTS.md

## Scope

Governs `ios/Lumen/AppIntents/`. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

This subtree exposes bounded Lumen operations to App Intents and Shortcuts without creating a second assistant runtime or bypassing native safety policy.

## Key Files And Entry Points

- `LumenAppShortcuts.swift`: shortcut registration/phrases.
- `LumenAskIntent.swift`: headless assistant request.
- `LumenAddMemoryIntent.swift` and `LumenMemorySearchIntent.swift`: memory operations.
- `LumenRunTriggerIntent.swift`: trigger execution boundary.
- `LumenDiagnosticsIntent.swift`: diagnostic surface.
- `LumenIntentPolicy.swift`: allowed headless/local behavior.
- `LumenIntentResultRenderer.swift`: stable result rendering.

## Public Interfaces

Intent identifiers, parameters, result/dialog text, shortcut phrases, and availability are operating-system-facing contracts. They are consumed outside the process by Shortcuts/Siri and may persist in user automations.

## Internal Structure

Each intent is guarded with `canImport(AppIntents)` where required, gathers a valid `ModelContext` from shared app infrastructure when persistence is needed, applies `LumenIntentPolicy`, and delegates to the kernel/store/orchestrator owning the operation.

## Incoming Dependencies

The operating system invokes intents, potentially while no app view is active.

## Outgoing Dependencies

Ask routes through the headless kernel path; memory intents use `MemoryStore`; trigger intent uses the trigger/background policy; diagnostics reads bounded diagnostic summaries. Shared persistence comes from `SharedContainer`.

## Data And Control Flow

OS parameters -> intent validation/policy -> owning kernel/store/service -> renderer -> App Intent result. Missing required input must clarify or fail explicitly, not invent a default action.

## Local Invariants

- Do not perform sensitive side effects directly in an intent implementation.
- Headless contexts cannot present arbitrary approval or permission UI; policy must reject/clarify unsupported operations.
- Intent construction and passive metadata discovery must not load the model.
- Keep stable identifiers and parameter meaning compatible with existing shortcuts.
- Return bounded, privacy-safe text; do not expose raw diagnostics or stored content unexpectedly.

## Coordinated Changes

Changing an intent identifier/parameter requires `LumenAppShortcuts.swift`, policy, renderer, tests, and release notes/status review. Memory schema changes require Models/Memory/Services coordination. Trigger changes require `Background/` and `Services/TriggerScheduler.swift` review.

## Safe Editing Rules

Add an intent only for an operation with an existing safe owner. Reuse kernel/store APIs and typed policy outcomes. Do not duplicate chat orchestration, permission prompts, or tool execution in this subtree.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/LumenIntentPolicyTests -only-testing:LumenTests/LumenAskIntentPolicyTests -only-testing:LumenTests/LumenAddMemoryIntentPolicyTests -only-testing:LumenTests/LumenRunTriggerIntentPolicyTests
```

Also review `LumenAppShortcutsTests.swift`, `LumenMemorySearchIntentTests.swift`, and `LumenIntentResultRendererTests.swift` when their surfaces change.

## Common Failure Modes

- A Shortcut-visible identifier changes silently.
- An intent bypasses kernel/tool policy because it appears local.
- A background invocation tries to present permission/approval UI.
- A missing shared `ModelContainer` is converted into empty success.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). No child guidance is needed.
