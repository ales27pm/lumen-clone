# AGENTS.md

## Scope

Governs `ios/Lumen/Developer/`, the in-app developer console/control tower and evidence-framework presentation. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

This subtree lets developers inspect diagnostics, launch bounded developer workflows, and distinguish static, simulator, live model, and device evidence. It must report runtime state honestly and remain separate from production assistant ownership.

## Key Files And Entry Points

- `DeveloperFramework.swift`: workflow/evidence-layer definitions and result semantics.
- `DeveloperConsoleView.swift`: developer action/diagnostic UI.
- `LumenControlTowerView.swift`: aggregated developer status UI.
- `ios/Lumen/Services/E2ETestRunner.swift`: live E2E producer consumed here.
- `ios/Lumen/Services/Diagnostics/EvidenceLayerExporter.swift`: exported evidence envelope.

## Public Interfaces

Developer action identifiers, evidence-layer/result models, export formats, and status labels are consumed by views, tests, and external artifact analysis. A green label is a semantic contract and must identify what actually ran.

## Internal Structure

Views call `DeveloperFramework`, which delegates to bounded diagnostics/E2E/export services. The framework classifies results rather than manufacturing runtime success.

## Incoming Dependencies

Developer-only UI and diagnostic intents invoke this subsystem.

## Outgoing Dependencies

It reads runtime diagnostics, manifest audits, E2E results, persistent trace summaries, and evidence exporters. It may request a real kernel run through the existing E2E owner but does not implement inference.

## Data And Control Flow

Developer action -> framework operation -> static/runtime/E2E producer -> typed evidence result -> redacted presentation/export. Layer identity and correlation metadata travel with the result.

## Local Invariants

- Static manifest audit is not model-backed evidence.
- Compile, simulator, live E2E, device runtime, signed archive, and upload remain distinct statuses.
- Do not turn missing trace/model correlation into success.
- UI copy must not expose raw prompts, memory, documents, tool arguments, or OAuth data.
- Developer actions remain bounded and cancellable.

## Coordinated Changes

Evidence schema changes require the exporter, E2E runner, persistent diagnostics, generated/runtime audit consumers, tests, and `docs/` evidence contracts. New actions must identify their execution layer and failure semantics.

## Safe Editing Rules

Use existing typed evidence models and explicit unavailable/failed states. Do not infer success from file presence or green-looking generated summaries. Keep developer-only behavior from influencing production routing.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/DeveloperFrameworkTests -only-testing:LumenTests/E2ETestRunnerHygieneTests -only-testing:LumenTests/PersistentRuntimeDiagnosticsTests
```

## Common Failure Modes

- A manifest comparison is labeled as live model validation.
- A stale audit artifact is presented as current.
- Missing correlated trace data is ignored.
- Developer UI directly invokes a fallback runtime unavailable in Release.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). Evidence export details are refined by [`../Services/Diagnostics/AGENTS.md`](../Services/Diagnostics/AGENTS.md); grounding trace details by [`../Services/AgentGrounding/AGENTS.md`](../Services/AgentGrounding/AGENTS.md).
