# AGENTS.md

## Scope

Governs `ios/Lumen/Services/Diagnostics/`, currently the evidence-layer exporter. Parent rules: [`../AGENTS.md`](../AGENTS.md).

## Role In The System

This boundary serializes typed validation/evidence results for developer workflows without conflating evidence layers or exposing private runtime content.

## Key Files And Entry Points

- `EvidenceLayerExporter.swift`: evidence envelope and export operation.
- `ios/Lumen/Developer/DeveloperFramework.swift`: producer/orchestrator of classified evidence.
- `ios/Lumen/Services/E2ETestRunner.swift`: live E2E source.
- `ios/Lumen/Services/AgentGrounding/AgentBehaviorTrace.swift`: redacted trace source.

## Public Interfaces

Envelope schema/version, evidence-layer identity, correlation metadata, timestamps, hashes, status, and failure taxonomy are consumed by exported files, audits, tooling, docs, and tests.

## Internal Structure

A typed framework/E2E/diagnostic result is mapped to an explicit evidence layer, privacy-filtered, encoded, and written. The exporter does not upgrade a result's evidentiary strength.

## Incoming Dependencies

Developer framework/control-tower actions call the exporter.

## Outgoing Dependencies

It consumes typed diagnostic models and file writing APIs. External tooling and humans consume the resulting envelope.

## Data And Control Flow

Typed result -> layer-preserving envelope -> privacy/hygiene checks -> atomic export -> external analysis. Encoding/write failures remain failures.

## Local Invariants

- Preserve distinctions among static audit, simulator test, live E2E, device runtime, and release evidence.
- Do not include raw prompt/document/memory/tool/OAuth content.
- Do not mark missing correlation, empty model stream, or fallback text as model-backed success.
- Schema changes are versioned and coordinated with readers.

## Coordinated Changes

Envelope changes require DeveloperFramework, E2E/persistent diagnostics, generated/runtime audit ingestion, documentation, and tests. Path/atomic-write changes require file-protection and cleanup review.

## Safe Editing Rules

Keep this exporter a serialization boundary, not a validation engine. Accept already typed evidence and preserve its status. Avoid permissive decoding/defaults that turn unknown fields into success.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/DeveloperFrameworkTests -only-testing:LumenTests/E2ETestRunnerHygieneTests -only-testing:LumenTests/PersistentRuntimeDiagnosticsTests
```

No dedicated exporter-only test file was found; these consumer/hygiene tests are the nearest verified coverage.

## Common Failure Modes

- Exported `success` means only encoding succeeded, not the underlying validation.
- Unknown evidence is defaulted to the strongest layer.
- Raw model/tool text leaks into an envelope.
- A reader and writer silently disagree on schema version.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). Evidence presentation: [`../../Developer/AGENTS.md`](../../Developer/AGENTS.md). Grounding trace: [`../AgentGrounding/AGENTS.md`](../AgentGrounding/AGENTS.md).
