# AGENTS.md

## Scope

Governs `ios/Lumen/Services/AgentGrounding/`, behavior-manifest loading/auditing, grounding resources, and redacted behavior traces. Parent rules: [`../AGENTS.md`](../AGENTS.md).

## Role In The System

This subtree binds deterministic generated agent metadata to runtime behavior and records privacy-safe trace evidence. It does not generate the source manifest and it does not itself prove a model-backed run.

## Key Files And Entry Points

- `AgentManifestStore.swift`: bundled/stored `AgentBehaviorManifest` loading.
- `RuntimeManifestAuditor.swift`: comparison of live registration to manifest expectations.
- `AgentBehaviorTrace.swift`: correlation, hashes/counts, timing, and redacted persisted trace.
- `generated/agent_manifest/AgentBehaviorManifest.json`: canonical generated source.
- `ios/Lumen/AgentBehaviorManifest.json`: sync-managed app mirror.
- `ios/Lumen/Scripts/copy_agent_grounding_resources.sh`: build bundle verification/copy.

## Public Interfaces

Manifest schema/version/hash, audit result categories, trace event taxonomy, correlation IDs, and redaction semantics are consumed by the kernel, E2E runner, developer framework, generated audits, and tests.

## Internal Structure

Build-time generator emits deterministic manifest -> sync script updates app mirror -> Xcode build copies/verifies bundled resources -> store loads manifest -> auditor compares runtime registration -> kernel/E2E records redacted correlated trace.

## Incoming Dependencies

`AssistantKernel`, `E2ETestRunner`, developer diagnostics, and app bootstrap/loaders use this code.

## Outgoing Dependencies

It reads bundled/local resource files, live tool/runtime registries, diagnostic persistence, and generated schema models. It must not call Python generation at runtime.

## Data And Control Flow

Manifest bytes -> validated decode/hash -> runtime comparison/grounding context. Runtime events -> redacted hash/count metadata -> detached persistence/export -> evidence consumers.

## Local Invariants

- Do not edit the app manifest mirror by hand.
- A manifest audit proves registration/shape agreement only, not live model/tool execution.
- Trace records must not contain raw prompts, documents, memories, arguments, observations, or final text.
- Correlation and primary-trace selection must fail closed when model-backed evidence is required.
- Generated-loop output is not runtime evidence.

## Coordinated Changes

Schema changes require crawler models/writer, canonical generated artifact, sync/copy scripts, app mirror, store/auditor, E2E/report readers, and regression tests. Trace taxonomy changes require evidence exporter, persistent diagnostics, developer UI, and docs.

## Safe Editing Rules

Keep decode/audit failures typed. Do not synthesize a runtime manifest in Release to hide a missing resource. Preserve deterministic hashing and privacy-safe fields.

## Validation

From the repository root:

```bash
cmp -s generated/agent_manifest/AgentBehaviorManifest.json ios/Lumen/AgentBehaviorManifest.json
bash scripts/check-lumen-integration-gate.sh
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/AgentGroundingRegressionTests -only-testing:LumenTests/AgentModeGroundingHardeningTests -only-testing:LumenTests/GroundingDiagnosticsTests
```

The byte comparison is non-mutating and verifies current mirror parity. The integration gate remains the authoritative checked-in consistency command.

## Common Failure Modes

- Canonical generated manifest and app mirror drift.
- A missing/corrupt manifest falls back to synthetic success.
- Trace content violates privacy despite a safe field name.
- An uncorrelated trace is selected as model evidence.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). Generator rules: [`../../../../tools/lumen_manifest_crawler/AGENTS.md`](../../../../tools/lumen_manifest_crawler/AGENTS.md). Generated rules: [`../../../../generated/AGENTS.md`](../../../../generated/AGENTS.md).
