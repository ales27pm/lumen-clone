# AGENTS.md

## Scope

Governs `ios/Lumen/Tools/`, secure native tool definitions, registration, schema bridging, approval policy, execution context/results, output limits, metrics, and built-in tool adapters. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

This subtree is the only production side-effect boundary for model-selected native tools. It turns a validated canonical invocation into a permission/approval-gated local action and bounded observation.

## Key Files And Entry Points

- `ToolRegistry.swift`: `SecureToolRegistry` and native tool-group registration/execution.
- `ToolSchemaBridge.swift`: manifest/tool schema conversion and exact structured-call validation.
- `ToolID.swift`, `SecureToolDefinition.swift`, `LocalTool.swift`: canonical identity and implementation contracts.
- `ToolApprovalPolicy.swift`: approval requirement.
- `ToolExecutionContext.swift`, `ToolInvocation.swift`, `ToolResult.swift`, `ToolExecutionError.swift`: execution API.
- `SafeToolOutputLimiter.swift`: observation size/hygiene boundary.
- `Builtin/`: Apple/local capability adapters.
- Coupled catalogs: `ios/Lumen/Models/ToolDefinition.swift` and `ios/Lumen/Services/ToolExecutor.swift`.

## Public Interfaces

Canonical tool IDs, aliases, JSON argument schema, required fields, enums, result/status/error categories, approval classification, and enabled registration are consumed by the kernel, generated manifest/training data, UI, diagnostics, and tests.

## Internal Structure

Model action -> schema bridge canonicalization -> enabled-tool lookup -> exact type/required/enum/extra-field validation -> approval policy -> permission gate -> registered `LocalTool` -> bounded `ToolResult` -> trusted kernel observation.

## Incoming Dependencies

`StructuredAgentKernelExecutor`, tool views/diagnostics, and direct tests invoke this boundary.

## Outgoing Dependencies

Built-ins call Permissions and platform services for contacts, calendar, communication, location, media/health, notifications, memory, RAG, files/knowledge, and URLs. Network-capable tools use bounded resilience services.

## Data And Control Flow

Untrusted JSON -> canonical typed invocation -> policy/permission/approval -> side effect/read -> classified bounded result. Missing required arguments clarify or fail; they do not become generic safe-failure success.

## Local Invariants

- Reject unknown tools, disabled tools, extra arguments, wrong JSON types, invalid enums, and missing required fields before execution.
- Approval occurs before side effects and cannot be inferred from model text.
- Permission state is checked through `Permissions/`; background/headless contexts cannot surprise-prompt.
- Tool IDs/aliases normalize in one canonical route contract.
- Bound output and redact diagnostics; do not log raw arguments/results.
- Return typed unavailable/permission/cancellation/failure states, not empty success.
- Built-ins remain local-first; optional network failure does not disable unrelated tools.

## Coordinated Changes

Any tool addition/change requires `Models/ToolDefinition.swift`, `ToolID.swift`, registry registration, schema bridge, `Services/ToolExecutor.swift`, permission/approval policy, generated manifest/datasets, app resource mirror, views/docs as applicable, and coverage/schema/policy tests.

## Safe Editing Rules

Place a new implementation under `Builtin/` and register it explicitly. Reuse services rather than accessing protected frameworks ad hoc. Do not loosen schema validation to accept model mistakes broadly; use bounded canonical aliases/repair with tests.

## Validation

From the repository root:

```bash
python3 tools/check_release_hardening.py
bash scripts/check-lumen-integration-gate.sh
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/ToolSchemaBridgeTests -only-testing:LumenTests/SecureToolRegistryTests -only-testing:LumenTests/ToolApprovalPolicyTests -only-testing:LumenTests/ToolRegistryCoverageTests
```

Also inspect `ToolApprovalBoundaryTests.swift`, `SafeToolOutputLimiterTests.swift`, `ToolOutputHygieneTests.swift`, and the affected built-in policy test.

## Common Failure Modes

- Registry, model catalog, route aliases, and generated manifest drift.
- Optional or extra fields bypass exact schema validation.
- Permission is mistaken for approval.
- A read failure returns `[]` and looks like no data.
- A tool result exceeds context/privacy bounds.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). Permission details: [`../Permissions/AGENTS.md`](../Permissions/AGENTS.md). Kernel execution: [`../Assistant/AGENTS.md`](../Assistant/AGENTS.md).
