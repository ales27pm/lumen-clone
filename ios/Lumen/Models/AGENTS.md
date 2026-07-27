# AGENTS.md

## Scope

Governs `ios/Lumen/Models/`, shared value types, tool definitions, output sanitization models, and SwiftData `@Model` declarations. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

These types are cross-subsystem contracts. `Conversation`, `ChatMessage`, `MemoryItem`, `RAGChunk`, `StoredModel`, and `Trigger` participate in persistence; tool and agent value types cross kernel/tool/UI boundaries.

## Key Files And Entry Points

- `Conversation.swift`, `ChatMessage.swift`, `MemoryItem.swift`, `RAGChunk.swift`, `StoredModel.swift`, `Trigger.swift`: persistent entities registered by `LumenApp.swift`.
- `ToolDefinition.swift`, `ToolApprovalState.swift`, `AgentStep.swift`, `AgentJSONValue.swift`: agent/tool contract models.
- `AssistantOutputSanitizer.swift`: user-facing output hygiene.
- `ChatAttachment.swift`, `MemoryContextItem.swift`, `WebRichContentPayload.swift`: cross-layer payloads.

## Public Interfaces

Stored property names/types, relationships, uniqueness/default semantics, Codable shapes, enum raw values, tool schema fields, and sanitizer outcomes are consumed throughout the app, tests, generated manifests, exports, and existing on-device stores.

## Internal Structure

Models should remain passive data/contracts. Persistence operations live in services; extraction/ranking lives in Memory/RAG; orchestration lives in Assistant; rendering lives in Views.

## Incoming Dependencies

Nearly every runtime subsystem imports these types. Existing user stores deserialize persistent models, making schema changes externally consequential even without a server API.

## Outgoing Dependencies

Models depend primarily on Foundation and SwiftData. Tool definitions may reference shared tool value types but must not execute services or call UI.

## Data And Control Flow

Runtime/store creates model -> SwiftData persists -> query/export/view consumes. Agent/tool payload models move from kernel to reducers and diagnostics. Sanitizers transform final output before presentation/persistence where called.

## Local Invariants

- Keep models free of service singletons and side-effect orchestration.
- Preserve persistent compatibility or supply an explicit migration strategy. No versioned SwiftData migration plan is tracked, so schema changes are high risk.
- Keep Codable/enum compatibility for stored/exported/generated data.
- Tool schema types must preserve exact JSON typing and reject unsupported values at the validation boundary.
- Do not store raw secrets or unnecessary diagnostic payloads.

## Coordinated Changes

Any `@Model` change requires the schema array in `LumenApp.swift`, owning store, fixtures/exports/views, and `PersistenceAuditTests.swift`. Tool model changes require Tools, `Services/ToolExecutor.swift`, crawler outputs, app manifest mirror, and schema/registry tests. Sanitizer changes require kernel finalization and sanitizer tests.

## Safe Editing Rules

Prefer additive, defaulted compatible fields only after checking existing-store behavior. Do not rename/remove persisted properties casually. New operational logic belongs in an owning service/policy, not a model initializer.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/PersistenceAuditTests -only-testing:LumenTests/AssistantOutputSanitizerTests -only-testing:LumenTests/RuntimeContractRegressionTests
```

Tool-model changes also require `ToolSchemaBridgeTests.swift` and `ToolRegistryCoverageTests.swift`.

## Common Failure Modes

- A new nonoptional persistent field breaks an existing store.
- An enum raw value change breaks saved/exported payloads.
- A model starts calling a singleton, creating circular ownership.
- A tool field changes in Swift but not in generated manifest/training data.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). No child file is needed; individual models share the same contract/migration risk.
