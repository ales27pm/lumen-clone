# AGENTS.md

## Scope

Governs `ios/Lumen/Memory/`, the memory extraction, capture, scoring, consolidation, provenance, and context-building policies. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

This subtree decides what candidate information is safe/useful to remember and how bounded memory context is assembled. Persistence is owned by `ios/Lumen/Services/MemoryStore.swift`; the persisted entity is `ios/Lumen/Models/MemoryItem.swift`.

## Key Files And Entry Points

- `MemoryEngine.swift`: main-actor memory orchestration.
- `MemoryCaptureQueue.swift`: queued candidate capture and later promotion.
- `MemoryExtractionPolicy.swift`: candidate acceptance/extraction policy.
- `MemoryScorer.swift` and `MemoryConsolidator.swift`: ranking/deduplication/consolidation.
- `MemoryContextBuilder.swift`: bounded context supplied to the kernel.
- `MemoryProvenance.swift` and `MemoryCandidate.swift`: source and candidate contracts.

## Public Interfaces

Candidate, provenance, recall/context, and diagnostic result types are consumed by the assistant, tools, views, and tests. Callers rely on empty recall being distinct from permission, embedding, model, persistence, cancellation, or corrupt-state failures.

## Internal Structure

Candidate extraction -> policy/provenance classification -> queue -> score/consolidate -> `MemoryStore` persistence. Recall -> store query/embedding -> rank/deduplicate -> bounded `MemoryContextBuilder` output.

## Incoming Dependencies

`AssistantKernel`, chat save flow, Add Memory intent, memory tools, and memory UI invoke this subsystem.

## Outgoing Dependencies

It calls `MemoryStore`, the shared embedding service through kernel/service APIs, SwiftData model types, and diagnostics. It may consume RAG-independent semantic utilities but does not own the vector store.

## Data And Control Flow

User-approved or policy-valid candidate -> queued capture -> persistence with provenance. Query -> explicit store/embedding result -> scorer -> context items -> kernel. Failures propagate as typed diagnostics rather than empty arrays.

## Local Invariants

- Do not persist raw transient content without the existing extraction/provenance policy.
- Preserve local storage and privacy-safe diagnostics.
- Keep queue promotion idempotent and cancellation-aware.
- Do not collapse an embedding/store failure into "no memories."
- Context output is bounded and deduplicated; it must not exceed kernel budget ownership.

## Coordinated Changes

Persistent fields require `Models/MemoryItem.swift`, `LumenApp.swift`, `MemoryStore.swift`, exports/views, and persistence tests. Scoring/extraction changes require queue, context builder, memory search tool, and regression tests. Embedding changes require RAG/vector metadata review.

## Safe Editing Rules

Put persistence operations in `MemoryStore`, not policy types. Keep `MemoryEngine` main-actor ownership and move heavy pure work off actor only with immutable inputs. Never log recalled content in normal diagnostics.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/MemoryCaptureQueueTests -only-testing:LumenTests/MemoryConsolidatorTests -only-testing:LumenTests/MemoryContextBuilderTests -only-testing:LumenTests/MemoryExtractionPolicyTests -only-testing:LumenTests/MemoryScorerTests
```

Also review `MemorySearchToolTests.swift`, `MemoryCascadeTests.swift`, and `PersistenceAuditTests.swift` when persistence/retrieval changes.

## Common Failure Modes

- Duplicate queue promotion creates repeated memory.
- Failed embedding is presented as zero matches.
- A context builder leaks unbounded or raw diagnostic content.
- A model-shape change works only on a fresh store.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). Store rules are in [`../Services/AGENTS.md`](../Services/AGENTS.md); persisted model rules are in [`../Models/AGENTS.md`](../Models/AGENTS.md).
