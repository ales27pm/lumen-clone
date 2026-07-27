# AGENTS.md

## Scope

Governs `ios/Lumen/RAG/`, document/chunking, indexing, embedding batches, maintenance policy, retrieval result, and context construction. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

This subtree transforms local documents/media into bounded retrieval context. Storage and vector persistence are owned by `Services/RAGStore.swift` and `Services/VectorIndex.swift`; the persisted chunk is `Models/RAGChunk.swift`.

## Key Files And Entry Points

- `RAGIndexer.swift`: extraction/chunking/index write orchestration.
- `RAGEngine.swift`: retrieval orchestration.
- `ChunkingStrategy.swift` and `EmbeddingBatcher.swift`: deterministic chunk/batch policy.
- `RAGContextBuilder.swift`: kernel-facing bounded context.
- `RAGDocument.swift`, `RAGSource.swift`, `RAGRetrievalResult.swift`: contracts/provenance.
- `RAGMaintenancePolicy.swift`: stale/maintenance behavior.

## Public Interfaces

Document/source identity, chunk metadata, retrieval status, ranked results, and context-item format are consumed by import flows, tools, views, the assistant, diagnostics, and tests.

## Internal Structure

Source extraction -> chunking -> batched embeddings -> store/vector flush with model metadata. Query -> semantic and lexical retrieval -> merge/deduplicate/rank -> context builder. Stale embedding identity/dimension must be detectable.

## Incoming Dependencies

File/import UI, `RAGSearchTool`, `AssistantKernel`, background maintenance, and source views call this subsystem.

## Outgoing Dependencies

The subsystem calls PDF/media extraction APIs where supported, `RAGStore`, `VectorIndex`, shared embedding/model services, SwiftData models, and diagnostics/resource policy.

## Data And Control Flow

Local source -> typed extraction result -> chunks with provenance -> embedding/index transaction -> persisted metadata. Query -> explicit retrieval result -> deduplicated bounded context -> kernel/tool result.

## Local Invariants

- Keep local source provenance and stable source/chunk identity.
- Distinguish empty retrieval from extraction, unsupported type, permission, model, embedding, index, persistence, and cancellation failures.
- Embedding model identity and vector dimension must match the index; stale data is rebuilt or reported, never mixed silently.
- Flush/index completion must reflect persisted state, not only queued work.
- Bound chunk size, batch work, result count, and context budget.

## Coordinated Changes

Chunking or IDs require index invalidation/dedup tests and store/model review. Embedding changes require LLM/model metadata, `VectorIndex`, Memory semantic behavior, and maintenance policy. Supported source changes require permission, extraction, UI, and diagnostics updates.

## Safe Editing Rules

Keep persistence in stores and pure policy in this subtree. Do not swallow corrupt index or extraction errors into `[]`. Avoid holding the main actor during CPU-heavy extraction/embedding while respecting `ModelContext` ownership.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/ChunkingStrategyTests -only-testing:LumenTests/EmbeddingBatcherPolicyTests -only-testing:LumenTests/RAGContextBuilderTests -only-testing:LumenTests/RAGRetrievalDedupTests
```

Also inspect `RAGSearchToolTests.swift`, `SemanticEmbeddingTextTests.swift`, and `PersistenceAuditTests.swift` for affected contracts.

## Common Failure Modes

- A changed embedding dimension reuses old vectors.
- Indexing reports completion before store flush.
- Lexical and semantic results duplicate the same chunk.
- Unsupported/corrupt input appears as a valid empty document.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). Store/model rules are in [`../Services/AGENTS.md`](../Services/AGENTS.md) and [`../Models/AGENTS.md`](../Models/AGENTS.md).
