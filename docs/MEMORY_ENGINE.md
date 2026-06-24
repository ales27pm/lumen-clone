# Memory Engine

Phase 6 adds `MemoryEngine` as a wrapper over existing `MemoryStore` and `MemoryItem`.

Key parts:
- deterministic candidate extraction (`remember that`, `I prefer`, `actually`, `for Lumen` patterns)
- deterministic scoring (`MemoryScorer`) with sensitivity-aware save decisions
- context building with strict char budget and ranking
- background-safe consolidation (`MemoryConsolidator`) with dedupe + runtime metrics
- offline-resilient App Intent capture (`MemoryCaptureQueue`) that stores approved memory text locally when embedding/indexing is unavailable, then promotes queued captures through `MemoryStore.remember` when local embedding runtime policy allows

Privacy:
- credential-like candidates are rejected
- health/legal/financial candidates default to ask-user in scorer
- bounded context snippets only
- pending captures stay in local app storage; no cloud or network fallback is used for memory indexing
