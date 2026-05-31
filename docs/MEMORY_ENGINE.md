# Memory Engine

Phase 6 adds `MemoryEngine` as a wrapper over existing `MemoryStore` and `MemoryItem`.

Key parts:
- deterministic candidate extraction (`remember that`, `I prefer`, `actually`, `for Lumen` patterns)
- deterministic scoring (`MemoryScorer`) with sensitivity-aware save decisions
- context building with strict char budget and ranking
- background-safe consolidation (`MemoryConsolidator`) with dedupe + runtime metrics

Privacy:
- credential-like candidates are rejected
- health/legal/financial candidates default to ask-user in scorer
- bounded context snippets only
