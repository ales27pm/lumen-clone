# Context Budgeting

`ContextBudgetAllocator` provides deterministic section budgets (char-based):
- system
- short-term history
- memories
- RAG
- tools
- runtime policy

`AssistantKernel.buildGroundingContext` uses Memory + RAG context builders and available secure tool definitions, then records compact counts/char totals in `AssistantGroundingContext`.

## Legacy bridge rendering
`PromptGroundingRenderer` emits bounded sections for memories, retrieved sources, tools, and runtime policy. `LegacyGroundingBridge` applies this renderer with strict character caps before injecting into legacy prompts.

`LegacyPromptAssembler` enforces profile caps (foreground/headless/role/slot) and guarantees bounded LOCAL MEMORY/LOCAL SOURCES/AVAILABLE LOCAL TOOLS/RUNTIME POLICY sections.

Interactive legacy services now apply `LegacyPromptAssembler` at run-entry to enforce bounded LOCAL MEMORY / LOCAL SOURCES / AVAILABLE LOCAL TOOLS / RUNTIME POLICY sections exactly once per request.

- Grounding injection now uses marker-based idempotency guard (`LUMEN_GROUNDING_V1`) to avoid duplicate blocks.
