# Legacy Agent Migration

Phase 7 introduces `LegacyGroundingBridge` for legacy agent/headless paths.

## Now using bridge
- `AgentRunner` headless path builds bounded grounding sections and secure-tool availability before constructing `AgentRequest`.

## Still legacy
- `AgentService`, `SlotAgentService`, and `RolePipelineAgentService` still execute through legacy planning/execution loops and legacy `ToolExecutor`.
- Migration is additive; behavior is preserved.

## Tool schema bridge
- `LegacyToolSchemaBridge` maps secure tool definitions into legacy `ToolDefinition` shape.
- New secure tools remain available via `AssistantKernel.executeTool(...)` and are surfaced to headless through mapped definitions.

## Risks remaining
- Some legacy-only tools continue to bypass new approval policy unless routed through secure invocation path.
- Full per-stage grounding reuse in role pipeline is pending deeper integration.

## Phase 7B migration progress
- Added `LegacyTurnGroundingCoordinator` as shared bounded grounding entrypoint.
- Added short-lived `LegacyGroundingCache` for reuse across role stages and to reduce repeated Memory/RAG lookups.
- Added `LegacyPromptInjectionPolicy` profiles for foreground/headless/role/slot flows.
- `AgentRunner` uses coordinator output; legacy services now route tool execution through `LegacySecureToolExecutor`.

## Prompt path status (Phase 7C)
- `AgentRunner` now uses `LegacyTurnGroundingCoordinator` + `LegacyPromptAssembler` + bridged secure tool definitions.
- `AgentService`/`SlotAgentService`/`RolePipelineAgentService` now execute tools through `LegacySecureToolExecutor`; prompt construction migration remains partial and tracked in `LEGACY_PROMPT_PATH_AUDIT.md`.

## Interactive services update
`AgentService`, `SlotAgentService`, and `RolePipelineAgentService` now enforce one bounded grounding assembly pass at run-entry using `LegacyPromptAssembler`, reducing duplicate/unbounded prompt injection. Full coordinator-in-service integration remains dependent on model-context plumbing.

- Added explicit UI ModelContext injection via `LegacyAgentRunOptions` in ChatView and VoiceModeView.
- Added idempotency guard marker/strip logic to prevent duplicate grounding sections.
