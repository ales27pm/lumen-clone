# Legacy Prompt Path Audit

## Status
- AgentRunner headless: fully migrated.
- AgentService interactive run path: fully migrated to `LegacyTurnGroundingCoordinator.prepareGroundedRequest(...)` with degraded fallback.
- SlotAgentService interactive run path: fully migrated to `LegacyTurnGroundingCoordinator.prepareGroundedRequest(...)` with degraded fallback.
- RolePipelineAgentService interactive run path: fully migrated to `LegacyTurnGroundingCoordinator.prepareGroundedRequest(...)` with degraded fallback.

## Remaining risks
- Some legacy request builders outside `run(_:)` still pass pre-grounded prompts and may double-ground if they are not normalized upstream.
- External legacy tools still depend on `LegacySecureToolExecutor` deny/allow heuristics when not present in `ToolRegistry`.

- Non-run path status: normalized where using assembler idempotency; remaining risky paths documented.
