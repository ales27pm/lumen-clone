# Agent Kernel Migration PR8: Secure Tool Registry Bridge

PR8 begins PR F by moving remaining legacy tool executions behind `SecureToolRegistry`.

## Goal

Stop production and compatibility paths from calling `ToolExecutor.shared.execute(...)` directly.

## What changed

- Add `LegacyToolExecutorLocalTool`, a temporary `LocalTool` adapter for each `ToolRegistry` legacy tool definition.
- Register legacy adapters in `SecureToolRegistry` so policy, approval, output limiting, and metrics wrap legacy execution.
- Add `SecureToolRegistry.executeLegacyTool(...)` as the single compatibility API for legacy tool IDs.
- Replace direct `ToolExecutor.shared.execute(...)` calls in:
  - `AgentService`
  - `SlotAgentService`
  - `LegacySecureToolExecutor`
  - `MessageBubble` user-approved tool execution
- Tighten `tools/check_agent_kernel_boundary.py` so `ToolExecutor.shared.execute(...)` is allowed only inside the temporary legacy LocalTool adapter.

## New path

```text
Agent / UI legacy tool call
  -> SecureToolRegistry.executeLegacyTool(...)
  -> SecureToolRegistry.execute(...)
  -> LegacyToolExecutorLocalTool.execute(...)
  -> ToolExecutor.shared.execute(...)
```

## Intentional limitation

This is a bridge PR. It does not yet split every legacy tool into a dedicated native `LocalTool`. The next PRs should replace groups of legacy adapters with native tools until `LegacyToolExecutorLocalTool` and `ToolExecutor` can be deleted.

## Boundary impact

The known legacy caller allowlist drops from 6 to 2. The remaining allowed entries are:

- `AgentService -> SlotAgentService.shared.run(...)` compatibility routing
- `LegacySecureToolExecutor` type declaration until that shim is deleted

## Next PR

`feat(kernel): port productivity tools to native LocalTool implementations`
