# Agent Kernel Migration PR3: ChatView Stream Migration

PR3 moves the foreground Agent Mode chat path from `AgentService.shared.run(...)` to `AssistantKernel.run(...)`.

## Goals

- Route `ChatView.runAgent(...)` through the Agent Kernel streaming boundary.
- Keep the existing ChatView UI rendering path by consuming `AgentKernelEvent.legacyAgentEvent` as a temporary adapter.
- Remove the ChatView legacy caller entry from `tools/check_agent_kernel_boundary.py`.
- Preserve cancellation, CPU watchdog, ResourceBudgetGate, final validation, memory persistence, developer traces, and runtime diagnostics around the migrated path.

## New foreground path

```text
ChatView.runAgent(...)
  -> AgentKernelRequest(source: .chat, task: .chat)
  -> AssistantKernel.run(..., modelContext:)
  -> AgentKernelEvent.legacyAgentEvent
  -> existing ChatView stream rendering/final persistence
```

## What changed

- Replaced `AgentService.shared.run(...)` in `ChatView.runAgent(...)` with `AssistantKernel.run(...)`.
- Converted recent conversation context into `AgentKernelMessage` history.
- Kept UI throttling, cancellation guards, schema-placeholder repair, final intent validation, memory extraction, and developer trace persistence.
- Removed the ChatView entry from the legacy boundary allowlist.

## Intentional limitation

This PR does not claim full tool parity for foreground Agent Mode. Tool execution still needs a later migration from legacy `ToolExecutor` into `SecureToolRegistry`/kernel-owned tool stages.

## Validation

```bash
python3 tools/check_agent_kernel_boundary.py
rg "AgentService\.shared\.run|SlotAgentService\.shared\.run|RolePipelineAgentService\.shared\.run|ToolExecutor\.shared\.execute|LegacySecureToolExecutor|AgentRunner\.runHeadless" ios/Lumen/Views/ChatView.swift
```

Expected results:

- Boundary guard passes.
- Known legacy caller allowlist drops from 16 to 15.
- `ChatView.swift` has no direct legacy runtime calls.

## Next PR

`feat(kernel): migrate voice path to Agent Kernel stream`
