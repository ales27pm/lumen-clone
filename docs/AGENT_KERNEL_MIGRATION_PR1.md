# Agent Kernel Migration PR 1: Kernel API Foundation

This PR establishes the first production-facing Agent Kernel migration contract without moving existing entrypoints yet.

## Goal

Make `AssistantKernel.run(...)` the target orchestration boundary for all future chat, voice, AppIntent, trigger, diagnostics, and benchmark work.

## Added contracts

- `AgentKernelRequest`
- `AgentKernelMessage`
- `AgentKernelSource`
- `AgentKernelOptions`
- `AgentKernelDiagnosticEvent`
- `AgentKernelEvent`
- `AgentKernelRunning`

## Added streaming entrypoint

`AssistantKernel.run(_:modelContext:)` emits kernel-native events:

- accepted step
- runtime-selection diagnostic
- optional grounding diagnostic
- final delta
- final text
- completion diagnostic
- done event
- error event

The existing `runTextTurn(_:)` implementation remains intact and is used under the new streaming boundary.

## Migration compatibility

`AgentKernelEvent.legacyAgentEvent` provides a short-lived compatibility bridge to the existing `AgentEvent` stream used by current UI and service code. It should be deleted once ChatView, voice, headless, diagnostics, and E2E have moved to kernel-native events.

## Boundary guard

`tools/check_agent_kernel_boundary.py` detects new direct calls to legacy runtime paths:

- `AgentService.shared.run`
- `SlotAgentService.shared.run`
- `RolePipelineAgentService.shared.run`
- `AgentRunner.runHeadless`
- `ToolExecutor.shared.execute`
- `LegacySecureToolExecutor`

The guard includes a snapshot allowlist for known legacy callers. Each migration PR should shrink that allowlist.

## Non-goals

This PR does not yet:

- wire AppLlamaService directly into the kernel runtime adapter;
- migrate ChatView;
- migrate voice;
- migrate AppIntents or TriggerScheduler;
- port legacy ToolExecutor tools into SecureToolRegistry;
- delete legacy services.

Those are the next migration PRs.
