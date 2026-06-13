# Agent Kernel Migration PR4: Voice Stream Migration

PR4 moves the foreground voice response path behind the Agent Kernel streaming boundary.

## Goals

- Replace voice callers with `VoiceAgentRuntimeBridge`, which keeps text-only voice turns on `AssistantKernel.run(...)`.
- Preserve tool-capable voice intents through an explicit `AssistantKernel.runLegacyAgentBridge(...)` compatibility path until kernel-native tool stages are complete.
- Preserve the existing voice UI stream by consuming `AgentKernelEvent.legacyAgentEvent` as a temporary bridge.
- Keep routing, memory recall, memory gating, cancellation, CPU watchdog, `ResourceBudgetGate`, speech playback, final validation, persistence, and memory extraction behavior intact.
- Remove both voice legacy caller entries from `tools/check_agent_kernel_boundary.py`, shrinking the allowlist from 15 to 13.

## New paths

```text
VoiceCommandRouter.routeFinalTranscript(...)
  -> VoiceAgentRuntimeBridge.streamVoiceTurn(...)
  -> text-only: AgentKernelRequest(source: .voice, task: .chat)
  -> text-only: AssistantKernel.run(..., modelContext:)
  -> AgentKernelEvent.legacyAgentEvent
```

```text
VoiceModeView.runAgent(...)
  -> VoiceAgentRuntimeBridge.streamVoiceTurn(...)
  -> tool-capable intent: AgentRequest(availableTools:)
  -> tool-capable intent: AssistantKernel.runLegacyAgentBridge(...)
  -> AgentKernelEvent.legacyAgentEvent
  -> existing voice streaming/speech/persistence path
```

## Intentional limitation

Kernel-native text generation does not yet provide full tool parity for voice. Tool-capable voice intents stay behind the kernel boundary through the legacy compatibility bridge, and tool execution still needs a later migration from legacy `ToolExecutor` into `SecureToolRegistry` and kernel-owned tool stages.

## Validation

```bash
python3 tools/check_agent_kernel_boundary.py
rg "AgentService\.shared\.run|SlotAgentService\.shared\.run|RolePipelineAgentService\.shared\.run|ToolExecutor\.shared\.execute|LegacySecureToolExecutor|AgentRunner\.runHeadless" ios/Lumen/Voice/VoiceCommandRouter.swift ios/Lumen/Views/VoiceModeView.swift
```

Expected result: the boundary guard passes with 13 known legacy callers still allowlisted, and `rg` returns no matches for the migrated voice files.

## Next PR

`feat(kernel): migrate AppIntent and trigger headless paths to Agent Kernel`
