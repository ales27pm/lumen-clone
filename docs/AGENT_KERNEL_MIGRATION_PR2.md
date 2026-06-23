# Agent Kernel Migration PR2: Llama Runtime Wiring

PR2 connects the Agent Kernel runtime router to the existing SwiftLlama/AppLlamaService path without migrating production entrypoints yet.

## Goals

- Make the default `AssistantRuntimeRouter` prefer a live llama adapter when heavy foreground runtime work is allowed.
- Keep `AppLlamaService` behind `LlamaRuntimeAdapter`, not called directly by new kernel clients.
- Preserve deterministic degraded mode when llama is selected but no chat model is loaded.
- Keep existing injected test adapters working.

## Runtime path

```text
AssistantKernel.run(...)
  -> AssistantKernel.runTextTurn(...)
  -> AssistantRuntimeRouter.selection(...)
  -> LlamaRuntimeAdapter.live(...)
  -> AppLlamaService.stream(..., slot: .mouth)
```

If the live llama adapter is selected but unavailable at generation time, the kernel records a runtime fallback event and runs `DeterministicFallbackRuntime`.

## Non-goals

This PR does not migrate `ChatView`, voice, AppIntents, triggers, diagnostics, or tool execution. It only wires the llama runtime behind the kernel adapter boundary.

## Follow-up

PR3 should migrate `ChatView` to call `AssistantKernel.run(...)` directly and consume native `AgentKernelEvent` values.
