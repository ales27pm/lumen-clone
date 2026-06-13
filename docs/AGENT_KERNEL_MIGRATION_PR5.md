# Agent Kernel Migration PR5: Headless AppIntent and Trigger Paths

PR5 migrates the headless AppIntent and scheduled-trigger paths away from direct `AgentRunner.runHeadless(...)` legacy routing.

## Goals

- Route `LumenAskIntent` through an Agent-Kernel-owned headless bridge.
- Route `TriggerScheduler` through the same kernel bridge.
- Remove the remaining `RolePipelineAgentService.shared.run(...)` implementation from `AgentRunner` by making `AgentRunner` a compatibility wrapper only.
- Treat user-initiated AppIntent turns as foreground for compute policy purposes while keeping scheduled triggers background-gated.
- Shrink the Agent Kernel boundary allowlist from 13 to 10.

## New path

```text
LumenAskIntent.perform(...)
  -> HeadlessAgentKernelRunner.run(source: .appIntent)
  -> AssistantKernel.run(...)
```

```text
TriggerScheduler.runTrigger(...)
  -> HeadlessAgentKernelRunner.run(source: .trigger)
  -> AssistantKernel.run(...)
```

`AgentRunner.runHeadless(...)` remains only as a short-lived compatibility wrapper and no longer calls `RolePipelineAgentService`.

## Intentional limitation

This PR does not claim full tool parity for headless turns. Tool execution still needs a later migration from legacy `ToolExecutor` into `SecureToolRegistry` and kernel-owned tool stages.

## Validation

```bash
python3 tools/check_agent_kernel_boundary.py
rg "AgentRunner\.runHeadless|RolePipelineAgentService\.shared\.run" ios/Lumen/AppIntents/LumenAskIntent.swift ios/Lumen/Services/TriggerScheduler.swift ios/Lumen/Services/AgentRunner.swift
swiftc -parse \
  ios/Lumen/Assistant/HeadlessAgentKernelRunner.swift \
  ios/Lumen/AppIntents/LumenAskIntent.swift \
  ios/Lumen/Services/TriggerScheduler.swift \
  ios/Lumen/Services/AgentRunner.swift
```

Expected guard result: 10 known legacy callers still allowlisted.

## Next PR

`feat(kernel): migrate diagnostics and e2e agent probes to Agent Kernel`
