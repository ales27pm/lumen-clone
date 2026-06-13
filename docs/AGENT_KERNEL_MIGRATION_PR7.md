# Agent Kernel Migration PR7: Grounding Audit Probe

PR7 migrates the Agent Grounding Audit live trace smoke test off a direct `RolePipelineAgentService.shared.run(...)` call and through the Agent Kernel compatibility bridge.

## Goal

Remove the remaining UI diagnostics probe that bypasses the Agent Kernel runtime boundary.

## Changed path

```text
AgentGroundingAuditView.runLiveTraceSmokeTest(...)
  -> AssistantKernel.shared.runLegacyAgentBridge(...)
  -> existing legacy agent stream behind kernel ownership
```

## Why this is still a bridge

The live trace smoke test intentionally exercises tool-capable legacy agent behavior and validates behavior trace recording. Full kernel-native tool execution depends on the later `ToolExecutor -> SecureToolRegistry` migration.

## Boundary impact

This removes the `AgentGroundingAuditView.swift` entry from `tools/check_agent_kernel_boundary.py`, reducing the known legacy caller allowlist from 7 to 6.

## Non-goals

This PR does not:

- port legacy `ToolExecutor` tools into `SecureToolRegistry`;
- delete `AgentService` or `SlotAgentService`;
- remove `AssistantKernel.runLegacyAgentBridge(...)`;
- change trace validation semantics.

## Next PR

`feat(kernel): move legacy tool execution behind SecureToolRegistry`
