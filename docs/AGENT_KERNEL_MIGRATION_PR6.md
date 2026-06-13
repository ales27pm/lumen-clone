# Agent Kernel Migration PR6: Diagnostics and E2E Probe Migration

PR6 moves diagnostics and E2E probe call sites off direct legacy agent services and through the Agent Kernel bridge.

## Goal

Shrink the production-facing legacy runtime allowlist by migrating diagnostic probe traffic into `AssistantKernel`.

## Changed paths

```text
PersistentRuntimeDiagnosticsRunner.scenarioLiveAgentStream(...)
  -> AssistantKernel.runLegacyAgentBridge(...)
```

```text
PersistentRuntimeDiagnosticsRunner.scenarioAgentCancellation(...)
  -> AssistantKernel.runLegacyAgentBridge(...)
```

```text
E2ETestRunner model-backed scenario probe
  -> AssistantKernel.runLegacyAgentBridge(...)
```

## Why this is still a bridge

These probes intentionally exercise agent/tool behavior. Until legacy `ToolExecutor` tools are ported into `SecureToolRegistry`, this PR routes them through the kernel-owned compatibility bridge rather than pretending full kernel tool parity exists.

The important boundary change is that diagnostics and E2E code no longer call `AgentService.shared.run(...)` or `SlotAgentService.shared.run(...)` directly.

## Boundary guard

This PR removes these entries from `tools/check_agent_kernel_boundary.py`:

- `PersistentRuntimeDiagnosticsRunner.swift` live agent stream probe
- `PersistentRuntimeDiagnosticsRunner.swift` cancellation probe
- `E2ETestRunner.swift` model-backed agent probe

Known legacy caller allowlist shrinks from 10 to 7.

## Non-goals

This PR does not yet:

- port legacy `ToolExecutor` tools into `SecureToolRegistry`;
- delete `AgentService` or `SlotAgentService`;
- migrate `AgentGroundingAuditView`;
- remove the kernel legacy bridge.

## Next PR

`feat(kernel): migrate grounding audit probe to Agent Kernel bridge`
