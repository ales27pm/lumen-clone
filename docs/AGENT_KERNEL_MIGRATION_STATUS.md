# Agent Kernel Migration Status

This document is the current roll-up for the PR1-PR10 Agent Kernel migration series. It distinguishes three migration states:

- **Behind kernel boundary**: callers enter through `AssistantKernel.run(...)`, `HeadlessAgentKernelRunner`, or a kernel-owned bridge rather than directly invoking legacy services.
- **Kernel-native complete**: the surface uses native `AgentKernelEvent` / `LocalTool` / `SecureToolRegistry` behavior without depending on legacy agent loops or the broad `LegacyToolExecutorLocalTool` adapter.
- **Compatibility bridge only**: the surface is no longer a direct external legacy caller, but the kernel still delegates to legacy services or legacy tool adapters behind the boundary.

## PR1-PR10 status

| PR | Objective | Files touched / documented | Current status | Limitations restantes |
| --- | --- | --- | --- | --- |
| PR1 | Establish the Agent Kernel API, streaming event contract, and boundary guard. | `ios/Lumen/Assistant/AgentKernelContracts.swift`, `ios/Lumen/Assistant/AssistantKernel+Streaming.swift`, `tools/check_agent_kernel_boundary.py`, `docs/AGENT_KERNEL_MIGRATION_PR1.md` | **Behind kernel boundary foundation**: contracts and guard exist. | No production surface migrated yet; legacy services remain. |
| PR2 | Wire the llama runtime behind the kernel router. | `ios/Lumen/Assistant/AssistantRuntimeRouter.swift`, `ios/Lumen/Assistant/LlamaRuntimeAdapter.swift`, `docs/AGENT_KERNEL_MIGRATION_PR2.md` | **Kernel-native runtime adapter** for text generation selection. | Entrypoints, tools, and UI streams still need migration. |
| PR3 | Move foreground chat from `AgentService.shared.run(...)` to `AssistantKernel.run(...)`. | `ios/Lumen/Views/ChatView.swift`, `ios/Lumen/Views/ChatKernelEventReducer.swift`, `docs/AGENT_KERNEL_MIGRATION_PR3.md` | **Kernel-native event consumption** for ChatView: the view enters through `AssistantKernel.run(...)` and reduces native `AgentKernelEvent` values with `ChatKernelEventReducer`. | The legacy AgentEvent adapter property is no longer consumed at the ChatView boundary; kernel-internal tool-required chat routing can still use the documented compatibility bridge until native tool stages are complete. |
| PR4 | Move voice response paths behind the kernel boundary. | `ios/Lumen/Voice/VoiceCommandRouter.swift`, `ios/Lumen/Views/VoiceModeView.swift`, `docs/AGENT_KERNEL_MIGRATION_PR4.md` | **Mixed**: text-only voice is behind `AssistantKernel.run(...)`; tool-capable voice is **compatibility bridge only** through `runLegacyAgentBridge(...)` and `LegacyAgentCompatibilityBridge`. | Tool-capable voice still depends on the legacy agent bridge. |
| PR5 | Move AppIntent and scheduled trigger headless paths to a kernel-owned runner. | `ios/Lumen/Assistant/HeadlessAgentKernelRunner.swift`, `ios/Lumen/AppIntents/LumenAskIntent.swift`, `ios/Lumen/Services/TriggerScheduler.swift`, `ios/Lumen/Services/AgentRunner.swift`, `docs/AGENT_KERNEL_MIGRATION_PR5.md` | **Behind kernel boundary** for AppIntent and trigger entrypoints. | Headless tool parity still depends on legacy tool compatibility until native tools are complete. |
| PR6 | Move diagnostics and E2E agent probes off direct legacy services. | `ios/Lumen/Diagnostics/PersistentRuntimeDiagnosticsRunner.swift`, `ios/Lumen/Services/E2ETestRunner.swift`, `docs/AGENT_KERNEL_MIGRATION_PR6.md` | **Compatibility bridge only** for live agent diagnostic probes. | Probes intentionally exercise legacy tool-capable behavior through `runLegacyAgentBridge(...)`. |
| PR7 | Move the grounding audit live trace smoke test behind the kernel bridge. | `ios/Lumen/Services/AgentGrounding/AgentGroundingAuditView.swift`, `docs/AGENT_KERNEL_MIGRATION_PR7.md` | **Compatibility bridge only** for the live grounding audit. | Still depends on legacy agent/tool behavior for trace validation. |
| PR8 | Route broad legacy tool execution through `SecureToolRegistry`. | `ios/Lumen/Tools/LegacyToolExecutorLocalTool.swift`, `ios/Lumen/Tools/ToolRegistry.swift`, `ios/Lumen/Services/AgentService.swift`, `ios/Lumen/Services/SlotAgentService.swift`, `ios/Lumen/Tools/LegacySecureToolExecutor.swift`, `ios/Lumen/Views/MessageBubble.swift`, `docs/AGENT_KERNEL_MIGRATION_PR8.md` | **Compatibility bridge only** for remaining legacy tool IDs; direct `ToolExecutor.shared.execute(...)` is isolated in the adapter. | Broad adapter still exists for unported legacy tools. |
| PR9 | Port productivity tools to native `LocalTool` implementations. | `ios/Lumen/Tools/Builtin/ProductivityLocalTools.swift`, `ios/Lumen/Tools/LegacyToolExecutorLocalTool.swift`, `ios/Lumen/Tools/ToolRegistry.swift`, `docs/AGENT_KERNEL_MIGRATION_PR9.md` | **Kernel-native complete** for the productivity tool group. | Non-productivity legacy adapter coverage remains. |
| PR10 | Port communication and Outlook tools to native `LocalTool` implementations. | `ios/Lumen/Tools/Builtin/CommunicationLocalTools.swift`, `ios/Lumen/Tools/LegacyToolExecutorLocalTool.swift`, `ios/Lumen/Tools/ToolRegistry.swift`, `docs/AGENT_KERNEL_MIGRATION_PR10.md` | **Kernel-native complete** for communication and Outlook tool groups. | Other tool groups still need native `LocalTool` ports before deleting the adapter. |

## Runtime surface status

| Runtime surface | Kernel entrypoint / owner | Current migration state | Remaining limitations |
| --- | --- | --- | --- |
| Chat | `ChatView.runAgent(...)` -> `AssistantKernel.run(...)` -> `ChatKernelEventReducer.reduce(...)` | **Kernel-native UI event consumption behind kernel boundary** | Native `AgentKernelEvent` cases feed ChatView directly; the legacy AgentEvent adapter property is not part of that stream. Tool-required chat turns may still be routed by `AssistantKernel+Streaming` through `runLegacyAgentBridge(...)` while native tool-stage parity remains incremental. |
| Voice | `VoiceAgentRuntimeBridge.streamVoiceTurn(...)` -> `AssistantKernel.run(...)` or `runLegacyAgentBridge(...)` -> `legacyEventStream(...)` | **Mixed** | Text-only turns are kernel-routed, but `VoiceCommandRouter` still adapts kernel events back to `AgentEvent` with `legacyAgentEvent`; tool-capable turns remain compatibility bridge only. |
| AppIntents | `LumenAskIntent.perform(...)` -> `HeadlessAgentKernelRunner.run(source: .appIntent)` | **Behind kernel boundary** | Native headless tool execution still follows the broader tool migration. |
| Triggers | `TriggerScheduler.runTrigger(...)` -> `HeadlessAgentKernelRunner.run(source: .trigger)` | **Behind kernel boundary** | Background-safe tool availability is still constrained by the secure/legacy bridge split. |
| Tools | `SecureToolRegistry.execute(...)` plus native `ProductivityLocalTool` and `CommunicationLocalTool` groups | **Mixed** | Productivity, communication, and Outlook groups are native; remaining legacy IDs can still flow through `LegacyToolExecutorLocalTool`. |
| Grounding | `AgentGroundingAuditView` live probe -> `AssistantKernel.shared.runLegacyAgentBridge(...)` | **Compatibility bridge only** for live smoke tests | Audit still validates legacy tool-capable traces; full kernel-native grounding probes are pending. |
| Diagnostics | `PersistentRuntimeDiagnosticsRunner` and `E2ETestRunner` probes -> `runLegacyAgentBridge(...)` -> `LegacyAgentCompatibilityBridge` | **Compatibility bridge only** for live agent probes | Diagnostic probes still intentionally exercise bridged legacy behavior. |

## Boundary guard counters and allowlists

`tools/check_agent_kernel_boundary.py` currently scans Swift files under `ios/Lumen` for these legacy patterns:

| Pattern | Expected status in normal mode |
| --- | --- |
| `AgentService.shared.run` | Allowed only in `ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift`. |
| `SlotAgentService.shared.run` | Allowed only in `ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift`. |
| `RolePipelineAgentService.shared.run` | No normal-mode production callers expected. |
| `AgentRunner.runHeadless` | No normal-mode production callers expected. |
| `ToolExecutor.shared.execute` | Allowed only in migration shim files. |
| `LegacySecureToolExecutor` | One line-specific allowlist entry remains. |

Expected normal-mode output:

```text
Agent Kernel boundary guard passed.
Documented compatibility bridges: 2
```

Exact documented compatibility bridge entries expected by the guard:

| File | Pattern | Reason | Removal condition |
| --- | --- | --- | --- |
| `ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift` | `AgentService.shared.run` | Temporary kernel-owned bridge for diagnostics and tool-capable legacy agent event streams. | Remove when Agent Kernel emits native tool stages for diagnostics and tool-capable paths. |
| `ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift` | `SlotAgentService.shared.run` | Temporary slot-agent bridge for diagnostics and deterministic compatibility responses. | Remove when diagnostics and deterministic compatibility responses are kernel-native. |

Strict mode reports the same documented bridge inventory and fails on any direct legacy call outside those exact entries.

Regression coverage: `ios/LumenTests/AgentKernelBoundaryGuardTests.swift` scans production Swift sources to keep `AgentService.shared.run`, `SlotAgentService.shared.run`, `RolePipelineAgentService.shared.run`, and `AgentRunner.runHeadless` confined to `LegacyAgentCompatibilityBridge.swift`.

## Legacy paths still present

Search command used for this inventory:

```bash
rg -n "runLegacyAgentBridge|LegacySecureToolExecutor|AgentService\.shared\.run|SlotAgentService\.shared\.run|RolePipelineAgentService\.shared\.run" ios/Lumen tools docs -S
```

Production Swift matches that remain:

| Pattern | Path(s) | Migration classification |
| --- | --- | --- |
| `runLegacyAgentBridge` | `ios/Lumen/Assistant/AssistantKernel+Streaming.swift`, `ios/Lumen/Voice/VoiceCommandRouter.swift`, `ios/Lumen/Diagnostics/PersistentRuntimeDiagnosticsRunner.swift`, `ios/Lumen/Services/AgentGrounding/AgentGroundingAuditView.swift`, `ios/Lumen/Services/E2ETestRunner.swift` | **Compatibility bridge only** for documented paths: kernel-internal tool-required chat turns in `AssistantKernel+Streaming`, tool-capable voice turns in `VoiceCommandRouter`, live diagnostics/E2E probes, and grounding audit smoke tests. These callers are behind the kernel boundary but still exercise legacy agent behavior through `LegacyAgentCompatibilityBridge`. |
| `LegacySecureToolExecutor` | No production Swift matches from the inventory command. | Removed from the current production inventory. |
| `AgentService.shared.run` | `ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift` | **Compatibility bridge only**; the direct call is contained in the named kernel bridge with a removal condition. |
| `SlotAgentService.shared.run` | `ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift` | **Compatibility bridge only**; the direct call is contained in the named kernel bridge with a removal condition. |
| `RolePipelineAgentService.shared.run` | No production Swift matches from the inventory command. | No current production caller found. |

## Definition of done

A surface should not be called simply “migrated” without the qualifier below:

1. **Behind kernel boundary** means external runtime callers enter through the kernel API or a kernel-owned runner/bridge. This is sufficient for boundary ownership, cancellation/resource policy consolidation, and preventing new direct legacy service dependencies.
2. **Kernel-native complete** means the surface no longer requires legacy agent services, `AgentKernelEvent.legacyAgentEvent`, `runLegacyAgentBridge(...)`, `LegacySecureToolExecutor`, `LegacyToolExecutorLocalTool`, or broad `ToolExecutor` compatibility to perform its normal behavior.
3. **Compatibility bridge only** means the old behavior is intentionally preserved behind a kernel-owned wrapper. This is safer than direct legacy calls, but it is not final migration completion and should keep a removal follow-up.

The remaining migration work is to shrink the documented compatibility bridge inventory from 2 to 0, delete `runLegacyAgentBridge(...)` and `LegacyAgentCompatibilityBridge` once all bridge-only surfaces are native, and continue replacing any remaining legacy tool compatibility coverage with dedicated native `LocalTool` implementations.
