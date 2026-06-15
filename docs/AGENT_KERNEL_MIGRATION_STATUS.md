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
| PR3 | Move foreground chat from `AgentService.shared.run(...)` to `AssistantKernel.run(...)`. | `ios/Lumen/Views/ChatView.swift`, `docs/AGENT_KERNEL_MIGRATION_PR3.md` | **Behind kernel boundary** for ChatView. | Chat still consumes the temporary legacy event adapter; full tool parity is not complete. |
| PR4 | Move voice response paths behind the kernel boundary. | `ios/Lumen/Voice/VoiceCommandRouter.swift`, `ios/Lumen/Views/VoiceModeView.swift`, `docs/AGENT_KERNEL_MIGRATION_PR4.md` | **Mixed**: text-only voice is behind `AssistantKernel.run(...)`; tool-capable voice is **compatibility bridge only** through `runLegacyAgentBridge(...)`. | Tool-capable voice still depends on the legacy agent bridge. |
| PR5 | Move AppIntent and scheduled trigger headless paths to a kernel-owned runner. | `ios/Lumen/Assistant/HeadlessAgentKernelRunner.swift`, `ios/Lumen/AppIntents/LumenAskIntent.swift`, `ios/Lumen/Services/TriggerScheduler.swift`, `ios/Lumen/Services/AgentRunner.swift`, `docs/AGENT_KERNEL_MIGRATION_PR5.md` | **Behind kernel boundary** for AppIntent and trigger entrypoints. | Headless tool parity still depends on legacy tool compatibility until native tools are complete. |
| PR6 | Move diagnostics and E2E agent probes off direct legacy services. | `ios/Lumen/Diagnostics/PersistentRuntimeDiagnosticsRunner.swift`, `ios/Lumen/Services/E2ETestRunner.swift`, `docs/AGENT_KERNEL_MIGRATION_PR6.md` | **Compatibility bridge only** for live agent diagnostic probes. | Probes intentionally exercise legacy tool-capable behavior through `runLegacyAgentBridge(...)`. |
| PR7 | Move the grounding audit live trace smoke test behind the kernel bridge. | `ios/Lumen/Services/AgentGrounding/AgentGroundingAuditView.swift`, `docs/AGENT_KERNEL_MIGRATION_PR7.md` | **Compatibility bridge only** for the live grounding audit. | Still depends on legacy agent/tool behavior for trace validation. |
| PR8 | Route broad legacy tool execution through `SecureToolRegistry`. | `ios/Lumen/Tools/LegacyToolExecutorLocalTool.swift`, `ios/Lumen/Tools/ToolRegistry.swift`, `ios/Lumen/Services/AgentService.swift`, `ios/Lumen/Services/SlotAgentService.swift`, `ios/Lumen/Tools/LegacySecureToolExecutor.swift`, `ios/Lumen/Views/MessageBubble.swift`, `docs/AGENT_KERNEL_MIGRATION_PR8.md` | **Compatibility bridge only** for remaining legacy tool IDs; direct `ToolExecutor.shared.execute(...)` is isolated in the adapter. | Broad adapter still exists for unported legacy tools. |
| PR9 | Port productivity tools to native `LocalTool` implementations. | `ios/Lumen/Tools/Builtin/ProductivityLocalTools.swift`, `ios/Lumen/Tools/LegacyToolExecutorLocalTool.swift`, `ios/Lumen/Tools/ToolRegistry.swift`, `docs/AGENT_KERNEL_MIGRATION_PR9.md` | **Kernel-native complete** for the productivity tool group. | Non-productivity legacy adapter coverage remains. |
| PR10 | Port communication and Outlook tools to native `LocalTool` implementations. | `ios/Lumen/Tools/Builtin/CommunicationLocalTools.swift`, `ios/Lumen/Tools/LegacyToolExecutorLocalTool.swift`, `ios/Lumen/Tools/ToolRegistry.swift`, `docs/AGENT_KERNEL_MIGRATION_PR10.md` | **Kernel-native complete** for communication and Outlook tool groups. | Other tool groups still need native `LocalTool` ports before deleting the adapter. |

## Runtime surface status

| Runtime surface | Kernel entrypoint / owner | Current migration state | Remaining limitations |
| --- | --- | --- | --- |
| Chat | `ChatView.runAgent(...)` -> `AssistantKernel.run(...)` | **Behind kernel boundary** | Uses `AgentKernelEvent.legacyAgentEvent` for the existing UI stream; tool execution parity remains incremental. |
| Voice | `VoiceAgentRuntimeBridge.streamVoiceTurn(...)` -> `AssistantKernel.run(...)` or `runLegacyAgentBridge(...)` | **Mixed** | Text-only turns are kernel-routed; tool-capable turns remain compatibility bridge only. |
| AppIntents | `LumenAskIntent.perform(...)` -> `HeadlessAgentKernelRunner.run(source: .appIntent)` | **Behind kernel boundary** | Native headless tool execution still follows the broader tool migration. |
| Triggers | `TriggerScheduler.runTrigger(...)` -> `HeadlessAgentKernelRunner.run(source: .trigger)` | **Behind kernel boundary** | Background-safe tool availability is still constrained by the secure/legacy bridge split. |
| Tools | `SecureToolRegistry.execute(...)` plus native `ProductivityLocalTool` and `CommunicationLocalTool` groups | **Mixed** | Productivity, communication, and Outlook groups are native; remaining legacy IDs can still flow through `LegacyToolExecutorLocalTool`. |
| Grounding | `AgentGroundingAuditView` live probe -> `AssistantKernel.shared.runLegacyAgentBridge(...)` | **Compatibility bridge only** for live smoke tests | Audit still validates legacy tool-capable traces; full kernel-native grounding probes are pending. |
| Diagnostics | `PersistentRuntimeDiagnosticsRunner` and `E2ETestRunner` probes -> `runLegacyAgentBridge(...)` | **Compatibility bridge only** for live agent probes | Diagnostic probes still intentionally exercise bridged legacy behavior. |

## Boundary guard counters and allowlists

`tools/check_agent_kernel_boundary.py` currently scans Swift files under `ios/Lumen` for these legacy patterns:

| Pattern | Expected status in normal mode |
| --- | --- |
| `AgentService.shared.run` | Allowed only in migration shim files. |
| `SlotAgentService.shared.run` | One line-specific allowlist entry remains. |
| `RolePipelineAgentService.shared.run` | No normal-mode production callers expected. |
| `AgentRunner.runHeadless` | No normal-mode production callers expected. |
| `ToolExecutor.shared.execute` | Allowed only in migration shim files. |
| `LegacySecureToolExecutor` | One line-specific allowlist entry remains. |

Expected normal-mode output:

```text
Agent Kernel boundary guard passed.
Known legacy callers still allowlisted: 2
```

Line-specific `ALLOWED_LEGACY_CALLERS` entries expected by the guard:

| File | Line | Pattern | Reason |
| --- | ---: | --- | --- |
| `ios/Lumen/Services/AgentService.swift` | 1324 | `SlotAgentService.shared.run` | Temporary compatibility routing from `AgentService` into `SlotAgentService`. |
| `ios/Lumen/Tools/LegacySecureToolExecutor.swift` | 5 | `LegacySecureToolExecutor` | Temporary shim type declaration until the legacy secure executor can be deleted. |

Migration shim files exempt in normal mode, but reported by `--strict`:

| File | Why it is exempt in normal mode |
| --- | --- |
| `ios/Lumen/Assistant/AgentKernelContracts.swift` | Kernel compatibility contracts. |
| `ios/Lumen/Assistant/AssistantKernel+Streaming.swift` | Owns `runLegacyAgentBridge(...)`, which delegates to the legacy agent service behind the kernel boundary. |
| `ios/Lumen/Tools/LegacyToolExecutorLocalTool.swift` | Temporary secure registry adapter around legacy `ToolExecutor`. |

Strict-mode inventory currently reports four findings: `AssistantKernel+Streaming.swift` bridge delegation, `AgentService.swift` slot compatibility routing, `LegacySecureToolExecutor.swift` shim declaration, and `LegacyToolExecutorLocalTool.swift` adapter execution.

## Legacy paths still present

Search command used for this inventory:

```bash
rg -n "runLegacyAgentBridge|LegacySecureToolExecutor|AgentService\.shared\.run|SlotAgentService\.shared\.run|RolePipelineAgentService\.shared\.run" ios/Lumen tools docs -S
```

Production Swift matches that remain:

| Pattern | Path(s) | Migration classification |
| --- | --- | --- |
| `runLegacyAgentBridge` | `ios/Lumen/Assistant/AssistantKernel+Streaming.swift`, `ios/Lumen/Voice/VoiceCommandRouter.swift`, `ios/Lumen/Diagnostics/PersistentRuntimeDiagnosticsRunner.swift`, `ios/Lumen/Services/AgentGrounding/AgentGroundingAuditView.swift`, `ios/Lumen/Services/E2ETestRunner.swift` | **Compatibility bridge only**; callers are behind the kernel boundary but still exercise legacy agent behavior. |
| `LegacySecureToolExecutor` | `ios/Lumen/Tools/LegacySecureToolExecutor.swift` | **Compatibility bridge only**; shim remains allowlisted. |
| `AgentService.shared.run` | `ios/Lumen/Assistant/AssistantKernel+Streaming.swift` | **Compatibility bridge only**; direct call is contained in the kernel bridge shim. |
| `SlotAgentService.shared.run` | `ios/Lumen/Services/AgentService.swift` | **Legacy allowlist remains**; direct compatibility routing is still outside full kernel-native completion. |
| `RolePipelineAgentService.shared.run` | No production Swift matches from the inventory command. | No current production caller found. |

## Definition of done

A surface should not be called simply “migrated” without the qualifier below:

1. **Behind kernel boundary** means external runtime callers enter through the kernel API or a kernel-owned runner/bridge. This is sufficient for boundary ownership, cancellation/resource policy consolidation, and preventing new direct legacy service dependencies.
2. **Kernel-native complete** means the surface no longer requires legacy agent services, `AgentKernelEvent.legacyAgentEvent`, `runLegacyAgentBridge(...)`, `LegacySecureToolExecutor`, `LegacyToolExecutorLocalTool`, or broad `ToolExecutor` compatibility to perform its normal behavior.
3. **Compatibility bridge only** means the old behavior is intentionally preserved behind a kernel-owned wrapper. This is safer than direct legacy calls, but it is not final migration completion and should keep a removal follow-up.

The remaining migration work is to shrink the normal allowlist from 2 to 0, make strict mode pass, delete `runLegacyAgentBridge(...)` once all bridge-only surfaces are native, and continue replacing `LegacyToolExecutorLocalTool` coverage with dedicated native `LocalTool` implementations.
