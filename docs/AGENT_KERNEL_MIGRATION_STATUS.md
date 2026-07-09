# Agent Kernel Migration Status

This is the current Release-facing status for Agent Kernel ownership. Historical PR documents remain as implementation history; this file describes what ships now.

## Release Surface

| Surface | Release status | Owner | Enforcement |
| --- | --- | --- | --- |
| Chat UI text turns | Shipped | `ChatView` -> `AssistantKernel.run(...)` -> `ChatKernelEventReducer` | ChatView consumes native `AgentKernelEvent` values. |
| Chat UI tool-required turns | Shipped for validated intent-routed tool actions | `AssistantKernel+Streaming` | Native kernel execution emits `toolInvocation` and `toolResult` events after intent routing, schema validation, approval policy, and `SecureToolRegistry` dispatch. |
| Voice text turns | Shipped | `VoiceCommandRouter` -> `AssistantKernel.run(...)` | Foreground-only voice text execution uses native kernel events. |
| Voice tool-required turns | Shipped for validated intent-routed tool actions | `VoiceCommandRouter` | Voice routes through `AssistantKernel.run(...)` and reuses the same native tool execution boundary as chat. |
| AppIntents | Shipped only for guarded local actions | `LumenAskIntent`, `LumenAddMemoryIntent`, `LumenMemorySearchIntent`, `LumenRunTriggerIntent` | Intents return degraded or open-app responses when required context is missing. |
| Background triggers/headless | Shipped only for background-safe coordination | `HeadlessAgentKernelRunner`, `BackgroundToolExecutionPolicy` | Background tasks cannot load model assets or prompt for permissions. Background-safe tool-only runs are policy-assessed and return explicit skip diagnostics when context is missing. |
| Diagnostics and E2E legacy live probes | DEBUG diagnostics only | `PersistentRuntimeDiagnosticsRunner`, `E2ETestRunner`, `AgentGroundingAuditView` | Release records skipped diagnostic events instead of running the legacy agent path. |
| Native tool groups | Shipped | `SecureToolRegistry` and `LocalTool` implementations | Tool execution requires registry lookup, policy approval, and schema-valid arguments. |

## Guardrails

`tools/check_agent_kernel_boundary.py` keeps direct calls to legacy services confined to the named migration wrapper. `tools/check_release_hardening.py` adds the Release rule: calls that exercise the legacy agent path must be inside `#if DEBUG`, and removed fallback/not-implemented runtime sentinels must not reappear in production source.

Expected checks:

```bash
python3 tools/check_agent_kernel_boundary.py
python3 tools/check_agent_kernel_boundary.py --strict
python3 tools/check_release_hardening.py
```

## Remaining DEBUG-Only Work

- Live runtime evidence for model-backed post-tool synthesis after native tool execution.
- Deleting the legacy agent migration wrapper after every DEBUG probe has a native replacement.
- Native live diagnostic scenarios that prove tool-capable kernel behavior without using the legacy agent path.
