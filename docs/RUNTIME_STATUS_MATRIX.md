# Runtime Status Matrix

This matrix is a short status index for the main Lumen runtime surfaces. It separates the **product target** adapter-first canonical runtime shape from the **current live or bridged implementation** for each surface, so migration documents do not imply that every entrypoint already runs the final adapter runtime.

Status values:

- `live`: the surface is wired through the intended current runtime boundary for that surface.
- `partial`: the surface has some target-runtime wiring but still depends on staged or incomplete behavior.
- `compatibility bridge`: the surface is intentionally routed through a bridge while legacy services remain in place.
- `planned`: the surface is documented as a target, but live runtime ownership is not complete.

| Surface | Entrypoint actuel | Statut | Runtime utilisé | Limitation observable | Document propriétaire |
|---|---|---|---|---|---|
| Chat foreground | `ios/Lumen/Views/ChatView.swift` -> `ios/Lumen/Assistant/AssistantKernel+Streaming.swift`; compatibility services remain in `ios/Lumen/Services/AgentService.swift` and `ios/Lumen/Services/SlotAgentService.swift` | `live` | kernel-native | Foreground chat is the primary kernel-owned surface, but adapter evidence still needs runtime traces to prove which role adapter was applied per turn. | `docs/DEVELOPER_IMPROVE_FRAMEWORK.md` |
| Voice text-only | `ios/Lumen/Views/VoiceModeView.swift` -> `ios/Lumen/Voice/VoiceCommandRouter.swift` -> `ios/Lumen/Assistant/AssistantKernel+Streaming.swift` | `live` | kernel-native | Voice remains foreground, user-initiated, speech-service backed, and text-turn oriented; scene transitions cancel active generation and require a new user action. | `docs/VOICE_MODE.md` |
| Voice tool-capable | `ios/Lumen/Voice/VoiceCommandRouter.swift`; `ios/Lumen/Views/VoiceModeView.swift`; tool execution via `ios/Lumen/Assistant/AssistantKernel.swift` when permitted | `partial` | kernel-native | Tool-capable voice depends on explicit command routing and foreground permission boundaries; it is not equivalent to unattended headless tool execution. | `docs/AGENT_KERNEL_MIGRATION_PR4.md` |
| AppIntent | `ios/Lumen/AppIntents/LumenAskIntent.swift`; `ios/Lumen/AppIntents/LumenRunTriggerIntent.swift`; `ios/Lumen/AppIntents/LumenAddMemoryIntent.swift`; `ios/Lumen/AppIntents/LumenMemorySearchIntent.swift` | `compatibility bridge` | legacy bridge | AppIntents are guarded and privacy-constrained; sensitive or ambiguous actions can return an open-app/approval response instead of executing fully in the extension-style path. | `docs/APP_INTENTS.md` |
| Trigger/headless | `ios/Lumen/Services/TriggerScheduler.swift` -> `ios/Lumen/Services/AgentRunner.swift`; manual trigger UI starts in `ios/Lumen/Views/TriggersView.swift` | `compatibility bridge` | legacy bridge | Scheduled/background turns are background-gated, cannot prompt for permissions, and do not yet claim full kernel tool parity. | `docs/BACKGROUND_PROCESSING.md` |
| Role pipeline | `ios/Lumen/Services/RolePipelineAgentService.swift`; slot execution through `ios/Lumen/Services/SlotAgentService.swift` | `partial` | legacy bridge | Role stages use bounded grounding and secure tool bridging, but the legacy planning/execution loop remains until the adapter runtime owns per-role stage execution end to end. | `docs/LEGACY_AGENT_MIGRATION.md` |
| REM | `ios/Lumen/Services/RemCycleService.swift`; runtime repair/audit artifacts under `generated/agent_improvement_loop/` | `planned` | adapter runtime | REM is a product target for repair, memory policy, regression samples, and training feedback; it must not be treated as blocking proof that the full role-adapter runtime is live on every surface. | `docs/ADAPTER_RUNTIME_IMPROVE_LOOP.md` |
| Diagnostics/E2E | `ios/Lumen/Diagnostics/DiagnosticsProvider.swift`; `ios/Lumen/Diagnostics/PersistentRuntimeDiagnosticsRunner.swift`; E2E evidence export via `ios/Lumen/Services/Diagnostics/EvidenceLayerExporter.swift` | `live` | deterministic fallback | Diagnostics are passive/status-oriented and Live E2E owns scenario pass/fail; static and local checks support investigation but cannot replace shipped-device evidence. | `docs/RUNTIME_AUDIT_BOUNDARIES.md` |

## Canonical runtime shape labeling

When other documents mention the canonical runtime shape, use these labels:

- **Product target:** the adapter-first shape: one shared Qwen3 chat base, one Qwen3 embedding model, and one active role LoRA adapter at a time for Cortex, Executor, Mouth, Mimicry, REM, and Fleet-oriented workflows.
- **Live state by surface:** the actual status in the table above. A surface marked `live` is live for its current boundary, not automatically proof that every role adapter in the product target is active there.
- **Bridge state:** `compatibility bridge` means the entrypoint is intentionally preserved through kernel or legacy compatibility while migration continues. It should not be described as the final adapter runtime.
- **Planned state:** `planned` means the document describes intended runtime ownership or training flow rather than shipped proof for that surface.
