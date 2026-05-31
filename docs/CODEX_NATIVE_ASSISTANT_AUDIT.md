# CODEX Native Assistant Audit (Phase 1)

## Date
- 2026-05-28

## Build System
- Native iOS app at `ios/Lumen.xcodeproj`.
- SwiftUI + SwiftData app lifecycle.
- Unit tests under `ios/LumenTests`.

## Current Startup / Lifecycle
- `LumenApp` uses staged startup through `AppStartupCoordinator` with explicit stages: container, bootstrap, grounding resources, model loader, triggers, REM cycle.
- `LumenAppDelegate` handles MSAL URL callbacks and memory warnings, delegating memory cleanup to `FleetRuntimeCleanup.unloadOptionalChatSlots()`.

## Model Runtime Baseline
- `ModelLoader` already performs assignment-first slot fleet setup and lazy role loading through `SlotModelRuntimeCoordinator`.
- `ModelRuntimeController` supports loaded path refresh, load/unload/reload for chat/embedding models.
- `AppLlamaService` provides llama.cpp / SwiftLlama-backed generation and embeddings with diagnostics.
- `SlotModelRuntimeCoordinator` manages slot assignments (`cortex`, `executor`, `mouth`, `mimicry`, `rem`) and runtime readiness.

## Agent / Trigger Baseline
- Agent services exist: `AgentService`, `RolePipelineAgentService`, `SlotAgentService`, `AgentRunner`.
- Background scheduling exists in `TriggerScheduler` and is registered at launch.

## Memory / RAG Baseline
- Data models and stores exist for memory and RAG (`MemoryStore`, `RAGStore`, `VectorIndex`, `MemoryItem`, `RAGChunk`).

## Tooling / Permissions / Voice Baseline
- Existing tool execution framework exists in `ToolExecutor` + `Services/Tools/*`.
- Permission center exists (`PermissionsCenter`).
- Voice integration exists (`VoiceService`).

## Delivered Assistant Runtime Scope
- Added `AssistantKernel` as a staged assistant entrypoint; legacy agent services still remain the primary production paths where noted below.
- Added `AssistantTurnContext`, `ComputePolicy`, `AssistantRuntimeProtocols`, `AssistantRuntimeRouter`, and runtime adapters for FoundationModels, CoreML embeddings, llama, and deterministic fallback.
- Added runtime-router and adapter tests under `ios/LumenTests`.
- This audit document tracks the delivered scope and remaining integration risks.

## Remaining Integration Risks
1. `AssistantKernel` is available and used by tests/tool entrypoints, but it is not yet the single fully integrated production orchestration boundary.
2. FoundationModels generation and direct llama generation through `AssistantKernel` are staged behind availability/unavailable errors until the production runtime bridge is validated on device.
3. Full SwiftData/AppIntents/concurrency validation still requires a macOS Xcode build.

## Phase 3/4 Foundation Update (this change)
- Added system capability snapshot/profiler with public API-only hardware/runtime capability inspection.
- Added thermal, power, and memory pressure monitors plus memory-pressure metrics emission.
- Added JSONL-backed runtime metrics store (`Application Support/Lumen/runtime-metrics.jsonl`).
- Added deterministic background policy and background execution lease actor.
- Added background orchestrator wrapper that preserves existing TriggerScheduler behavior and identifiers; launch wiring remains staged to avoid duplicate BGTaskScheduler registration for the same identifiers until Xcode/device validation.
- Added entitlement validator for BG task IDs and usage description keys with non-fatal warnings.

## Phase 5 Update
- Added centralized permission domain/state/registry abstractions.
- Added secure tool framework with approval policy, bounded outputs, and metrics recorder.
- Integrated AssistantKernel tool execution through `SecureToolRegistry` while preserving legacy `ToolExecutor` path.

## Phase 5C Completion Update
- Added deferred built-in tools: `MemorySearchTool`, `RAGSearchTool`, `CalendarReadTool`, `ContactsLookupTool`, `LocationSnapshotTool`.
- Added background-safe tool filtering in `ToolRegistry` and assistant kernel tool execution tests.
- Kept legacy `ToolExecutor` intact while integrating secure path via `AssistantKernel.executeTool(...)`.

## Phase 6 Update
- Added MemoryEngine wrapper stack (`MemoryCandidate`, `MemoryScorer`, extraction policy, context builder, consolidator).
- Added RAGEngine wrapper stack (`RAGEngine`, `RAGIndexer`, `ChunkingStrategy`, `RAGContextBuilder`, maintenance policy).
- Added deterministic context budgeting (`ContextBudgetAllocator`) and `AssistantGroundingContext`.
- Integrated Memory/RAG wrappers into the staged `AssistantKernel` grounding path and background maintenance hooks.

## Phase 7 Integration Update
- Added `LegacyGroundingBridge`, `PromptGroundingRenderer`, and `PromptGroundingSection` for bounded grounding in legacy paths.
- Integrated `AgentRunner` headless flow with coordinator/bridge-generated grounding and secure-tool availability mapping.
- Added `LegacyToolSchemaBridge` for secure-to-legacy tool schema compatibility.

## Phase 7B.1/7B.2 Update
- Added shared legacy grounding coordinator + cache + prompt injection policy.
- Added legacy secure tool executor wrapper to reduce policy bypass risk in legacy services.
- Added legacy tool audit document with migration status classification.

## Phase 7C Update
- Added `LegacyPromptAssembler` and `LEGACY_PROMPT_PATH_AUDIT`.
- Headless `AgentRunner` now uses coordinator + assembler consistently.
- Legacy services route migrated tool execution through secure wrapper; prompt path migration still partial in interactive pipelines.

## Phase 7D Update
- Interactive legacy services now call a shared bounded grounding assembly helper at run-entry.
- Prompt path audit updated with explicit partial/fully-migrated status and remaining blockers.

- UI ModelContext injection audit added in `docs/MODEL_CONTEXT_INJECTION_AUDIT.md`.

- Concrete AppIntent entrypoints implemented with bounded/degraded-safe behavior and open-app-required gates for sensitive actions.

- DiagnosticsProvider and diagnostics UI surfaces implemented with metadata-only snapshots (no raw user content).

- VoiceModeView migration completed to VoiceSessionController primary state/runtime with legacy VoiceService bridge for compatibility.

- Build hardening pass documented synchronized Xcode project membership, static validation script, and SecureToolRegistry/legacy ToolRegistry symbol split.
- Build hardening also split new assistant symbols from legacy names: `SecureToolCategory`, `AssistantPermissionState`, and `AssistantDeviceCapabilitySnapshot` avoid collisions with existing UI/runtime types.

## PR #232 Review Remediation Update
- Runtime adapters no longer echo prompts; unavailable/not-yet-wired runtimes now fail with typed, sanitized errors instead of returning user input.
- `AssistantKernel.runTextTurn` is limited to text-generation tasks and records sanitized metric error codes.
- Background, metrics, permission, tool, AppIntent, Memory, and RAG review fixes are tracked in `docs/PR_REVIEW_REMEDIATION.md`.

## Final PR #232 Cleanup Update
- Final cleanup made the staged AssistantKernel `@MainActor` for ModelContext-dependent grounding/tool helpers, removed `LegacyAgentRunOptions` Sendable conformance, clarified staged BackgroundOrchestrator status, and fixed remaining legacy grounding/test hygiene comments tracked in `docs/PR_REVIEW_FINAL_CLEANUP.md`.
