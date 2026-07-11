# Feature Complete Validation

## PR #81 follow-up structured executor hardening (local validation)

The Release structured-agent path now registers classic and continued-processing BGTask handlers synchronously during `didFinishLaunching`, before launch completion is recorded. Registration outcomes retain identifier, success, launch timing, and sanitized failure metadata, and failed registrations remain retryable.

Structured tool definitions now fail closed against `SecureToolRegistry`; structured streaming and completed generation traces flow through the llama adapter injected into `AssistantKernel`. Memory save/recall synthesis requires successful trusted observations, final-step degraded location requests continue directly to `maps.search` when valid, and live evidence validation uses parser-derived trace fields rather than reparsing bounded diagnostic output.

Dataset export now enforces `modelBackedRequired`, `policyFirstAllowed`, and `routingOnly` independently. RAG chunks persist embedding format, model identifier, and dimension; legacy or mismatched vectors are excluded with `rag_reindex_required` until the user runs the existing reindex workflow. Hybrid retrieval preserves lexical-side degradation diagnostics even when semantic results remain usable.

Local `build-for-testing` compilation succeeded. Fresh signed Release, TestFlight, real-device BGTask registration, local model generation, live E2E, RAG reindex, and embedding-model-load evidence remain required before release-readiness claims.

## Executive Summary

This pass hardened Release routing and validation around the highest-risk completion gaps: production runtime selection, deterministic fallback behavior, unavailable GGUF registration, structured tool-call validation, legacy Agent Kernel bridge exposure, shipped-status documentation, and release-readiness gates.

A follow-up kernel pass replaced the Release exclusion branch for tool-capable chat and voice turns with native kernel tool execution. Tool-backed kernel turns now route through deterministic intent planning, `StructuredToolCallValidator`, `ToolRouteGuard` canonicalization, `SecureToolRegistry`, approval/permission policy, `toolInvocation` events, `toolResult` events, typed diagnostics, and honest non-success statuses.

A follow-up schema pass extended tool contracts beyond scalar-only validation. Tool argument contracts now carry array/object/enum types and enum allowed values through Swift runtime manifests and the static manifest crawler, and validation rejects nested JSON values, arrays, wrong scalar types, extra arguments, enum values outside the declared contract, and prose/markdown-noisy action payloads before execution.

A follow-up RAG/memory pass replaced product use of several lossy wrappers with diagnostic result APIs. Release-visible RAG search/indexing, memory recall, memory context building, memory cascade recall, duplicate memory save checks, and Sources UI indexing now distinguish empty success from fetch failure, permission denial, embedding failure, persistence failure, empty imports, empty photo library, empty text, cancellation/maintenance budget, and degraded fallback states.

A follow-up privacy/logging pass removed public raw diagnostic logging for prompts, queries, file/note titles, local paths, backend log lines, MSAL callback messages, and raw error descriptions. Persistent agent trace JSONL now records content hashes and character counts instead of raw prompt, raw output, tool argument, or path text.

The latest runtime-hardening pass removed remaining production `precondition` crash paths from persistent model storage, imported-file storage, secure tool registry construction, generated E2E scenario collection, GGUF construction, and default router creation. Release-visible failure now returns typed errors, diagnostics, duplicate-definition reports, or de-duped generated scenarios instead of terminating the app.

The latest slot-agent pass removed the non-diagnostic resource-budget path that returned deterministic compatibility text as a completed answer. Foreground/product budget denial now emits an `AgentEvent.error` with a typed resource-budget message and no tool execution, final delta, or done event; deterministic compatibility remains limited to diagnostic probes and explicit compatibility paths.

The latest AppIntent memory-search pass removed another Release-visible empty-success mask. `LumenMemorySearchIntent` now renders `MemoryEngine.searchWithDiagnostics(...)` results so fetch/embedding/degraded failures become explicit degraded responses, while true empty-query or empty-store states remain honest "No memories found." responses.

The latest headless model-routing pass removed an empty-fleet fallback from AppIntent/trigger entrypoints. `HeadlessAgentKernelRunner` now surfaces stored-model catalog fetch failures as explicit skipped runs with sanitized error codes instead of resolving an empty model fleet and continuing as if no models were installed.

The latest trigger persistence pass removed a generic AppIntent/UI result after trigger-save failure. Trigger runs now surface a sanitized persistence failure message instead of returning `nil` and allowing callers to display `"No result."`.

The latest model-bootstrap pass removed stored-model catalog empty fallbacks from runtime artifact preparation. `ModelLaunchBootstrap` now stops with sanitized diagnostics when SwiftData cannot fetch `StoredModel` records instead of treating the fleet as empty, all-missing, or `0 / N` ready without context; the live E2E preflight report also carries the diagnostic.

The latest trigger-tool pass removed ignored persistence and fetch-empty behavior from tool-executed trigger create/list/cancel. `TriggerTools` now surfaces fetch/save failures with sanitized diagnostics instead of reporting scheduled, cancelled, empty-list, or no-match success.

The latest REM maintenance pass removed a stored-model catalog empty fallback from diagnostic repair-cycle reporting. `RemCycleService` now records a sanitized model-catalog diagnostic when SwiftData cannot fetch `StoredModel` records instead of reporting `storedModelCount: 0` without evidence.

The latest live-E2E setup pass removed another stored-model catalog empty fallback from Settings-driven test execution. `SettingsView` now uses a diagnostic `ModelLoadSnapshot` builder and writes a single model-catalog preflight failure report when SwiftData fetch fails instead of running scenarios as if no models were installed.

The latest trigger scheduler pass removed silent trigger-fetch no-ops from background firing and local next-fire refresh. `TriggerScheduler.fireDueTriggers(...)` and `refreshNextFireTimes(...)` now return sanitized trigger fetch or persistence failures instead of quietly returning when SwiftData fetch fails.

The latest imported-file pass removed empty-list masking from local file storage and file-backed RAG indexing. `FileStore` now exposes `importedFilesWithDiagnostics(...)`; `RAGStore.indexImportedFilesWithDiagnostics(...)` and `FilesTools.readImportedFile(...)` use it so imports-directory and directory-list failures are reported as sanitized diagnostics instead of `"no imported files"` success.

The latest RAG file-extraction pass removed lossy read/decode masking inside file indexing. `RAGStore.indexFileWithDiagnostics(...)` now uses an extraction result that distinguishes file read failure, RTF read failure, RTF decode failure, PDF open failure, generic text decode failure, and true empty text with sanitized diagnostics.

The latest memory-capture queue pass removed fake pending-count fallbacks from AppIntent and queued-memory drain paths. Queue read failures now surface sanitized `pending_count_failed` diagnostics and an explicit unknown remaining count instead of becoming `0` remaining or `1` pending.

The latest file-tool read pass removed generic imported-file read/open failure text. `FilesTools.readImportedFile(...)` now routes matched files through `readMatchedFileWithDiagnostics(...)`, distinguishing file read failure, text decode failure, PDF open failure, and empty readable text with sanitized diagnostics.

The latest model-integrity pass removed another artifact-state mask from runtime selection. `ModelFileIntegrity` now exposes sanitized diagnostic codes for missing, too-small, invalid-GGUF, and unreadable model files; localized failure text no longer leaks raw local paths, and embedding candidate selection logs invalid artifact diagnostics instead of dropping candidates through a boolean filter.

The latest memory-save pass removed `try? await MemoryStore.remember(...)` from Release UI paths. Chat, voice, bookmark, and manual memory saves now use a structured diagnostics result, preserve skipped/failed reasons, and manual save stays open with a sanitized error instead of dismissing as if persistence or pinning succeeded.

The latest Settings model-directory pass removed another no-model mask from Release-visible diagnostics. Settings now uses `ModelStorage.modelFilesWithDiagnostics(...)` so model-directory/list failures remain sanitized diagnostics instead of empty file lists or unavailable success, while a genuinely empty Models directory remains a distinct loaded-empty state.

The latest Settings imported-file pass removed a matching import-storage mask from Release-visible diagnostics. Settings now uses `FileStore.importedFilesWithDiagnostics(...)` for launch diagnostics and developer checks, so imports-directory/list failures surface sanitized diagnostics instead of `0` imported files or failed read/write checks without context.

The latest imported-file write pass removed silent import-copy and attachment-metadata masking from Chat and Sources. `FileStore.importFileWithDiagnostics(...)` now distinguishes imports-directory failure, destination replacement failure, and copy failure with sanitized diagnostics; Chat shows import/picker/metadata failures instead of dropping attachments or inventing a zero-byte size, and Sources counts import failures as degraded/failed files instead of silently skipping them.

The latest attachment extraction pass removed another empty-content mask from chat prompt assembly. `AttachmentResolver.extractTextWithDiagnostics(...)` now distinguishes file read failure, PDF open failure, attributed decode failure, true empty attachment text, and successful extraction with sanitized diagnostics; `PromptAssembler` carries that state into prompt context and attachment preview instead of treating every extraction failure as an empty file.

The latest developer-trace persistence pass closed a DEBUG persistence privacy gap. `DeveloperTraceCodec` now encodes a redacted copy before JSON persistence, and chat trace context stores attachment/history hashes, counts, and summaries instead of raw prompt text, model output, tool arguments, memory text, attachment names, or local paths.

The latest secure-tool alias pass fixed the real XCTest failure found during full-suite investigation. `SecureToolRegistry` now resolves catalog-facing aliases such as `memory.recall`, `rag.search`, `calendar.list`, and `contacts.search` to their dedicated secure execution implementations before running a tool, so a background `memory.recall` without `ModelContext` returns the typed `.unavailable`/`no_model_context` result instead of falling into a generic failed compatibility path.

The latest Release-status documentation pass removed stale shipped-surface wording that still described chat/voice tool turns as Release-excluded and background execution as bridge-backed. The current matrix now reflects the implemented native kernel tool path for validated intent-routed chat and voice tool actions, and the Release hardening guard now rejects stale background bridge policy names in shipped docs.

The latest developer-console diagnostics pass removed another Release-visible empty-success/privacy mask. `DeveloperConsoleModel.logsText()` now reports imported/model storage using diagnostic result APIs, includes mode and diagnostic codes, and hashes the model directory path instead of listing raw local paths or collapsing directory/list failures to `0` files.

The latest Settings diagnostics privacy pass removed the same raw local-path leak from the Release-visible Settings diagnostics report. `SettingsView.logsText` still surfaces model/import storage counts, modes, and diagnostic codes, but the model directory is now represented as a SHA-256 path summary rather than a raw filesystem path, and the Release hardening gate rejects the old interpolation.

The latest legacy-bridge compile-surface pass made the entire legacy compatibility bridge file DEBUG-only. `LegacyAgentCompatibilityBridge` still exists for diagnostic migration probes, but its direct `AgentService.shared.run` and `SlotAgentService.shared.run` calls plus the `AgentEvent` conversion helper are no longer compiled into non-DEBUG builds; Release keeps only the explicit `runLegacyAgentBridge` unavailable diagnostic branch, and the Release hardening gate now rejects any non-comment source in that bridge file outside `#if DEBUG`.

The latest model-catalog surface pass removed fallback/unavailable wording from Release-visible model descriptors. TinyIntent can still exist as an internal bundled intent-classification engine, and Qwen/Nomic GGUF entries remain selectable model artifacts, but shipped built-in, family, and fleet catalogs no longer tag or describe models as fallback routes. The Release hardening gate now rejects fallback/mock/staged/unavailable/not-implemented wording in non-DEBUG catalog entries.

The latest deterministic-compatibility execution pass closed a Release-compiled compatibility path that still read the raw `allowDeterministicCompatibility` option. Product AgentService and SlotAgentService paths now use DEBUG-only effective gates, parse-failure deterministic recovery is disabled outside DEBUG, and the Release hardening gate rejects raw flag checks in Release-compiled execution services.

The latest legacy bridge API pass removed the Release-compiled `runLegacyAgentBridge` shim entirely. Diagnostic callers remain DEBUG-only, `AssistantKernel.runLegacyAgentBridge(...)` now exists only inside `#if DEBUG`, and the Release hardening gate rejects any future Release-visible declaration of that API instead of allowing a non-DEBUG method that returns a diagnostic error.

The latest memory tool privacy pass removed raw content and raw localized-error echoing from the Release-visible `memory.save` tool result. Memory saves now use `MemoryStore.rememberWithDiagnostics(...)`, return generic success text, preserve duplicate/empty/skipped/failed diagnostics, and the Release hardening gate rejects restoring `Saved: \(trimmed)`, raw localized save errors, or the throwing `MemoryStore.remember(...)` path in `MemoryTools`.

The latest trigger AppIntent result pass removed the remaining generic `"No result."` fallback from the Release-visible run-trigger AppIntent. Empty or nil trigger results now render a sanitized degraded diagnostic, non-empty results are still trimmed and bounded, and the Release hardening gate rejects restoring the generic no-result text.

The latest calendar reminder privacy pass removed raw localized provider errors from Release-visible reminder tool failures. Reminder add/list failures now return sanitized retry text, the Calendar policy tests assert the helper does not leak raw provider details, and the Release hardening gate rejects restoring `error.localizedDescription` in `CalendarTools`.

The latest native tool error privacy pass removed the same raw localized provider-error echoing from Release-visible alarm, health, and contacts tool failures. Alarm authorization/read/mutation/schedule/countdown failures, Health authorization failure, and Contacts search failure now use sanitized retry text, and the Release hardening gate rejects restoring `error.localizedDescription` in those tool files.

The latest live-E2E preflight workflow pass fixed a bad gate rather than masking a runtime failure. Executor readiness remains fatal for missing adapters, unavailable models, budget denial, and no-output runtime failures, but a malformed generative JSON smoke-probe response after readiness passes is now recorded as non-fatal telemetry so the actual live scenario can run. The App Store Connect lane also now treats upload-log validation errors as failed submissions even if the upload tool exits cleanly, and the submitted build number was bumped before the replacement upload.

The latest planner/schema parity pass fixed deterministic plans that were still emitting string values for numeric and boolean schema fields. Outlook list/search limits, Outlook unread filters, calendar start offsets, alarm durations, trigger relative schedules, and photo-index month windows now emit typed `AgentJSONValue` values that pass `StructuredToolCallValidator`; photo indexing also supplies the required default `months: 6` argument.

The latest native approval-boundary pass fixed the remaining known-bad native tool path for approval-required actions. Native kernel chat turns now stop after validation, emit an `.approvalBoundary` step with the validated canonical tool ID and arguments, return approval-required final text, and do not emit `toolInvocation`, `toolResult`, or execute `SecureToolRegistry` for message, mail, calendar, trigger, phone, alarm, or Outlook mutation actions before user approval.

The latest Chat approval UI mapping pass connected that native approval-boundary step to persisted chat state. `ChatView` now converts the sanitized `.approvalBoundary` step into a pending `.tool` `ChatMessage`, enqueues the canonical tool ID plus exact validated arguments in `ToolApprovalQueue`, and includes the `pendingActionID` payload required for the approval UI to verify and execute the queued action after user confirmation.

The latest strict-boundary pass removed the Release-compiled `StructuredAgentKernelExecutor` bridge to `AgentService.shared.run`. Live E2E model-backed probes now enter through `AssistantKernel.run(...)`, `check_agent_kernel_boundary.py --strict` fails on documented compatibility entries unless the legacy call is inside `#if DEBUG`, and scanner regression tests cover DEBUG versus Release branch detection.

The latest live-E2E completeness pass closed the remaining known scoring gaps from the post-merge live reports. Final-output hygiene now rejects dangling completions such as `an`, `a`, `the`, `with`, `because`, and `you do not need an`; weather observation turns repair truncated finals from the trusted tool observation before scoring. Model-backed training runs check CPU watchdog state before entering generation and emit one non-actionable runtime-preflight result when degraded. Deterministic maps planning keeps `maps.search` actionable for nearby search prompts, including degraded `location.current` observations, and missing-argument `files.read` requests clarify with `Which file should I read?` instead of returning generic safe failure text.

This is not a claim that every future product target is complete. The Release product surface now excludes experimental or legacy paths that are not release-safe, and documents those exclusions explicitly. Hardware, TestFlight, signed archive/export, and real model/device checks still require Apple credentials and physical device coverage.

## Files Changed

- Runtime adapters and routing: `ios/Lumen/Assistant/AssistantRuntimeAdapters.swift`, `ios/Lumen/Assistant/AssistantRuntimeRouter.swift`, `ios/Lumen/Assistant/AssistantKernel.swift`, `ios/Lumen/Services/LLM/LLMEngineFactory.swift`, `ios/Lumen/Services/LLM/GGUF/GGUFEngine.swift`, `ios/Lumen/Services/LLM/GGUF/UnavailableGGUFNativeBridge.swift`, `ios/Lumen/Services/LLM/GGUF/Native/LumenGGUFBridge.h`, `ios/Lumen/System/RuntimeMetric.swift`
- Agent Kernel and legacy bridge exposure: `ios/Lumen/Assistant/AssistantKernel+Streaming.swift`, `ios/Lumen/Voice/VoiceCommandRouter.swift`, `ios/Lumen/Diagnostics/PersistentRuntimeDiagnosticsRunner.swift`, `ios/Lumen/Services/AgentGrounding/AgentGroundingAuditView.swift`, `ios/Lumen/Services/E2ETestRunner.swift`, `ios/Lumen/Services/AgentService.swift`
- Tool-call validation: `ios/Lumen/Tools/ToolSchemaBridge.swift`, `ios/Lumen/Models/ToolDefinition.swift`, `ios/Lumen/Services/AgentGrounding/AgentBehaviorManifest.swift`
- Tests: `ios/LumenTests/RuntimeRouterTests.swift`, `ios/LumenTests/ToolSchemaBridgeTests.swift`, `ios/LumenTests/AssistantKernelLlamaRuntimeAdapterTests.swift`, `ios/LumenTests/AssistantKernelRunContractTests.swift`, `ios/LumenTests/AssistantKernelTextTurnRemediationTests.swift`, `ios/LumenTests/AssistantRuntimeAdapterRemediationTests.swift`, `ios/LumenTests/RuntimeContractRegressionTests.swift`, `ios/LumenTests/ToolApprovalBoundaryTests.swift`, `ios/LumenTests/AgentGroundingRegressionTests.swift`
- Gates/docs: `tools/check_release_hardening.py`, `tools/check_adapter_runtime_invariants.py`, `scripts/check-lumen-integration-gate.sh`, `README.md`, `CLAUDE.md`, `docs/RUNTIME_STATUS_MATRIX.md`, `docs/AGENT_KERNEL_MIGRATION_STATUS.md`, `docs/VALIDATION.md`

Follow-up kernel tool execution files:

- Native kernel execution: `ios/Lumen/Assistant/AssistantKernel+Streaming.swift`
- Voice routing: `ios/Lumen/Voice/VoiceCommandRouter.swift`
- Tests: `ios/LumenTests/AssistantKernelRunContractTests.swift`, `ios/LumenTests/AgentKernelBoundaryGuardTests.swift`

Follow-up native approval-boundary files:

- Native kernel approval boundary: `ios/Lumen/Assistant/AssistantKernel+Streaming.swift`
- Approval boundary regression test: `ios/LumenTests/AssistantKernelRunContractTests.swift`

Follow-up Chat approval UI mapping files:

- Chat approval persistence: `ios/Lumen/Views/ChatView.swift`
- Approval payload mapper and queue integration: `ios/Lumen/Models/ToolApprovalState.swift`
- Approval mapper regression test: `ios/LumenTests/ToolApprovalQueueTests.swift`

Follow-up strict-boundary files:

- Live E2E kernel entrypoint: `ios/Lumen/Services/E2ETestRunner.swift`
- Removed Release bridge: `ios/Lumen/Assistant/StructuredAgentKernelExecutor.swift`
- Boundary guard: `tools/check_agent_kernel_boundary.py`
- Tests: `ios/LumenTests/AgentKernelBoundaryGuardTests.swift`, `tools/pipeline/tests/test_check_agent_kernel_boundary.py`
- Shipped-state docs: `docs/AGENT_KERNEL_MIGRATION_STATUS.md`, `docs/LEGACY_AGENT_MIGRATION.md`, `docs/CODEX_NATIVE_ASSISTANT_AUDIT.md`

Follow-up live-E2E completeness files:

- Final hygiene, preflight, pacing, and quarantine scoring: `ios/Lumen/Services/E2ETestRunner.swift`
- Maps/files deterministic planning and clarification: `ios/Lumen/Services/DeterministicToolPlanner.swift`, `ios/Lumen/Services/SlotAgentService.swift`, `ios/Lumen/Services/IntentRouter.swift`
- Regression tests: `ios/LumenTests/E2ETestRunnerHygieneTests.swift`, `ios/LumenTests/DeterministicToolPlannerTests.swift`, `ios/LumenTests/IntentClassifierPolicyTests.swift`
- Boundary guard documentation: `tools/check_agent_kernel_boundary.py`

Follow-up tool JSON hardening files:

- Swift contracts and runtime manifest: `ios/Lumen/Models/ToolDefinition.swift`, `ios/Lumen/Services/AgentGrounding/AgentBehaviorManifest.swift`
- Validator: `ios/Lumen/Tools/ToolSchemaBridge.swift`
- Static manifest crawler: `tools/lumen_manifest_crawler/lumen_manifest_crawler/manifest.py`, `tools/lumen_manifest_crawler/lumen_manifest_crawler/swift_extractors/tool_definition.py`, `tools/lumen_manifest_crawler/lumen_manifest_crawler/validators.py`
- Crawler tests: `tools/lumen_manifest_crawler/tests/test_tool_extraction.py`, `tools/lumen_manifest_crawler/tests/test_manifest_validation.py`

Follow-up RAG/memory diagnostic files:

- RAG diagnostics: `ios/Lumen/Services/RAGStore.swift`, `ios/Lumen/RAG/RAGEngine.swift`, `ios/Lumen/Services/VectorIndex.swift`, `ios/Lumen/Tools/Builtin/RAGSearchTool.swift`
- Memory diagnostics: `ios/Lumen/Services/MemoryStore.swift`, `ios/Lumen/Services/MemoryRecall.swift`, `ios/Lumen/Services/MemoryCascade.swift`, `ios/Lumen/Memory/MemoryEngine.swift`, `ios/Lumen/Memory/MemoryContextBuilder.swift`, `ios/Lumen/Memory/MemoryConsolidator.swift`, `ios/Lumen/Services/Tools/MemoryTools.swift`, `ios/Lumen/Services/VectorIndex.swift`
- Sources UI: `ios/Lumen/Views/SourcesView.swift`
- Tests: `ios/LumenTests/MemoryToolsTests.swift`, `ios/LumenTests/PersistenceAuditTests.swift`, `ios/LumenTests/RAGSearchToolTests.swift`
- Release hardening gate: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up privacy/logging files:

- Sanitized public logging: `ios/Lumen/Services/RAGStore.swift`, `ios/Lumen/Services/MemoryStore.swift`, `ios/Lumen/Services/VectorIndex.swift`, `ios/Lumen/Services/ModelLaunchBootstrap.swift`, `ios/Lumen/Services/TriggerScheduler.swift`, `ios/Lumen/Services/SlotModelRuntimeCoordinator.swift`, `ios/Lumen/Services/LlamaService.swift`, `ios/Lumen/Services/ModelDownloader.swift`, `ios/Lumen/Background/BackgroundContinuedProcessingCoordinator.swift`, `ios/Lumen/LumenApp.swift`, `ios/Lumen/KnowledgeGraph/KnowledgeGraphService.swift`, `ios/Lumen/Services/MicrosoftGraph/MicrosoftGraphAuthManager.swift`
- Persistent trace redaction: `ios/Lumen/Services/AgentGrounding/AgentBehaviorTrace.swift`, `ios/Lumen/Services/AgentService.swift`
- Tests/gate: `ios/LumenTests/AgentGroundingRegressionTests.swift`, `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up headless/AppIntent/background Release surface files:

- Background tool execution policy: `ios/Lumen/Assistant/BackgroundToolExecutionPolicy.swift` renamed from `ios/Lumen/Assistant/BackgroundToolBridgePolicy.swift`
- Kernel/headless naming and execution: `ios/Lumen/Assistant/AssistantKernel+Streaming.swift`, `ios/Lumen/Assistant/AgentKernelContracts.swift`, `ios/Lumen/Assistant/HeadlessAgentKernelRunner.swift`
- Secure tool command execution rename: `ios/Lumen/Tools/ToolRegistry.swift`, `ios/Lumen/Services/SlotAgentService.swift`, `ios/Lumen/Services/AgentService.swift`, `ios/Lumen/Views/MessageBubble.swift`
- DEBUG-only probe wording: `ios/Lumen/Services/AgentGrounding/AgentGroundingAuditView.swift`, `ios/Lumen/Services/E2ETestRunner.swift`, `ios/Lumen/Diagnostics/PersistentRuntimeDiagnosticsRunner.swift`
- Shipped docs and gates: `docs/BACKGROUND_PROCESSING.md`, `docs/TOOL_SECURITY_MODEL.md`, `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`, `ios/LumenTests/AgentKernelBoundaryGuardTests.swift`

Follow-up production unfinished-marker gate files:

- Release catalog cleanup: `ios/Lumen/Services/LLM/Models/BuiltInModelCatalog.swift`
- Catalog regression test: `ios/LumenTests/LLMModelStorageTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`
- Integration readiness gate: `scripts/check-ios-build-readiness.sh`

Follow-up GGUF Release crash-path files:

- GGUF typed unavailable path: `ios/Lumen/Services/LLM/GGUF/GGUFEngine.swift`
- Default router unavailable registration policy: `ios/Lumen/Services/LLM/LLMEngineFactory.swift`
- Release crash regression gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up production precondition crash-path files:

- Persistent model directory diagnostics: `ios/Lumen/Services/ModelStorage.swift`, `ios/Lumen/Models/StoredModel.swift`, `ios/Lumen/Services/ModelDownloader.swift`, `ios/Lumen/Views/SettingsView.swift`
- Imported-file directory diagnostics: `ios/Lumen/Services/SharedContainer.swift`, `ios/Lumen/Views/SettingsView.swift`
- Duplicate-definition resilience: `ios/Lumen/Tools/ToolRegistry.swift`, `ios/Lumen/Services/E2ETestRunner.swift`
- Tests/gates: `ios/LumenTests/RuntimeHardeningTests.swift`, `ios/LumenTests/SecureToolRegistryTests.swift`, `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up slot-agent resource-budget hardening files:

- Slot-agent runtime stream behavior: `ios/Lumen/Services/SlotAgentService.swift`
- Runtime contract regression test: `ios/LumenTests/RuntimeContractRegressionTests.swift`

Follow-up AppIntent memory-search diagnostics files:

- AppIntent renderer: `ios/Lumen/AppIntents/LumenMemorySearchIntent.swift`
- AppIntent regression tests: `ios/LumenTests/LumenMemorySearchIntentTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up headless model-routing diagnostics files:

- Headless runner: `ios/Lumen/Assistant/HeadlessAgentKernelRunner.swift`
- Headless regression tests: `ios/LumenTests/AgentRunnerHeadlessPromptGroundingTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up trigger persistence diagnostics files:

- Trigger execution: `ios/Lumen/Services/TriggerScheduler.swift`
- Trigger persistence regression tests: `ios/LumenTests/PersistenceAuditTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up model-bootstrap catalog diagnostics files:

- Runtime artifact bootstrap: `ios/Lumen/Services/ModelLaunchBootstrap.swift`
- Live E2E artifact preflight reporting: `ios/Lumen/Services/E2ETestRunner.swift`, `ios/Lumen/Views/SettingsView.swift`
- Regression tests: `ios/LumenTests/PersistenceAuditTests.swift`, `ios/LumenTests/E2ETestRunnerHygieneTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up trigger tool persistence diagnostics files:

- Trigger tool execution: `ios/Lumen/Services/Tools/TriggerTools.swift`
- Trigger tool persistence regression tests: `ios/LumenTests/PersistenceAuditTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up REM model-catalog diagnostics files:

- REM maintenance reporting: `ios/Lumen/Services/RemCycleService.swift`
- REM persistence regression tests: `ios/LumenTests/PersistenceAuditTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up Settings live-E2E model-catalog diagnostics files:

- Model-load snapshot diagnostics: `ios/Lumen/Services/ModelLoader.swift`
- Settings live-E2E setup: `ios/Lumen/Views/SettingsView.swift`
- Live-E2E blocked report: `ios/Lumen/Services/E2ETestRunner.swift`
- Regression tests: `ios/LumenTests/PersistenceAuditTests.swift`, `ios/LumenTests/E2ETestRunnerHygieneTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up trigger scheduler fetch diagnostics files:

- Trigger scheduler execution: `ios/Lumen/Services/TriggerScheduler.swift`
- Trigger scheduler regression tests: `ios/LumenTests/PersistenceAuditTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up imported-file diagnostics files:

- File storage diagnostics: `ios/Lumen/Services/SharedContainer.swift`
- File-backed RAG indexing: `ios/Lumen/Services/RAGStore.swift`
- File read tool: `ios/Lumen/Services/Tools/FilesTools.swift`
- File storage regression tests: `ios/LumenTests/RuntimeHardeningTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up RAG file-extraction diagnostics files:

- File extraction diagnostics: `ios/Lumen/Services/RAGStore.swift`
- RAG persistence regression tests: `ios/LumenTests/PersistenceAuditTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up memory-capture queue diagnostics files:

- Queue diagnostics: `ios/Lumen/Memory/MemoryCaptureQueue.swift`
- Queue metrics summary: `ios/Lumen/Memory/MemoryConsolidator.swift`
- AppIntent pending capture renderer: `ios/Lumen/AppIntents/LumenAddMemoryIntent.swift`
- Regression tests: `ios/LumenTests/MemoryCaptureQueueTests.swift`, `ios/LumenTests/LumenAddMemoryIntentPolicyTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up file-tool read diagnostics files:

- File tool diagnostics: `ios/Lumen/Services/Tools/FilesTools.swift`
- File tool regression tests: `ios/LumenTests/RuntimeHardeningTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up model-integrity diagnostics files:

- Model artifact integrity diagnostics: `ios/Lumen/Services/ModelFileIntegrity.swift`
- Runtime candidate selection: `ios/Lumen/Services/SlotModelRuntimeCoordinator.swift`
- Model integrity regression tests: `ios/LumenTests/LLMModelStorageTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up memory-save diagnostics files:

- Memory persistence result API: `ios/Lumen/Services/MemoryStore.swift`
- Memory-save UI callers: `ios/Lumen/Views/ChatView.swift`, `ios/Lumen/Views/VoiceModeView.swift`, `ios/Lumen/Views/MessageBubble.swift`, `ios/Lumen/Views/MemoryView.swift`
- Memory persistence regression tests: `ios/LumenTests/PersistenceAuditTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up Settings model-directory diagnostics files:

- Model directory listing diagnostics: `ios/Lumen/Services/ModelStorage.swift`
- Settings diagnostics UI: `ios/Lumen/Views/SettingsView.swift`
- Runtime hardening tests: `ios/LumenTests/RuntimeHardeningTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up Settings imported-file diagnostics files:

- Imported-file diagnostics shape: `ios/Lumen/Services/SharedContainer.swift`
- Settings diagnostics UI: `ios/Lumen/Views/SettingsView.swift`
- Runtime hardening tests: `ios/LumenTests/RuntimeHardeningTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up imported-file write diagnostics files:

- Imported-file write diagnostics shape: `ios/Lumen/Services/SharedContainer.swift`
- Attachment metadata resolver: `ios/Lumen/Models/ChatAttachment.swift`
- Chat attachment import UI: `ios/Lumen/Views/ChatView.swift`
- Sources import/index UI: `ios/Lumen/Views/SourcesView.swift`
- Runtime hardening tests: `ios/LumenTests/RuntimeHardeningTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up attachment extraction diagnostics files:

- Attachment extraction diagnostics: `ios/Lumen/Models/ChatAttachment.swift`
- Prompt assembly state: `ios/Lumen/Services/PromptBudget.swift`
- Attachment preview UI: `ios/Lumen/Views/ChatView.swift`
- Runtime hardening tests: `ios/LumenTests/RuntimeHardeningTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up developer trace persistence files:

- Developer trace persistence redaction: `ios/Lumen/Services/LLM/DeveloperTrace.swift`
- Chat trace context hashing/summarization: `ios/Lumen/Views/ChatView.swift`
- Regression tests and release gate: `ios/LumenTests/AgentGroundingRegressionTests.swift`, `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up secure tool alias execution files:

- Secure alias execution routing: `ios/Lumen/Tools/ToolRegistry.swift`
- Regression tests: `ios/LumenTests/SecureToolRegistryBackgroundFilteringTests.swift`, `ios/LumenTests/AssistantKernelRunContractTests.swift`

Follow-up Release status documentation files:

- Release matrix and migration status: `docs/RUNTIME_STATUS_MATRIX.md`, `docs/AGENT_KERNEL_MIGRATION_STATUS.md`
- Background policy wording: `docs/RUNTIME_POLICY.md`
- Release doc gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up developer-console diagnostics files:

- Developer console storage diagnostics: `ios/Lumen/Developer/DeveloperFramework.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up Settings diagnostics privacy files:

- Settings diagnostics report: `ios/Lumen/Views/SettingsView.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up legacy bridge compile-surface files:

- DEBUG-only legacy bridge implementation and helper extension: `ios/Lumen/Assistant/LegacyAgentCompatibilityBridge.swift`
- Boundary regression test: `ios/LumenTests/AgentKernelBoundaryGuardTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up model catalog Release-surface files:

- Built-in model catalog copy: `ios/Lumen/Services/LLM/Models/BuiltInModelCatalog.swift`
- Fleet/family catalog copy: `ios/Lumen/Services/ModelFamilySelection.swift`, `ios/Lumen/Services/ModelFleetCatalog.swift`
- Catalog regression tests: `ios/LumenTests/LLMModelStorageTests.swift`, `ios/LumenTests/LumenFleetTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up deterministic compatibility execution files:

- Effective run options: `ios/Lumen/Assistant/LegacyAgentRunOptions.swift`
- Release-compiled execution call sites: `ios/Lumen/Services/AgentService.swift`, `ios/Lumen/Services/SlotAgentService.swift`
- Regression tests and release gate: `ios/LumenTests/LegacyAgentRunOptionsTests.swift`, `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up trigger AppIntent result files:

- Run-trigger AppIntent result rendering: `ios/Lumen/AppIntents/LumenRunTriggerIntent.swift`
- AppIntent regression tests: `ios/LumenTests/LumenRunTriggerIntentPolicyTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up calendar reminder privacy files:

- Reminder tool failure rendering: `ios/Lumen/Services/Tools/CalendarTools.swift`
- Calendar policy regression tests: `ios/LumenTests/CalendarReadToolPolicyTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up native tool error privacy files:

- Alarm failure rendering: `ios/Lumen/Services/Tools/AlarmTools.swift`
- Health failure rendering: `ios/Lumen/Services/Tools/HealthTools.swift`
- Contacts failure rendering: `ios/Lumen/Services/Tools/ContactsTools.swift`
- Native tool privacy regression tests: `ios/LumenTests/SecureToolRegistryTests.swift`, `ios/LumenTests/ContactsLookupToolPolicyTests.swift`
- Release hardening gate/tests: `tools/check_release_hardening.py`, `tools/pipeline/tests/test_check_release_hardening.py`

Follow-up planner/schema parity files:

- Deterministic typed tool planning: `ios/Lumen/Services/DeterministicToolPlanner.swift`
- Planner/schema regression tests: `ios/LumenTests/DeterministicToolPlannerTests.swift`

## Previous Gaps Closed

- Production-selectable deterministic fallback: Release routing no longer selects the deterministic runtime by default; DEBUG can still exercise it for diagnostics.
- FoundationModels/CoreML experimental adapters: both are explicitly non-selectable and report experimental Release exclusion instead of staged implementation wording.
- GGUF unavailable native bridge: the unavailable bridge is DEBUG-only, and the default factory cannot register it in Release.
- Runtime failure fallback: `AssistantKernel.runTextTurn` no longer catches a selected local runtime failure and turns it into deterministic assistant text.
- Legacy Agent Kernel bridge exposure: chat tool turns, voice tool turns, diagnostics, E2E, and grounding audit live probes are DEBUG-only where they still need legacy behavior; Release emits explicit unavailable/skipped states.
- Tool-call schema safety: parsed actions now pass central validation for known tool, availability, required arguments, JSON value types, and extra arguments before any tool execution.
- Shipped-status docs: current Release docs no longer label shipped surfaces as partial, planned, or bridge-backed.
- Release tool-turn exclusion: chat and voice tool-capable turns no longer return a Release exclusion message; the kernel executes validated native tool invocations and emits explicit results.
- Enum and nested JSON validation: `trigger.create.schedule` is declared as an enum with `absolute`, `interval`, and `relative` allowed values; nested objects/arrays are preserved during validation so alias normalization cannot coerce invalid JSON into strings.
- Strict action framing: prose/markdown-noisy parsed actions are converted to typed `noisyOutput` failures before tool selection, and the one-tool missing-action repair refuses noisy payloads.
- RAG/memory empty-success masking: product RAG/memory paths now use diagnostic result APIs instead of interpreting `0`, `[]`, `[:]`, or `"[]"` as success. Empty stores remain explicit empty states; fetch, persist, embedding, permission, cancellation, export, vector-index load, and maintenance-budget failures remain diagnostics.
- RAG/memory regression gate: `tools/check_release_hardening.py` now fails Release source scans that reintroduce product calls to lossy RAG/memory wrappers, `try? FetchDescriptor<RAGChunk|MemoryItem> ?? []`, or `"[]"` memory export fallbacks.
- Unsafe diagnostics/logging: public logs now use sanitized error codes, hashes, counts, private privacy annotations, or operational status fields instead of raw prompts, queries, file/note titles, local paths, backend messages, MSAL messages, and raw error descriptions. Persistent agent trace files hash raw content before writing.
- Unsafe diagnostics regression gate: `tools/check_release_hardening.py` now fails Release source scans that reintroduce public raw error diagnostics, public sensitive interpolations, raw sensitive logger interpolations, direct raw persistent diagnostic fields, or raw developer trace encoding outside `#if DEBUG`.
- Background/headless Release wording: background tool-only execution is now named as execution policy rather than bridge policy; shipped docs no longer describe a compatibility bridge or unsupported bridge mappings.
- Legacy command API naming: migrated string-command secure tool execution now uses `SecureToolRegistry.executeToolCommand(...)`; `executeLegacyTool(...)` is rejected by the Release hardening gate.
- Release legacy diagnostic wording: the non-DEBUG `runLegacyAgentBridge` branch now reports a DEBUG-only migration diagnostic instead of a shipped legacy bridge exclusion message, and tests/gates reject restoring the old wording.
- Future embedding catalog descriptor: the `nomic-embed-text-local` descriptor is now DEBUG-only and no longer appears in the Release built-in catalog.
- Production unfinished markers: Release source scans now fail on `TODO`, `FIXME`, `XXX`, `stub`, `not implemented`, `not-implemented`, and `unimplemented` outside `#if DEBUG`; the iOS readiness gate also hard-fails production unfinished markers under `ios/Lumen`.
- GGUF Release construction crash: `GGUFEngine()` no longer calls `preconditionFailure` in Release when no native bridge is compiled. It constructs with a nil bridge and returns typed `LLMEngineError.backendUnavailable("GGUF native backend is not compiled.")` from load/generation paths.
- Unavailable GGUF router flag: `LLMEngineFactory.makeDefaultRouter(includeUnavailableGGUF:)` no longer uses a Release `precondition`; the unavailable bridge registration remains DEBUG-only and Release simply does not register the unavailable GGUF backend.
- GGUF crash regression gate: `tools/check_release_hardening.py` now rejects GGUF-related `precondition` and `preconditionFailure` crash paths.
- Persistent app-data crash paths: `ModelStorage` and `FileStore` no longer terminate when the app cannot resolve or create persistent directories. Throwing APIs surface directory failures to model download/import callers, while compatibility accessors use isolated temporary fallback directories instead of crashing.
- Model download unavailable-directory handling: model download start/resume/finish paths now return `.persistentDirectoryUnavailable` or fail the download with a sanitized persistent-directory diagnostic instead of continuing with an invalid destination.
- Duplicate secure tool definitions: `SecureToolRegistry` no longer crashes on duplicate tool IDs. It keeps the first definition, reports duplicate IDs through `duplicateDefinitionIDs()`, and preserves deterministic registry behavior.
- Duplicate generated E2E scenarios: live tool coverage scenario generation de-dupes by scenario ID instead of using a production precondition.
- Production precondition regression gate: `tools/check_release_hardening.py` now rejects production `precondition` and `preconditionFailure` usage under Release-scanned source.
- Slot-agent resource-budget denial: non-diagnostic SlotAgent runs no longer convert a denied heavy-model budget into deterministic compatibility answer text. They emit a typed resource-budget error and finish the stream without `finalDelta`, `done`, or tool execution.
- Deterministic resource fallback source cleanup: the unused deterministic compatibility fallback helper was deleted after the non-diagnostic branch stopped using it.
- AppIntent memory search empty-success masking: `LumenMemorySearchIntent` no longer calls the lossy `MemoryEngine().search(...)` wrapper. Failed or degraded memory searches render degraded output with sanitized diagnostics instead of `"No memories found."`; true empty stores and empty queries still render as empty results.
- Lossy memory engine wrapper regression gate: `tools/check_release_hardening.py` now fails product calls to `MemoryEngine().search(...)` outside the owner implementation and allows `searchWithDiagnostics(...)`.
- Headless empty-fleet masking: AppIntent and trigger entrypoints no longer use `try? FetchDescriptor<StoredModel>() ?? []` before resolving the model fleet. Stored-model fetch failure now returns `Headless agent skipped: model catalog fetch failed (...)` with a sanitized error code.
- Headless model-fetch regression gate: `tools/check_release_hardening.py` rejects reintroducing the lossy `StoredModel` fetch-empty fallback in `HeadlessAgentKernelRunner`.
- Trigger persistence nil masking: `TriggerScheduler.runTrigger(...)` no longer returns `nil` when saving the updated trigger state fails. It now returns `Trigger failed: persistence save failed (...)`, so AppIntents and UI callers do not collapse the failure into `"No result."`.
- Trigger persistence regression gate: `tools/check_release_hardening.py` rejects restoring `catch { return nil }` in `TriggerScheduler`.
- Model-bootstrap empty catalog masking: `ModelLaunchBootstrap` no longer uses `try? FetchDescriptor<StoredModel>() ?? []` while repairing, linking, polling, or reporting live runtime artifacts. Fetch failure now logs a sanitized error code, updates boot diagnostics where an `AppState` is available, and stops the path instead of presenting empty/missing artifact state as fact.
- Live E2E artifact preflight diagnostics: runtime artifact blocked reports now include the readiness diagnostic in final text and metadata when model-catalog fetch failed.
- Model-bootstrap regression gate: `tools/check_release_hardening.py` rejects reintroducing stored-model fetch-empty fallback in `ModelLaunchBootstrap`.
- Trigger tool ignored-save masking: `trigger.create` and `trigger.cancel` no longer report scheduled or cancelled when `ctx.save()` fails.
- Trigger tool fetch-empty masking: `trigger.list` and `trigger.cancel` no longer turn fetch failure into `"No scheduled runs."` or `"No trigger matching ..."`.
- Trigger tool regression gate: `tools/check_release_hardening.py` rejects restoring `try? ctx.save()` and trigger fetch-empty fallback in `TriggerTools`.
- REM empty-fleet masking: `RemCycleService` no longer turns `StoredModel` fetch failure into a diagnostic report with `storedModelCount: 0` and no failure context.
- REM model-fetch regression gate: `tools/check_release_hardening.py` rejects reintroducing the lossy `StoredModel` fetch-empty fallback in `RemCycleService`.
- Settings live-E2E no-model masking: `SettingsView` no longer turns a `StoredModel` fetch failure into an empty `ModelLoadSnapshot` that later reports generic `"no chat model loaded"`.
- Settings live-E2E model-fetch regression gate: `tools/check_release_hardening.py` rejects reintroducing the lossy `StoredModel` fetch-empty fallback in `SettingsView`.
- Trigger scheduler fetch no-op masking: `fireDueTriggers(...)` and `refreshNextFireTimes(...)` no longer silently return when `Trigger` fetch fails.
- Trigger scheduler fetch regression gate: `tools/check_release_hardening.py` rejects restoring `guard let all = try? context.fetch(FetchDescriptor<Trigger>()) else { return }` in `TriggerScheduler`.
- Imported-file empty-list masking: RAG file indexing and the local file read tool no longer turn imports-directory or directory-list failure into empty imported-file results.
- Imported-file regression gate: `tools/check_release_hardening.py` rejects `FileStore.importedFiles()` and directory `try? contentsOfDirectory(...) ?? []` in `RAGStore` and `FilesTools`.
- RAG file extraction read/decode masking: `RAGStore.indexFileWithDiagnostics(...)` no longer uses `try? Data(contentsOf:)` or `try? NSAttributedString(...)` while indexing files. Read failures, RTF decode failures, PDF open failures, and generic text decode failures now produce distinct sanitized diagnostics.
- RAG file extraction regression gate: `tools/check_release_hardening.py` rejects lossy `try? Data(contentsOf:)` and `try? NSAttributedString(...)` inside `RAGStore.swift`.
- Memory-capture pending-count masking: AppIntent queued memory capture and queue drain skip paths no longer use `(try? pendingCount(...)) ?? 0/1`. Queue read failures now produce unknown remaining/pending counts plus sanitized diagnostics.
- Memory-capture queue regression gate: `tools/check_release_hardening.py` rejects pending-count `try?` fallbacks in `MemoryCaptureQueue` and `LumenAddMemoryIntent`.
- File-tool read/open masking: `FilesTools.readImportedFile(...)` no longer uses `try? Data(contentsOf:)` or generic `Couldn't read/open` responses for matched imported files. Read failure, text decode failure, PDF open failure, and empty text now produce sanitized diagnostics.
- File-tool read regression gate: `tools/check_release_hardening.py` rejects lossy file-tool read fallbacks and generic file-tool open/read failure text in `FilesTools.swift`.
- Installed model integrity masking: embedding candidate selection no longer uses `candidates.filter { ModelFileIntegrity.validateInstalledFile($0) }`, which dropped missing/corrupt/unreadable artifacts without an explicit reason. Invalid candidates now emit sanitized `skip_invalid_artifact` diagnostics.
- Model integrity privacy leak: `ModelFileIntegrity.Failure.localizedDescription` no longer includes raw local file paths for missing, unreadable, or invalid GGUF files; path context is carried only as a short SHA-256 diagnostic.
- Model integrity regression gate: `tools/check_release_hardening.py` rejects restoring the lossy installed-model integrity boolean filter in `SlotModelRuntimeCoordinator`.
- Memory save failure masking: chat, voice, bookmark, and manual memory-save paths no longer use `try? await MemoryStore.remember(...)`. Failed or skipped memory saves now return structured mode/diagnostic data, and manual save/pin errors remain visible instead of dismissing the sheet as success.
- Memory save regression gate: `tools/check_release_hardening.py` rejects `try? await MemoryStore.remember(...)` in Release-scanned source.
- Settings model-directory masking: Settings launch diagnostics no longer turn `ModelStorage.modelsDirectoryURLOrThrow()` or model-file directory-list failure into `unavailable`/`0 files` without a diagnostic. The UI now reports model file count, mode, sanitized diagnostic, and path when available.
- Settings model-directory regression gate: `tools/check_release_hardening.py` rejects reintroducing lossy Settings `ModelStorage.modelsDirectoryURLOrThrow()` and `contentsOfDirectory` `try?` fallbacks.
- Settings imported-file masking: Settings launch diagnostics and developer checks no longer call `FileStore.importedFiles()` or `try? FileStore.importsDirectoryOrThrow(...)`. Import directory/list failures now carry mode, directory when available, and sanitized diagnostics.
- Settings imported-file regression gate: `tools/check_release_hardening.py` rejects reintroducing lossy Settings imported-file wrappers and imports-directory `try?` fallbacks.
- Imported-file write masking: Chat and Sources no longer call the optional `FileStore.importFile(...)` wrapper. Directory, destination replacement, copy, and attachment metadata failures now surface sanitized diagnostics instead of disappearing as a skipped attachment/file or becoming a fake zero-byte attachment.
- Imported-file write regression gate: `tools/check_release_hardening.py` rejects restoring `FileStore.importFile(...)` in Chat or Sources product paths, rejects attachment-size `?? 0` fallback in `ChatAttachment`, and allows `FileStore.importFileWithDiagnostics(...)`.
- Attachment extraction masking: Chat prompt assembly no longer calls a raw string-returning attachment extractor that turns read/open/decode failures into `""`. Failed extraction produces sanitized diagnostics in prompt context and preview state, while genuinely empty text remains `empty_attachment_text`.
- Attachment extraction regression gate: `tools/check_release_hardening.py` rejects raw attachment extraction wrappers, `try? Data(contentsOf:)`, `try? NSAttributedString(...)`, and PDF-open `return ""` fallbacks in attachment extraction code.
- Developer trace raw persistence: DEBUG developer trace JSON persistence no longer writes raw system/developer/user prompts, resolved context, retrieved memory, tool arguments, model output, reasoning text, visible answer, parser warnings, or raw errors. The codec redacts before encoding, and Chat trace context records hashes/counts or sanitized summaries for history, attachments, and memory.
- Developer trace privacy regression gate: `tools/check_release_hardening.py` rejects raw `encoder.encode(trace)`, raw attachment name/path trace context, and raw history/memory content assignments in developer trace surfaces.
- Secure tool alias execution: catalog IDs now execute their dedicated secure implementations where one exists. In particular, `memory.recall` executes the context-aware memory search path, so background/no-context execution returns typed `.unavailable` with `no_model_context` instead of generic `.failed`.
- Release status doc drift: shipped docs no longer describe chat or voice tool turns as Release-excluded now that the native kernel tool path is implemented for validated intent-routed actions.
- Stale background bridge doc wording: shipped docs now name `BackgroundToolExecutionPolicy`, and `tools/check_release_hardening.py` rejects `BackgroundToolBridgePolicy`, `BackgroundToolBridgeAssessment`, or `bridgeMappingUnavailable` in Release-status docs.
- Developer console storage masking: the developer console no longer uses `FileStore.importedFiles()` or `try? contentsOfDirectory(...) ?? []` for its report counts. It reports imported/model storage mode and sanitized diagnostic codes, and exposes only a SHA-256 path summary for the model directory.
- Settings raw model path exposure: Settings launch diagnostics no longer print `modelFilesResult.directory?.path`; they keep counts/modes/diagnostics and expose only a SHA-256 path summary for the model directory.
- Legacy bridge Release compile surface: the whole `LegacyAgentCompatibilityBridge.swift` file is wrapped in `#if DEBUG`, so its direct legacy service calls and event conversion helper are not part of non-DEBUG compilation, and `tools/check_release_hardening.py` rejects restoring the bridge enum outside a DEBUG-only block.

## Runtime Adapters Final Status

- Canonical production text path: SwiftLlama/AppLlamaService through `LlamaRuntimeAdapter.live(...)`.
- FoundationModels: excluded from Release routing until a real generation implementation exists.
- CoreML embeddings: excluded from Release routing until real embedding extraction exists.
- Deterministic runtime: DEBUG diagnostics only.
- Mock backend: no default factory registration path.

## GGUF/Native/Local Model Final Status

- `GGUFEngine` still owns lifecycle, prompt building, cancellation, and validation tests.
- Release builds require a compiled native bridge to load and use GGUF. Constructing `GGUFEngine()` without that bridge returns typed unavailable-backend failures instead of crashing.
- `UnavailableGGUFNativeBridge` is wrapped in `#if DEBUG`.
- The model fleet and adapter lifecycle checks remain covered by `tools/check_adapter_runtime_invariants.py`.

## Structured JSON/Tool-Calling Final Status

- Existing structured generation still requests constrained JSON and uses bounded retry paths for empty/incomplete output.
- New `StructuredToolCallValidator` blocks unknown tools, unavailable tools, missing required arguments, wrong JSON types, and extra arguments.
- `StructuredToolCallValidator` now supports array/object/enum argument declarations and returns a typed enum-value failure when a string is outside the declared allowed set.
- Tool execution is reached only after validation returns a canonical tool ID and normalized argument dictionary.
- Kernel-native tool turns emit `toolInvocation` and `toolResult` events after validation and registry policy checks.
- `SecureToolRegistry` resolves catalog aliases to preferred secure execution IDs before dispatch, preserving catalog-facing IDs while avoiding generic compatibility implementations for `memory.recall`, `rag.search`, `calendar.list`, and `contacts.search`.
- Added adversarial tests for unknown tool, unavailable manifest tool, missing key, wrong type, nested invalid args, array-vs-number mismatch, enum failure, extra argument, and benign alias normalization.
- Static manifest extraction now preserves `allowedValues` so generated schema manifests do not silently lose enum constraints.
- AgentService now treats noisy parsed action payloads as non-executable `.noisyOutput` turns. The parser can still recover noisy final text for diagnostics/user-facing repair, but tools are not run from prose- or markdown-wrapped action JSON.

## Agent Kernel Migration Final Status

- Shipped chat/voice text turns use native kernel entrypoints.
- Tool-capable chat/voice turns use native kernel tool execution for validated intent-planned actions.
- Live legacy diagnostic probes remain DEBUG-only.
- `tools/check_release_hardening.py` enforces that the legacy compatibility bridge file stays fully DEBUG-only, and that legacy bridge calls and unavailable GGUF construction stay DEBUG-only.

## Model Fleet Final Status

- Qwen3 shared-base plus role-adapter lifecycle remains the canonical local model fleet shape.
- Existing invariants verify adapter-first catalog shape, role adapter switching, missing adapter hard failures, and active adapter diagnostics.
- Real-device model load and role-adapter switching still require physical-device validation with actual artifacts.

## RAG/Memory Final Status

- `RAGStore` now exposes diagnostic APIs for search, counts, chunks, imported-file indexing, photo indexing, note indexing, and file indexing. Backward-compatible wrappers still exist, but product/UI/tool callers added in this pass use diagnostic APIs.
- `RAGEngine.retrieveWithDiagnostics` preserves search mode and diagnostics for Release-visible search callers.
- `RAGSearchTool` no longer uses `try? fetch ?? []` lexical fallback; fetch/persist/permission/corruption diagnostics become failed tool results instead of empty success.
- `MemoryTools.recall`, RAG search, RAG file indexing, and RAG photo indexing now preserve diagnostic user-facing messages.
- `MemoryRecall`, `MemoryCascade`, `MemoryEngine`, `MemoryContextBuilder`, and `MemoryConsolidator` preserve or record diagnostic state instead of silently converting memory fetch failures to empty context.
- `MemoryStore.remember` no longer proceeds after duplicate-check fetch failure; it logs a sanitized diagnostic and throws.
- `MemoryStore.exportJSONWithDiagnostics` reports export failures explicitly, and the compatibility `exportJSON` wrapper now returns a diagnostic error JSON object instead of `"[]"`.
- `RAGVectorIndex` and `MemoryVectorIndex` return sanitized load diagnostics instead of treating fetch failure as an empty loaded index; failed loads remain retryable.
- `SourcesView` no longer reports file/photo/note indexing as successful when diagnostic indexing fails or is skipped.
- `LumenMemorySearchIntent` renders diagnostic memory search results, distinguishing failed/degraded search from a true empty memory store.
- Remaining task 5 work is broader proof/gating, not a known current product caller still using the specific lossy RAG/memory wrappers patched in this pass.

## Voice/AppIntent/Headless Final Status

- Voice text turns: shipped through the kernel.
- Voice tool turns: routed through the same native kernel tool execution path as chat.
- AppIntents: shipped only for guarded local actions and degraded/open-app responses.
- Headless/background: shipped only for background-safe coordination without unavailable model loading.

## Privacy/Logging Final Status

- Runtime fallback logging uses prompt hashes and prompt sizes, not raw prompts.
- Persistent diagnostics record prompt SHA-256 and byte counts for live probes.
- Agent behavior, parse-failure, and parse-noise JSONL recorders persist redacted copies with content hashes and character counts for prompt, raw output, streamed text, selected JSON, tool arguments, stop sequences, and model/runtime paths.
- DEBUG developer trace JSON persistence also encodes a redacted copy; attachment/history trace context stores hashes and counts, and memory trace context stores sanitized summaries.
- RAG/memory/model/runtime/startup logs publish typed error codes and stable status fields; sensitive paths, backend messages, query text, file/note titles, MSAL callback messages, and raw error descriptions are private or hashed.
- `check-lumen-integration-gate.sh` continues to run privacy/build-hardening checks and now includes `tools/check_release_hardening.py`.

## Tests Added or Updated

- Release-style runtime router tests prove diagnostic fallback is not selected when `allowDiagnosticFallbackSelection` is false.
- Tool schema tests prove malformed or untrusted tool actions are rejected before execution.
- Runtime adapter tests now assert experimental Release exclusion wording and sanitized error codes.
- Kernel runtime tests now assert selected runtime failures propagate instead of deterministic fallback success.
- RAG/memory tests now assert diagnostic index messages preserve failure reason, RAG retrieval diagnostics survive empty/degraded states, counts/chunks diagnostic APIs distinguish loaded empty stores, normalized memory recall preserves empty-query diagnostics, vector-index load failures remain diagnostic/retryable, and memory export failure does not become an empty JSON-array success.
- Release hardening tests now assert lossy RAG/memory product paths are rejected and diagnostic RAG/memory product paths are allowed.
- Privacy/logging tests now assert persistent agent behavior traces and parse traces redact raw prompt, raw output, tool argument, and path content before file persistence.
- Release hardening tests now assert unsafe public diagnostics are rejected and private/hash/error-code diagnostics are allowed.
- Release hardening tests now assert removed legacy command APIs, background bridge policy names, shipped compatibility-bridge docs, and old Release legacy-bridge exclusion wording are rejected.
- Agent kernel boundary tests now assert the old Release legacy-bridge exclusion string is not restored.
- Release hardening tests now assert production unfinished markers are rejected outside `#if DEBUG` and allowed inside DEBUG-only diagnostic source.
- Model catalog tests now assert the local Nomic descriptor remains non-embedding in DEBUG and is absent from non-DEBUG builds.
- Release hardening tests now assert GGUF-related `precondition` and `preconditionFailure` crash paths are rejected.
- Runtime hardening tests now assert persistent model directory failures throw typed errors and imported-file compatibility accessors do not crash when persistent directories are unavailable.
- Secure tool registry tests now assert duplicate tool IDs are reported without crashing and that the first definition remains the executable definition.
- Release hardening tests now assert production `precondition` and `preconditionFailure` crash paths are rejected.
- Runtime contract tests now assert production SlotAgent resource-budget denial emits exactly `SlotAgentService.resourceBudgetDeniedMessage()`, never emits `finalDelta` or `done`, and does not mention deterministic fallback behavior.
- AppIntent memory-search tests now assert failed search and degraded empty search do not render as `"No memories found."`, while a true empty memory store still does.
- Release hardening tests now assert product calls to `MemoryEngine().search(...)` are rejected and `MemoryEngine().searchWithDiagnostics(...)` is allowed.
- Headless runner tests now assert stored-model fetch failure is not rendered as "local model not loaded" and does not expose a raw persistence error.
- Release hardening tests now assert headless `StoredModel` fetch-empty fallback is rejected and throwing fetch is allowed.
- Trigger persistence tests now assert save failure is explicit, does not render as `"No result"`, and does not expose the raw persistence error.
- Release hardening tests now assert trigger persistence `catch { return nil }` is rejected and the sanitized failure-message path is allowed.
- Model-bootstrap tests now assert catalog fetch failure messaging is explicit and sanitized.
- E2E hygiene tests now assert live runtime artifact preflight reports preserve the model-catalog diagnostic in metadata and final text.
- Release hardening tests now assert `ModelLaunchBootstrap` stored-model fetch-empty fallback is rejected and the diagnostic fetch path is allowed.
- Trigger tool persistence tests now assert create/list/cancel failure messages are explicit, sanitized, and cannot be confused with scheduled, cancelled, no-match, or empty-list success.
- Release hardening tests now assert `TriggerTools` ignored-save and fetch-empty fallbacks are rejected while explicit catch paths are allowed.
- REM persistence tests now assert model-catalog fetch failure is explicit, sanitized, and preserved in `RemCycleReport.modelCatalogDiagnostic`.
- Release hardening tests now assert `RemCycleService` stored-model fetch-empty fallback is rejected and the diagnostic snapshot path is allowed.
- Model-load snapshot tests now assert stored-model fetch failure produces a sanitized diagnostic instead of a snapshot with an empty model list.
- E2E hygiene tests now assert Settings/live-E2E model-catalog failure produces a single preflight report with `failureKind=liveModelCatalogFetchFailed` and does not render as generic no-model output.
- Release hardening tests now assert `SettingsView` stored-model fetch-empty fallback is rejected and `ModelLoader.modelLoadSnapshot(...)` is allowed.
- Trigger scheduler tests now assert trigger fetch failure renders an explicit sanitized message and cannot be confused with `"No result"` or empty-list success.
- Release hardening tests now assert `TriggerScheduler` silent fetch returns are rejected and explicit fetch failure messages are allowed.
- FileStore tests now assert imports-directory failure, contents-list failure, and true empty imports are distinct states with sanitized diagnostics.
- Release hardening tests now assert imported-file empty fallback paths are rejected and `importedFilesWithDiagnostics(...)` is allowed in RAG/file tool product code.
- RAG persistence tests now assert file read, RTF decode, and PDF open failures are distinct sanitized extraction diagnostics and do not leak raw local paths.
- Release hardening tests now assert lossy RAG file read/decode fallbacks are rejected and the diagnostic extraction path is allowed.
- Memory capture queue tests now assert queue read failure does not become `0` remaining and does not leak raw local file paths.
- AppIntent memory tests now assert queued-memory copy reports pending captures as `unknown` with a diagnostic instead of inventing a count.
- Release hardening tests now assert lossy memory-capture pending-count fallbacks are rejected and diagnostic pending-count paths are allowed.
- Runtime hardening tests now assert file-tool read failure, text decode failure, PDF open failure, and empty text are distinct sanitized diagnostics.
- Release hardening tests now assert generic file-tool read/open failures are rejected and the diagnostic read path is allowed.
- Model storage tests now assert missing model files and invalid GGUF magic produce distinct sanitized integrity failures without raw path leakage.
- Release hardening tests now assert installed-model integrity boolean filters are rejected and diagnostic integrity selection is allowed.
- Persistence tests now assert `MemoryStore.rememberWithDiagnostics(...)` returns a skipped `empty_content` diagnostic for empty memory input.
- Release hardening tests now assert lossy `try? await MemoryStore.remember(...)` calls are rejected and `rememberWithDiagnostics(...)` is allowed.
- Runtime hardening tests now assert model-directory failure, model-file list failure, and true empty model directories are distinct states with sanitized diagnostics.
- Release hardening tests now assert Settings model-directory/file-list `try?` fallbacks are rejected and `ModelStorage.modelFilesWithDiagnostics(...)` is allowed.
- Runtime hardening tests now assert imported-file directory/list failures preserve directory state where available and do not leak raw paths.
- Release hardening tests now assert Settings imported-file wrappers and imports-directory `try?` fallbacks are rejected, while `FileStore.importedFilesWithDiagnostics(...)` is allowed.
- Runtime hardening tests now assert imported-file write directory failure, destination replacement failure, copy failure, attachment metadata failure, and success are distinct and sanitized.
- Release hardening tests now assert Chat/Sources imported-file write wrappers and fake zero-byte attachment metadata fallbacks are rejected, while `FileStore.importFileWithDiagnostics(...)` product paths are allowed.
- Runtime hardening tests now assert attachment text read failure, PDF open failure, attributed decode failure, true empty text, and prompt-assembly failure surfacing are distinct and sanitized.
- Release hardening tests now assert attachment extraction empty fallbacks and raw extraction wrapper usage are rejected, while `AttachmentResolver.extractTextWithDiagnostics(...)` paths are allowed.
- Agent grounding regression tests now assert persisted developer trace JSON redacts raw prompts, memory text, tool arguments, attachment names, and local paths while retaining hashes and tool IDs.
- Release hardening tests now assert raw developer trace encoding and raw attachment/history/memory trace context assignments are rejected.
- Secure tool registry tests now assert `memory.recall` executes the secure memory search implementation and returns `.unavailable`/`no_model_context` without a model context.
- The focused `AssistantKernelRunContractTests.testBackgroundTriggerToolIntentUsesNativeBackgroundSafeExecution` rerun now passes after the alias-routing fix.
- Release hardening tests now assert shipped docs reject stale background bridge policy names as well as unproven shipped-status vocabulary.
- Release hardening tests now assert developer-console storage diagnostics reject lossy file-count fallbacks and raw model path output, while allowing diagnostic APIs plus hashed path summaries.
- Release hardening tests now assert Settings diagnostics reject raw model-directory path interpolation and allow diagnostic APIs plus hashed path summaries.
- Release hardening tests now assert the legacy compatibility bridge file rejects non-comment source outside `#if DEBUG`, and the agent-kernel boundary test asserts the bridge file starts and ends with that DEBUG gate.
- Release hardening tests now assert non-DEBUG built-in, family, and fleet model catalog entries reject fallback/mock/staged/unavailable surface wording, while DEBUG-only diagnostic descriptors may keep that wording.
- Legacy run-option tests now assert deterministic compatibility execution and parse-failure deterministic recovery are DEBUG-only effective gates, and Release hardening tests reject raw `options.allowDeterministicCompatibility` checks in Release-compiled AgentService/SlotAgentService execution code.
- AppIntent trigger policy tests now assert empty run-trigger results render a degraded diagnostic instead of `"No result."`, and Release hardening tests reject restoring that generic fallback.
- Calendar policy tests now assert reminder failure text is sanitized, and Release hardening tests reject raw `error.localizedDescription` in `CalendarTools`.
- Secure tool registry and contacts policy tests now assert alarm, health, and contacts failure text is sanitized, and Release hardening tests reject raw `error.localizedDescription` in `AlarmTools`, `HealthTools`, and `ContactsTools`.
- Deterministic planner tests now validate representative planned actions through `StructuredToolCallValidator` and assert numeric/bool schema values for Outlook, calendar, alarm, trigger, and RAG photo-index arguments.

## Commands Run

| Command | Result |
| --- | --- |
| `git diff --check` | Passed |
| `python -m compileall tools scripts` | Failed: `python` executable not found in this shell |
| `python3 -m compileall tools scripts` | Passed |
| `bash scripts/check-lumen-integration-gate.sh` | Passed |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` | Passed: 170 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` | Passed: 187 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` | Passed after fixing a Swift string interpolation error in `ToolSchemaBridge.swift` |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO` | Completed with exit code 0 |

Integration-gate note: `check-ios-build-readiness.sh` now hard-fails production unfinished markers in `ios/Lumen`. It still prints informational placeholder/logging review lines for existing docs/tests and `RuntimeFallbackLogger.swift`; the script exits successfully.

## Current Follow-up Commands

| Command | Result |
| --- | --- |
| `git diff --check` | Passed |
| `python3 -m compileall tools scripts` | Passed |
| `python3 tools/check_release_hardening.py` | Passed |
| `bash scripts/check-lumen-integration-gate.sh` | Passed |
| `uv run --python 3.12 pytest tests/test_tool_extraction.py tests/test_manifest_validation.py` | Passed: 18 passed |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` | Passed: 179 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` | Passed |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test -only-testing:LumenTests/AssistantKernelRunContractTests -only-testing:LumenTests/AgentKernelBoundaryGuardTests CODE_SIGNING_ALLOWED=NO` | Interrupted after 181.7s; Xcode reported `TEST INTERRUPTED` while waiting for simulator install/launch workers. No executed XCTest pass is claimed for this follow-up. |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO` | Interrupted manually after the run went silent during simulator test execution. No full XCTest pass is claimed for this follow-up. |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after strict noisy-action guard | Passed |
| `git diff --check` after RAG/memory diagnostics | Passed |
| `python3 -m compileall tools scripts` after RAG/memory diagnostics | Passed |
| `python3 tools/check_release_hardening.py` after RAG/memory diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after RAG/memory diagnostics | Passed; advisory TODO/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after RAG/memory diagnostics | Passed: 179 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after RAG/memory diagnostics | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after RAG/memory diagnostics | First run failed on Swift inference in `MemoryRecall`; after adding an explicit closure return type, rerun passed |
| `git diff --check` after vector-index/export diagnostics | Passed |
| `python3 -m compileall tools scripts` after vector-index/export diagnostics | Passed |
| `python3 tools/check_release_hardening.py` after vector-index/export diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after vector-index/export diagnostics | Passed; advisory TODO/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after vector-index/export diagnostics | Passed: 179 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after vector-index/export diagnostics | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after vector-index/export diagnostics | Passed |
| `git diff --check` after lossy RAG/memory release gate | Passed |
| `python3 tools/check_release_hardening.py` after lossy RAG/memory release gate | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` | Passed: 9 passed |
| `python3 -m compileall tools scripts` after lossy RAG/memory release gate | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after lossy RAG/memory release gate | Passed; advisory TODO/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after lossy RAG/memory release gate | Passed: 181 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after lossy RAG/memory release gate | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after lossy RAG/memory release gate | Passed |
| `git diff --check` after privacy/logging hardening | Passed |
| `python3 -m compileall tools scripts` after privacy/logging hardening | Passed |
| `python3 tools/check_release_hardening.py` after privacy/logging hardening | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after privacy/logging hardening | Passed: 11 passed |
| `bash scripts/check-lumen-integration-gate.sh` after privacy/logging hardening | Passed; advisory TODO/placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after privacy/logging hardening | Passed: 183 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after privacy/logging hardening | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after privacy/logging hardening | Passed |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO` after privacy/logging hardening | Inconclusive; interrupted/killed after 470.9s. Xcode reported `Testing started` but blocked waiting for simulator install/launch workers to materialize and `runningDidFinish`. No XCTest pass is claimed. |
| `python3 tools/check_release_hardening.py` after headless/background Release surface cleanup | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after headless/background Release surface cleanup | Passed: 13 passed |
| `git diff --check` after headless/background Release surface cleanup | Passed |
| `python3 -m compileall tools scripts` after headless/background Release surface cleanup | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after headless/background Release surface cleanup | Passed; advisory TODO/placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after headless/background Release surface cleanup | Passed: 185 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after headless/background Release surface cleanup | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after headless/background Release surface cleanup | Failed before compilation with exit code 70: no available device matched `platform=iOS Simulator,name=iPhone 16`; Xcode listed only generic iOS and generic iOS Simulator destinations. |
| `xcrun simctl list devices available` during headless/background Release surface cleanup | Eventually returned iOS 26.3 shutdown devices `Lumen Focused Test iPhone` and `monGARS Test iPhone`; no `iPhone 16` device was available. |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' build-for-testing CODE_SIGNING_ALLOWED=NO` after headless/background Release surface cleanup | Failed before compilation with exit code 70: Xcode still listed only generic iOS and generic iOS Simulator destinations and rejected the simctl-listed device. |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'generic/platform=iOS Simulator' build-for-testing CODE_SIGNING_ALLOWED=NO` after headless/background Release surface cleanup | Inconclusive; build began, removed stale `BackgroundToolBridgePolicy` artifacts, compiled into `Lumen`, then went idle with no compiler child processes and was interrupted after about 3 minutes. No compile pass is claimed. |
| `python3 tools/check_release_hardening.py` after production unfinished-marker gate | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after production unfinished-marker gate | Passed: 15 passed |
| `rg -n -e "TODO" -e "FIXME" -e "XXX" -e "stub" -e "not implemented" -e "not-implemented" -e "unimplemented" ios/Lumen -g '*.swift' -g '*.h' -g '*.m' -g '*.mm'` after production unfinished-marker gate | Passed: no matches |
| `git diff --check` after production unfinished-marker gate | Passed |
| `python3 -m compileall tools scripts` after production unfinished-marker gate | Passed |
| `bash scripts/check-ios-build-readiness.sh` after production unfinished-marker gate | Passed; production unfinished markers are now hard failures, while historical/test placeholder and logger-presence notices remain informational |
| `bash scripts/check-lumen-integration-gate.sh` after production unfinished-marker gate | Passed; includes the stricter Release hardening guard and iOS readiness production unfinished-marker check |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after production unfinished-marker gate | Passed: 187 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after production unfinished-marker gate | Passed: 189 tests collected |
| `python3 tools/check_release_hardening.py` after GGUF Release crash-path cleanup | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after GGUF Release crash-path cleanup | Passed: 16 passed |
| `git diff --check` after GGUF Release crash-path cleanup | Passed |
| `python3 -m compileall tools scripts` after GGUF Release crash-path cleanup | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after GGUF Release crash-path cleanup | Passed; includes stricter GGUF crash regression gate |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after GGUF Release crash-path cleanup | Passed: 188 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after GGUF Release crash-path cleanup | Passed: 189 tests collected |
| `xcrun simctl create 'iPhone 16' com.apple.CoreSimulator.SimDeviceType.iPhone-16 com.apple.CoreSimulator.SimRuntime.iOS-26-3` during GGUF Release crash-path cleanup | Inconclusive; command hung and was interrupted, but the simulator was created before interruption. Xcode subsequently listed `iPhone 16` as an available destination. |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -sdk iphonesimulator -configuration Debug build-for-testing CODE_SIGNING_ALLOWED=NO` after GGUF Release crash-path cleanup | First run failed on `BuiltInModelCatalog` conditional array syntax; after rewriting the catalog as an append closure, rerun proceeded without filtered Swift compiler errors but was interrupted after `actool`/`AssetCatalogSimulatorAgent` hung. No full build pass is claimed. |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after GGUF Release crash-path cleanup | Inconclusive; exact destination was accepted after simulator creation, but build hung in `actool`/`AssetCatalogSimulatorAgent` and was interrupted. No build pass is claimed. |
| `rg -n -e "preconditionFailure\\(" -e "precondition\\(" ios/Lumen -g '*.swift'` after production precondition crash-path cleanup | Passed: no matches |
| `python3 tools/check_release_hardening.py` after production precondition crash-path cleanup | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after production precondition crash-path cleanup | Passed: 16 passed |
| `git diff --check` after production precondition crash-path cleanup | Passed |
| `python3 -m compileall tools scripts` after production precondition crash-path cleanup | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after production precondition crash-path cleanup | Passed; includes production precondition crash regression gate |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after production precondition crash-path cleanup | Passed: 188 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after production precondition crash-path cleanup | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after production precondition crash-path cleanup | Passed: `TEST BUILD SUCCEEDED` |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO` after production precondition crash-path cleanup | Inconclusive; reached `Testing started`, then stalled waiting for `runningDidFinish` and was interrupted. Xcode reported `TEST INTERRUPTED`. No XCTest pass is claimed. |
| `git diff --check` after SlotAgent budget-denial hardening | Passed |
| `python3 tools/check_release_hardening.py` after SlotAgent budget-denial hardening | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after SlotAgent budget-denial hardening | Passed: 16 passed |
| `rg -n "fallbackBehavior: \"return deterministic compatibility response\"\|resource-budget-fallback\|deterministicCompatibilityFallback\\(\\)" ios/Lumen ios/LumenTests -g '*.swift'` after SlotAgent budget-denial hardening | Passed: no matches |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after SlotAgent budget-denial hardening | Passed: `TEST BUILD SUCCEEDED` |
| `python3 -m compileall tools scripts` after SlotAgent budget-denial hardening | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after SlotAgent budget-denial hardening | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after SlotAgent budget-denial hardening | Passed: 188 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after SlotAgent budget-denial hardening | Passed: 189 tests collected |
| `git diff --check` after AppIntent memory-search diagnostics | Passed |
| `python3 tools/check_release_hardening.py` after AppIntent memory-search diagnostics | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after AppIntent memory-search diagnostics | Passed: 16 passed |
| `rg -n "MemoryEngine\\(\\)\\.search\\(|MemoryStore\\.recall\\(" ios/Lumen ios/LumenTests -g '*.swift'` after AppIntent memory-search diagnostics | Passed: no matches |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after AppIntent memory-search diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `python3 -m compileall tools scripts` after AppIntent memory-search diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after AppIntent memory-search diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after AppIntent memory-search diagnostics | Passed: 188 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after AppIntent memory-search diagnostics | Passed: 189 tests collected |
| `git diff --check` after headless model-fetch diagnostics | Passed |
| `python3 tools/check_release_hardening.py` after headless model-fetch diagnostics | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after headless model-fetch diagnostics | Passed: 18 passed |
| `rg -n "try\\?[^\\n]*fetch\\(FetchDescriptor<StoredModel>\\(\\)\\)\\) \\?\\? \\[\\]" ios/Lumen/Assistant/HeadlessAgentKernelRunner.swift ios/LumenTests/AgentRunnerHeadlessPromptGroundingTests.swift` after headless model-fetch diagnostics | Passed: no matches |
| `python3 -m compileall tools scripts` after headless model-fetch diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after headless model-fetch diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after headless model-fetch diagnostics | Passed: 190 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after headless model-fetch diagnostics | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after headless model-fetch diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `git diff --check` after trigger persistence diagnostics | Passed |
| `python3 tools/check_release_hardening.py` after trigger persistence diagnostics | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after trigger persistence diagnostics | Passed: 20 passed |
| `python3 -m compileall tools scripts` after trigger persistence diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after trigger persistence diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after trigger persistence diagnostics | Passed: 192 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after trigger persistence diagnostics | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after trigger persistence diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `git diff --check` after model-bootstrap catalog diagnostics | Passed |
| `python3 tools/check_release_hardening.py` after model-bootstrap catalog diagnostics | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after model-bootstrap catalog diagnostics | Passed: 22 passed |
| `rg -n "try\\?[^\\n]*fetch\\(FetchDescriptor<StoredModel>\\|FetchDescriptor<StoredModel>\\(\\)\\)\\) \\?\\? \\[\\]" ios/Lumen/Services/ModelLaunchBootstrap.swift ios/Lumen/Assistant/HeadlessAgentKernelRunner.swift` after model-bootstrap catalog diagnostics | Passed: no matches |
| `python3 -m compileall tools scripts` after model-bootstrap catalog diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after model-bootstrap catalog diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after model-bootstrap catalog diagnostics | Passed: 194 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after model-bootstrap catalog diagnostics | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after model-bootstrap catalog diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `git diff --check` after trigger tool persistence diagnostics | Passed |
| `python3 tools/check_release_hardening.py` after trigger tool persistence diagnostics | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after trigger tool persistence diagnostics | Passed: 24 passed |
| `rg -n "try\\?\\s*ctx\\.save\\(|try\\?\\s*ctx\\.fetch\\(FetchDescriptor<Trigger>\\(\\)\\)\\) \\?\\? \\[\\]" ios/Lumen/Services/Tools/TriggerTools.swift tools/pipeline/tests/test_check_release_hardening.py` after trigger tool persistence diagnostics | Passed for product source: only unsafe test fixtures matched |
| `python3 -m compileall tools scripts` after trigger tool persistence diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after trigger tool persistence diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after trigger tool persistence diagnostics | Passed: 196 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after trigger tool persistence diagnostics | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after trigger tool persistence diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `python3 tools/check_release_hardening.py` after REM model-catalog diagnostics | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after REM model-catalog diagnostics | Passed: 26 passed |
| `rg -n "try\\?[^\\n]*fetch\\(FetchDescriptor<StoredModel>\\(\\)\\)\\) \\?\\? \\[\\]" ios/Lumen/Services/RemCycleService.swift ios/Lumen/Services/ModelLaunchBootstrap.swift ios/Lumen/Assistant/HeadlessAgentKernelRunner.swift tools/pipeline/tests/test_check_release_hardening.py` after REM model-catalog diagnostics | Passed for product source: only unsafe test fixtures matched |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after REM model-catalog diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `git diff --check` after REM model-catalog diagnostics | Passed |
| `python3 -m compileall tools scripts` after REM model-catalog diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after REM model-catalog diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after REM model-catalog diagnostics | Passed: 198 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after REM model-catalog diagnostics | Passed: 189 tests collected |
| `git diff --check` after Settings live-E2E model-catalog diagnostics | Passed |
| `python3 tools/check_release_hardening.py` after Settings live-E2E model-catalog diagnostics | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after Settings live-E2E model-catalog diagnostics | Passed: 28 passed |
| `rg -n "try\\?[^\\n]*fetch\\(FetchDescriptor<StoredModel>\\(\\)\\)\\) \\?\\? \\[\\]" ios/Lumen/Views/SettingsView.swift ios/Lumen/Services/RemCycleService.swift ios/Lumen/Services/ModelLaunchBootstrap.swift ios/Lumen/Assistant/HeadlessAgentKernelRunner.swift tools/pipeline/tests/test_check_release_hardening.py` after Settings live-E2E model-catalog diagnostics | Passed for product source: only unsafe test fixtures matched |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after Settings live-E2E model-catalog diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `python3 -m compileall tools scripts` after Settings live-E2E model-catalog diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after Settings live-E2E model-catalog diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after Settings live-E2E model-catalog diagnostics | Passed: 200 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after Settings live-E2E model-catalog diagnostics | Passed: 189 tests collected |
| `git diff --check` after trigger scheduler fetch diagnostics | Passed |
| `python3 tools/check_release_hardening.py` after trigger scheduler fetch diagnostics | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after trigger scheduler fetch diagnostics | Passed: 30 passed |
| `rg -n "guard\\s+let\\s+\\w+\\s*=\\s*try\\?[^\\n]*fetch\\(FetchDescriptor<Trigger>\\(\\)\\)\\s*else\\s*\\{\\s*return\\s*\\}" ios/Lumen/Services/TriggerScheduler.swift tools/pipeline/tests/test_check_release_hardening.py` after trigger scheduler fetch diagnostics | Passed for product source: only the unsafe test fixture matched |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after trigger scheduler fetch diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `python3 -m compileall tools scripts` after trigger scheduler fetch diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after trigger scheduler fetch diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after trigger scheduler fetch diagnostics | Passed: 202 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after trigger scheduler fetch diagnostics | Passed: 189 tests collected |
| `git diff --check` after imported-file diagnostics | Passed |
| `python3 tools/check_release_hardening.py` after imported-file diagnostics | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after imported-file diagnostics | Passed: 32 passed |
| `rg -n "FileStore\\.importedFiles\\(|try\\?[^\\n]*contentsOfDirectory\\([^\\n]*\\)\\) \\?\\? \\[\\]" ios/Lumen/Services/RAGStore.swift ios/Lumen/Services/Tools/FilesTools.swift tools/pipeline/tests/test_check_release_hardening.py` after imported-file diagnostics | Passed for product source: only unsafe test fixtures matched |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after imported-file diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `python3 -m compileall tools scripts` after imported-file diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after imported-file diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after imported-file diagnostics | Passed: 204 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after imported-file diagnostics | Passed: 189 tests collected |
| `python3 tools/check_release_hardening.py` after RAG file-extraction diagnostics | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after RAG file-extraction diagnostics | Passed: 34 passed |
| `rg -n "try\\?\\s*(Data\\s*\\(\\s*contentsOf|NSAttributedString\\s*\\()" ios/Lumen/Services/RAGStore.swift tools/pipeline/tests/test_check_release_hardening.py` after RAG file-extraction diagnostics | Passed for product source: only unsafe test fixtures matched |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after RAG file-extraction diagnostics | Passed: `TEST BUILD SUCCEEDED` after fixing test actor isolation and a closure escaping signature |
| `git diff --check` after RAG file-extraction diagnostics | Passed |
| `python3 -m compileall tools scripts` after RAG file-extraction diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after RAG file-extraction diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after RAG file-extraction diagnostics | Passed: 206 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after RAG file-extraction diagnostics | Passed: 189 tests collected |
| `python3 tools/check_release_hardening.py` after memory-capture queue diagnostics | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after memory-capture queue diagnostics | Passed: 36 passed |
| `rg -n "try\\?[^\\n]*pendingCount|drain\\.remaining\\)" ios/Lumen/Memory/MemoryCaptureQueue.swift ios/Lumen/Memory/MemoryConsolidator.swift ios/Lumen/AppIntents/LumenAddMemoryIntent.swift tools/pipeline/tests/test_check_release_hardening.py` after memory-capture queue diagnostics | Passed for product source: only unsafe test fixtures matched |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after memory-capture queue diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `git diff --check` after memory-capture queue diagnostics | Passed |
| `python3 -m compileall tools scripts` after memory-capture queue diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after memory-capture queue diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after memory-capture queue diagnostics | Passed: 208 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after memory-capture queue diagnostics | Passed: 189 tests collected |
| `python3 tools/check_release_hardening.py` after file-tool read diagnostics | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after file-tool read diagnostics | Passed: 38 passed |
| `rg -n "try\\?\\s*Data\\s*\\(\\s*contentsOf|Couldn['’]?t read|Couldn['’]?t open PDF" ios/Lumen/Services/Tools/FilesTools.swift tools/pipeline/tests/test_check_release_hardening.py` after file-tool read diagnostics | Passed for product source: only unsafe test fixtures matched |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after file-tool read diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `git diff --check` after file-tool read diagnostics | Passed |
| `python3 -m compileall tools scripts` after file-tool read diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after file-tool read diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after file-tool read diagnostics | Passed: 210 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after file-tool read diagnostics | Passed: 189 tests collected |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after model-integrity diagnostics | Passed: 40 passed |
| `python3 tools/check_release_hardening.py` after model-integrity diagnostics | Passed |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after model-integrity diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `git diff --check` after model-integrity diagnostics | Passed |
| `python3 -m compileall tools scripts` after model-integrity diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after model-integrity diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `rg -n "filter \\{ ModelFileIntegrity\\.validateInstalledFile\\(\\$0\\) \\}|Model file is missing:|Downloaded file is not a GGUF model:|Model file is unreadable:" ios/Lumen/Services/ModelFileIntegrity.swift ios/Lumen/Services/SlotModelRuntimeCoordinator.swift tools/pipeline/tests/test_check_release_hardening.py` after model-integrity diagnostics | Passed: no matches |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after model-integrity diagnostics | Passed: 212 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after model-integrity diagnostics | Passed: 189 tests collected |
| `rg -n "try\\?\\s+await\\s+MemoryStore\\.remember|try\\?\\s+MemoryStore\\.remember" ios/Lumen -g '*.swift'` after memory-save diagnostics | Passed: no matches |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after memory-save diagnostics | Passed: 42 passed |
| `python3 tools/check_release_hardening.py` after memory-save diagnostics | Passed |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after memory-save diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `git diff --check` after memory-save diagnostics | Passed |
| `python3 -m compileall tools scripts` after memory-save diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after memory-save diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after memory-save diagnostics | Passed: 214 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after memory-save diagnostics | Passed: 189 tests collected |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after Settings model-directory diagnostics | Passed: 44 passed |
| `python3 tools/check_release_hardening.py` after Settings model-directory diagnostics | Passed |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after Settings model-directory diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `git diff --check` after Settings model-directory diagnostics | Passed |
| `python3 -m compileall tools scripts` after Settings model-directory diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after Settings model-directory diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after Settings model-directory diagnostics | Passed: 216 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after Settings model-directory diagnostics | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO` after Settings model-directory diagnostics | Inconclusive; command reached `Testing started`, then the simulator launch path failed/stalled. Xcode reported `NSMachErrorDomain Code=-308` from `IDELaunchiPhoneSimulatorLauncher`, blocked waiting for test workers/log finalization, and ended with `BUILD INTERRUPTED` after terminating the stuck `xcodebuild` process. No executed XCTest pass is claimed. |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after Settings imported-file diagnostics | Passed: 46 passed |
| `python3 tools/check_release_hardening.py` after Settings imported-file diagnostics | Passed |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after Settings imported-file diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `git diff --check` after Settings imported-file diagnostics | Passed |
| `python3 -m compileall tools scripts` after Settings imported-file diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after Settings imported-file diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after Settings imported-file diagnostics | Passed: 218 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after Settings imported-file diagnostics | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO` after Settings imported-file diagnostics | Not rerun in this pass; the immediately preceding full simulator test attempt for the Settings diagnostics work hit `NSMachErrorDomain Code=-308` in `IDELaunchiPhoneSimulatorLauncher` and was recorded above. No executed XCTest pass is claimed. |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after imported-file write diagnostics | Passed: 50 passed |
| `python3 tools/check_release_hardening.py` after imported-file write diagnostics | Passed |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after imported-file write diagnostics | First run failed on a Swift shadowing compile error in `SharedContainer.swift`; after the fix and final attachment metadata guard it passed: `TEST BUILD SUCCEEDED` |
| `git diff --check` after imported-file write diagnostics | Passed |
| `python3 -m compileall tools scripts` after imported-file write diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after imported-file write diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after imported-file write diagnostics | Passed: 222 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after imported-file write diagnostics | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO` after imported-file write diagnostics | Interrupted/inconclusive before the final attachment metadata guard. The run reached `Testing started`, then stalled with Xcode blocked on `waiting for workers to materialize`, unfinished `IDEInstalliPhoneSimulatorWorker` and `IDELaunchiPhoneSimulatorLauncher`, and `Waiting for -runningDidFinish call`; `xcodebuild` reported `TEST INTERRUPTED` after 173.006 seconds, with logs at `~/Library/Developer/Xcode/DerivedData/Lumen-gafqarfynlsuiseecxqmygoyalln/Logs/Test/Test-Lumen-2026.07.07_05-20-12--0400.xcresult`. It was not rerun after the metadata guard to avoid repeated simulator launches after the same stuck worker condition. No executed XCTest pass is claimed for this slice. |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after attachment extraction diagnostics | Passed: 52 passed |
| `python3 tools/check_release_hardening.py` after attachment extraction diagnostics | Passed |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after attachment extraction diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `git diff --check` after attachment extraction diagnostics | Passed |
| `python3 -m compileall tools scripts` after attachment extraction diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after attachment extraction diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after attachment extraction diagnostics | Passed: 224 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after attachment extraction diagnostics | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO` after attachment extraction diagnostics | Not rerun in this pass; the immediately preceding full simulator test attempt reached `Testing started`, stalled on `waiting for workers to materialize`/`Waiting for -runningDidFinish call`, and ended `TEST INTERRUPTED`. No executed XCTest pass is claimed for this slice. |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after developer trace persistence redaction | Passed: 54 passed |
| `python3 tools/check_release_hardening.py` after developer trace persistence redaction | Passed |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after developer trace persistence redaction | Passed: `TEST BUILD SUCCEEDED` |
| `git diff --check` after developer trace persistence redaction | Passed |
| `python3 -m compileall tools scripts` after developer trace persistence redaction | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after developer trace persistence redaction | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after developer trace persistence redaction | Passed: 226 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after developer trace persistence redaction | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/AgentGroundingRegressionTests/developerTraceCodecRedactsPromptsMemoryToolArgumentsAndAttachmentPathsBeforePersistence -parallel-testing-enabled NO` | Inconclusive for that Swift Testing case; Xcode reported `TEST SUCCEEDED` but Swift Testing executed 0 tests. No pass is claimed for that selector. |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO -parallel-testing-enabled NO` after developer trace persistence redaction | Interrupted/inconclusive after about 2551s. The run emitted one real failure in `AssistantKernelRunContractTests.testBackgroundTriggerToolIntentUsesNativeBackgroundSafeExecution`: expected `.unavailable`, got `.failed`; after that, Xcode blocked waiting for workers to materialize/`runningDidFinish` and ended `BUILD INTERRUPTED`. |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after secure tool alias routing | Passed: `TEST BUILD SUCCEEDED` |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO -parallel-testing-enabled NO -only-testing:LumenTests/AssistantKernelRunContractTests/testBackgroundTriggerToolIntentUsesNativeBackgroundSafeExecution -only-testing:LumenTests/SecureToolRegistryBackgroundFilteringTests/testCatalogMemoryRecallExecutesSecureMemorySearchImplementation` | Passed: 2 tests executed, 0 failures |
| `git diff --check` after secure tool alias routing | Passed |
| `python3 tools/check_release_hardening.py` after secure tool alias routing | Passed |
| `python3 -m compileall tools scripts` after secure tool alias routing | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after secure tool alias routing | Passed: 54 passed |
| `bash scripts/check-lumen-integration-gate.sh` after secure tool alias routing | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `python3 tools/check_release_hardening.py` after Release status doc cleanup | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after Release status doc cleanup | Passed: 54 passed |
| `rg -n "BackgroundToolBridgePolicy\|BackgroundToolBridgeAssessment\|bridgeMappingUnavailable\|compatibility bridge\|\bpartial\b\|\bplanned\b" README.md CLAUDE.md docs/RUNTIME_STATUS_MATRIX.md docs/AGENT_KERNEL_MIGRATION_STATUS.md docs/VALIDATION.md docs/APP_INTENTS.md docs/BACKGROUND_PROCESSING.md docs/TOOL_SECURITY_MODEL.md docs/RUNTIME_POLICY.md` after Release status doc cleanup | Passed: no matches |
| `git diff --check` after Release status doc cleanup | Passed |
| `python3 -m compileall tools scripts` after Release status doc cleanup | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after Release status doc cleanup | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after Release status doc cleanup | Passed: 226 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after Release status doc cleanup | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after Release status doc cleanup | Passed: `TEST BUILD SUCCEEDED` |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO` after Release status doc cleanup | Not rerun for this docs/Python-only slice; the preceding full simulator XCTest attempt and focused XCTest fix are recorded above. No new full-suite pass is claimed. |
| `python3 tools/check_release_hardening.py` after developer-console diagnostics | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after developer-console diagnostics | Passed: 56 passed |
| `git diff --check` after developer-console diagnostics | Passed |
| `python3 -m compileall tools scripts` after developer-console diagnostics | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after developer-console diagnostics | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after developer-console diagnostics | Passed: 228 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after developer-console diagnostics | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after developer-console diagnostics | Passed: `TEST BUILD SUCCEEDED` |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO` after developer-console diagnostics | Not rerun for this narrow Swift diagnostics slice because repeated full simulator execution has been stalling at the worker/finalization layer; no new full-suite pass is claimed. |
| `python3 tools/check_release_hardening.py` after Settings diagnostics path privacy | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after Settings diagnostics path privacy | Passed: 57 passed |
| `git diff --check` after Settings diagnostics path privacy | Passed |
| `python3 -m compileall tools scripts` after Settings diagnostics path privacy | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after Settings diagnostics path privacy | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after Settings diagnostics path privacy | Passed: 229 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after Settings diagnostics path privacy | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after Settings diagnostics path privacy | Passed: `TEST BUILD SUCCEEDED` |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO` after Settings diagnostics path privacy | Not rerun for this narrow Swift diagnostics slice because repeated full simulator execution has been stalling at the worker/finalization layer; no new full-suite pass is claimed. |
| `python3 tools/check_release_hardening.py` after legacy bridge compile-surface gating | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after legacy bridge compile-surface gating | Passed: 59 passed |
| `git diff --check` after legacy bridge compile-surface gating | Passed |
| `python3 -m compileall tools scripts` after legacy bridge compile-surface gating | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after legacy bridge compile-surface gating | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after legacy bridge compile-surface gating | Passed: 231 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after legacy bridge compile-surface gating | Passed: 189 tests collected |
| `xcodebuild -project Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build-for-testing CODE_SIGNING_ALLOWED=NO` after legacy bridge compile-surface gating | Failed immediately: root-level `Lumen.xcodeproj` does not exist |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build-for-testing CODE_SIGNING_ALLOWED=NO` after legacy bridge compile-surface gating | Failed immediately: no available simulator named `iPhone 17 Pro`; available destinations included `Lumen Focused Test iPhone`, `iPhone 16`, and `monGARS Test iPhone` |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone,OS=26.3.1' build-for-testing CODE_SIGNING_ALLOWED=NO` after legacy bridge compile-surface gating | Passed: `TEST BUILD SUCCEEDED` |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone,OS=26.3.1' test CODE_SIGNING_ALLOWED=NO` after legacy bridge compile-surface gating | Not rerun for this narrow compile-surface slice because repeated full simulator execution has been stalling at the worker/finalization layer; no new full-suite pass is claimed. |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -configuration Release -destination 'generic/platform=iOS Simulator' build CODE_SIGNING_ALLOWED=NO` after full-file legacy bridge DEBUG gating | Passed: `BUILD SUCCEEDED`; confirms non-DEBUG simulator compilation succeeds with the entire bridge implementation file gated out |
| `git diff --check` after full-file legacy bridge DEBUG gating | Passed |
| `python3 -m compileall tools scripts` after full-file legacy bridge DEBUG gating | Passed |
| `python3 tools/check_release_hardening.py` after full-file legacy bridge DEBUG gating | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after full-file legacy bridge DEBUG gating | Passed: 60 passed |
| `bash scripts/check-lumen-integration-gate.sh` after full-file legacy bridge DEBUG gating | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after full-file legacy bridge DEBUG gating | Passed: 231 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after full-file legacy bridge DEBUG gating | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after full-file legacy bridge DEBUG gating | Passed: `TEST BUILD SUCCEEDED` |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO` after full-file legacy bridge DEBUG gating | Not rerun in this slice because repeated full simulator execution has been stalling at the worker/finalization layer; no new full-suite pass is claimed. |
| `python3 tools/check_release_hardening.py` after bridge full-file static guard | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after bridge full-file static guard | Passed: 60 passed |
| `git diff --check` after bridge full-file static guard | Passed |
| `python3 tools/check_release_hardening.py` after model-catalog fallback surface cleanup | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after model-catalog fallback surface cleanup | Passed: 62 passed |
| `git diff --check` after model-catalog fallback surface cleanup | Passed |
| `python3 -m compileall tools scripts` after model-catalog fallback surface cleanup | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after model-catalog fallback surface cleanup | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after model-catalog fallback surface cleanup | Passed: 234 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after model-catalog fallback surface cleanup | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after model-catalog fallback surface cleanup | Passed: `TEST BUILD SUCCEEDED` |
| `python3 tools/check_release_hardening.py` after fleet/family catalog surface cleanup | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after fleet/family catalog surface cleanup | Passed: 63 passed |
| `git diff --check` after fleet/family catalog surface cleanup | Passed |
| `rg -n "fallback\|unavailable\|mock\|staged\|not implemented\|unimplemented" ios/Lumen/Services/ModelFamilySelection.swift ios/Lumen/Services/ModelFleetCatalog.swift ios/Lumen/Services/LLM/Models/BuiltInModelCatalog.swift ios/LumenTests/LumenFleetTests.swift ios/LumenTests/LLMModelStorageTests.swift` after fleet/family catalog surface cleanup | Passed: remaining matches are test assertions/fixtures only |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after fleet/family catalog surface cleanup | Passed: `TEST BUILD SUCCEEDED` |
| `python3 -m compileall tools scripts` after fleet/family catalog surface cleanup | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after fleet/family catalog surface cleanup | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after fleet/family catalog surface cleanup | Passed: 235 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after fleet/family catalog surface cleanup | Passed: 189 tests collected |
| `python3 tools/check_release_hardening.py` after deterministic compatibility execution gating | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after deterministic compatibility execution gating | Passed: 64 passed |
| `git diff --check` after deterministic compatibility execution gating | Passed |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after deterministic compatibility execution gating | Passed: `TEST BUILD SUCCEEDED` |
| `python3 -m compileall tools scripts` after deterministic compatibility execution gating | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after deterministic compatibility execution gating | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after deterministic compatibility execution gating | Passed: 236 passed, 31 deselected |
| `uv run --python 3.11 --with pytest --with ./tools/lumen_manifest_crawler pytest --collect-only -q tools/lumen_manifest_crawler/tests` after deterministic compatibility execution gating | Passed: 189 tests collected |
| `git diff --check` on 2026-07-07 after dedicated simulator validation attempt | Passed |
| `python3 tools/check_release_hardening.py` on 2026-07-07 after dedicated simulator validation attempt | Passed |
| `python3 -m compileall tools scripts` on 2026-07-07 after dedicated simulator validation attempt | Passed |
| `bash scripts/check-lumen-integration-gate.sh` on 2026-07-07 after dedicated simulator validation attempt | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` on 2026-07-07 after dedicated simulator validation attempt | Passed: 236 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` on 2026-07-07 after dedicated simulator validation attempt | Passed: 189 tests collected |
| `AGENT_GROUNDING_RESOURCE_MODE=minimal xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,id=D27B287A-7B93-4A8C-B65B-E6F75C121857' build-for-testing CODE_SIGNING_ALLOWED=NO` on 2026-07-07 with dedicated `Lumen Focused Test iPhone` | Passed: `TEST BUILD SUCCEEDED`; log: `/tmp/lumen-dedicated-build-for-testing.log` |
| Dedicated `Lumen Focused Test iPhone` full XCTest run setup on 2026-07-07 | Blocked before `xcodebuild test`: the first cleanup `simctl terminate` call hung, CoreSimulatorService was restarted, the device was erased and rebooted, then `xcrun simctl bootstatus ... -b` remained stuck at `Status=4` / `Waiting on System App` through `Elapsed=03:13`. The device was shut down afterward and no Xcode/simctl workers remained. No full XCTest pass is claimed. |
| `python3 tools/check_release_hardening.py` after legacy bridge API DEBUG-only gating | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after legacy bridge API DEBUG-only gating | Passed: 66 passed |
| `git diff --check` after legacy bridge API DEBUG-only gating | Passed |
| `python3 -m compileall tools scripts` after legacy bridge API DEBUG-only gating | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after legacy bridge API DEBUG-only gating | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `AGENT_GROUNDING_RESOURCE_MODE=minimal xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -configuration Release -destination 'generic/platform=iOS Simulator' build CODE_SIGNING_ALLOWED=NO` after legacy bridge API DEBUG-only gating | Passed: `BUILD SUCCEEDED`; log: `/tmp/lumen-release-generic-build-after-legacy-api-gate.log` |
| `python3 tools/check_release_hardening.py` after memory tool privacy hardening | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after memory tool privacy hardening | Passed: 68 passed |
| `git diff --check` after memory tool privacy hardening | Passed |
| Dedicated `Lumen Focused Test iPhone` full XCTest run on 2026-07-07 using `xcodebuild ... -destination 'platform=iOS Simulator,id=D27B287A-7B93-4A8C-B65B-E6F75C121857' test` | Blocked in Xcode/CoreSimulator before any test executed. First attempt let the scheme create `Clone 1` and `Clone 2` workers; both session logs reached `Installing app at path: .../Lumen.app`, stdout files were 0 bytes, no `xctest` child launched, and stopping produced `NSMachErrorDomain Code=-308` from `IDELaunchiPhoneSimulatorLauncher` after about 939-940s. |
| Dedicated `Lumen Focused Test iPhone` serialized full XCTest run on 2026-07-07 using `-parallel-testing-enabled NO -maximum-concurrent-test-simulator-destinations 1` | Blocked on the real dedicated device at the same `Installing app at path: .../Lumen.app` line with no test stdout or `xctest` process; stopping produced `NSMachErrorDomain Code=-308` from `IDELaunchiPhoneSimulatorLauncher` after about 142.5s. |
| Dedicated `Lumen Focused Test iPhone` erased serialized full XCTest retry on 2026-07-07 | Blocked again after `xcrun simctl erase D27B287A-7B93-4A8C-B65B-E6F75C121857`; the session log again stopped at `Installing app at path: .../Lumen.app`, no test stdout was written, and stopping produced `NSMachErrorDomain Code=-308` after about 93.9s. The device was left `Shutdown`; no full XCTest pass is claimed. |
| `python3 tools/check_release_hardening.py` after trigger AppIntent no-result hardening | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after trigger AppIntent no-result hardening | Passed: 70 passed |
| `git diff --check` after trigger AppIntent no-result hardening | Passed |
| `AGENT_GROUNDING_RESOURCE_MODE=minimal xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,id=D27B287A-7B93-4A8C-B65B-E6F75C121857' build-for-testing CODE_SIGNING_ALLOWED=NO` after trigger AppIntent no-result hardening | Passed: `TEST BUILD SUCCEEDED`; log: `/tmp/lumen-dedicated-build-for-testing-after-trigger-intent.log` |
| `python3 tools/check_release_hardening.py` after calendar reminder privacy hardening | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after calendar reminder privacy hardening | Passed: 72 passed |
| `git diff --check` after calendar reminder privacy hardening | Passed |
| `AGENT_GROUNDING_RESOURCE_MODE=minimal xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,id=D27B287A-7B93-4A8C-B65B-E6F75C121857' build-for-testing CODE_SIGNING_ALLOWED=NO` after calendar reminder privacy hardening | Passed: `TEST BUILD SUCCEEDED`; log: `/tmp/lumen-dedicated-build-for-testing-after-calendar-reminder.log` |
| `python3 tools/check_release_hardening.py` after native tool error privacy hardening | Passed |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_release_hardening.py` after native tool error privacy hardening | Passed: 74 passed |
| `git diff --check` after native tool error privacy hardening | Passed |
| `AGENT_GROUNDING_RESOURCE_MODE=minimal xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,id=D27B287A-7B93-4A8C-B65B-E6F75C121857' build-for-testing CODE_SIGNING_ALLOWED=NO` after native tool error privacy hardening | Passed: `TEST BUILD SUCCEEDED`; log: `/tmp/lumen-dedicated-build-for-testing-after-native-tool-error-privacy.log` |
| `git diff --check -- ios/Lumen/Services/ExecutorRuntimePreflight.swift ios/LumenTests/RuntimeContractRegressionTests.swift scripts/build_and_submit_appstoreconnect.sh scripts/run_focused_simulator_tests.sh scripts/validate_lumen_ios.sh ios/Lumen.xcodeproj/project.pbxproj` on 2026-07-08 after preflight/upload workflow hardening | Passed |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' build-for-testing CODE_SIGNING_ALLOWED=NO` on 2026-07-08 after preflight/upload workflow hardening | Passed: `TEST BUILD SUCCEEDED` |
| `xcodebuild test-without-building -xctestrun ... -destination 'platform=iOS Simulator,id=8C613E1F-22F1-4A0E-88B9-01031856659B' -only-testing:LumenTests/ExecutorPreflightTests` on 2026-07-08 | Passed: Swift Testing executed 7 `ExecutorPreflightTests` with 0 failures after simulator readiness was established through the SpringBoard/backboardd probe. |
| `bash scripts/build_and_submit_appstoreconnect.sh` on 2026-07-08 after bumping `CURRENT_PROJECT_VERSION` to `20260708192500` | Passed: archive/export/Info.plist/entitlement checks succeeded, upload reported `UPLOAD SUCCEEDED with no errors`, Delivery UUID `e1158bc3-6d83-47f5-aa18-2494f24733fe`. |
| `git diff --check` after planner/schema parity | Passed |
| `python3 tools/check_release_hardening.py` after planner/schema parity | Passed |
| `python3 tools/check_agent_kernel_boundary.py --strict` after planner/schema parity | Passed |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' build-for-testing CODE_SIGNING_ALLOWED=NO` after planner/schema parity | Passed: `TEST BUILD SUCCEEDED`; log: `/tmp/lumen-build-for-testing-planner-schema.log` |
| `xcrun simctl boot 8C613E1F-22F1-4A0E-88B9-01031856659B` plus SpringBoard/backboardd probe before focused planner tests | Passed: probe ready after 1 second without waiting on the long System App bootstatus path |
| `xcodebuild test-without-building -xctestrun /Users/ales27pm/Library/Developer/Xcode/DerivedData/Lumen-gafqarfynlsuiseecxqmygoyalln/Build/Products/Lumen_Lumen_iphonesimulator26.2-x86_64.xctestrun -destination 'platform=iOS Simulator,id=8C613E1F-22F1-4A0E-88B9-01031856659B' -only-testing:LumenTests/DeterministicToolPlannerTests` after planner/schema parity | Inconclusive; two compiled-output attempts stalled before XCTest execution. The first never booted the dedicated simulator; the second followed the SpringBoard/backboardd readiness probe but remained stuck in Xcode test-manager handoff with repeated `IDERunDestination: Supported platforms for the buildables in the current scheme is empty` warnings. No executed XCTest pass is claimed. |
| `git diff --check` after native approval-boundary gating | Passed |
| `python3 tools/check_release_hardening.py` after native approval-boundary gating | Passed |
| `python3 tools/check_agent_kernel_boundary.py --strict` after native approval-boundary gating | Passed; only documented DEBUG/transition bridge allowlist entries remained |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' build-for-testing CODE_SIGNING_ALLOWED=NO` after native approval-boundary gating | Passed: `TEST BUILD SUCCEEDED`; log: `/tmp/lumen-build-for-testing-approval-boundary-final.log` |
| `xcrun simctl boot 8C613E1F-22F1-4A0E-88B9-01031856659B` plus SpringBoard/backboardd probe before focused approval-boundary test | Passed: probe ready after 1 second without waiting on the long System App bootstatus path |
| `xcodebuild test-without-building -xctestrun /Users/ales27pm/Library/Developer/Xcode/DerivedData/Lumen-gafqarfynlsuiseecxqmygoyalln/Build/Products/Lumen_Lumen_iphonesimulator26.2-x86_64.xctestrun -destination 'platform=iOS Simulator,id=8C613E1F-22F1-4A0E-88B9-01031856659B' -only-testing:LumenTests/AssistantKernelRunContractTests/testNativeToolTurnApprovalRequiredActionsStopBeforeExecution -parallel-testing-enabled NO -jobs 1` after native approval-boundary gating | Passed: executed 1 XCTest with 0 failures; `TEST EXECUTE SUCCEEDED`; log: `/tmp/lumen-approval-boundary-test-without-building-final.log` |
| `git diff --check` after strict boundary enforcement | Passed |
| `python3 -m compileall tools scripts` after strict boundary enforcement | Passed |
| `python3 tools/check_release_hardening.py` after strict boundary enforcement | Passed |
| `python3 tools/check_agent_kernel_boundary.py --strict` after strict boundary enforcement | Passed; remaining documented calls are inside the DEBUG-only `LegacyAgentCompatibilityBridge` |
| `uv run --python 3.12 pytest tools/pipeline/tests/test_check_agent_kernel_boundary.py` after strict boundary enforcement | Passed: 3 passed |
| `bash scripts/check-lumen-integration-gate.sh` after strict boundary enforcement | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 pytest -m "not slow and not e2e"` after strict boundary enforcement | Passed: 249 passed, 31 deselected |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only` after strict boundary enforcement | Passed: 189 tests collected |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO` after strict boundary enforcement | Passed: `TEST BUILD SUCCEEDED`; log: `/tmp/lumen-build-for-testing-strict-boundary.log` |
| `xcrun simctl boot 710EFEFE-73D8-400B-B5D1-D5F43662B536` plus SpringBoard/backboardd probe before focused boundary tests | Passed: probe ready after 1 second without waiting on the long System App bootstatus path |
| `xcodebuild test-without-building -xctestrun /Users/ales27pm/Library/Developer/Xcode/DerivedData/Lumen-gafqarfynlsuiseecxqmygoyalln/Build/Products/Lumen_Lumen_iphonesimulator26.2-x86_64.xctestrun -destination 'platform=iOS Simulator,id=710EFEFE-73D8-400B-B5D1-D5F43662B536' -only-testing:LumenTests/AgentKernelBoundaryGuardTests -parallel-testing-enabled NO -jobs 1` after strict boundary enforcement | Passed: executed 4 XCTest cases with 0 failures; `TEST EXECUTE SUCCEEDED`; log: `/tmp/lumen-boundary-guard-test-without-building.log` |
| `git diff --check` after Chat approval UI mapping | Passed |
| `python3 tools/check_release_hardening.py` after Chat approval UI mapping | Passed |
| `python3 tools/check_agent_kernel_boundary.py --strict` after Chat approval UI mapping | Passed; remaining documented calls are inside the DEBUG-only `LegacyAgentCompatibilityBridge` |
| `SIM_UDID=8C613E1F-22F1-4A0E-88B9-01031856659B TEST_TIMEOUT_SECONDS=3600 SIM_BOOT_TIMEOUT_SECONDS=1200 SIM_READY_PROBE_TIMEOUT_SECONDS=20 bash scripts/run_focused_simulator_tests.sh --only-testing LumenTests/ToolApprovalQueueTests/testApprovalBoundaryStepCreatesPendingToolMessageForChatView` after Chat approval UI mapping | Passed with `pipefail`: dedicated `Lumen Focused Test iPhone` readiness probe succeeded without the long System App wait, `TEST BUILD SUCCEEDED`, and `TEST EXECUTE SUCCEEDED`; executed 1 XCTest with 0 failures. Log: `/tmp/lumen-chat-approval-mapper-focused-clean.log`; result bundle: `~/Library/Developer/Xcode/DerivedData/Lumen-chhvizdiogdwpghffmgflhfgheel/Logs/Test/Test-Lumen-2026.07.09_03-17-54--0400.xcresult` |
| `git diff --check` after live-E2E completeness hardening | Passed |
| `python3 -m compileall tools scripts` after live-E2E completeness hardening | Passed |
| `python3 tools/check_release_hardening.py` after live-E2E completeness hardening | Passed |
| `bash scripts/check-lumen-integration-gate.sh` after live-E2E completeness hardening | Passed; advisory placeholder/logging review lines still printed by readiness checks |
| `uv run --python 3.12 --with pytest python -m pytest tools/pipeline/tests/test_check_agent_kernel_boundary.py` after live-E2E completeness hardening | Passed: 3 passed |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' build-for-testing CODE_SIGNING_ALLOWED=NO` after live-E2E completeness hardening | Failed before compile: requested simulator `Lumen Focused Test iPhone` was unavailable on this host. |
| `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'generic/platform=iOS Simulator' build-for-testing CODE_SIGNING_ALLOWED=NO` after live-E2E completeness hardening | Passed: `TEST BUILD SUCCEEDED` |
| `xcrun simctl list devices available` after live-E2E completeness hardening | Interrupted after hanging with no output; no executed simulator XCTest pass is claimed. |

## Remaining DEBUG-Only Experimental Items

- Deterministic diagnostic runtime.
- Legacy agent bridge probes for migration/debug evidence.
- Unavailable GGUF native bridge.
- REM autonomous repair workflows.

## Remaining Release Evidence Gaps

- Executed XCTest proof now exists for the new native approval-boundary regression test using `test-without-building` on the dedicated `Lumen Focused Test iPhone`; broader full-suite simulator XCTest proof is still missing.
- Executed XCTest proof now exists for the Chat approval UI mapping regression test using the bounded focused simulator runner on the dedicated `Lumen Focused Test iPhone`; broader full-suite simulator XCTest proof is still missing.
- Executed XCTest proof now exists for the strict Agent Kernel boundary guard tests using `test-without-building` on `iPhone 16`; broader full-suite simulator XCTest proof is still missing.
- Full simulator XCTest proof for the current follow-up remains missing; the attempted full `xcodebuild test` run was interrupted after it stopped producing output.
- Executed XCTest proof for the new RAG/memory diagnostic tests is still missing; the non-launching `build-for-testing` checkpoint compiled them successfully.
- Executed XCTest proof for the new privacy/logging redaction tests is still missing; the non-launching `build-for-testing` checkpoint compiled them successfully.
- Current Xcode `build-for-testing` proof after the headless/background Release surface cleanup, production catalog cleanup, GGUF Release crash-path cleanup, and production precondition crash-path cleanup now exists for the requested `iPhone 16` simulator destination. Full simulator XCTest execution is still missing for the latest Swift changes; the latest attempt reached `Testing started`, then Xcode stalled waiting for `runningDidFinish` and reported `TEST INTERRUPTED`.
- Executed XCTest proof for the new AppIntent memory-search renderer tests is still missing; the non-launching `build-for-testing` checkpoint compiled them successfully.
- Executed XCTest proof for the new headless model-fetch renderer test is still missing until the next Xcode test run; the non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new trigger persistence renderer test is still missing until the next Xcode test run; the non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new model-bootstrap and E2E preflight diagnostics tests is still missing until the next Xcode test run; the non-launching `build-for-testing` checkpoint compiled them successfully.
- Executed XCTest proof for the new TriggerTools persistence messages is still missing until the next Xcode test run; the non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new REM model-catalog diagnostic test is still missing until the next Xcode test run; the non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new Settings live-E2E model-catalog diagnostics is still missing until the next Xcode test run; the non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new trigger scheduler fetch diagnostics is still missing until the next Xcode test run; the non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new imported-file diagnostics is still missing until the next Xcode test run; the non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new RAG file-extraction diagnostics is still missing until the next Xcode test run; the non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new memory-capture queue diagnostics is still missing until the next Xcode test run; the non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new file-tool read diagnostics is still missing until the next Xcode test run; the non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new model-integrity diagnostics is still missing until the next Xcode test run; the non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new memory-save diagnostics is still missing until the next Xcode test run; the non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new Settings model-directory diagnostics is still missing until the simulator launch path is stable; the non-launching `build-for-testing` checkpoint compiled it successfully, while the attempted full `xcodebuild test` hit `NSMachErrorDomain Code=-308` in `IDELaunchiPhoneSimulatorLauncher`.
- Executed XCTest proof for the new Settings imported-file diagnostics is still missing until the simulator launch path is stable; the non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new imported-file write and attachment metadata diagnostics is still missing until the simulator launch path is stable; the non-launching `build-for-testing` checkpoint compiled it successfully, while the attempted full `xcodebuild test` stalled waiting for test workers and ended `TEST INTERRUPTED`.
- Executed XCTest proof for the new attachment extraction diagnostics is still missing until the simulator launch path is stable; the non-launching `build-for-testing` checkpoint compiled it successfully, and the full simulator XCTest command was not rerun immediately after the same worker-materialization stall recorded above.
- Executed XCTest proof for the new developer trace persistence redaction test is still missing because the focused Swift Testing selector reported success while executing 0 tests; the static scanner, Python tests, and non-launching `build-for-testing` checkpoint passed.
- Full simulator XCTest proof after the secure alias-routing fix is still missing. The failing `AssistantKernelRunContractTests` case and the new secure registry alias case passed in a focused XCTest run, but the preceding full-suite attempt was interrupted after Xcode blocked on test workers and `runningDidFinish`.
- Full simulator XCTest proof on the dedicated `Lumen Focused Test iPhone` is still missing. The latest dedicated-device attempts reached `Testing started` but never launched `xctest`; Xcode/CoreSimulator blocked at app installation and reported `NSMachErrorDomain Code=-308` from `IDELaunchiPhoneSimulatorLauncher`, including after a serialized clean-erased retry. The device was shut down afterward.
- Executed XCTest proof for the new trigger AppIntent no-result renderer tests is still missing until the simulator launch/install path is stable; the dedicated-device non-launching `build-for-testing` checkpoint compiled them successfully.
- Executed XCTest proof for the new calendar reminder privacy test is still missing until the simulator launch/install path is stable; the dedicated-device non-launching `build-for-testing` checkpoint compiled it successfully.
- Executed XCTest proof for the new alarm, health, and contacts privacy tests is still missing until the simulator launch/install path is stable; the dedicated-device non-launching `build-for-testing` checkpoint compiled them successfully.
- Executed XCTest proof for the new planner/schema parity tests is still missing. The dedicated simulator reached SpringBoard/backboardd readiness through the probe fallback, but `test-without-building` stalled in Xcode's test-manager handoff before any XCTest output; the non-launching `build-for-testing` checkpoint compiled the tests successfully.
- Fresh executed proof now exists for `ExecutorPreflightTests`, but the attached live E2E weather scenario was not rerun after the preflight gate fix in this documentation pass. Do not claim live model-backed scenario success until a new report proves it.
- Live device/TestFlight evidence is still needed for model-backed tool-call loops, voice, AppIntent, RAG, memory, permissions, and real local model artifacts.
- The native kernel path currently executes validated intent-planned tool actions and surfaces the tool result honestly; model-driven post-tool synthesis still needs live runtime evidence before claiming full parity with every shipped workflow.

## Manual Validations Still Required

These require Apple credentials, signing assets, TestFlight, or physical hardware:

- Signed archive/export with current App Store signing profile.
- Signed entitlements inspection on exported `.ipa`.
- Privacy manifest validation on the submitted archive.
- TestFlight or real-device smoke test.
- Real-device local model load with actual model artifacts.
- Real-device role-adapter switching.
- Live tool-call validation for any tool-capable surface enabled in that build.
- Live RAG indexing/search with user files/photos where permissions are granted.
- Live memory extraction/storage with real model embeddings.
- Voice and AppIntent flows for the exact submitted Release build.
