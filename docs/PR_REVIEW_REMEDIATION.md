# PR #232 Review Remediation

Review threads could not be queried in this runner because the GitHub CLI is unavailable. This pass therefore used the unresolved review checklist from the PR prompt and current source inspection after commit `d5be063fc572165d3958ec633645cc7695555249`.

## Addressed comments

| Review summary | File(s) | Priority | Action taken | Validation |
| --- | --- | --- | --- | --- |
| Llama/Foundation adapters echoed prompts and CoreML returned empty embeddings. | `ios/Lumen/Assistant/AssistantRuntimeAdapters.swift` | High | Adapters now report unavailable/not implemented and throw typed errors instead of echoing prompts or returning `[]`. CoreML availability checks configured file existence. | `rg -n "request.prompt|return \[\]" ios/Lumen` |
| `AssistantKernel.runTextTurn` accepted non-text tasks and leaked raw error descriptions to metrics. | `ios/Lumen/Assistant/AssistantKernel.swift`, `ios/Lumen/System/RuntimeMetric.swift` | High | Added `KernelError`, rejects embedding/safety tasks, rejects CoreML for text, and records sanitized error codes. | Static `rg` and `git diff --check` |
| Runtime router selected llama even when heavy runtime was disallowed. | `ios/Lumen/Assistant/AssistantRuntimeRouter.swift` | High | Llama is selected only when `allowHeavyRuntime` and availability are both true. | Static `rg` and router tests pending Xcode |
| Metrics file creation silently ignored `createFile` failure. | `ios/Lumen/System/RuntimeMetricsStore.swift` | Medium | `createFile` Bool is checked and falls back to atomic `Data.write`. | `rg -n "createFile" ios/Lumen` |
| Notification observers were not retained/removed. | `PowerModeMonitor`, `ThermalStateMonitor`, `MemoryPressureMonitor` | Medium | Stored observer tokens and removed them in `deinit`; added `MemoryPressureMonitor.shared`. | Static source inspection |
| Background task policy allowed serious thermal/network-required denial mismatch. | `ios/Lumen/Background/BackgroundTaskPolicy.swift` | High | Denies serious/critical thermal states and denies network-required work when network cannot be allowed. | Static source inspection |
| Background orchestrator released leases via unawaited `Task` and marked empty RAG maintenance as failure. | `ios/Lumen/Background/BackgroundOrchestrator.swift`, `ios/Lumen/RAG/RAGEngine.swift` | High | Release is awaited after trigger scan; RAG maintenance now returns `maintenance_success_empty`, `maintenance_success_work_done`, or `maintenance_failed`. | Static source inspection |
| Permission registry continuation race and EventKit `writeOnly` read access. | `ios/Lumen/Permissions/PermissionRegistry.swift` | High | Location continuation is installed before requesting authorization and concurrent requests are rejected; `writeOnly` maps to `.limited`. | Static source inspection |
| Tool execution context had invalid `Sendable`; tool policy ignored allowlist; tool metrics were hardcoded. | `ios/Lumen/Tools/*` | High | Removed `Sendable`, enforced non-empty allowlist, routed metrics through context store, recorded missing-tool metrics. | Static source inspection |
| Safe tool output could exceed max and OpenURL always returned success. | `SafeToolOutputLimiter`, `OpenURLTool` | High | Clamping now never exceeds max; URL opening awaits system completion and fails on rejection. | Static source inspection |
| Memory/RAG comments about empty query, prompt extraction, chunking offsets, RAG dedupe, empty maintenance and indexing errors. | `ios/Lumen/Memory/*`, `ios/Lumen/RAG/*` | High | User-only memory extraction, non-query match handling, throwing save paths, validated chunking config, stable RAG keys, explicit indexing errors. | Static `rg`, `git diff --check` |
| AppIntent memory save used `try?`, memory search exposed content without opening app, trigger fetch/matching was ambiguous. | `ios/Lumen/AppIntents/*` | High | Memory save now reports degraded on failure; memory search opens app; trigger fetch errors and ambiguous matches are surfaced. | `rg -n "openAppWhenRun" ios/Lumen/AppIntents` |
| Readiness script could fail under `set -e` when privacy logging scan had no matches. | `scripts/check-ios-build-readiness.sh` | Medium | Added explicit non-match handling. | `./scripts/check-ios-build-readiness.sh` |
| Voice UI phase sync, live partial transcript polling and background interruption precedence. | `ios/Lumen/Views/VoiceModeView.swift`, `ios/Lumen/Voice/VoiceSessionController.swift` | High | VoiceModeView now mirrors controller state instead of eagerly setting listening; controller polls partial transcripts while listening and parenthesizes the background interruption condition. | Static source inspection; Xcode unavailable |

## Skipped / already resolved

| Review summary | File | Reason |
| --- | --- | --- |
| Manual Xcode project source membership additions requested. | `ios/Lumen.xcodeproj/project.pbxproj` | Skipped because the project uses `PBXFileSystemSynchronizedRootGroup`; converting to manual source lists would violate the PR instruction and project model. |
| Full legacy grounding cache/idempotency deep changes. | `ios/Lumen/Assistant/*` | Partially out of scope for this pass; no direct unresolved thread data was available locally. Existing static checks did not find escaped interpolation regressions in services. |
| Brittle source-reading test rewrites. | `ios/LumenTests/*` | Skipped for this focused production remediation because Xcode is unavailable to validate broad test rewrites safely. |

## Final cleanup note
A final narrow cleanup pass is tracked in `docs/PR_REVIEW_FINAL_CLEANUP.md`, covering remaining documentation, ModelContext isolation, contacts provider errors, RAG vector freshness, legacy grounding role/idempotency, test hygiene, and readiness-script no-match handling.
