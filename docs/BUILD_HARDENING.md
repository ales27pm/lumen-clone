# iOS Build Hardening

## Project membership

`ios/Lumen.xcodeproj` uses `PBXFileSystemSynchronizedRootGroup` entries for `Lumen`, `LumenTests`, and `LumenUITests`. New Swift files under those folders are expected to be discovered by Xcode's synchronized groups rather than by manually adding `PBXBuildFile` source entries. Do not convert these groups to manual membership without a dedicated project migration.

## Local validation commands

Run on macOS with Xcode installed:

```sh
./scripts/check-lumen-integration-gate.sh
./scripts/check-ios-build-readiness.sh
cd ios
xcodebuild -list -project Lumen.xcodeproj
xcodebuild -project Lumen.xcodeproj -scheme Lumen -destination 'generic/platform=iOS Simulator' build
xcodebuild -project Lumen.xcodeproj -scheme Lumen -destination 'generic/platform=iOS Simulator' test
```

`scripts/check-lumen-integration-gate.sh` is the local pre-PR gate for this repo. It runs the Agent Kernel boundary guards, adapter runtime invariants, LoRA hardening invariants, MSAL release validation, signing capability validation, shell-subprocess security scan, iOS build-readiness checks, and `git diff --check` in a deterministic fail-fast order.

Run full pre-PR validation from a real git checkout so the `git diff --check` whitespace guard runs against the worktree. In exported ZIP archives or other non-git directories, the integration gate prints `Skipping git diff --check: not a git worktree` and continues with the non-git static checks.

If a named simulator is required locally, run `xcodebuild -showdestinations -project Lumen.xcodeproj -scheme Lumen` and choose an installed simulator.

## Static checks in non-macOS runners

When `xcodebuild` is unavailable, `scripts/check-ios-build-readiness.sh` performs static checks only:
- verifies project path exists;
- reports synchronized group project membership;
- counts app/test Swift files;
- checks AppIntents references;
- checks Info.plist usage string build settings and BGTask identifiers;
- scans privacy-sensitive additions for logging APIs.

Static checks are not compile validation.

## Common failure points

- AppIntents availability must remain guarded for supported deployment targets.
- SwiftData `ModelContext` usage must remain on main actor or explicit contexts.
- Generated Info.plist keys must match microphone, speech, calendar, contacts, location, photo, notification, and background task usage.
- Background task identifiers must include `com.27pm.lumenclone.agent.refresh` and `com.27pm.lumenclone.agent.process`.
- Secure tool registry and legacy tool catalog intentionally use distinct type names (`SecureToolRegistry` and `ToolRegistry`) to avoid duplicate symbols.

## Symbol collision hardening

The build-hardening pass reserves legacy names for existing production UI/services and gives new assistant subsystems explicit prefixes:
- `SecureToolRegistry` and `SecureToolCategory` are the secure-tool layer; existing `ToolRegistry` and `ToolCategory` remain the legacy catalog used by current UI and tool routing.
- `AssistantPermissionState` is used by the new permission registry; existing `PermissionState` remains the `PermissionsCenter` UI state.
- `AssistantDeviceCapabilitySnapshot` is used by diagnostics/system profiling; existing LLM policy `DeviceCapabilitySnapshot` remains unchanged.

## Runtime resource-kill hardening

- App launch must not load chat, embedding, FoundationModels, or tokenizer assets. `AppStartupCoordinator` now marks model runtime as deferred until an explicit foreground user chat or voice turn.
- Scene phase transitions to inactive/background must only do nonblocking cancellation and optional cleanup; no synchronous memory compaction, RAG indexing, trigger firing, or model probing may run from scene-update handlers.
- Before TestFlight submission, run the duplicate Swift filename and duplicate compatibility-type checks listed in the validation section of the runtime resource-kill analysis.
