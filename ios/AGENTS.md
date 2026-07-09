# AGENTS.md

## iOS Scope

This directory contains the shipped iOS app, XCTest targets, and Xcode project. Guidance here extends the repository root instructions for Swift, runtime, and simulator work.

## Swift And Runtime Rules

- Release builds must not select deterministic fallback, mock, staged, unavailable, experimental, legacy-bridge, or not-compiled runtimes.
- Keep diagnostic fallback, unavailable GGUF bridge implementations, live legacy probes, and experimental adapters behind `#if DEBUG`.
- Prefer typed runtime errors, typed diagnostics, and explicit readiness states over generic fallback text.
- Do not convert production failures into empty arrays, zero counts, false booleans, or vague user-facing copy.
- Tool-capable chat, voice, AppIntent, trigger, and headless paths must be Agent Kernel-native in Release or excluded from Release.
- Tool calls must be schema-validated before execution. Never execute unknown tools, schema-invalid arguments, malformed JSON, or extra dangerous arguments.
- Logs and diagnostics must redact raw prompts, user documents, memory contents, and raw tool arguments unless the user explicitly exports diagnostics.

## XCTest And Xcode Validation

Use `build-for-testing` as the default simulator checkpoint:

```bash
xcodebuild -project ios/Lumen.xcodeproj \
  -scheme Lumen \
  -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' \
  build-for-testing \
  CODE_SIGNING_ALLOWED=NO
```

If a previous `build-for-testing` already produced the needed `.xctestrun`, run focused simulator checks with `test-without-building` instead of rebuilding the app.

Run full simulator tests only when the task requires executed XCTest proof and CoreSimulator is healthy:

```bash
xcodebuild -project ios/Lumen.xcodeproj \
  -scheme Lumen \
  -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' \
  test \
  CODE_SIGNING_ALLOWED=NO
```

Prefer focused tests when changing a narrow subsystem. Examples:

```bash
xcodebuild -project ios/Lumen.xcodeproj \
  -scheme Lumen \
  -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' \
  test \
  -only-testing:LumenTests/RuntimeRouterTests \
  CODE_SIGNING_ALLOWED=NO
```

Preferred focused execution pattern:

```bash
bash scripts/run_focused_simulator_tests.sh --only-testing LumenTests/<SuiteName>
```

The focused runner uses the dedicated simulator, minimal AgentGrounding resources, a generated focused `.xctestrun`, disabled parallel workers, and bounded boot/test phases. The default simulator test timeout is `2400` seconds.

Useful command recap:

| Command | Purpose | When to use |
| --- | --- | --- |
| `xcodebuild ... build-for-testing CODE_SIGNING_ALLOWED=NO` | Compile app and tests without signing | Default iOS validation checkpoint |
| `xcodebuild test-without-building -xctestrun ... -only-testing:LumenTests/<Suite>` | Execute focused tests from an existing build | Rerun narrow tests without recompiling |
| `xcodebuild ... test -only-testing:LumenTests/<Suite>` | Focused XCTest execution | Narrow Swift behavior changes |
| `xcodebuild ... test CODE_SIGNING_ALLOWED=NO` | Full simulator XCTest | Only when execution proof is required and simulator is healthy |
| `bash scripts/validate_lumen_ios.sh` | Repo iOS validation wrapper | Release-candidate style local validation |

Avoid during normal agent work:

- Do not repeatedly launch a failing simulator. Switch to non-launching validation or a focused bounded runner.
- Do not wait for a long `bootstatus -b` System App timeout when the device is already Booted and SpringBoard/backboardd are running.
- Do not change signing, entitlements, bundle IDs, or App Store settings as a side effect of compile fixes.
- Do not add production mock adapters to satisfy tests.
- Do not treat generic fallback text as acceptable runtime UX.

## Manual Evidence Boundary

Do not treat a simulator compile or XCTest pass as proof of TestFlight, signed Release, real-device local model load, live RAG, live memory, voice, or AppIntent behavior. State those gaps explicitly unless fresh evidence was produced.
