# Lumen Validation

`python3 -m lumen_manifest_crawler developer-cycle --root .` is the top-level validation entrypoint. It preserves the lower-level commands for targeted use:

```bash
bash scripts/check-lumen-integration-gate.sh
bash scripts/check-ios-build-readiness.sh
bash scripts/validate_lumen_ios.sh
RUN_SIMULATOR_TESTS=1 TEST_TIMEOUT_SECONDS=900 bash scripts/validate_lumen_ios.sh
bash scripts/run_focused_simulator_tests.sh --only-testing LumenTests/AgentGroundingRegressionTests
python3 -m lumen_manifest_crawler framework status --root .
python3 -m lumen_manifest_crawler framework plan --root .
python3 -m lumen_manifest_crawler framework ingest improve-loop --root .
python3 -m lumen_manifest_crawler improve-loop --root . --output generated/agent_manifest --loop-output generated/agent_improvement_loop
```

## Validation Levels

- **Portable validation** is static-source safe and works in non-git ZIP/export, Linux/Codex, and macOS environments. It can skip Git-only and Xcode-only checks with explicit reasons.
- **Local validation** adds repo-aware checks such as `git diff --check` when a git worktree is present.
- **Release-candidate validation** requires `xcodebuild` and `scripts/validate_lumen_ios.sh`. By default this performs `build-for-testing` on a generic iOS Simulator destination with minimal AgentGrounding resources and skips simulator XCTest execution, because full simulator handoff is not deterministic on every Xcode/CoreSimulator stack. If Xcode is unavailable, the report must say skipped or failed. It must not claim compile validation passed.
- **Simulator XCTest validation** is opt-in. Use `RUN_SIMULATOR_TESTS=1` for the full validation script or `scripts/run_focused_simulator_tests.sh` for a pinned focused test. Focused simulator runs use a reusable warmed simulator by default, minimal AgentGrounding resources, a focused `.xctestrun`, disabled parallel workers, and bounded boot/test phases. Set `PREWARM_ONLY=1` to create/boot the reusable simulator without running tests. Set `USE_DISPOSABLE_SIMULATOR=1` only when isolation is worth paying the first-boot migration cost. The focused runner accepts normal `bootstatus` completion or, when `bootstatus` stalls at System App, a Booted device with SpringBoard and backboardd running.
- **CoreSimulator runtime health** matters. On this host the recurring focused-runner blocker was CoreSimulator runtime/device readiness, visible as simulator-runtime `Info.plist missing` lines, slow MobileInstallation `Preflight/Patch` timings, and fresh simulators spending more than 7 minutes in `bootstatus` migration before XCTest could start. `simctl runtime list -v`, `simctl runtime verify`, and direct `codesign --verify` can disagree on cryptex-mounted simulator runtimes, so the focused runner pins an installed runtime and treats a Booted device with SpringBoard and backboardd running as usable even when `bootstatus` does not reach terminal readiness. Runtime cleanup, `xcodebuild -downloadPlatform iOS`, dyld-cache rebuilds, prewarming a reusable simulator, and attaching Simulator.app before `bootstatus` are host repair tools before treating simulator XCTest timeout as an app regression.
- **Runtime validation** comes from exported runtime audit/TestFlight/E2E evidence. Missing runtime evidence is not a runtime pass, and simulator XCTest success is not a substitute for live runtime evidence.
- **Training/HF validation** is opt-in and never runs by default.

## Feature-Complete Release Gate

Run these commands before claiming a Release hardening pass:

```bash
git diff --check
python -m compileall tools scripts
bash scripts/check-lumen-integration-gate.sh
uv run --python 3.12 pytest -m "not slow and not e2e"
cd tools/lumen_manifest_crawler && uv run --python 3.12 pytest --collect-only
```

On a macOS/Xcode runner with the requested simulator installed:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO
```

Release submission validation also requires credentialed or physical-device checks:

- signed archive and export
- signed entitlement inspection
- privacy manifest validation
- TestFlight or real-device smoke test
- real-device local model load
- live tool-call validation
- live RAG indexing and search
- live memory extraction and storage
- voice/AppIntent flows that are enabled in the submitted build

## Failure Flags

- `--fail-on-static` exits non-zero for static validation failures.
- `--fail-on-validation` exits non-zero for manifest, improvement-loop, Xcode, or training validation failures.
- `--fail-on-gaps` exits non-zero when improvement-loop gaps remain or runtime evidence is missing.
- `--require-runtime-audit` exits non-zero when no runtime audit evidence is ingested.
- `--with-xcode` exits non-zero when Xcode validation cannot pass.

Reports are written to:

```text
generated/developer_framework/developer_cycle_report.json
generated/developer_framework/DEVELOPER_CYCLE_REPORT.md
generated/developer_framework/framework_report.json
generated/developer_framework/runtime_report_index.json
```
