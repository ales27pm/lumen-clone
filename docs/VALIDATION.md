# Lumen Validation

Structured-agent release validation must confirm that refresh and processing handlers register before application launch completion. For iOS 26 continued processing, confirm that the Info.plist advertises the wildcard pattern and that a user-initiated request dynamically registers and submits the same fully composed concrete identifier. Model-visible tools must remain a fail-closed subset of `SecureToolRegistry`, and exported E2E evidence is evaluated by `evidenceMode`. A `policyFirstAllowed` scenario may use correlated deterministic policy-first evidence, while `modelBackedRequired` remains strict.

After an embedding format, model, or dimension change, run the in-app reindex workflow and confirm stale chunks report `rag_reindex_required` before reindex and become searchable afterward. Simulator compilation is not proof of real-device background registration, local model loading, or live E2E behavior.

On an Apple host with a connected development device, use signed physical-device validation as the primary iOS lane. Resolve the device immediately before each run, keep its identifier out of tracked documentation, and use simulator validation only as an explicitly labeled fallback. A signed device build, `build-for-testing`, focused XCTest execution, install, launch, and live feature exercise are separate checkpoints; record only the checkpoints that actually completed.

`python3 -m lumen_manifest_crawler developer-cycle --root .` is the top-level validation entrypoint. It preserves the lower-level commands for targeted use:

```bash
bash scripts/check-lumen-integration-gate.sh
bash scripts/check-ios-build-readiness.sh
bash scripts/validate_lumen_ios.sh
RUN_SIMULATOR_TESTS=1 TEST_TIMEOUT_SECONDS=2400 bash scripts/validate_lumen_ios.sh
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
- **Signed physical-device validation** is the preferred Apple-host lane when a development iPhone is connected. Run a signed app build and `build-for-testing` against the freshly resolved device, then execute risk-focused XCTest suites. Treat install, launch, smoke, and live permission/model/tool flows as later, independently recorded checkpoints. A device `build-for-testing` does not prove that the app launched, and focused device XCTest does not prove unexercised product flows.
- **Simulator XCTest validation** is opt-in. Use `RUN_SIMULATOR_TESTS=1` for the full validation script or `scripts/run_focused_simulator_tests.sh` for a pinned focused test. Focused simulator runs use the dedicated `Lumen Focused Test iPhone` simulator by default, minimal AgentGrounding resources, a focused `.xctestrun`, disabled parallel workers, and bounded boot/test phases. Set `PREWARM_ONLY=1` to create/boot the reusable simulator without running tests. Set `USE_DISPOSABLE_SIMULATOR=1` only when isolation is worth paying the first-boot migration cost. The focused runner accepts normal `bootstatus` completion or, when `bootstatus` stalls at System App, a Booted device with SpringBoard and backboardd running.
- **Optimized simulator reruns** should reuse compiled products. Run `build-for-testing` once, locate the produced `.xctestrun`, then use `xcodebuild test-without-building -xctestrun ... -only-testing:LumenTests/<Suite>` for focused reruns. Do not recompile just to rerun a narrow simulator suite.
- **CoreSimulator runtime health** matters. On this host the recurring focused-runner blocker was CoreSimulator runtime/device readiness, visible as simulator-runtime `Info.plist missing` lines, slow MobileInstallation `Preflight/Patch` timings, and fresh simulators spending more than 7 minutes in `bootstatus` migration before XCTest could start. `simctl runtime list -v`, `simctl runtime verify`, and direct `codesign --verify` can disagree on cryptex-mounted simulator runtimes, so the focused runner pins an installed runtime and treats a Booted device with SpringBoard and backboardd running as usable even when `bootstatus` does not reach terminal readiness. Runtime cleanup, `xcodebuild -downloadPlatform iOS`, dyld-cache rebuilds, prewarming a reusable simulator, and attaching Simulator.app before `bootstatus` are host repair tools before treating simulator XCTest timeout as an app regression.
- **Runtime validation** comes from exported runtime audit/TestFlight/E2E evidence. Missing runtime evidence is not a runtime pass, and simulator XCTest success is not a substitute for live runtime evidence.
- **Persistent evidence privacy validation** requires `redacted-v1` E2E results/exports, behavior traces, and in-app grounding packages. Run `python3 tools/check_runtime_audit_privacy.py` before accepting a checked-in runtime artifact. The version label and privacy check do not establish runtime freshness, model attribution, or a behavioral pass.
- **Live E2E scoring** treats incomplete user-visible finals as failures. Dangling endings such as `an`, `a`, `the`, `with`, `because`, or `you do not need an` must be repaired from trusted tool observations when possible or fail hygiene. Tool-backed missing-argument scenarios should clarify rather than pass through generic safe failure text.
- **Model-backed training pacing** checks runtime budget and CPU watchdog state before generation. If degraded before a valid generation begins, emit one non-actionable runtime-preflight result with `trainingSignal=false` instead of entering agent-json and accumulating trainable failures.
- **Training/HF validation** is opt-in and never runs by default.

## Offline Adapter Qualification Boundaries

- Cortex's five-field route and `actionStep` JSON shapes are training and frozen-evaluation contracts, not the shipped iOS response wire. Release iOS continues to use its current constrained `action` or `final` JSON contract, and the runtime owns routing, clarification, canonical manifest validation, approval, and action persistence.
- Schema 1.1 contamination evidence is hash-only. It checks exact record or segment fingerprints, 13-token near-overlap shingles, and 4-token short-window containment over non-system content; raw held-out text is not written into the report. Validation also verifies the report self-hash, training/evaluation corpus hashes and counts, the public-evaluation fingerprint bundle SHA/count, variant-manifest bindings, and zero contamination matches.
- Grounded Mouth and Mimicry text cases accept only audited relation frames under narrow case, ASCII-whitespace, apostrophe/dash, and optional terminal-period normalization. Semantic symbols, markup, emoji, format controls, interrogative endings, relation inversion, unsupported qualifiers, and appended clauses fail closed. Structured Mimicry and REM evaluation prompts use the same JSON-only instruction as training, while text-mode Mimicry must not carry it.
- Smoke accounting keeps `fullCaseCount` and `generatedCaseCount` separate. The scored report covers only the generated cohort: report generated passes as `passedCaseCount`, generated failures as `generatedCaseCount - passedCaseCount`, and missing generated outputs as `missingOutputCount`. Ungenerated cases are `fullCaseCount - generatedCaseCount` and are not inserted into the scored report. `criticalFailureCount` counts critical failures within the generated cohort, not all generated failures. `smoke_complete` is not a full-suite quality-gate pass.
- A future independently verified `603/603` aggregate would prove the offline six-adapter evaluation and bound artifact lineage only. It would not prove that those artifacts are installed or selected by iOS, and it would not replace signed-build, real-device, TestFlight, or live-runtime validation.

## Feature-Complete Release Gate

Run these commands before claiming a Release hardening pass:

```bash
git diff --check
python3 -m compileall tools scripts
bash scripts/check-lumen-integration-gate.sh
uv run --python 3.12 --with-editable ./tools/lumen_manifest_crawler --with pytest --with pydantic --with typer --with rich pytest -m "not slow and not e2e"
cd tools/lumen_manifest_crawler && uv run --python 3.12 --with pytest pytest --collect-only
```

On an Apple host with a connected iPhone, prefer a freshly resolved physical-device destination. Keep the actual identifier machine-local:

```bash
LUMEN_DEVICE_UDID='<fresh connected-device UDID>'
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -configuration Debug \
  -destination "platform=iOS,id=$LUMEN_DEVICE_UDID" \
  -derivedDataPath build/DerivedData-DeviceProductionReady \
  build-for-testing
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -configuration Debug \
  -destination "platform=iOS,id=$LUMEN_DEVICE_UDID" \
  -derivedDataPath build/DerivedData-DeviceProductionReady \
  test-without-building -only-testing:LumenTests/<SuiteName>
```

Resolve the device again before install/launch work rather than treating a previously recorded identifier as durable. Report a successful build, executed tests, install, launch, and smoke separately.

On a macOS/Xcode runner with the requested simulator installed:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' test CODE_SIGNING_ALLOWED=NO
```

On this host, prefer the dedicated simulator when it is available:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' build-for-testing CODE_SIGNING_ALLOWED=NO
bash scripts/run_focused_simulator_tests.sh --only-testing LumenTests/<SuiteName>
```

The simulator commands above are fallback/coverage tools. Do not combine their outcome with physical-device evidence or present them as a device pass.

## Current Hardening Evidence (2026-08-09)

- A signed Debug build and signed `build-for-testing` completed on a physical iPhone 16 Pro running iOS 26.6. The device identifier is intentionally not recorded here.
- The latest runtime-hardening physical-device risk suite passed 237 of 237 tests with 0 failures and 0 skips. This supersedes the earlier 57-test and 16-test subsets as the current risk-focused XCTest checkpoint.
- The corrected diagnostics resource-budget regression passed on the physical device, confirming that persistent runtime diagnostics use the canonical heavy-model-work gate.
- The signed app was installed and launched with `devicectl`, and its process was verified alive. That install/launch checkpoint is process-liveness evidence only; it did not itself exercise an interactive product flow.
- A separate DEBUG-only physical interactive model/tool validation ran from source `95174d975da515cf8625212592721cd0baa7bfa5`, build `20260810031810`, on the iPhone 16 Pro running iOS 26.6. Its dedicated physical-device XCUITest passed 1/1. The actual local model emitted the correlated `alarm.authorization_status` action, `SecureToolRegistry` executed that permission-safe status read and returned the trusted observation, and the model-authored final matched the observation.
- The fail-closed host verifier accepted the corresponding redacted package. Its selected byte identity had SHA-256 `d15676774b3d28feef7eca63b67e06afc753cb4ef58d1d0956cd135ba46c610f`; this digest records the verified package and does not imply that the exported package is tracked in Git.
- No simulator was used as evidence for this hardening checkpoint.
- A no-upload Release archive/export completed from exact source `5df950aeb58088d2cf5a371a0b373a8ca9377d5e` with build number `20260810035524`. `xcodebuild` reported `** ARCHIVE SUCCEEDED **` for `build/Lumen-20260809-235633.xcarchive` and `** EXPORT SUCCEEDED **` for `build/export-Lumen-20260809-235633/Lumen.ipa`; the IPA had SHA-256 `760c1a4f8f7444211229bf73d7804fb1e5dd90ff36082c7f32615b35b27c9a83`.
- The release wrapper verified archive and IPA Info.plist/provenance, the app privacy manifest, bundled `AgentBehaviorManifest.json` SHA-256 `f050e3b696cc73a82a2c1a9c613efa19ecd16027f0419b81c81ffcafd45817c8`, MSAL `1.9.0`, and signed entitlements. The exported signature used Apple Distribution identity `CF9FD3FF6D932AD1EB2822261611570904AE5641`, team `52T7P32J34`, with `get-task-allow=false`.
- Xcode selected an automatically managed App Store profile with UUID `f16a0f00-406d-476a-9210-ce27e7df387c`, no provisioned devices, and expiration `2027-08-09`. It is distinct from the separately installed profile UUID `64b7a123-7fe8-48f4-a4a3-8d1aa8bfb91a`.
- Export completed despite a nonfatal `Upload Symbols Failed` warning: the embedded MSAL framework UUID `67EC8882-4D2F-33F6-88ED-8B8CEAED65B3` has no matching dSYM in the archive. Lumen and llama symbol packages are present; MSAL crash symbolication remains unproven.
- `LUMEN_NO_UPLOAD=1` was a hard boundary. No upload, TestFlight submission, processing, or App Store acceptance occurred, and the archive remains bound to source `5df950aeb58088d2cf5a371a0b373a8ca9377d5e` rather than any later documentation commit.
- The checked-in configuration pins MSAL exactly to `1.9.0`; the link and release validators enforce the exact requirement.
- Release/minimal AgentGrounding packaging fails closed on missing, drifted, invalid, warning-bearing, failed, or hash-mismatched runtime resources and on missing, unsafe, or hash-mismatched source files declared by `sourceIntegrity`. Only an explicit Debug diagnostic build may produce an empty fallback bundle.
- Background trigger/headless execution now distinguishes completed, deferred, failed, and cancelled outcomes. A background task succeeds only for a completed trigger scan; `beforeNextEvent` creation reports typed unavailability.
- Persisted E2E results/exports, behavior traces, and in-app grounding packages are versioned `redacted-v1`; raw-content legacy filenames are purged and the current-tree privacy checker rejects legacy checked-in E2E artifacts.

These checkpoints now include one exact DEBUG local-model/action/native-observation/model-final flow, but they are not a final Release qualification. The proof applies only to its source, build, device, and `alarm.authorization_status` scenario; the product and release checks below remain unproven unless separately recorded for the exact submitted Release build.

Release submission validation also requires credentialed or physical-device checks:

- TestFlight or real-device smoke test of the exact submitted Release build
- real-device local-model load and interactive generation through a production user surface in that submitted Release build
- live tool-call validation for submitted Release surfaces beyond the one DEBUG `alarm.authorization_status` proof
- live RAG indexing and search
- live memory extraction and storage
- live voice and AppIntent flows that are enabled in the submitted build
- real background scheduling and execution
- Microsoft OAuth and mailbox behavior
- App Store upload, processing, and acceptance

## App Store Connect Submission Evidence

Use the repo-native submission lane only when explicitly requested:

```bash
bash scripts/build_and_submit_appstoreconnect.sh
```

Before archiving, confirm `CURRENT_PROJECT_VERSION` in `ios/Lumen.xcodeproj/project.pbxproj` is higher than the latest uploaded build number or provide a fresh timestamp-style override accepted by the release lane. After upload, require all of these before documenting success:

- archive completed with `** ARCHIVE SUCCEEDED **`
- export completed with `** EXPORT SUCCEEDED **`
- archived and exported `CFBundleVersion` match the intended build number
- entitlement checks passed for the archive and exported IPA
- upload output contains `UPLOAD SUCCEEDED with no errors`
- upload output includes a `Delivery UUID`

If the upload log contains `ERROR:`, `Failed to upload`, `ENTITY_ERROR`, or a duplicate-build/version rejection, document the submission as failed and fix the root cause before retrying.

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
