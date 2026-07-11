# AGENTS.md

## Project Overview

Lumen is a native iOS local AI assistant. Treat this repository as a release-grade Swift/iOS product, not a prototype. The app should remain local-first, private, resilient, diagnosable, and Apple-quality.

Codex reads this root file first, then any closer `AGENTS.md` file under the current working directory. Keep this file focused on repository-wide rules. Put subtree-specific instructions in nested files so they override or extend this guidance only where relevant.

Primary source areas:

- `ios/Lumen/`: iOS application source.
- `ios/LumenTests/`: XCTest coverage for runtime, routing, tools, diagnostics, and services.
- `scripts/`: local validation, readiness, release, and static gates.
- `tools/`: Python validation and manifest/dataset tooling.
- `tools/lumen_manifest_crawler/`: manifest, dataset, audit, and developer-cycle pipeline.
- `docs/`: architecture, runtime, validation, and release-readiness notes.

Nested instruction files:

- `ios/AGENTS.md`: Swift, Xcode, runtime routing, and XCTest expectations.
- `docs/AGENTS.md`: shipped-state documentation rules.
- `tools/lumen_manifest_crawler/AGENTS.md`: Python crawler and generated-artifact rules.

## Non-Negotiable Engineering Rules

- Do not add placeholder, fake, mock, staged, or not-implemented production behavior.
- Do not weaken or delete tests to make validation pass.
- Do not hide real failures behind broad catch blocks or empty success values such as `[]`, `[:]`, `false`, `0`, `"{}"`, or `"[]"`.
- Do not allow Release builds to route to mock, deterministic fallback, unavailable, experimental, or not-compiled backends.
- Do not rely on prompt-only constraints for structured tool calling. Validate generated tool calls before execution.
- Do not log raw prompts, raw user documents, raw memory contents, or raw tool arguments unless the user explicitly exports diagnostics.
- Do not claim live runtime, device, TestFlight, or model-backed validation unless fresh evidence was actually produced.
- Learn from run to run. When a validated failure exposes a bad workflow assumption, update the default procedure and do not repeat the same mistake in later runs.

## Validation Policy

Use local validation. Do not trigger, rerun, or rely on GitHub Actions workflows unless the user explicitly asks for that.

Fast local checks:

```bash
git diff --check
python3 -m compileall tools scripts
python3 tools/check_release_hardening.py
bash scripts/check-lumen-integration-gate.sh
uv run --python 3.12 --with-editable ./tools/lumen_manifest_crawler --with pytest --with pydantic --with typer --with rich pytest -m "not slow and not e2e"
cd tools/lumen_manifest_crawler && uv run --python 3.12 --with pytest pytest --collect-only
```

The repository documentation may list `python -m compileall tools scripts` for parity with generic environments. On this host, `python` may be unavailable; record that exact failure if required, then run `python3 -m compileall tools scripts`.

Useful command recap:

| Command | Purpose | When to use |
| --- | --- | --- |
| `git diff --check` | Whitespace and patch sanity | Before every commit or handoff |
| `python3 -m compileall tools scripts` | Python syntax check | After script/tool edits |
| `python3 tools/check_release_hardening.py` | Release routing/static hardening guard | After runtime, docs, or gate edits |
| `bash scripts/check-lumen-integration-gate.sh` | Repo integration gate | Before claiming a hardening pass |
| `uv run --python 3.12 --with-editable ./tools/lumen_manifest_crawler --with pytest --with pydantic --with typer --with rich pytest -m "not slow and not e2e"` | Python unit suite | After Python/tooling changes |
| `cd tools/lumen_manifest_crawler && uv run --python 3.12 --with pytest pytest --collect-only` | Crawler collection check | After crawler or import-path changes |

Stable Xcode compile checkpoint:

```bash
xcodebuild -project ios/Lumen.xcodeproj \
  -scheme Lumen \
  -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' \
  build-for-testing \
  CODE_SIGNING_ALLOWED=NO
```

Use the dedicated `Lumen Focused Test iPhone` simulator when it exists. A generic simulator destination is acceptable for non-launching compile checks, but executed XCTest requires a concrete simulator.

Only run full simulator XCTest when needed and when CoreSimulator is healthy:

```bash
xcodebuild -project ios/Lumen.xcodeproj \
  -scheme Lumen \
  -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' \
  test \
  CODE_SIGNING_ALLOWED=NO
```

Optimized simulator workflow:

- Prefer `build-for-testing` first.
- Reuse the compiled `.xctestrun` with `test-without-building` for focused simulator reruns instead of recompiling.
- Prefer the SpringBoard/backboardd readiness probe over waiting for a long `simctl bootstatus -b` System App timeout. Treat `bootstatus` as an opportunistic short check, not the only readiness signal.
- Keep simulator execution bounded. The repo defaults `TEST_TIMEOUT_SECONDS=2400` for focused and validation wrapper runs.
- If simulator install/launch/test-manager handoff stalls, report it as an environment boundary and keep compile/build-for-testing evidence separate from executed XCTest evidence.

For release-readiness claims, also document manual checks that cannot be proven locally: signed archive/export, entitlement inspection, privacy manifest validation, TestFlight or real-device smoke test, real-device local model load, live tool calls, live RAG, live memory, voice, and AppIntent flows.

Commands and flows agents must not run unless the user explicitly asks:

- Do not run or rerun GitHub Actions workflows.
- Do not push just to update a PR when the user has warned that Actions usage matters.
- Do not run App Store Connect upload or release scripts such as `scripts/build_and_submit_appstoreconnect.sh`.
- Do not run unbounded simulator launch/test loops. Prefer `build-for-testing` first.
- Do not run broad cleanup commands that delete generated artifacts, DerivedData, or simulator state without stating the impact first.

When the user explicitly asks to build and submit:

- Use `bash scripts/build_and_submit_appstoreconnect.sh`; do not rely on the executable bit.
- Check or bump `CURRENT_PROJECT_VERSION` before archiving. The submitted `CFBundleVersion` must be higher than the latest App Store Connect upload.
- Treat App Store Connect upload output as authoritative. If `altool` prints validation errors, duplicate-build errors, `ENTITY_ERROR`, or `Failed to upload`, the upload failed even if the process exits cleanly.
- Report the archive path, IPA path, `CFBundleVersion`, and `Delivery UUID` only when the terminal output includes `UPLOAD SUCCEEDED with no errors`.

## Runtime And Tool-Calling Rules

- Release runtime routing must fail closed.
- Diagnostic deterministic fallback, unavailable GGUF native bridge paths, legacy bridge probes, and experimental adapters must remain DEBUG-only.
- Structured tool calls must be parsed and validated before execution: canonical tool lookup, enabled-tool check, required fields, exact JSON types, enum/tool-name validation through the manifest, extra-argument rejection, bounded repair, and typed failure after retry exhaustion.
- RAG and memory paths must distinguish empty results from permission, embedding, model, SwiftData, persistence, corrupt index, unsupported type, cancellation, disabled, extraction, and save failures.
- User-facing runtime state should use explicit readiness and recovery states, not vague fallback copy.
- Live E2E and training scoring must reject incomplete final text. Dangling endings such as `an`, `a`, `the`, `with`, `because`, and `you do not need an` cannot pass; synthesize from trusted tool observations only when the observation supports the answer.
- Missing required tool arguments must clarify or be scored as clarification, not generic safe failure coverage evidence. For example, `files.read` without a filename/path asks `Which file should I read?`.
- Before model-backed training scenarios, respect runtime budget and CPU watchdog degradation. If degradation is known before generation, emit a single non-actionable runtime-preflight result with `trainingSignal=false`.

## Documentation Rules

- Keep shipped-state docs honest. Do not describe shipped Release features as partial, planned, staged, or compatibility-bridge backed.
- Separate DEBUG-only or experimental surfaces from Release behavior.
- Update `FEATURE_COMPLETE_VALIDATION.md` when a hardening pass changes runtime status, validation evidence, or remaining manual checks.
- Keep `README.md`, `docs/VALIDATION.md`, `docs/RUNTIME_STATUS_MATRIX.md`, and `docs/AGENT_KERNEL_MIGRATION_STATUS.md` aligned with actual code.

## Git And PR Rules

- Preserve unrelated user changes. Stage explicit paths when the worktree is mixed.
- Avoid generated manifests, datasets, audit exports, and validation byproducts in unrelated commits.
- `uv run` in `tools/lumen_manifest_crawler` can create `tools/lumen_manifest_crawler/uv.lock`; remove it if it was only a validation byproduct.
- Do not push to a PR branch if doing so would trigger unwanted GitHub Actions. Ask first or keep the change local.
- When a PR is requested, use a draft PR by default and include local validation evidence plus any manual validation gaps.
- If a docs-only or instruction-only change is requested after a PR already exists, leave it unpushed unless the user explicitly accepts another remote run.

## Agent Setup Notes

- Start from the repository root when possible so Codex loads this file plus nested subtree files.
- If behavior seems stale, start a new Codex session in the target directory; project instructions are loaded at session start.
- Add nested `AGENTS.md` files only when a subtree has materially different commands or policies. Keep them short and operational.
- Use `AGENTS.override.md` only for temporary local overrides; do not commit one unless the task explicitly requires it.

## Code Style

- Prefer existing Swift patterns, typed errors, typed diagnostics, explicit state models, and small composable services.
- Keep changes focused to the requested behavior and surrounding ownership boundary.
- Add or update tests in proportion to risk and blast radius.
- Use `rg` for searches.
- Use `apply_patch` or normal editor patches for manual edits; avoid broad generated rewrites unless the task requires them.
