# Lumen Clone

Lumen is an iOS app with on-device assistant runtime, tool routing, diagnostics,
runtime-audit exports, and local pipeline tooling for manifest, dataset, and
validation workflows.

## Repository Layout

- `ios/Lumen/`: iOS app source.
- `ios/LumenTests/`: XCTest coverage for app services and runtime wiring.
- `docs/`: architecture, runtime-audit, hardening, and validation notes.
- `scripts/`: build, readiness, release, and static validation helpers.
- `tools/lumen_manifest_crawler/`: Python pipeline for manifests, datasets,
  improve-loop artifacts, and audit ingestion.
- `tools/pipeline/`: Python validation utilities used by release/readiness
  checks.

## Common Checks

Run static iOS readiness checks:

```sh
bash scripts/check-ios-build-readiness.sh
```

Run the repository integration gate:

```sh
bash scripts/check-lumen-integration-gate.sh
```

Run the local Python validation set:

```sh
git diff --check
python3 -m compileall tools scripts
uv run --python 3.12 --with-editable ./tools/lumen_manifest_crawler --with pytest --with pydantic --with typer --with rich pytest -m "not slow and not e2e"
cd tools/lumen_manifest_crawler && uv run --python 3.12 --with pytest pytest --collect-only
```

Run the stable simulator compile checkpoint:

```sh
xcodebuild -project ios/Lumen.xcodeproj \
  -scheme Lumen \
  -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' \
  build-for-testing \
  CODE_SIGNING_ALLOWED=NO
```

When CoreSimulator is healthy enough for focused XCTest execution, prefer the bounded runner:

```sh
bash scripts/run_focused_simulator_tests.sh --only-testing LumenTests/<SuiteName>
```

When a compiled `.xctestrun` already exists, use `xcodebuild test-without-building` for focused reruns instead of recompiling.

Only run full simulator XCTest when the task requires full execution proof:

```sh
xcodebuild -project ios/Lumen.xcodeproj \
  -scheme Lumen \
  -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' \
  test \
  CODE_SIGNING_ALLOWED=NO
```

Collect the Python test suite with the expected interpreter and local package:

```sh
uv run --python 3.12 \
  --with-editable ./tools/lumen_manifest_crawler \
  --with pytest --with pydantic --with typer --with rich \
  python -m pytest --collect-only
```

Validate generated JSONL artifacts and compatibility aliases:

```sh
python3 scripts/check-generated-jsonl-artifacts.py
```

## Release Submission

Use the repo-native App Store Connect lane only when a release upload is explicitly requested:

```sh
bash scripts/build_and_submit_appstoreconnect.sh
```

Before submitting, ensure `CURRENT_PROJECT_VERSION` in `ios/Lumen.xcodeproj/project.pbxproj` is higher than the latest uploaded build. Treat App Store Connect upload output as authoritative: success requires `UPLOAD SUCCEEDED with no errors` plus a `Delivery UUID`; duplicate-build or `ENTITY_ERROR` output is a failed upload even if a wrapper script continues.

## Generated Artifacts

Generated manifests and datasets are deterministic pipeline outputs, not
hand-authored source. Keep heavyweight artifacts under their canonical family
directories. Compatibility paths under `generated/agent_manifest/dataset/` for
embedding and reranker datasets must be symlinks to the canonical
`generated/agent_manifest/embedding/` and `generated/agent_manifest/reranker/`
files, so checkout size does not double while older tooling paths still work.

## Reference Docs

- `AGENTS.md`
- `ios/AGENTS.md`
- `docs/AGENTS.md`
- `tools/lumen_manifest_crawler/AGENTS.md`
- `docs/VALIDATION.md`
- `docs/DEVELOPER_WORKFLOW.md`
- `docs/DEVELOPER_IMPROVE_FRAMEWORK.md`
- `docs/RELEASE_WORKFLOW.md`
- `docs/RUNTIME_AUDIT_BOUNDARIES.md`
- `docs/RUNTIME_STATUS_MATRIX.md`
- `FEATURE_COMPLETE_VALIDATION.md`
