# AGENTS.md

## Scope

Governs `scripts/`, including local gates, Xcode/simulator runners, manifest sync, release/upload automation, training launchers, diagnostics helpers, hotfixes, and the unrelated removable-media installer. Parent rules: [`../AGENTS.md`](../AGENTS.md).

## Role In The System

Scripts operationalize validation, generation, release, training, and exceptional repair. Their side effects vary substantially; a script name is not proof that it is read-only or safe for routine use.

## Key Files And Entry Points

- `check-lumen-integration-gate.sh`: broad local integration/static gate.
- `validate_lumen_ios.sh`: bounded iOS build-for-testing/default validation wrapper.
- `run_focused_simulator_tests.sh`: bounded simulator build/test execution using stable DerivedData and `.xctestrun`.
- `check-ios-build-readiness.sh`: static/Xcode readiness; static mode is not compilation.
- `sync-agent-manifest-resource.sh`: canonical generated manifest -> app resource mirror.
- `archive_lumen_stable.sh`: signed archive creation; `build_and_submit_appstoreconnect.sh`: archive/export plus optional upload and inline upload-output interpretation.
- `ubuntu_train_lumen_full_pipeline.sh`: controlled external Ubuntu/GPU training launcher.
- `tripleboot_aio.sh`: destructive, standalone removable-media installer unrelated to routine Lumen validation.
- Hotfix/repair scripts: source/project mutation, not validation.

## Public Interfaces

Command-line arguments, environment variables, exit behavior, emitted artifact paths, logs, and success markers are consumed by developers, workflows, release operators, and documentation.

## Internal Structure

Validation scripts compose Python checks and xcodebuild. Sync scripts compare/copy generated resources. Release scripts can patch project/linker settings, increment/build, archive/export, and upload. Training scripts stage controlled inputs and remote/local execution. Exceptional scripts may deliberately mutate source or devices.

## Incoming Dependencies

Humans, documentation, and `.github/workflows/` invoke these scripts.

## Outgoing Dependencies

Scripts call git/path tools, Python, uv, Xcode/xcrun/simctl, signing/upload tools, Docker/SSH/GPU tooling, and repository generators. They may write build, generated, export, project, or device state.

## Data And Control Flow

Operator intent -> argument/preflight validation -> explicit side effects -> authoritative tool output -> exit/report. Wrapper exit status is insufficient when an underlying uploader prints validation/duplicate/entity errors.

## Local Invariants

- Default local validation does not trigger GitHub Actions, App Store upload, destructive cleanup, hotfixes, or removable-media changes.
- Keep simulator loops bounded; the focused runner uses stable DerivedData and performs `build-for-testing` before `test-without-building`.
- Release/upload scripts run only on explicit request. After successful archive/export, non-upload flows may report the archive, IPA, and build number; report a Delivery UUID only after confirmed, error-free upload completion.
- Sync/generation scripts fail on drift; they do not silently accept nondeterministic/runtime-evidence manifests as app resources.
- Destructive target/device scripts require explicit target confirmation and clear impact.
- Never echo credentials or signing/token material.

## Coordinated Changes

Changing a validation script requires workflow/docs/AGENTS command review. Sync changes require crawler/generated/app resource and Xcode post-build review. Release changes require project settings, entitlements, versioning, export/upload semantics, and release docs. Training launcher changes require Unsloth lineage/tests and artifact docs.

## Safe Editing Rules

Use `set -euo pipefail` where compatible, quote paths/variables, validate required inputs before mutation, and preserve underlying stderr/output. Make dry-run/no-upload defaults explicit for network/destructive tools. Do not use a hotfix script as a substitute for a reviewed source patch.

## Validation

From the repository root after script edits:

```bash
bash -n scripts/*.sh
python3 -m compileall scripts
bash scripts/check-lumen-integration-gate.sh
git diff --check -- scripts
```

Run only the edited script's safe/local mode. Do not run archive, upload, training, cleanup, hotfix, or `tripleboot_aio.sh` merely to validate syntax.

## Common Failure Modes

- A static readiness check is reported as Xcode compilation.
- An upload process exits zero while its log contains `Failed to upload`, `ENTITY_ERROR`, duplicate build, or validation errors.
- A validation run creates an unrequested lockfile/generated artifact.
- A release helper silently mutates `project.pbxproj`.
- A generic device variable points at the wrong removable disk.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). Crawler generation rules are in [`../tools/lumen_manifest_crawler/AGENTS.md`](../tools/lumen_manifest_crawler/AGENTS.md); training rules are in [`../tools/fine_tuning/unsloth/AGENTS.md`](../tools/fine_tuning/unsloth/AGENTS.md).
