This repository is an iOS app plus local validation and dataset tooling.

Use `AGENTS.md` as the canonical agent instruction file.

Start with `README.md` for layout and common checks. Prefer the repo scripts
over ad hoc commands when validating:

- `bash scripts/check-ios-build-readiness.sh`
- `bash scripts/check-lumen-integration-gate.sh`
- `xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=iPhone 16' build-for-testing CODE_SIGNING_ALLOWED=NO`
- `uv run --python 3.12 --with-editable ./tools/lumen_manifest_crawler --with pytest --with pydantic --with typer --with rich python -m pytest --collect-only`

Keep generated audit, manifest, and dataset artifacts out of unrelated commits
unless the task explicitly asks to regenerate them.
