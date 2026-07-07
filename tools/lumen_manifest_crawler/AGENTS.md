# AGENTS.md

## Crawler Scope

This package owns deterministic source crawling, manifest generation, dataset policy checks, audit ingestion, and developer-cycle reports.

## Python Environment

- Use `uv run --python 3.12` for tests in this package unless the task requires a different interpreter.
- The package supports Python 3.11+, but repo validation commonly standardizes on Python 3.12.
- Do not assume `python` exists on this host. Use `python3` for local compile checks when `python` is missing.

Common checks:

```bash
uv run --python 3.12 --with pytest pytest --collect-only
uv run --python 3.12 --with pytest pytest -m "not slow and not e2e"
```

From the repository root, the editable-package collect check is:

```bash
uv run --python 3.12 \
  --with-editable ./tools/lumen_manifest_crawler \
  --with pytest --with pydantic --with typer --with rich \
  python -m pytest --collect-only
```

Useful command recap:

| Command | Purpose | When to use |
| --- | --- | --- |
| `uv run --python 3.12 --with pytest pytest --collect-only` | Import and collection validation | After module, fixture, or dependency changes |
| `uv run --python 3.12 --with pytest pytest -m "not slow and not e2e"` | Local crawler tests | After crawler logic changes |
| `python3 -m compileall tools scripts` | Repo Python syntax check | After Python edits outside this package too |
| `python3 -m lumen_manifest_crawler developer-cycle --root .` | Top-level developer-cycle report | Only when the task asks for framework/report regeneration |

## Generated Artifacts

- Do not hand-edit generated manifests, datasets, or audit reports unless the task explicitly asks for artifact surgery.
- Keep heavyweight regenerated outputs out of unrelated commits.
- Preserve deterministic output paths and compatibility symlinks described in the root `README.md`.
- `uv run` may create `tools/lumen_manifest_crawler/uv.lock`; remove it if it was only a local validation byproduct.
- Do not regenerate `generated/` trees as drive-by validation. Regenerate only when the requested change affects generated artifacts or evidence.

## Evidence Rules

Developer-cycle reports and runtime audit artifacts are evidence, not proof by naming. Inspect the artifact shape before querying it, and distinguish skipped, failed, passed, and missing evidence states.
