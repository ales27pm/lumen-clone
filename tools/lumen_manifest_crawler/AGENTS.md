# AGENTS.md

## Scope

Governs `tools/lumen_manifest_crawler/`, its `lumen_manifest_crawler/` package, package tests, configuration, and crawler-owned generated contracts. Parent rules: [`../AGENTS.md`](../AGENTS.md).

## Role In The System

This package deterministically crawls current Swift source and runtime definitions, builds a typed behavior manifest, emits audits/prompts/datasets/fine-tuning inputs, ingests explicitly supplied runtime evidence, and orchestrates improve/developer-cycle reports. The root `lumen_manifest_crawler/` package is only a compatibility redirect into this implementation.

## Key Files And Entry Points

- `pyproject.toml`: Python `>=3.11`, Hatchling build, runtime dependencies, and pytest configuration.
- `lumen_manifest_crawler/cli.py` and package entrypoint: Typer commands including generation, improve loop, developer cycle, and framework workflows.
- Crawler/parser modules: derive source/runtime/tool definitions while excluding generated trees from the source digest.
- Writer/artifact modules: own canonical generated paths, stable serialization, aliases, SHA files, prompts, datasets, and runtime grounding.
- Runtime-ingest modules: normalize explicit runtime evidence without promoting generated output.
- Dataset/fine-tuning modules: schema, contamination, evaluation, split, and training artifacts.
- `tests/`: contract, determinism, schema, path, evidence, and CLI coverage.

## Public Interfaces

CLI commands/options, Pydantic schemas, generated filenames/layout, manifest schema/version/source digest, SHA sidecars, report JSON/Markdown shapes, dataset row formats, and failure diagnostics are consumed by scripts, iOS loaders, training, docs, and audits.

## Internal Structure

Source discovery -> Swift/config parser -> normalized typed records -> deterministic manifest -> writers for canonical artifact families. Runtime evidence is a separate ingest path with explicit provenance/evidence layer. Improve/developer-cycle orchestration consumes these typed products and emits reports; it does not prove iOS runtime execution itself.

## Incoming Dependencies

Root/scripts/workflows invoke the CLI or editable package. It reads `ios/Lumen/`, selected project/config/docs, prior explicit runtime exports, and public corpus manifests.

## Outgoing Dependencies

It writes canonical scopes under `generated/`, the public corpus under `datasets/` when that builder is invoked, and reports/audits. `scripts/sync-agent-manifest-resource.sh` then owns the app mirror; the crawler must not directly mutate production Swift.

## Data And Control Flow

Current tracked source -> deterministic source digest -> parsed contracts -> validated Pydantic model -> stable writer -> SHA/schema checks. Optional runtime export -> runtime ingest -> evidence-tagged report. Fine-tuning output -> contamination/eval/manifest gates -> controlled training consumer.

## Local Invariants

- Never use generated output or codebase snapshots as crawler source input for the source digest.
- Stable input produces stable ordering, bytes, hashes, and paths.
- Canonical fine-tuning root is `generated/fine_tuning/`; reject `generated/agent_manifest/fine_tuning/` as a stale duplicate.
- Runtime evidence must carry explicit source/layer/correlation and cannot be synthesized from generated loop results.
- Missing fields, schema drift, contamination, stale hashes, or incomplete artifacts fail explicitly.
- Preserve compatibility aliases only when they do not duplicate canonical payloads.
- Generated resources remain privacy-safe and contain no raw user runtime data.

## Coordinated Changes

Manifest schema changes require generated outputs, `ios/Lumen/AgentBehaviorManifest.json`, sync/copy scripts, Swift decoders/auditor, tests, and docs. Dataset schema changes require Unsloth consumers/evals/contamination tests. CLI/path changes require scripts, workflows, compatibility shim, and documentation.

## Safe Editing Rules

Keep parsers pure and writers centralized. Add typed fields to Pydantic models before emitting them. Preserve deterministic sort/serialization and source exclusions. Do not manually patch generated JSON/JSONL to make tests pass.

## Validation

From the repository root:

```bash
python3 -m compileall tools/lumen_manifest_crawler lumen_manifest_crawler
uv run --python 3.12 --with-editable ./tools/lumen_manifest_crawler --with pytest --with pydantic --with typer --with rich pytest -m "not slow and not e2e"
```

From the package directory:

```bash
cd tools/lumen_manifest_crawler && uv run --python 3.12 --with pytest pytest --collect-only
```

Before running either `uv` command, record whether `tools/lumen_manifest_crawler/uv.lock` exists. Remove that explicit path afterward only if it was absent before the run and this validation created it; preserve any pre-existing tracked or untracked lockfile.

If a full regeneration is explicitly required, validate the whole regenerated artifact scope and do not mix unrelated generated changes.

## Common Failure Modes

- Parser includes `generated/` and changes its own digest recursively.
- A writer emits nondeterministic map/set order.
- A compatibility path becomes a second canonical payload.
- Runtime ingestion accepts a generated report as live evidence.
- A schema change updates JSON but not Swift decoder/training consumer.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). Generated output rules: [`../../generated/AGENTS.md`](../../generated/AGENTS.md). Training consumer: [`../fine_tuning/unsloth/AGENTS.md`](../fine_tuning/unsloth/AGENTS.md). No additional child file is needed; package submodules share one deterministic schema/writer boundary.
