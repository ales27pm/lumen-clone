# AGENTS.md

## Scope

Governs `tools/`: Python static checks, the manifest crawler/generator, fine-tuning/training, Hugging Face artifact/Space tooling, intent-classifier tooling, model conversion/audit utilities, and associated tests. Parent rules: [`../AGENTS.md`](../AGENTS.md).

## Role In The System

These programs analyze current source/configuration, enforce hardening, generate deterministic runtime/training artifacts, train/evaluate models, and publish or assemble external artifacts. They are build-time/operational code and must not be confused with the iOS runtime.

## Key Files And Entry Points

- `check_release_hardening.py`, `check_agent_kernel_boundary.py`, `check_adapter_runtime_invariants.py`, `check_ios_lora_hardening_invariants.py`: static contract gates.
- `lumen_manifest_crawler/`: installable Typer-based crawler/generator package; child guidance applies.
- `fine_tuning/unsloth/`: controlled SFT/DPO/evaluation/export pipeline; child guidance applies.
- `hf_zerogpu/`: external Space assembly; child guidance applies.
- `hf_artifacts/publish_hf_artifacts.py`: artifact resolution/publication workflow.
- Intent-classifier tools: dataset/training/export pipeline, optionally using scikit-learn/Core ML tooling.
- Top-level `lumen_manifest_crawler/`: compatibility import shim outside this directory.

## Public Interfaces

CLI arguments, exit codes, report schemas, generated directory layouts, dataset schemas, SHA/revision lineage, and check failure text are consumed by scripts, workflows, docs, iOS resource sync, external training, and release decisions.

## Internal Structure

Static checks parse source/project/docs. The crawler maps Swift/runtime definitions to deterministic models/writers. Training consumes frozen generated datasets/configs and emits lineage-bound artifacts. HF tools resolve immutable revisions/hashes and perform opt-in network publication.

## Incoming Dependencies

Developers, local scripts, workflows, and external controlled training hosts invoke these tools.

## Outgoing Dependencies

Tools read Swift/Xcode/docs/generated inputs and write `generated/`, `datasets/`, audit/export directories, or external services when explicitly enabled. Python package dependencies are declared by the nearest `pyproject.toml` or command; do not assume a global environment.

## Data And Control Flow

Tracked source/config -> parser/check/generator -> typed intermediate model -> deterministic artifact/report -> validation/hash -> optional controlled training/publication. Runtime evidence enters only through explicit ingestion and retains its layer/provenance.

## Local Invariants

- Checks fail closed on missing/ambiguous required evidence; do not catch and emit empty success.
- Generation is deterministic for identical source/config and records source digest/schema/provenance.
- Generated artifacts, training rows, and runtime evidence remain distinct data classes.
- Canonical fine-tuning output is `generated/fine_tuning/`; stale duplicate roots fail validation.
- Network upload/publication is opt-in, private by default where supported, and uses immutable revision/hash identity.
- Do not embed credentials, private user data, or unredacted runtime payloads in generated/training output.
- Optional dependencies produce explicit unavailable errors, not fabricated artifacts.

## Coordinated Changes

A check change requires its tests, scripts/workflows, and docs. Generator schema changes require readers, generated outputs, app resource mirror, runtime decoders, and package tests. Training changes require configs, dataset/eval/contamination contracts, lineage, resume tests, and runtime-binding evidence.

## Safe Editing Rules

Use typed models and explicit diagnostics. Keep source parsing separate from writing. Preserve stable ordering/serialization. Do not rewrite large generated scopes while making an unrelated tool fix. Do not add a dependency without updating the owning manifest/controlled environment.

## Validation

From the repository root:

```bash
python3 -m compileall tools scripts
python3 tools/check_release_hardening.py
python3 tools/check_agent_kernel_boundary.py --strict
python3 tools/check_adapter_runtime_invariants.py
python3 tools/check_ios_lora_hardening_invariants.py
python3 scripts/check-generated-jsonl-artifacts.py
uv run --python 3.12 --with-editable ./tools/lumen_manifest_crawler --with pytest --with pydantic --with typer --with rich pytest -m "not slow and not e2e"
```

Run subtree tests for training/HF changes as specified by child guidance. Do not run uploads or heavyweight training as routine validation.

## Common Failure Modes

- A check scans a stale snapshot instead of tracked source.
- Serialization order changes create large nondeterministic diffs.
- Generated-loop output is ingested as live runtime evidence.
- An optional package/network failure becomes an empty artifact.
- `uv` creates an unrequested crawler `uv.lock`.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). Children: [`lumen_manifest_crawler/AGENTS.md`](lumen_manifest_crawler/AGENTS.md), [`fine_tuning/unsloth/AGENTS.md`](fine_tuning/unsloth/AGENTS.md), and [`hf_zerogpu/AGENTS.md`](hf_zerogpu/AGENTS.md). Other tool subdirectories inherit this file because their validation/provenance rules do not justify another layer.
