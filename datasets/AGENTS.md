# AGENTS.md

## Scope

Governs `datasets/`, currently the checked-in public adapter corpus and its provenance/licensing material. Parent rules: [`../AGENTS.md`](../AGENTS.md).

## Role In The System

The public corpus supplies licensed, pinned, integrity-checked records to crawler/fine-tuning preparation. It is not user memory, runtime telemetry, or a place for ad hoc training examples.

## Key Files And Entry Points

- `datasets/public_adapter_corpus/records.jsonl`: generated corpus records.
- Corpus manifest/provenance files in `datasets/public_adapter_corpus/`: source repositories/revisions, hashes, record counts, and processing metadata.
- Third-party attribution/license documentation in the corpus directory.
- Public-corpus builder and validators under `tools/lumen_manifest_crawler/`.
- `scripts/check-generated-jsonl-artifacts.py`: structural JSONL guard.

## Public Interfaces

Record schema, stable IDs, source/revision/license metadata, content hashes, split/category labels, and exact caps/counts are consumed by crawler dataset builders, contamination/eval gates, and Unsloth training.

## Internal Structure

Pinned public source -> license/provenance validation -> deterministic normalization/filtering -> JSONL records -> manifest/hash/count checks -> generated fine-tuning consumer.

## Incoming Dependencies

The crawler's public-corpus builder is the owning writer. Source datasets are external and must be pinned/attributed.

## Outgoing Dependencies

Crawler fine-tuning modules and Unsloth preparation consume records under explicit caps/splits. No iOS runtime reads this corpus directly.

## Data And Control Flow

Immutable external revision -> verified license/hash -> deterministic records -> contamination/dedup/cap checks -> training mix. Changes propagate through generated fine-tuning manifests and lineage.

## Local Invariants

- Do not hand-edit `records.jsonl`; regenerate from the builder.
- Preserve source revision, license, attribution, hash, and deterministic record ID.
- Do not add private user data, runtime prompts, emails, documents, memories, secrets, or unclear-license content.
- Exact caps/splits and contamination checks remain enforced.
- A source revision change is a new provenance event and must update hashes/manifests.

## Coordinated Changes

Schema/content changes require crawler dataset code/tests, generated fine-tuning outputs, Unsloth row/cap/lineage tests, attribution/licenses, and artifact status review.

## Safe Editing Rules

Treat the corpus as generated, reviewable supply-chain input. Add sources through the builder with immutable revisions and verified license compatibility. Keep large record diffs isolated from unrelated code changes.

## Validation

From the repository root:

```bash
python3 scripts/check-generated-jsonl-artifacts.py
uv run --python 3.12 --with-editable ./tools/lumen_manifest_crawler --with pytest --with pydantic --with typer --with rich pytest -m "not slow and not e2e"
git diff --check -- datasets
```

Run Unsloth public-corpus/cap tests when the corpus feeds a training change.

## Common Failure Modes

- A mutable upstream branch is recorded instead of an immutable revision.
- Records change without manifest/hash/count updates.
- License/attribution is missing.
- Runtime/private content is mixed into public training data.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). Generator: [`../tools/lumen_manifest_crawler/AGENTS.md`](../tools/lumen_manifest_crawler/AGENTS.md). Training consumer: [`../tools/fine_tuning/unsloth/AGENTS.md`](../tools/fine_tuning/unsloth/AGENTS.md).
