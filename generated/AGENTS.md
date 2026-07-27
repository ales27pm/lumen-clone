# AGENTS.md

## Scope

Governs `generated/`, all checked-in derived manifests, prompts, datasets, audits, reports, visualizations, fine-tuning inputs, hashes, and resource payloads. Parent rules: [`../AGENTS.md`](../AGENTS.md).

## Role In The System

This directory is the canonical output surface for the manifest crawler and related deterministic tooling. Some files are bundled into the iOS build or consumed by training, but none are authoritative over their source generator and none alone prove live runtime behavior.

## Key Files And Entry Points

- `generated/agent_manifest/`: canonical behavior manifest, prompt/grounding resources, audits, and metadata.
- `generated/fine_tuning/`: canonical fine-tuning datasets/manifests/evaluation inputs.
- Generated embedding/reranker resources under `generated/agent_manifest/`: canonical generated model-support data where present.
- SHA/hash/manifest sidecars: provenance and integrity contracts.
- `tools/lumen_manifest_crawler/`: primary generator.
- `docs/ARTIFACT_STATUS.md`: interpretation, canonical paths, and freshness policy.
- `scripts/sync-agent-manifest-resource.sh`: app resource mirror step.

## Public Interfaces

Paths, filenames, schema versions, JSON/JSONL field shapes, ordering, hashes, source digests, runtime-evidence flags, and report semantics are consumed by Swift resource loaders, sync/copy scripts, training, audits, docs, and external tools.

## Internal Structure

Crawler input/model -> deterministic writer -> canonical artifact family -> hash/schema checks -> downstream sync/training/audit. Historical or sample reports remain labeled by producer/evidence layer.

## Incoming Dependencies

Crawler/generator, dataset builders, improve/developer-cycle tooling, and explicit runtime-ingest operations write this directory.

## Outgoing Dependencies

The iOS build consumes selected grounding resources; Unsloth consumes fine-tuning data; docs/audits consume reports; scripts compare mirrors/hashes.

## Data And Control Flow

Tracked source/config/evidence input -> generator -> complete artifact scope -> validation -> checked-in review -> downstream consumer. Never reverse this flow by treating a generated file as the source to patch.

## Local Invariants

- Do not hand-edit generated files. Change the generator/input and regenerate the coherent scope.
- Stable inputs must produce stable bytes/order/hashes.
- `generated/fine_tuning/` is canonical; `generated/agent_manifest/fine_tuning/` must not reappear.
- App behavior manifest is deterministic/source-derived and has `runtimeEvidence=false`; live evidence remains separate.
- Generated loop/report output is not live model, device, or tool evidence.
- Keep canonical and compatibility paths from duplicating payloads.
- Preserve provenance, licenses, source digest, schema, and SHA sidecars together.

## Coordinated Changes

Manifest changes require crawler, app mirror/copy scripts, Swift decoder/auditor, and grounding tests. Fine-tuning changes require crawler schema, public corpus provenance, Unsloth configs/tests, and artifact lineage. Embedding/reranker changes require iOS model identity/dimension and retrieval tests.

## Safe Editing Rules

Regenerate only for a requested generator/input change. Review the full regenerated scope rather than staging a misleading subset. Do not normalize/reformat large JSONL manually. Do not delete historical evidence to make freshness checks pass.

## Validation

From the repository root:

```bash
python3 scripts/check-generated-jsonl-artifacts.py
bash scripts/check-lumen-integration-gate.sh
git diff --check -- generated
```

For crawler-owned changes, also run the crawler suite in [`../tools/lumen_manifest_crawler/AGENTS.md`](../tools/lumen_manifest_crawler/AGENTS.md).

## Common Failure Modes

- Manual JSON edit leaves SHA/source digest stale.
- A partial regeneration mixes schema versions.
- A generated report is presented as fresh runtime evidence.
- The stale fine-tuning duplicate path returns.
- App mirror and canonical manifest diverge.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). Generator: [`../tools/lumen_manifest_crawler/AGENTS.md`](../tools/lumen_manifest_crawler/AGENTS.md). Training consumer: [`../tools/fine_tuning/unsloth/AGENTS.md`](../tools/fine_tuning/unsloth/AGENTS.md). No child file is needed because all subtrees share generated ownership.
