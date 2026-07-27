# AGENTS.md

## Scope

Governs `runtime-audits/`, checked-in runtime/evidence export snapshots and their metadata. Parent rules: [`../AGENTS.md`](../AGENTS.md).

## Role In The System

This directory preserves historical evidence for analysis and regression comparison. Artifacts describe the run that produced them; they are not current runtime state and cannot substitute for fresh model/device/tool evidence.

## Key Files And Entry Points

- Audit/report JSON, JSONL, Markdown, and metadata files under `runtime-audits/`.
- `docs/ARTIFACT_STATUS.md`: ownership/freshness interpretation.
- `docs/RUNTIME_AUDIT_BOUNDARIES.md`: evidence-layer rules.
- `ios/Lumen/Services/E2ETestRunner.swift`, persistent diagnostics, behavior trace, and exporter: runtime producers.
- Crawler runtime-ingest modules: typed consumers/normalizers.

## Public Interfaces

Schema/version, timestamp, commit/build/model identity, evidence layer, correlation IDs, scenario IDs, trace status, hashes, and failure taxonomy are consumed by tooling and human reviewers.

## Internal Structure

Fresh runtime/developer export -> redacted evidence envelope/report -> optional checked-in snapshot -> later ingest/comparison with provenance retained. No source code imports these files as production behavior.

## Incoming Dependencies

Explicit runtime/developer export workflows create candidate artifacts. Human review decides whether a snapshot belongs in version control.

## Outgoing Dependencies

Crawler improve/developer-cycle tools and documentation may analyze snapshots. They must preserve age/source/evidence labels.

## Data And Control Flow

Executed run -> correlated redacted trace/result -> export -> snapshot -> later analysis. Any newer app build, manifest, registry, model, dataset, scenario, or evidence contract can invalidate freshness.

## Local Invariants

- Never call a historical snapshot current without a fresh matching run.
- Preserve producer, timestamp, commit/build/model/schema, evidence layer, and correlation metadata.
- Do not edit failures into successes or manufacture missing trace/model fields.
- Do not include raw prompts, documents, memory, tool arguments/results, OAuth/mail data, or secrets.
- Keep static/generated audit results distinct from live E2E/device evidence.

## Coordinated Changes

Schema changes require runtime exporter/E2E/persistent diagnostics, crawler ingest, docs, and tests. Adding a snapshot requires checking privacy, provenance, freshness, size, and whether it is actually needed rather than generated locally.

## Safe Editing Rules

Prefer producing a new immutable export to editing an old one. Do not delete adverse historical evidence to improve a summary. Keep bulk audit updates isolated and explain their exact producer/run.

## Validation

From the repository root:

```bash
git diff --check -- runtime-audits
python3 scripts/check-generated-jsonl-artifacts.py
```

These validate text/JSONL structure only. They do not re-run or authenticate the underlying runtime evidence.

## Common Failure Modes

- Old evidence is compared against a new build without invalidation.
- Generated/static output is mislabeled as live model evidence.
- An uncorrelated trace is chosen because it looks successful.
- Privacy-sensitive payloads are checked in with diagnostics.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). Runtime producer rules: [`../ios/Lumen/Services/Diagnostics/AGENTS.md`](../ios/Lumen/Services/Diagnostics/AGENTS.md) and [`../ios/Lumen/Services/AgentGrounding/AGENTS.md`](../ios/Lumen/Services/AgentGrounding/AGENTS.md). No child file is needed.
