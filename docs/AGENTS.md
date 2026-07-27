# AGENTS.md

## Scope

Governs `docs/` only. Repository-wide rules remain in [`../AGENTS.md`](../AGENTS.md).

## Role In The System

This directory records architecture, current runtime status, validation/evidence boundaries, generated artifact interpretation, training/release procedures, and historical migrations. Documentation must distinguish shipped behavior from plans, historical snapshots, and manual gaps.

## Key Files And Entry Points

- `docs/VALIDATION.md`: validation workflow and evidence interpretation.
- `docs/RUNTIME_STATUS_MATRIX.md`: current runtime capability status.
- `docs/AGENT_KERNEL_MIGRATION_STATUS.md`: current kernel migration state.
- `docs/ARTIFACT_STATUS.md`: canonical generated/runtime artifact ownership and freshness.
- `docs/RUNTIME_AUDIT_BOUNDARIES.md`: static/live/device evidence boundaries.
- `docs/DEVELOPER_IMPROVE_FRAMEWORK.md`: developer-cycle/evidence workflow.
- `FEATURE_COMPLETE_VALIDATION.md` and root `README.md`: cross-directory shipped-state summaries that must remain aligned.
- Historical migration/PR documents are evidence of their time, not current authority.

## Public Interfaces

Commands, path names, capability claims, status matrices, evidence labels, and manual-check lists are consumed by developers, release operators, and coding agents. Incorrect prose can cause unsafe commands or false release claims.

## Internal Structure

Current-state documents should point to executable source/configuration and fresh evidence. Historical documents retain context but must be labeled so they cannot override current status. Artifact documents distinguish generated examples, historical runtime exports, and fresh live proof.

## Incoming Dependencies

Code/runtime/tooling/release changes require documentation updates when they alter shipped status, validation evidence, generated ownership, or remaining manual work.

## Outgoing Dependencies

Agents and scripts use documented commands and paths. Documentation itself does not define runtime behavior; executable files remain authoritative.

## Data And Control Flow

Verified code/config/evidence -> current status/validation documentation -> developer/release decision. Historical artifact -> provenance/freshness label -> limited claim, never automatic promotion to current proof.

## Local Invariants

- Describe shipped Release behavior in present tense only when code and current evidence support it.
- Separate DEBUG/experimental/compatibility paths from Release behavior.
- Separate compile, static audit, simulator XCTest, live model E2E, device, signed archive, TestFlight, and upload evidence.
- Do not describe generated manifest/training output as live runtime proof.
- Keep `README.md`, `FEATURE_COMPLETE_VALIDATION.md`, `docs/VALIDATION.md`, `docs/RUNTIME_STATUS_MATRIX.md`, and `docs/AGENT_KERNEL_MIGRATION_STATUS.md` aligned after status changes.
- Historical `LEGACY_*` or migration prose can be stale; current code/status files win. A verified contradiction should be marked, not silently copied.
- Do not include secrets, raw personal data, or private runtime payloads.

## Coordinated Changes

Runtime routing changes require status matrix/kernel migration/feature validation review. Tool/manifest changes require artifact/validation docs. Release scripts or evidence semantics require validation/release docs. Training pipeline changes require artifact status, training runbook, lineage, and deployment-evidence wording.

## Safe Editing Rules

Use exact commands and paths found in current scripts/configuration. State working directory and what a command proves. Preserve unresolved manual checks and contradictions. Do not update non-AGENTS documentation during an AGENTS-only task.

## Validation

From the repository root:

```bash
git diff --check -- docs README.md FEATURE_COMPLETE_VALIDATION.md
rg -n -i 'generated/agent_manifest/fine_tuning|legacy.*shipped|partial|planned|staged' docs README.md FEATURE_COMPLETE_VALIDATION.md
```

The `rg` command is a review aid, not an automatic failure rule: inspect each historical/current context. Validate every added command against the referenced script/manifest before publishing it.

## Common Failure Modes

- A historical migration snapshot is cited as current architecture.
- "Build passed" is rewritten as "runtime validated."
- The stale duplicate fine-tuning path is presented as canonical.
- A manual privacy/signing/device check is deleted because local tests passed.

## Parent And Child Guidance

Parent: [`../AGENTS.md`](../AGENTS.md). No nested files are needed; documents share one shipped-state/evidence policy.
