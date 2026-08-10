# Artifact Status and Freshness

This document classifies generated or exported artifact paths used by the Lumen developer-improvement loop. It prevents stale exports, regenerated datasets, and expected-but-absent loop outputs from being mistaken for current live proof.

## Artifact classes

| Path | Status | How to use it | Regeneration or refresh command |
|---|---|---|---|
| `runtime-audits/` | Historical, dated, privacy-reviewed runtime exports and quarantine receipts. | Treat each artifact as an archived snapshot, never current proof by default. Legacy raw-content E2E formats are forbidden; only versioned privacy-safe exports may be added after provenance/freshness review. | Export a fresh `redacted-v1` in-app dataset package or E2E report from the candidate build, label it with build/commit/device/time, run `python3 tools/check_runtime_audit_privacy.py`, then pass it with `--runtime-audit` to `generate` or `improve-loop`. For the DEBUG physical correlated package, also run `tools/verify_interactive_model_tool_evidence.py` with the expected source revision, build number, and freshness bound. |
| `tools/lumen_manifest_crawler/generated/` | Regenerable crawler and dataset-pipeline outputs. | Use these as checked-in examples or local tool outputs, not as hand-authored source of truth. They can be replaced by rerunning the crawler from Swift source and selected runtime inputs. | From `tools/lumen_manifest_crawler`: `python -m lumen_manifest_crawler generate --root ../.. --output generated/agent_manifest --pretty --generate-system-prompts --generate-agent-fine-tuning --fine-tuning-output generated/fine_tuning`. |
| `generated/fine_tuning/` | Canonical checked-in fine-tuning datasets. | Use this as the single checked-in per-agent fine-tuning output root. Do not commit duplicate files under `generated/agent_manifest/fine_tuning/`; those are stale compatibility outputs when `--fine-tuning-output` is omitted. | Regenerate with the manifest crawler command above and keep `--fine-tuning-output generated/fine_tuning` explicit. |
| `ios/Lumen/AgentBehaviorManifest.json` | Bundled deterministic app resource, not live evidence. | This is the manifest copy shipped inside the iOS app bundle so the app can audit live runtime state against the manifest-derived training truth. Its `artifactStatus.runtimeEvidence=false` and `artifactStatus.deterministicBuild=true` fields mean the file is not proof of a live TestFlight run. Do not edit it by hand. | Regenerate `generated/agent_manifest/AgentBehaviorManifest.json` with the crawler, then run `scripts/sync-agent-manifest-resource.sh` to copy/sync that JSON into `ios/Lumen/AgentBehaviorManifest.json` as part of the app resource update. |
| `generated/agent_improvement_loop/` | Expected improvement-loop outputs. | Documentation and runbooks may mention these files before they exist in a fresh clone. Absence means the loop has not been run in that workspace, not that the paths are invalid. | Run `python -m lumen_manifest_crawler improve-loop --root . --output generated/agent_manifest --loop-output generated/agent_improvement_loop --runtime-audit <fresh-audit-or-report> --generate-system-prompts --generate-agent-fine-tuning`. |
| Other `generated/...` paths mentioned in docs | Expected generated outputs. | Treat them as products of the manifest crawler, dataset compiler, adapter export, or improvement loop. They are not guaranteed to be present until the corresponding command has run. | Use the command attached to the owning pipeline: `generate` for manifest/datasets/fleet prompts, `improve-loop` for loop state/runbooks, and adapter export/training commands for model artifacts. |

## Generated dataset alias policy

Embedding and reranker datasets have canonical homes under:

- `generated/agent_manifest/embedding/`
- `generated/agent_manifest/reranker/`

The older compatibility paths under `generated/agent_manifest/dataset/` must be
symlinks to those canonical files, not second full JSONL copies. This keeps
legacy docs and tooling paths usable while avoiding duplicate tens-of-MB
checkout payloads. `scripts/check-generated-jsonl-artifacts.py` enforces this
alongside the zero-byte JSONL guard.

Per-agent fine-tuning JSONL files have a separate canonical root:
`generated/fine_tuning/`. The repository must not also check in
`generated/agent_manifest/fine_tuning/`, because that creates stale parallel
training corpora with different source snapshots and large duplicate payloads.

## AgentBehaviorManifest timestamp and evidence status

`AgentBehaviorManifest.json` is generated from source by the manifest crawler. It is intentionally a deterministic source artifact, so `app.generatedAt` may be fixed to the Unix epoch (`1970-01-01T00:00:00+00:00`) when no real build metadata is being injected. That fixed timestamp is not a placeholder for TestFlight activity; it exists so repeated crawler runs with identical inputs produce stable JSON and hashes.

Source provenance is intentionally non-self-referential. `sourceIntegrity.baseCommit`
identifies the checked-out commit on which generation started,
`sourceIntegrity.workingTreeDigest` identifies the current non-generated repository
snapshot, and `sourceIntegrity.dirtyState` says whether that snapshot differs from
the base commit. Generated output trees and the synced iOS manifest are excluded
from the working-tree digest so writing the manifest cannot change its own identity.
The legacy `sourceIntegrity.commit` key is accepted when older manifests are read,
but new manifests do not emit it as though a dirty pre-commit generation already
belonged to a future commit.

The manifest must declare:

```json
"artifactStatus": {
  "artifactStatus": "deterministic_source_manifest",
  "deterministicBuild": true,
  "runtimeEvidence": false,
  "generatedAtPolicy": "unix_epoch_for_reproducible_source_artifact",
  "liveTestFlightProof": false
}
```

If the manifest is ever changed to represent a concrete app build instead of a deterministic source snapshot, the build pipeline must inject the real bundle identifier, build/version label, build timestamp, and source revision at build time. Until then, this manifest is never live TestFlight proof; only selected runtime audits or live E2E reports can make live device/TestFlight claims.

## Freshness rule

A runtime proof is current only when all of the following are true:

1. The audit/report file is explicitly selected by the current runbook, command line, CI job, or PR notes.
2. The selected file identifies the app build, TestFlight build label, export time, or scenario run time closely enough to order it against other exports.
3. The selected file was captured after the code, manifest, entitlement, dataset, or model-adapter change being validated.
4. The selected file covers the scenario or surface being claimed; device-runtime grounding traces support diagnosis, while `live_e2e` reports own live scenario pass/fail.
5. No newer selected audit for the same app build/scenario contradicts it.

For the current exact physical checkpoint, the host verifier accepted the redacted correlated package for source `95174d975da515cf8625212592721cd0baa7bfa5`, build `20260810031810`, produced by the 1/1 DEBUG scenario on an iPhone 16 Pro running iOS 26.6. The selected package bytes had SHA-256 `d15676774b3d28feef7eca63b67e06afc753cb4ef58d1d0956cd135ba46c610f`. The digest identifies the package verified during that run; it does not state that the exported package is committed, and it does not promote DEBUG evidence to Release or TestFlight proof.

To identify the latest valid audit, prefer an explicit runbook selection first. If no runbook pins a file, sort candidate `runtime-audits/` inputs by embedded export timestamp/build label when available, then by filename date, then by filesystem modification time as a last resort. Record the chosen file in the next `improve-loop` invocation with `--runtime-audit` so the evidence selection is reproducible.

A proof becomes obsolete when a newer app build is shipped, the bundled manifest changes, the runtime registry or adapter binding changes, the dataset/model artifacts being evaluated are regenerated, the relevant scenario definition changes, or a newer audit/E2E report supersedes the same surface. Obsolete evidence can still be ingested as training feedback, but it must not be cited as current live pass/fail proof.

## Command pointers

The crawler README contains concrete command examples for the two main refresh paths:

- Manifest and dataset generation: `tools/lumen_manifest_crawler/README.md` section “Run locally”.
- Closed improvement loop with runtime-audit ingestion: `tools/lumen_manifest_crawler/README.md` section “Run one closed improvement-loop cycle”.
- Runtime audit ingestion for existing reports: `tools/lumen_manifest_crawler/README.md` section “Include in-app runtime audit data”.

The same commands are implemented by the Typer CLI in `tools/lumen_manifest_crawler/lumen_manifest_crawler/cli.py` under `generate` and `improve-loop`.
