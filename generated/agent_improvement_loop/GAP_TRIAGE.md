# Agent Improvement Gap Triage

- Total gaps: `1`
- Raw runtime failures: `0`
- Fresh runtime failures: `0`
- Skipped live model generation: `0`
- Classification rule: skippedLiveModelRun=true remains a gap but is not counted as a fresh runtime failure.

## Root Cause Counts

- `stale_audit_evidence`: `1`

## Failure Groups

### Runtime audit export does not prove the current TestFlight build

- Count: `1`
- Status: `deferred`
- Root causes: `{"stale_audit_evidence": 1}`
- Categories: `{"testflight_runtime_build_mismatch": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Install build 20260629064751, run Agent Grounding in that TestFlight app, export the TestFlight + Agent Grounding package JSON, and ingest only that current-build package.

  - `stale_audit_evidence` | skipped=`None` | prompt:  | actual:
