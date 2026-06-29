# Agent Improvement Gap Triage

- Total gaps: `1`
- Raw runtime failures: `0`
- Fresh runtime failures: `0`
- Skipped live model generation: `0`
- Classification rule: skippedLiveModelRun=true remains a gap but is not counted as a fresh runtime failure.

## Root Cause Counts

- `stale_audit_evidence`: `1`

## Failure Groups

### TestFlight in-app audit export has not been ingested yet

- Count: `1`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"stale_audit_evidence": 1}`
- Categories: `{"testflight_runtime_pending": 1}`
- Fresh runtime failures: `0`
- Skipped live model generation: `0`
- Recommended action: Compile/distribute the TestFlight build, run Agent Grounding in the app, export the TestFlight + Agent Grounding package JSON, then rerun improve-loop with --runtime-audit <json>.

  - `stale_audit_evidence` | skipped=`None` | prompt:  | actual:
