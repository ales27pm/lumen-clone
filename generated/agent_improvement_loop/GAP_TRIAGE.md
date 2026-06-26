# Agent Improvement Gap Triage

- Total gaps: `2`
- Raw runtime failures: `0`
- Fresh runtime failures: `0`
- Skipped live model generation: `0`
- Classification rule: skippedLiveModelRun=true remains a gap but is not counted as a fresh runtime failure.

## Root Cause Counts

- `stale_audit_evidence`: `1`
- `stale_or_unclassified_runtime_evidence`: `1`

## Failure Groups

### Empty dataset family: runtime_audit_repairs

- Count: `1`
- Status: `deferred`
- Root causes: `{"stale_or_unclassified_runtime_evidence": 1}`
- Categories: `{"dataset_coverage": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Add generators or runtime inputs that produce runtime_audit_repairs records.

  - `stale_or_unclassified_runtime_evidence` | skipped=`None` | prompt:  | actual:

### TestFlight in-app audit export has not been ingested yet

- Count: `1`
- Status: `deferred`
- Root causes: `{"stale_audit_evidence": 1}`
- Categories: `{"testflight_runtime_pending": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Compile/distribute the TestFlight build, run Agent Grounding in the app, export the in-app dataset package JSON, then rerun improve-loop with --runtime-audit <json>.

  - `stale_audit_evidence` | skipped=`None` | prompt:  | actual:
