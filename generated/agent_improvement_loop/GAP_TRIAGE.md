# Agent Improvement Gap Triage

- Total gaps: `1`
- Raw runtime failures: `7`
- Fresh runtime failures: `7`
- Skipped live model generation: `0`
- Classification rule: skippedLiveModelRun=true remains a gap but is not counted as a fresh runtime failure.

## Root Cause Counts

- `stale_or_unclassified_runtime_evidence`: `1`

## Failure Groups

### Empty dataset family: runtime_audit_repairs

- Count: `1`
- Status: `deferred`
- Root causes: `{"stale_or_unclassified_runtime_evidence": 1}`
- Categories: `{"dataset_coverage": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Add current-build runtime inputs that produce runtime_audit_repairs records.

  - `stale_or_unclassified_runtime_evidence` | skipped=`None` | prompt:  | actual:
