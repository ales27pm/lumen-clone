# Agent Improvement Gap Triage

- Total gaps: `2`
- Raw runtime failures: `1`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Classification rule: skippedLiveModelRun=true remains a gap but is not counted as a fresh runtime failure.

## Root Cause Counts

- `no_correlated_model_turn`: `1`
- `stale_audit_evidence`: `1`

## Failure Groups

### Runtime audit export does not prove the current TestFlight build

- Count: `1`
- Status: `deferred`
- Root causes: `{"stale_audit_evidence": 1}`
- Categories: `{"testflight_runtime_build_mismatch": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Install build 20260629060414, run Agent Grounding in that TestFlight app, export the TestFlight + Agent Grounding package JSON, and ingest only that current-build package.

  - `stale_audit_evidence` | skipped=`None` | prompt:  | actual:

### preflight

- Count: `1`
- Status: `deferred`
- Root causes: `{"no_correlated_model_turn": 1}`
- Categories: `{"no_correlated_model_turn": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `no_correlated_model_turn` | skipped=`False` | prompt: What is the weather here and should I carry an umbrella? | actual:
