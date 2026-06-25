# Lumen Agent Improvement Loop Report

- Passed: `True`
- Tools: `53`
- Intents: `22`
- Model slots: `6`
- Dataset records: `34768`
- Runtime audit reports: `2`
- Runtime failures: `0`
- Raw runtime failures: `8`
- Skipped live model generation: `8`
- TestFlight status: `runtime-audit-ingested`
- TestFlight scenarios: `120`
- Gaps: `9`
- Next action prompts: `9`

## TestFlight handoff

Run `TESTFLIGHT_RUNBOOK.md` in the real TestFlight app, export the in-app dataset package JSON, then rerun this command with `--runtime-audit <exported-json>`.

## Top gaps

### WARNING — Empty dataset family: runtime_audit_repairs

- Category: `dataset_coverage`
- Recommendation: Add generators or runtime inputs that produce runtime_audit_repairs records.

### WARNING — e2e_runtime_environment_deferred

- Category: `runtime_environment_deferred`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_runtime_environment_deferred

- Category: `runtime_environment_deferred`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_runtime_environment_deferred

- Category: `runtime_environment_deferred`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_runtime_environment_deferred

- Category: `runtime_environment_deferred`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_runtime_environment_deferred

- Category: `runtime_environment_deferred`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_runtime_environment_deferred

- Category: `runtime_environment_deferred`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_runtime_environment_deferred

- Category: `runtime_environment_deferred`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_runtime_environment_deferred

- Category: `runtime_environment_deferred`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.
