# Lumen Agent Improvement Loop Report

- Passed: `False`
- Tools: `53`
- Intents: `22`
- Model slots: `6`
- Dataset records: `34452`
- Runtime audit reports: `5`
- Runtime failures: `6`
- Raw runtime failures: `60`
- Skipped live model generation: `54`
- TestFlight status: `runtime-audit-ingested`
- TestFlight scenarios: `120`
- Gaps: `60`
- Next action prompts: `60`

## TestFlight handoff

Run `TESTFLIGHT_RUNBOOK.md` in the real TestFlight app, export the in-app dataset package JSON, then rerun this command with `--runtime-audit <exported-json>`.

## Top gaps

### ERROR — e2e_response_quality_calendar

- Category: `runtime_permission_config`
- Recommendation: Convert this failure into a REM repair sample and add a regression eval.

### ERROR — e2e_response_quality_calendar

- Category: `runtime_permission_config`
- Recommendation: Convert this failure into a REM repair sample and add a regression eval.

### ERROR — e2e_response_quality_calendar

- Category: `runtime_permission_config`
- Recommendation: Convert this failure into a REM repair sample and add a regression eval.

### ERROR — e2e_response_quality_calendar

- Category: `runtime_permission_config`
- Recommendation: Convert this failure into a REM repair sample and add a regression eval.

### ERROR — e2e_response_quality_calendar

- Category: `runtime_permission_config`
- Recommendation: Convert this failure into a REM repair sample and add a regression eval.

### ERROR — e2e_response_quality_calendar

- Category: `runtime_permission_config`
- Recommendation: Convert this failure into a REM repair sample and add a regression eval.

### WARNING — e2e_response_quality_chat

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_emaildraft

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_maps

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_maps

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_maps

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_maps

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_maps

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_maps

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_maps

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_maps

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_maps

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_maps

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_memory

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_memory

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_memory

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_memory

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_memory

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_memory

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_memory

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_memory

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_memory

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_memory

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_memory

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

### WARNING — e2e_response_quality_memory

- Category: `skipped_live_model_generation`
- Recommendation: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.
