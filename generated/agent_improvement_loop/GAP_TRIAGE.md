# Agent Improvement Gap Triage

- Total gaps: `9`
- Raw runtime failures: `8`
- Fresh runtime failures: `0`
- Skipped live model generation: `8`
- Classification rule: skippedLiveModelRun=true remains a gap but is not counted as a fresh runtime failure.

## Root Cause Counts

- `runtime_environment_deferred`: `8`
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

### chat

- Count: `1`
- Status: `deferred`
- Root causes: `{"runtime_environment_deferred": 1}`
- Categories: `{"runtime_environment_deferred": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `runtime_environment_deferred` | skipped=`True` | prompt: Explain tradeoffs between precision and recall in retrieval systems in plain English. | actual: I couldn't complete the structured agent turn because agent-json produced no JSON output. Reason: cancelledBeforeFirstToken.

### emailDraft

- Count: `1`
- Status: `deferred`
- Root causes: `{"runtime_environment_deferred": 1}`
- Categories: `{"runtime_environment_deferred": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `runtime_environment_deferred` | skipped=`True` | prompt: Draft an email to Alex with a professional update and ask one clarifying question. | actual: I couldn't complete the structured agent turn because agent-json produced no JSON output. Reason: cancelledBeforeFirstToken.

### memory

- Count: `1`
- Status: `deferred`
- Root causes: `{"runtime_environment_deferred": 1}`
- Categories: `{"runtime_environment_deferred": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `runtime_environment_deferred` | skipped=`True` | prompt: Remember that I prefer concise bullet points, then tell me what you remembered. | actual: Memory tool output could not be validated.

### preflight

- Count: `1`
- Status: `deferred`
- Root causes: `{"runtime_environment_deferred": 1}`
- Categories: `{"runtime_environment_deferred": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `runtime_environment_deferred` | skipped=`True` | prompt: What is the weather here and should I carry an umbrella? | actual:

### rag

- Count: `1`
- Status: `deferred`
- Root causes: `{"runtime_environment_deferred": 1}`
- Categories: `{"runtime_environment_deferred": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `runtime_environment_deferred` | skipped=`True` | prompt: Search my files for architecture notes and summarize key modules. | actual: I couldn't complete the structured agent turn because agent-json produced no JSON output. Reason: cancelledBeforeFirstToken.

### trigger

- Count: `1`
- Status: `deferred`
- Root causes: `{"runtime_environment_deferred": 1}`
- Categories: `{"runtime_environment_deferred": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `runtime_environment_deferred` | skipped=`True` | prompt: Schedule a trigger to summarize reminders tonight and confirm what will run. | actual: I couldn't complete the structured agent turn because agent-json produced no JSON output. Reason: cancelledBeforeFirstToken.

### weather

- Count: `1`
- Status: `deferred`
- Root causes: `{"runtime_environment_deferred": 1}`
- Categories: `{"runtime_environment_deferred": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `runtime_environment_deferred` | skipped=`True` | prompt: What is the weather here and should I carry an umbrella? | actual: Weather tool output could not be validated. Try again or provide a city.

### webSearch

- Count: `1`
- Status: `deferred`
- Root causes: `{"runtime_environment_deferred": 1}`
- Categories: `{"runtime_environment_deferred": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `runtime_environment_deferred` | skipped=`True` | prompt: Search the web for two recent Swift concurrency best practices and summarize them. | actual: No direct answer from web search. Try a different phrasing, or provide a URL to fetch directly.
