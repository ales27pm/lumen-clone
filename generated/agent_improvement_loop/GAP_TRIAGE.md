# Agent Improvement Gap Triage

- Total gaps: `46`
- Raw runtime failures: `46`
- Fresh runtime failures: `33`
- Skipped live model generation: `13`
- Classification rule: skippedLiveModelRun=true remains a gap but is not counted as a fresh runtime failure.

## Root Cause Counts

- `manifest_mismatch`: `29`
- `skipped_live_model_generation`: `13`
- `stale_or_unclassified_runtime_evidence`: `4`

## Failure Groups

### argument_mismatch

- Count: `27`
- Status: `manifest_reconciliation_required`
- Root causes: `{"manifest_mismatch": 27}`
- Categories: `{"manifest_mismatch": 27}`
- Fresh runtime failures: `27`
- Skipped live model generation: `0`
- Recommended action: Regenerate executor schema cards and add missing-argument clarification examples.

  - `manifest_mismatch` | skipped=`None` | prompt: messages.draft | actual: recipient:string:required=false
  - `manifest_mismatch` | skipped=`None` | prompt: messages.draft | actual: message:string:required=false
  - `manifest_mismatch` | skipped=`None` | prompt: messages.draft | actual: text:string:required=false

### chat

- Count: `4`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 4}`
- Categories: `{"skipped_live_model_generation": 4}`
- Fresh runtime failures: `0`
- Skipped live model generation: `4`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Explain why a sharp chisel is safer than a dull one. | actual: I hit an internal response-format issue. Please try again.
  - `skipped_live_model_generation` | skipped=`True` | prompt: Give me three tips for fitting a door hinge cleanly. | actual: I hit an internal response-format issue. Please try again.
  - `skipped_live_model_generation` | skipped=`True` | prompt: Explain actor isolation in Swift in simple terms. | actual: I hit an internal response-format issue. Please try again.

### persistent_diagnostics_scenario_not_passed

- Count: `4`
- Status: `deferred`
- Root causes: `{"stale_or_unclassified_runtime_evidence": 4}`
- Categories: `{"runtime_drift": 4}`
- Fresh runtime failures: `4`
- Skipped live model generation: `0`
- Recommended action: Fix the diagnostics scenario or app runtime path, then rerun persistent diagnostics before using the artifact.

  - `stale_or_unclassified_runtime_evidence` | skipped=`None` | prompt: liveAgentStream | actual: interrupted
  - `stale_or_unclassified_runtime_evidence` | skipped=`None` | prompt: liveAgentStream | actual: interrupted
  - `stale_or_unclassified_runtime_evidence` | skipped=`None` | prompt: agentToolPrompt | actual: skipped

### emaildraft

- Count: `2`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 2}`
- Categories: `{"skipped_live_model_generation": 2}`
- Fresh runtime failures: `0`
- Skipped live model generation: `2`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Draft a email | actual: I hit an internal response-format issue. Please try again.  what should  who should
  - `skipped_live_model_generation` | skipped=`True` | prompt: Draft an email to Alex with a professional update and ask one clarifying question. | actual: I hit an internal response-format issue. Please try again.  One clarifying question: what specific deadline, priority, or next step should I align this with?

### weather

- Count: `2`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 2}`
- Categories: `{"skipped_live_model_generation": 2}`
- Fresh runtime failures: `0`
- Skipped live model generation: `2`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: What is the weather here? | actual: Weather tool output could not be validated. Try again or provide a city.
  - `skipped_live_model_generation` | skipped=`True` | prompt: What is the weather here and should I carry an umbrella? | actual: Weather tool output could not be validated. Try again or provide a city.

### websearch

- Count: `2`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 2}`
- Categories: `{"skipped_live_model_generation": 2}`
- Fresh runtime failures: `0`
- Skipped live model generation: `2`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Search web for diy underground shelter | actual: No direct answer from web search. Try a different phrasing, or provide a URL to fetch directly.
  - `skipped_live_model_generation` | skipped=`True` | prompt: Search the web for two recent Swift concurrency best practices and summarize them. | actual: I hit an internal response-format issue. Please try again.  swift

### memory

- Count: `1`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 1}`
- Categories: `{"skipped_live_model_generation": 1}`
- Fresh runtime failures: `0`
- Skipped live model generation: `1`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Remember that I prefer concise bullet points, then tell me what you remembered. | actual: I hit an internal response-format issue. Please try again.  I remember that you prefer concise bullet points.

### missing_live_argument

- Count: `1`
- Status: `manifest_reconciliation_required`
- Root causes: `{"manifest_mismatch": 1}`
- Categories: `{"manifest_mismatch": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Regenerate executor schema cards and add missing-argument clarification examples.

  - `manifest_mismatch` | skipped=`None` | prompt: trigger.create | actual:

### rag

- Count: `1`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 1}`
- Categories: `{"skipped_live_model_generation": 1}`
- Fresh runtime failures: `0`
- Skipped live model generation: `1`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Search my files for architecture notes and summarize key modules. | actual: I hit an internal response-format issue. Please try again. Key modules: core module details were retrieved from local file snippets [1].  Key modules: core module details were retrieved from local file snippets [1].

### trigger

- Count: `1`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 1}`
- Categories: `{"skipped_live_model_generation": 1}`
- Fresh runtime failures: `0`
- Skipped live model generation: `1`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Schedule a trigger to summarize reminders tonight and confirm what will run. | actual: I hit an internal response-format issue. Please try again.  trigger

### unmanifested_live_argument

- Count: `1`
- Status: `manifest_reconciliation_required`
- Root causes: `{"manifest_mismatch": 1}`
- Categories: `{"manifest_mismatch": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Regenerate the manifest from Swift source, then add unknown-tool DPO contrast samples.

  - `manifest_mismatch` | skipped=`None` | prompt: trigger.create | actual: plus
