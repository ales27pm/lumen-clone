# Agent Improvement Gap Triage

- Total gaps: `60`
- Raw runtime failures: `60`
- Fresh runtime failures: `6`
- Skipped live model generation: `54`
- Classification rule: skippedLiveModelRun=true remains a gap but is not counted as a fresh runtime failure.

## Root Cause Counts

- `permission_config_issue`: `6`
- `skipped_live_model_generation`: `54`

## Failure Groups

### memory

- Count: `13`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 13}`
- Categories: `{"skipped_live_model_generation": 13}`
- Fresh runtime failures: `0`
- Skipped live model generation: `13`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Tell me what style I asked you to use. | actual: Memory tool output could not be validated.
  - `skipped_live_model_generation` | skipped=`True` | prompt: What do you remember about my response style preference? | actual: Memory tool output could not be validated.
  - `skipped_live_model_generation` | skipped=`True` | prompt: Use Recall Memory, but ask for clarification if required details are missing. | actual: Memory tool output could not be validated.

### outlook

- Count: `13`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 13}`
- Categories: `{"skipped_live_model_generation": 13}`
- Fresh runtime failures: `0`
- Skipped live model generation: `13`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Show attachments on the latest Outlook email. | actual: Outlook tool output could not be validated.
  - `skipped_live_model_generation` | skipped=`True` | prompt: List attachments for my latest Outlook email. | actual: Outlook tool output could not be validated.
  - `skipped_live_model_generation` | skipped=`True` | prompt: Use List Outlook Attachments, but ask for clarification if required details are missing. | actual: Outlook tool output could not be validated.

### maps

- Count: `10`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 10}`
- Categories: `{"skipped_live_model_generation": 10}`
- Fresh runtime failures: `0`
- Skipped live model generation: `10`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Give me directions to the nearest hardware store. | actual: I couldn’t safely complete the maps/location request.
  - `skipped_live_model_generation` | skipped=`True` | prompt: Give me directions to the nearest hardware store. | actual: I couldn’t safely complete the maps/location request.
  - `skipped_live_model_generation` | skipped=`True` | prompt: Find coffee near me. | actual: I couldn’t safely complete the maps/location request.

### calendar

- Count: `6`
- Status: `code_or_configuration_fix_required`
- Root causes: `{"permission_config_issue": 6}`
- Categories: `{"runtime_permission_config": 6}`
- Fresh runtime failures: `6`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `permission_config_issue` | skipped=`False` | prompt: What's on my schedule today? | actual: Calendar event tools are unavailable in this build right now.
  - `permission_config_issue` | skipped=`False` | prompt: Show my next calendar events. | actual: Calendar event tools are unavailable in this build right now.
  - `permission_config_issue` | skipped=`False` | prompt: Use List Events, but ask for clarification if required details are missing. | actual: Calendar event tools are unavailable in this build right now.

### messagedraft

- Count: `6`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 6}`
- Categories: `{"skipped_live_model_generation": 6}`
- Fresh runtime failures: `0`
- Skipped live model generation: `6`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Text 5551234567 that I am late. | actual: What should the message say?
  - `skipped_live_model_generation` | skipped=`True` | prompt: Text 5551234567 that approval boundary works. | actual: What should the message say?
  - `skipped_live_model_generation` | skipped=`True` | prompt: Draft a message to 5551234567 saying I am running late. | actual: Who should I message, and what should it say?

### motion

- Count: `6`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 6}`
- Categories: `{"skipped_live_model_generation": 6}`
- Fresh runtime failures: `0`
- Skipped live model generation: `6`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Am I walking or stationary right now? | actual: I couldn’t safely complete the motion/activity request.
  - `skipped_live_model_generation` | skipped=`True` | prompt: What motion activity is detected right now? | actual: I couldn’t safely complete the motion/activity request.
  - `skipped_live_model_generation` | skipped=`True` | prompt: Use Motion Activity, but ask for clarification if required details are missing. | actual: I couldn’t safely complete the motion/activity request.

### chat

- Count: `1`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 1}`
- Categories: `{"skipped_live_model_generation": 1}`
- Fresh runtime failures: `0`
- Skipped live model generation: `1`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Explain tradeoffs between precision and recall in retrieval systems in plain English. | actual: Precision is about how many returned results are actually relevant. Recall is about how many of all relevant results the system managed to find. Higher precision avoids clutter; higher recall avoids missing useful matches.

### emaildraft

- Count: `1`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 1}`
- Categories: `{"skipped_live_model_generation": 1}`
- Fresh runtime failures: `0`
- Skipped live model generation: `1`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Draft an email to Alex with a professional update and ask one clarifying question. | actual: Approval required for mail.draft. I can prepare the email draft after you approve it. One clarifying question: should the update emphasize timeline, blockers, or next steps?

### rag

- Count: `1`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 1}`
- Categories: `{"skipped_live_model_generation": 1}`
- Fresh runtime failures: `0`
- Skipped live model generation: `1`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Search my files for architecture notes and summarize key modules. | actual: {"intent":"rag","module":"photos","module(s)":["Photos"],"nextModel":"rag","reasoningSummary":"RAG search results: [1] Photos · Photos 2026-06 Photos (2026-06): 98 items between Jun 3, 2026 and Jun 18, 2026. 0 favorites, 0 videos, 80 screen

### trigger

- Count: `1`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 1}`
- Categories: `{"skipped_live_model_generation": 1}`
- Fresh runtime failures: `0`
- Skipped live model generation: `1`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Schedule a trigger to summarize reminders tonight and confirm what will run. | actual: Approval required for trigger.create. Trigger request prepared for: Schedule a trigger to summarize reminders tonight and confirm what will run.. It will run the scheduled agent prompt after approval.

### weather

- Count: `1`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 1}`
- Categories: `{"skipped_live_model_generation": 1}`
- Fresh runtime failures: `0`
- Skipped live model generation: `1`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: What is the weather here and should I carry an umbrella? | actual: Weather update: Weather at your location: · overcast · 18°C · feels like 18°C · humidity 90% · wind 19 km/h · gusts 40 km/h · cloud cover 100% · updated 2026-06-18T20:45

### websearch

- Count: `1`
- Status: `needs_fresh_runtime_evidence`
- Root causes: `{"skipped_live_model_generation": 1}`
- Categories: `{"skipped_live_model_generation": 1}`
- Fresh runtime failures: `0`
- Skipped live model generation: `1`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `skipped_live_model_generation` | skipped=`True` | prompt: Search the web for two recent Swift concurrency best practices and summarize them. | actual: Search results for: Search the web for two recent Swift concurrency best practices and summarize them.  1. Swift Concurrency Architecture: Patterns and Best Practices https://www.atelier-socle.com/en/articles/swift-concurrency-architecture
