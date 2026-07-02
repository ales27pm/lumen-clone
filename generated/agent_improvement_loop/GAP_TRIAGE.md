# Agent Improvement Gap Triage

- Total gaps: `164`
- Raw runtime failures: `164`
- Fresh runtime failures: `160`
- Skipped live model generation: `4`
- Classification rule: skippedLiveModelRun=true remains a gap but is not counted as a fresh runtime failure.

## Root Cause Counts

- `agent_json_parse_error`: `1`
- `deterministic_compatibility_not_live_evidence`: `156`
- `runtime_environment_deferred`: `4`
- `stale_or_unclassified_runtime_evidence`: `3`

## Failure Groups

### outlook

- Count: `48`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 48}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 48}`
- Fresh runtime failures: `48`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Show attachments on the latest Outlook email. | actual: Outlook attachments: No attachments found for message AQMkADAwATZiZmYAZC1mN2E5LTMwOWUtMDACLTAwCgBGAAADrd0FawXuTEmw3Fyok-zgwwcA7r1XmTxuJkam7HO44tJDmgAAAgEMAAAA7r1XmTxuJkam7HO44tJDmgAIHZqYiwAAAA==.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: List attachments for my latest Outlook email. | actual: Outlook attachments: No attachments found for message AQMkADAwATZiZmYAZC1mN2E5LTMwOWUtMDACLTAwCgBGAAADrd0FawXuTEmw3Fyok-zgwwcA7r1XmTxuJkam7HO44tJDmgAAAgEMAAAA7r1XmTxuJkam7HO44tJDmgAIHZqYiwAAAA==.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use List Outlook Attachments, but ask for clarification if required details are missing. | actual: Outlook attachments: No attachments found for message AQMkADAwATZiZmYAZC1mN2E5LTMwOWUtMDACLTAwCgBGAAADrd0FawXuTEmw3Fyok-zgwwcA7r1XmTxuJkam7HO44tJDmgAAAgEMAAAA7r1XmTxuJkam7HO44tJDmgAIHZqYiwAAAA==.

### alarm

- Count: `30`
- Status: `deferred`
- Root causes: `{"agent_json_parse_error": 1, "deterministic_compatibility_not_live_evidence": 29}`
- Categories: `{"agent_json_parse_error": 1, "deterministic_compatibility_not_live_evidence": 29}`
- Fresh runtime failures: `30`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `agent_json_parse_error` | skipped=`False` | prompt: Set an alarm for tomorrow at 7. | actual: I couldn’t safely complete the alarm/timer request.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Show alarm permission status. | actual: Alarm authorization status: Alarm authorization status: authorized.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Check alarm authorization status. | actual: Alarm authorization status: Alarm authorization status: authorized.

### rag

- Count: `10`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 6, "runtime_environment_deferred": 3, "stale_or_unclassified_runtime_evidence": 1}`
- Categories: `{"architecture_finalizer_failure": 1, "deterministic_compatibility_not_live_evidence": 6, "runtime_environment_deferred": 3}`
- Fresh runtime failures: `10`
- Skipped live model generation: `0`
- Recommended action: Quarantine this architecture/runtime/finalizer failure from SFT; add a deterministic regression test or runtime diagnostic instead.

  - `stale_or_unclassified_runtime_evidence` | skipped=`False` | prompt: Search my files for architecture notes and summarize key modules. | actual: I'm ready. Please ask again or tell me what you'd like to do next. Key modules: core module details were retrieved from local file snippets [1].
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Refresh the file retrieval index. | actual: Local file index updated: RAG indexing failed: no chunks were indexed. Check embedding model readiness and imported file contents.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Reindex my imported files. | actual: Local file index updated: RAG indexing failed: no chunks were indexed. Check embedding model readiness and imported file contents.

### maps

- Count: `9`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 9}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 9}`
- Fresh runtime failures: `9`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Where am I right now? | actual: Current location: Current location: 46.0061, -73.1658 (±39m)
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Where are we? | actual: Current location: Current location: 46.0061, -73.1653 (±20m)
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Current Location, but ask for clarification if required details are missing. | actual: Current location: Current location: 46.0061, -73.1653 (±20m)

### trigger

- Count: `9`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 9}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 9}`
- Fresh runtime failures: `9`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Cancel trigger named nightly summary. | actual: Approval required for trigger.cancel. I did not cancel nightly summary yet.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Cancel trigger named nightly summary. | actual: Approval required for trigger.cancel. I did not cancel nightly summary yet.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Cancel the trigger called nightly summary. | actual: Approval required for trigger.cancel. I did not cancel nightly summary yet.

### calendar

- Count: `6`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 6}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 6}`
- Fresh runtime failures: `6`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Set an appointment for tomorrow morning at nine in my calendar. | actual: Approval required for calendar.create. I did not create an event yet.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Create a calendar event called Approval Test in 10 minutes. | actual: Approval required for calendar.create. I did not create an event yet.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Create a calendar event called Test tomorrow in 30 minutes. | actual: Approval required for calendar.create. I did not create an event yet.

### memory

- Count: `6`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 6}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 6}`
- Fresh runtime failures: `6`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Tell me what style I asked you to use. | actual: Memory recall: • I prefer concise bullet points
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: What do you remember about my response style preference? | actual: Memory recall: • I prefer concise bullet points
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Recall Memory, but ask for clarification if required details are missing. | actual: Memory recall: • I prefer concise bullet points

### reminder

- Count: `6`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 6}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 6}`
- Fresh runtime failures: `6`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Create a reminder to review logs. | actual: Approval required for reminders.create. I did not create a reminder yet.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Remind me to check approval boundary. | actual: Approval required for reminders.create. I did not create a reminder yet.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Remind me to check Lumen logs. | actual: Approval required for reminders.create. I did not create a reminder yet.

### websearch

- Count: `6`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 6}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 6}`
- Fresh runtime failures: `6`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Read this URL: https://example.com. | actual: Fetched page summary: Example Domain https://example.com  Example Domain This domain is for use in documentation examples without needing permission. Avoid use in operations. Learn more
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Fetch https://example.com and summarize the page. | actual: Example Domain https://example.com  Example Domain This domain is for use in documentation examples without needing permission. Avoid use in operations. Learn more
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Fetch the page at https://example.com. | actual: Fetched page summary: Example Domain https://example.com  Example Domain This domain is for use in documentation examples without needing permission. Avoid use in operations. Learn more

### emaildraft

- Count: `4`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 4}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 4}`
- Fresh runtime failures: `4`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Draft a email | actual: Who should I send it to, and what should it say?
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Write a quick email to alex@example.com saying I will send it tonight. | actual: Approval required for mail.draft. I can prepare the email draft after you approve it. One clarifying question: should the update emphasize timeline, blockers, or next steps?
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Draft an email to alex@example.com saying approval boundary works. | actual: Approval required for mail.draft. I can prepare the email draft after you approve it. One clarifying question: should the update emphasize timeline, blockers, or next steps?

### camera

- Count: `3`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 3}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 3}`
- Fresh runtime failures: `3`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Take a photo now. | actual: Approval required for camera.capture. I did not open the camera yet.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Take a picture now. | actual: Approval required for camera.capture. I did not open the camera yet.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Open the camera to take a picture. | actual: Approval required for camera.capture. I did not open the camera yet.

### contactsearch

- Count: `3`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 3}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 3}`
- Fresh runtime failures: `3`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Find Alex in my contacts. | actual: Contact search results: No contacts match "my contacts.".
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Search my contacts for Alex. | actual: Contact search results: • Alexandre Tremblay — (514) 779-9054 • Alexis Boulet — 14504943059 • Alexis Boulet — no phone • Alexandre Thivierge — 1 (418) 559-7084 • Alexandra Boulet — +14505614086
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Search Contacts, but ask for clarification if required details are missing. | actual: Contact search results: No contacts match "clarification if required details are missing.".

### files

- Count: `3`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 3}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 3}`
- Fresh runtime failures: `3`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Open local document diagnostics.txt. | actual: File result: File not found. Available:
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Read document diagnostics.txt from imported files. | actual: File result: File not found. Available:
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Read File, but ask for clarification if required details are missing. | actual: File result: No imported files. Tap the paperclip to add one.

### health

- Count: `3`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 3}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 3}`
- Fresh runtime failures: `3`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Read my steps and sleep summary. | actual: Health summary: Today's Health: 10,348 steps · 6.37 km
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Show my health summary. | actual: Health summary: Today's Health: 10,348 steps · 6.37 km
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Health Summary, but ask for clarification if required details are missing. | actual: Health summary: Today's Health: 10,348 steps · 6.37 km

### messagedraft

- Count: `3`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 3}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 3}`
- Fresh runtime failures: `3`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Text 5551234567 that I am late. | actual: Approval required for messages.draft. I can prepare the message after you approve it. What tone should I use?
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Text 5551234567 that approval boundary works. | actual: Approval required for messages.draft. I can prepare the message after you approve it. What tone should I use?
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Draft a message to 5551234567 saying I am running late. | actual: Approval required for messages.draft. I can prepare the message after you approve it. What tone should I use?

### motion

- Count: `3`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 3}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 3}`
- Fresh runtime failures: `3`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Am I walking or stationary right now? | actual: Motion activity: Today's motion — 10348 steps · 6.37 km · 7 floors · activity: 241m driving, 197m stationary, 164m walking, 18m cycling
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: What motion activity is detected right now? | actual: Motion activity: Today's motion — 10348 steps · 6.37 km · 7 floors · activity: 241m driving, 197m stationary, 164m walking, 18m cycling
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Motion Activity, but ask for clarification if required details are missing. | actual: Motion activity: Today's motion — 10348 steps · 6.37 km · 7 floors · activity: 241m driving, 197m stationary, 164m walking, 18m cycling

### phonecall

- Count: `3`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 3}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 3}`
- Fresh runtime failures: `3`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Dial 5551234567. | actual: Approval required for phone.call. Contact found; I can call 5551234567 after you approve it. I did not place the call yet.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Call 5551234567 now. | actual: Approval required for phone.call. Contact found; I can call 5551234567 after you approve it. I did not place the call yet.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Call 5551234567. | actual: Approval required for phone.call. Contact found; I can call 5551234567 after you approve it. I did not place the call yet.

### photos

- Count: `3`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 3}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 3}`
- Fresh runtime failures: `3`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Find receipt pictures in my photo library. | actual: Photo search results: Found 500 photos matching "my photo library.". Recent dates: Jul 1, 2026 at 10:06 PM, Jul 1, 2026 at 9:12 PM, Jul 1, 2026 at 9:06 PM, Jul 1, 2026 at 8:35 PM, Jul 1, 2026 at 8:35 PM. {"matches":[{"creationDate":"2026-07
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Search my photos for receipts. | actual: Photo search results: Found 500 photos matching "receipts.". Recent dates: Jul 1, 2026 at 10:06 PM, Jul 1, 2026 at 9:12 PM, Jul 1, 2026 at 9:06 PM, Jul 1, 2026 at 8:35 PM, Jul 1, 2026 at 8:35 PM. {"matches":[{"creationDate":"2026-07-02T02:0
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Search Photos, but ask for clarification if required details are missing. | actual: Photo search results: Found 500 photos matching "clarification if required details are missing.". Recent dates: Jul 1, 2026 at 10:06 PM, Jul 1, 2026 at 9:12 PM, Jul 1, 2026 at 9:06 PM, Jul 1, 2026 at 8:35 PM, Jul 1, 2026 at 8:35 PM. {"match

### weather

- Count: `3`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 3}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 3}`
- Fresh runtime failures: `3`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Will it rain here today? | actual: Weather update: Weather at your location: · overcast · 24°C · feels like 26°C · humidity 78% · wind 11 km/h · gusts 29 km/h · cloud cover 100% · updated 2026-07-01T22:15
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: What is the weather here right now? | actual: Weather update: Weather at your location: · overcast · 24°C · feels like 26°C · humidity 78% · wind 11 km/h · gusts 29 km/h · cloud cover 100% · updated 2026-07-01T22:15
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Current Weather, but ask for clarification if required details are missing. | actual: Weather update: Couldn't find weather coordinates for "clarification if required details are missing.". Try a more specific city, region, or address.

### webSearch

- Count: `2`
- Status: `deferred`
- Root causes: `{"stale_or_unclassified_runtime_evidence": 2}`
- Categories: `{"architecture_finalizer_failure": 2}`
- Fresh runtime failures: `2`
- Skipped live model generation: `0`
- Recommended action: Quarantine this architecture/runtime/finalizer failure from SFT; add a deterministic regression test or runtime diagnostic instead.

  - `stale_or_unclassified_runtime_evidence` | skipped=`False` | prompt: Search the web for two recent Swift concurrency best practices and summarize them. | actual: No direct answer from web search. Try a different phrasing, or provide a URL to fetch directly.
  - `stale_or_unclassified_runtime_evidence` | skipped=`False` | prompt: Search web for diy underground shelter | actual: Check out Battlbox.com's guide on building an underground shelter: https://www.battlbox.com/blogs/outdoors/how-to-build-an-underground-shelter-a-comprehensive-guide

### preflight

- Count: `1`
- Status: `deferred`
- Root causes: `{"runtime_environment_deferred": 1}`
- Categories: `{"runtime_environment_deferred": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `runtime_environment_deferred` | skipped=`True` | prompt: Schedule a trigger to summarize reminders tonight and confirm what will run. | actual: device thermal state serious; cool device and retry
