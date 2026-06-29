# Agent Improvement Gap Triage

- Total gaps: `167`
- Raw runtime failures: `166`
- Fresh runtime failures: `165`
- Skipped live model generation: `1`
- Classification rule: skippedLiveModelRun=true remains a gap but is not counted as a fresh runtime failure.

## Root Cause Counts

- `deterministic_compatibility_not_live_evidence`: `165`
- `runtime_environment_deferred`: `1`
- `stale_audit_evidence`: `1`

## Failure Groups

### outlook

- Count: `48`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 48}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 48}`
- Fresh runtime failures: `48`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Show attachments on the latest Outlook email. | actual: Outlook attachments: No attachments found for message AQMkADAwATZiZmYAZC1mN2E5LTMwOWUtMDACLTAwCgBGAAADrd0FawXuTEmw3Fyok-zgwwcA7r1XmTxuJkam7HO44tJDmgAAAgEMAAAA7r1XmTxuJkam7HO44tJDmgAIG7kimgAAAA==.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: List attachments for my latest Outlook email. | actual: Outlook attachments: No attachments found for message AQMkADAwATZiZmYAZC1mN2E5LTMwOWUtMDACLTAwCgBGAAADrd0FawXuTEmw3Fyok-zgwwcA7r1XmTxuJkam7HO44tJDmgAAAgEMAAAA7r1XmTxuJkam7HO44tJDmgAIG7kimgAAAA==.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use List Outlook Attachments, but ask for clarification if required details are missing. | actual: Outlook attachments: No attachments found for message AQMkADAwATZiZmYAZC1mN2E5LTMwOWUtMDACLTAwCgBGAAADrd0FawXuTEmw3Fyok-zgwwcA7r1XmTxuJkam7HO44tJDmgAAAgEMAAAA7r1XmTxuJkam7HO44tJDmgAIG7kimgAAAA==.

### alarm

- Count: `30`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 30}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 30}`
- Fresh runtime failures: `30`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Show alarm permission status. | actual: I couldn’t safely complete the alarm/timer request.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Check alarm authorization status. | actual: I couldn’t safely complete the alarm/timer request.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Alarm Auth Status, but ask for clarification if required details are missing. | actual: I couldn’t safely complete the alarm/timer request.

### maps

- Count: `9`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 9}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 9}`
- Fresh runtime failures: `9`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Where am I right now? | actual: Current location: Current location: 46.0062, -73.1650 (±10m)
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Where are we? | actual: Current location: Current location: 46.0062, -73.1650 (±10m)
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Current Location, but ask for clarification if required details are missing. | actual: Current location: Current location: 46.0062, -73.1650 (±10m)

### rag

- Count: `9`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 9}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 9}`
- Fresh runtime failures: `9`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Refresh the file retrieval index. | actual: Local file index updated: Indexed 23 chunks from imported files.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Reindex my imported files. | actual: Local file index updated: RAG indexing failed: no chunks were indexed. Check embedding model readiness and imported file contents.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Reindex Files, but ask for clarification if required details are missing. | actual: Local file index updated: RAG indexing failed: no chunks were indexed. Check embedding model readiness and imported file contents.

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

### websearch

- Count: `7`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 7}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 7}`
- Fresh runtime failures: `7`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Search web for diy underground shelter | actual: Web search results: Search results for: Search web for diy underground shelter 1. 10 Steps to Build a DIY Underground Bushcraft Survival Shelter https://www.bushcraftbasecamp.com/10-steps-to-build-a-diy-underground-bushcraft-survival-shelte
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Read this URL: https://example.com. | actual: Fetched page summary: Example Domain https://example.com  Example Domain This domain is for use in documentation examples without needing permission. Avoid use in operations. Learn more
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Fetch https://example.com and summarize the page. | actual: Example Domain https://example.com  Example Domain This domain is for use in documentation examples without needing permission. Avoid use in operations. Learn more

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

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Tell me what style I asked you to use. | actual: Memory recall: • I prefer concise bullet points • Use Save Memory, but ask for clarification if required details are missing • Keep in mind that I like short answers • User asked: Show. Assistant: Le résultat de la géolocalisation est : 45.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: What do you remember about my response style preference? | actual: Memory recall: • Keep in mind that I like short answers • I prefer concise bullet points • Use Save Memory, but ask for clarification if required details are missing • User asked: Show. Assistant: Le résultat de la géolocalisation est : 45.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Recall Memory, but ask for clarification if required details are missing. | actual: Memory recall: • Use Save Memory, but ask for clarification if required details are missing • I prefer concise bullet points • Keep in mind that I like short answers • User asked: Show itinerary on map. Assistant: Le résultat de la géolocal

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

### weather

- Count: `4`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 4}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 4}`
- Fresh runtime failures: `4`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: What is the weather here? | actual: Weather update: Weather at your location: · clear sky · 17°C · feels like 18°C · humidity 86% · wind 5 km/h · gusts 12 km/h · cloud cover 5% · updated 2026-06-28T23:00
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Will it rain here today? | actual: Weather update: Weather at your location: · clear sky · 17°C · feels like 18°C · humidity 86% · wind 5 km/h · gusts 12 km/h · cloud cover 5% · updated 2026-06-28T23:00
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: What is the weather here right now? | actual: Weather update: Weather at your location: · clear sky · 17°C · feels like 18°C · humidity 86% · wind 5 km/h · gusts 12 km/h · cloud cover 5% · updated 2026-06-28T23:00

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

### chat

- Count: `3`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 3}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 3}`
- Fresh runtime failures: `3`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Explain why a sharp chisel is safer than a dull one. | actual: A sharp chisel is safer because it needs less force, follows the cut more predictably, and is less likely to slip. A dull edge makes you push harder, which reduces control.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Give me three tips for fitting a door hinge cleanly. | actual: Three hinge-fitting tips: mark the leaf with a sharp knife, pare to the line in thin passes, and test-fit often so the hinge sits flush without deep gaps.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Explain actor isolation in Swift in simple terms. | actual: Actor isolation means Swift protects data owned by an actor so only that actor can touch it directly. Other code has to await access, which helps prevent races.

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

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Open local document diagnostics.txt. | actual: File result: File not found. Available: Dogme_et_rituel_de_la_[...]Constant_Alphonse_bpt6k205564w.pdf, Dogme_et_rituel_de_la_[...]Constant_Alphonse_bpt6k2055658.pdf
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Read document diagnostics.txt from imported files. | actual: File result: File not found. Available: Dogme_et_rituel_de_la_[...]Constant_Alphonse_bpt6k205564w.pdf, Dogme_et_rituel_de_la_[...]Constant_Alphonse_bpt6k2055658.pdf
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Read File, but ask for clarification if required details are missing. | actual: File result: Imported files: • Dogme_et_rituel_de_la_[...]Constant_Alphonse_bpt6k205564w.pdf • Dogme_et_rituel_de_la_[...]Constant_Alphonse_bpt6k2055658.pdf

### health

- Count: `3`
- Status: `deferred`
- Root causes: `{"deterministic_compatibility_not_live_evidence": 3}`
- Categories: `{"deterministic_compatibility_not_live_evidence": 3}`
- Fresh runtime failures: `3`
- Skipped live model generation: `0`
- Recommended action: Convert this failure into a REM repair sample and add a regression eval.

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Read my steps and sleep summary. | actual: Health summary: Today's Health: 2,485 steps · 1.72 km
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Show my health summary. | actual: Health summary: Today's Health: 2,485 steps · 1.72 km
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Health Summary, but ask for clarification if required details are missing. | actual: Health summary: Today's Health: 2,485 steps · 1.72 km

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

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Am I walking or stationary right now? | actual: Motion activity: Today's motion — 2485 steps · 1.72 km · 1 floors · activity: 698m stationary, 190m driving, 20m walking, 2m cycling
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: What motion activity is detected right now? | actual: Motion activity: Today's motion — 2485 steps · 1.72 km · 1 floors · activity: 698m stationary, 190m driving, 20m walking, 2m cycling
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Motion Activity, but ask for clarification if required details are missing. | actual: Motion activity: Today's motion — 2485 steps · 1.72 km · 1 floors · activity: 698m stationary, 190m driving, 20m walking, 2m cycling

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

  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Find receipt pictures in my photo library. | actual: Photo search results: Found 500 photos matching "my photo library.". Recent dates: Jun 28, 2026 at 2:50 AM, Jun 28, 2026 at 12:14 AM, Jun 28, 2026 at 12:14 AM, Jun 28, 2026 at 12:14 AM, Jun 28, 2026 at 12:14 AM.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Search my photos for receipts. | actual: Photo search results: Found 500 photos matching "receipts.". Recent dates: Jun 28, 2026 at 2:50 AM, Jun 28, 2026 at 12:14 AM, Jun 28, 2026 at 12:14 AM, Jun 28, 2026 at 12:14 AM, Jun 28, 2026 at 12:14 AM.
  - `deterministic_compatibility_not_live_evidence` | skipped=`False` | prompt: Use Search Photos, but ask for clarification if required details are missing. | actual: Photo search results: Found 500 photos matching "clarification if required details are missing.". Recent dates: Jun 28, 2026 at 2:50 AM, Jun 28, 2026 at 12:14 AM, Jun 28, 2026 at 12:14 AM, Jun 28, 2026 at 12:14 AM, Jun 28, 2026 at 12:14 AM.

### Runtime audit export does not prove the current TestFlight build

- Count: `1`
- Status: `deferred`
- Root causes: `{"stale_audit_evidence": 1}`
- Categories: `{"testflight_runtime_build_mismatch": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Install build 20260629054657, run Agent Grounding in that TestFlight app, export the in-app dataset package JSON, and ingest only that current-build package.

  - `stale_audit_evidence` | skipped=`None` | prompt:  | actual:

### preflight

- Count: `1`
- Status: `deferred`
- Root causes: `{"runtime_environment_deferred": 1}`
- Categories: `{"runtime_environment_deferred": 1}`
- Fresh runtime failures: `1`
- Skipped live model generation: `0`
- Recommended action: Rerun this scenario through the live app/model path and export fresh E2E evidence before treating it as a tool failure.

  - `runtime_environment_deferred` | skipped=`True` | prompt: What is the weather here and should I carry an umbrella? | actual: device thermal state serious; cool device and retry
