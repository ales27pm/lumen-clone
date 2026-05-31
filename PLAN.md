# Fill in all stubs and simplified features across Lumen

I audited the project and found ~20 incomplete or simplified spots. Here's everything I'll finish.

**Messaging & Mail (real composers, not fake strings)**

- Tapping "send a message" / "draft email" will now open the real iOS Messages or Mail composer pre-filled with recipient, subject, and body. If the device can't send (e.g. no mail accounts), it falls back to opening the system app via sms:/mailto:.

**Photos**

- "Selfies" search now uses the real Selfies album instead of a broken filter.
- The photo-indexing selfie count is replaced with an accurate one.
- Photos search also supports "live photos" and "portrait" as filters.

**File attachments (Chat)**

- File picker now accepts Markdown, RTF, CSV, JSON, and rich text alongside plain text and PDF.
- The RAG indexer can decode all of these.
- Attaching a file no longer forces a "summarize it" prompt — it adds a subtle chip to the composer and injects the file as context for whatever the user actually types.

**Chat "Stop" button**

- Messages stopped mid-generation get a small "Stopped" badge instead of a literal "…[stopped]" string appended to the text. Streaming task is properly awaited before state resets so no trailing tokens leak in.

**Hands-free voice mode (the big one)**

- After the assistant finishes speaking, the mic automatically re-opens for the next turn — a true continuous conversation loop.
- Auto-restart only happens when Hands-free is on; a single tap still ends the session.
- Voice catalog locale filter bug fixed so non-English system voices appear correctly in Settings.
- Dead speak-queue property removed.

**Model downloads**

- Pause and Resume actions added (with resume data preserved across app launches).
- Double-tapping Download shows a clear "already downloading" state instead of silently ignoring.
- Fixed a potential deadlock in the download-finished handler.

**Triggers / background scheduling**

- When the app comes to the foreground, any overdue triggers fire immediately instead of waiting for the next background refresh.
- The processing-task lane now actually submits a background processing request (or it's cleanly removed).
- Notification permission result is surfaced to the UI so the user knows if it was denied.

**Memory**

- Pinned memories are blended into recall results by score instead of always crowding out more relevant items.
- Per-turn fact extraction cap lifted from 3 to a more reasonable limit with deduping.

**Agent loop**

- Incremental Thought/Reflection fragments now flush a final "step" event on completion so the Agent Steps panel shows the tail end of every step (currently it gets dropped).
- Tool-call argument stringification uses the same robust helper as the rest of the agent (no more "Optional(5)"-style values).

After all changes I'll build the project and fix any compile errors before handing back.

automatic model loading when the application opens. Add error handling with multiple fallbacks to make sure both models ( main and embedding) are loaded and ready at launch 

