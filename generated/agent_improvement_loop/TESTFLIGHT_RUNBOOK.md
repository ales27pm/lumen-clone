# TestFlight In-App Runtime Runbook

This is the live-runtime phase of the Lumen improvement loop. Do not replace this with mocked unit tests. The point is to run the current app build through TestFlight, then export what the shipped app observed.

## Build identity

- Manifest fingerprint: `06ec4a4d253a01a28995c7e25cbbc856ab48e3d929f558f190515014896d6b8a`
- Manifest commit: `65446a07258bc6fa06111e5f968527101570bbe6`
- Build label: `None`
- Expected export: `lumen-agent-grounding-audit-*.json or lumen-live-e2e-report-*.json from Agent Grounding > Export Runtime Audit Package or End-to-end tests > Export Live E2E Report JSON`

## Required app flow

1. Compile/archive the app and distribute it through TestFlight.
2. Install or update that TestFlight build on the device.
3. Use the normal app surface for scenario prompts. Do not use a mocked harness for this pass.
4. Open the in-app Agent Grounding screen.
5. Tap `Run Agent Grounding Audit`.
6. Tap `Export In-App Dataset Package`.
7. Share/save the produced `lumen-agent-grounding-audit-*.json or lumen-live-e2e-report-*.json` file.
8. Feed it into the next loop:

```bash
python -m lumen_manifest_crawler improve-loop --root /home/ales27pm/Lumen --output /home/ales27pm/Lumen/generated/agent_manifest --loop-output /home/ales27pm/Lumen/generated/agent_improvement_loop --runtime-audit '<exported-testflight-json>'
```

## Scenario queue

Full machine-readable queue: `testflight_scenarios.jsonl`

### 1. runtime_trace_export_coverage

- Agent: `runtime`
- Source: `trace_export_coverage`
- Prompt: Trigger intent `alarm` with a realistic request that should select one of: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 2. runtime_trace_export_coverage

- Agent: `runtime`
- Source: `trace_export_coverage`
- Prompt: Trigger intent `calendar` with a realistic request that should select one of: calendar.create, calendar.list.

### 3. runtime_trace_export_coverage

- Agent: `runtime`
- Source: `trace_export_coverage`
- Prompt: Trigger intent `camera` with a realistic request that should select one of: camera.capture.

### 4. runtime_trace_export_coverage

- Agent: `runtime`
- Source: `trace_export_coverage`
- Prompt: Ask a normal chat-only question that should not call tools, then verify the exported runtime traces still include prompt prefixes and parse diagnostics.

### 5. runtime_trace_integrity

- Agent: `runtime`
- Source: `trace_integrity`
- Prompt: Run one tool-backed task and verify the exported dataset shows `traceParseErrorCount` does not increase unexpectedly.

### 6. runtime_trace_integrity

- Agent: `runtime`
- Source: `trace_integrity`
- Prompt: Run a mixed batch of chat and tool prompts, then verify the export includes both `traceSelectedToolAllowedCount` and `traceParseErrorCount`.

### 7. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `alarm`, select only an allowed tool. Forbidden candidates: calendar.create, calendar.list, camera.capture, contacts.search, files.read.

### 8. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `calendar`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 9. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `camera`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 10. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `chat`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 11. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `contactSearch`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 12. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `emailDraft`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 13. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `files`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 14. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `health`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 15. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `maps`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 16. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `memory`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 17. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `messageDraft`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 18. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `motion`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 19. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `note`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 20. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `outlook`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 21. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `phoneCall`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 22. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `photos`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 23. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `rag`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 24. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `reminder`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 25. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `trigger`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 26. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `unknown`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 27. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `weather`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 28. routing_matrix_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: For intent `webSearch`, select only an allowed tool. Forbidden candidates: alarm.authorization_status, alarm.cancel, alarm.countdown, alarm.list, alarm.pause.

### 29. tool_schema_adherence

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: Generate a Tool Executor JSON call for `calendar.create` using only required arguments from the manifest.

### 30. tool_runtime_scenario_selection

- Agent: `runtime`
- Source: `eval_scenarios`
- Prompt: Generate a manifest-valid action step for `calendar.create`.

Additional scenarios omitted from this Markdown view: `90`. Use `testflight_scenarios.jsonl` for the full queue.
