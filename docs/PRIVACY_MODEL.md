# Privacy Model

- Raw prompts, messages, tool observations, final text, and failure details may exist only in memory while an explicit diagnostic run is executing. They are not a persistent evidence format.
- Persisted E2E results and exported E2E reports use the versioned `redacted-v1` format. Free-form fields are replaced with SHA-256/count summaries before encoding; raw correlation UUIDs are omitted.
- Persisted behavior traces use `agent-behavior-traces-redacted-v1.jsonl`. Prompts, outputs, tool arguments, and non-allowlisted runtime paths are reduced to hashes and counts before disk write.
- Agent parse diagnostics use `agent-parse-failures-redacted-v1.jsonl` and `agent-parse-noise-redacted-v1.jsonl`. Persisted identifiers are rotated, model/error labels and all free-form fields are one-way summaries, and disk readers reject raw records.
- E2E, trace, and parse-diagnostic files use complete file protection. App startup deletes known legacy raw-content and unversioned parse-diagnostic filenames before normal runtime work begins.
- Release builds do not expose the Developer Console in Settings and disable iTunes/Finder Documents file sharing. DEBUG builds retain the local diagnostic workflow.
- `python3 tools/check_runtime_audit_privacy.py` rejects checked-in legacy E2E, behavior-trace, and in-app grounding-package names, validates redaction/hash-summary fields in versioned exports, and scans private `exports/` paths reachable from `HEAD`. A deleted export remains a failure while its object is reachable from release history. The checker runs inside `scripts/check-lumen-integration-gate.sh` and fails closed if the history query fails.
- Tool diagnostics expose categories, permission/approval/background state, bounded counts, and sanitized failure codes only. Background and network diagnostics use identifier/status summaries rather than payload content.

- Voice capture remains foreground/user-initiated; background scene transition interrupts active voice sessions.

Historical raw runtime exports were removed from the current tree and recorded by path and hash in `runtime-audits/PRIVACY_QUARANTINE.md`. The HEAD-reachable history scan currently detects these four legacy private export paths:

- `exports/lumen-agent-grounding-audit-2026-05-09T19-16-04Z-edf67849-4911-45e3-bf18-00f4af251488.json`
- `exports/lumen-agent-grounding-audit-2026-05-09T19-40-57Z-b6b16997-41aa-42b7-b52f-00c5bd5bafe3.json`
- `exports/lumen-live-e2e-report-2026-05-09T19-12-05Z-2cc5f76e-e20f-46bb-9de3-931bbe49b1a8.json`
- `exports/lumen-live-e2e-report-2026-05-09T19-15-24Z-a8be4863-bfe6-4ea7-ac44-ff424f8a33c7.json`

This is an unresolved privacy gate, not proof of containment. Those Git objects and any downstream copies remain reachable until an explicitly coordinated history purge and downstream resynchronization are completed.
