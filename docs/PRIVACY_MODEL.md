# Privacy Model

- Raw prompts, messages, tool observations, final text, and failure details may exist only in memory while an explicit diagnostic run is executing. They are not a persistent evidence format.
- Persisted E2E results and exported E2E reports use the versioned `redacted-v1` format. Free-form fields are replaced with SHA-256/count summaries before encoding; raw correlation UUIDs are omitted.
- Persisted behavior traces use `agent-behavior-traces-redacted-v1.jsonl`. Prompts, outputs, tool arguments, and non-allowlisted runtime paths are reduced to hashes and counts before disk write.
- E2E and trace files use complete file protection when first written. App startup deletes known legacy raw-content filenames before normal runtime work begins.
- Release builds do not expose the Developer Console in Settings and disable iTunes/Finder Documents file sharing. DEBUG builds retain the local diagnostic workflow.
- `python3 tools/check_runtime_audit_privacy.py` rejects checked-in legacy E2E, behavior-trace, and in-app grounding-package names and validates redaction/hash-summary fields in versioned exports. It runs inside `scripts/check-lumen-integration-gate.sh`.
- Tool diagnostics expose categories, permission/approval/background state, bounded counts, and sanitized failure codes only. Background and network diagnostics use identifier/status summaries rather than payload content.

- Voice capture remains foreground/user-initiated; background scene transition interrupts active voice sessions.

Historical raw runtime exports were removed from the current tree and recorded by path and hash in `runtime-audits/PRIVACY_QUARANTINE.md`. Prior Git objects and remote clones still retain those bytes until an explicitly coordinated history rewrite and downstream resynchronization is completed.
