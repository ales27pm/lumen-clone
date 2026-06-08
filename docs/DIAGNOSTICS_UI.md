# Diagnostics UI

Diagnostics are the observation layer of the developer improve framework in
`docs/DEVELOPER_IMPROVE_FRAMEWORK.md`. They provide bounded on-device signals
for debugging and gap diagnosis; they do not replace TestFlight/device runtime
evidence or live E2E pass/fail evidence.

The primary app entry point for these surfaces is Settings -> Developer Console.

Implemented diagnostics surfaces:
- Runtime
- Permissions
- Tools
- Background
- Grounding
- Privacy

Data is collected via `DiagnosticsProvider` from on-device status sources only:
- `DeviceCapabilityProfiler`
- `PermissionRegistry`
- `ToolRegistry`
- `RuntimeMetricsStore`
- `BackgroundEntitlementValidator`

Diagnostics intentionally exclude raw prompts, transcripts, messages, memory bodies, and raw RAG chunk text.
