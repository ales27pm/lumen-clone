# Diagnostics UI

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
