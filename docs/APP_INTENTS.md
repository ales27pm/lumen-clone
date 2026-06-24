# App Intents

Implemented concrete intents:
- Ask Lumen (`LumenAskIntent`)
- Search Lumen Memory (`LumenMemorySearchIntent`)
- Add Lumen Memory (`LumenAddMemoryIntent`)
- Run Lumen Trigger (`LumenRunTriggerIntent`)
- Check Lumen Diagnostics (`LumenDiagnosticsIntent`)
- App shortcuts provider (`LumenAppShortcuts`)

Safety rules:
- No sensitive tool execution directly from intents.
- Sensitive operations return: "Open Lumen to approve."
- No external network by default.
- Outputs are bounded and compact.
- If model/store context is unavailable, intents return degraded responses and do not fake success.
- Add Memory saves through the local vector-backed memory store when embeddings are available. If local embedding/indexing is unavailable, it writes a bounded pending capture to local app storage and reports later indexing instead of using cloud fallback or claiming indexed memory.
- Diagnostics intent is passive: it reads cached local capability state, persistent diagnostic run/remediation state, and metadata-only pending local memory capture health. It does not start model loading, memory indexing, background jobs, or diagnostic campaigns.

## Build integration notes

AppIntents are guarded with `canImport(AppIntents)` and `@available(iOS 16.0, *)`. Run `scripts/check-ios-build-readiness.sh` and Xcode builds locally to verify target integration.

Memory search opens Lumen before running because it can reveal saved memory content; the bounded output behavior remains in place after foreground launch.
