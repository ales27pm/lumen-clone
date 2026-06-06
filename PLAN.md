# Fix App Store crash (watchdog 0x8BADF00D + 100% CPU) and implement async model loading

## What's Causing the Crash

Two crash types, same root cause — **heavy synchronous work blocking the main thread**:

1. **Watchdog kill (0x8BADF00D)**: The app takes >10 seconds to render its first screen because the startup bootstrap runs heavy operations on the main actor — specifically parsing a 492KB JSON file (`fleet_system_prompts.json`) and iterating the fleet model catalog with synchronous file I/O and database operations.
2. **CPU overuse kill**: The same code path spins at 100% CPU for 90 seconds straight, flagged by iOS's resource monitoring.

Both crashes trace to the same call chain: `completeTaskWithClosure` → `defaultBootstrap` → grounding resource loading / fleet model iteration — all running on `@MainActor`.

## The Fix

**Move heavy startup work off the main actor so the app renders its first screen in under 1 second**, then load models asynchronously in the background with proper fallbacks.

### Changes

- [x] 1. **Make grounding resource loading non-blocking** — Created `GroundingResourceLoader` actor. The 492KB JSON parse and manifest verification now run on a background actor via `Task.detached`.
- [x] 2. **Make fleet model checks non-blocking** — Optimized `repairFleet` to fetch all stored models once instead of per-catalog-item. The bootstrap now runs in a `Task.detached` context.
- [x] 3. **Implement async model loading at startup** — Models load in the background after the UI renders with:
  - Full context size first, 2048 fallback if that fails
  - Per-candidate fallback (tries preferred model, then next available)
  - Chat + embedding models loaded independently
  - App stays fully usable even if models fail to load
- [x] 4. **Show loading progress in the UI** — Boot splash overlay shows real-time step progress and auto-dismisses 1.5s after all models are ready.

### Files changed

- `ios/Lumen/LumenApp.swift` — Restructured `AppStartupCoordinator` to split critical boot (container creation, UI render) from background bootstrap (grounding resources, fleet checks, model loading). Added `loadChatModelWithFallbacks` and `loadEmbeddingModelWithFallbacks`.
- `ios/Lumen/Services/AgentGrounding/BundledAgentGroundingStore.swift` — Added `GroundingResourceLoader` actor for background JSON parsing.
- `ios/Lumen/Services/ModelLaunchBootstrap.swift` — Optimized `repairFleet` to fetch stored models once and reuse across the catalog iteration. Added overloaded `storedModel(for:in:)` and `missingModels(from:allStored:)`.
- `ios/Lumen/State/RuntimeState.swift` — Added `grounding` boot step.
