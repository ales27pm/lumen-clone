# Lumen scene watchdog analysis

## Crash evidence

App: Lumen (`com.27pm.lumenclone`), version 1.0.0 build 2, device `iPhone17,1`, iPhone OS 26.4 (23E244), slice UUID `098200ab-a47e-3f5a-ba3f-4c350efaca50`.

Primary incident `B086FE71-01AD-4D58-8489-20A81451C823` terminated with `EXC_CRASH SIGKILL`, FRONTBOARD code `0x8BADF00D`, because the app exhausted the 10.00 second wall-clock allowance during `WatchdogEvent: scene-update` while backgrounded. The main thread was implicated. Unsymbolicated offsets included `Lumen +1592924`, `+2421896`, `+72365`, `+48769`, and `+451581`.

Correlated reports showed sustained CPU (`90s` CPU over `132s` wall, `68%` average, footprint about `850 MB`) and excessive file-backed dirty writes (`4294.98 MB` over `10497s`, average `409.18 KB/s`, above the reported `49.71 KB/s` limit). Memory pressure named Lumen as the largest process (`55244` resident 16 KB pages, lifetime max `89674` pages), but the fatal kill was watchdog rather than jetsam.

## Why `0x8BADF00D` means scene-update watchdog

`0x8BADF00D` is the iOS watchdog termination code used when an app fails to return control to the system within a required lifecycle/event deadline. Here FrontBoard explicitly reported `WatchdogEvent: scene-update` and `WatchdogVisibility: Background`, so scene/lifecycle work exceeded the background scene-update wall-clock budget. The fix is to make all scene transition handlers synchronous, non-awaiting, and bounded.

## Why lifecycle handlers must be nonblocking

Scene phase changes, `applicationWillResignActive`, `applicationDidEnterBackground`, and termination callbacks run while UIKit/SwiftUI is coordinating app visibility and suspension. Awaiting model unloads, memory extraction, RAG indexing, diagnostics scans, file compaction, or large persistence flushes inside those paths can monopolize the main actor or keep Swift concurrency work alive past the watchdog allowance. Lumen now treats lifecycle callbacks as hard real-time paths: record state, cancel existing work, enqueue future maintenance, and return.

## Symbolicating the listed offsets

Use the helper script added at `scripts/symbolicate-lumen-offsets.sh` on macOS with Xcode tools:

```bash
scripts/symbolicate-lumen-offsets.sh --archive /path/to/Lumen.xcarchive
# or
scripts/symbolicate-lumen-offsets.sh --dsym /path/to/Lumen.app.dSYM --base 0x100d88000 0x184E5C 0x24F488 0x11AAD 0xBE81 0x6E3FD
```

The exact hex offsets for the provided decimal crash offsets are `1592924 = 0x184E5C`, `2421896 = 0x24F488`, `72365 = 0x11AAD`, `48769 = 0xBE81`, and `451581 = 0x6E3FD`. The script validates `atos`, locates the DWARF image in either an archive or direct dSYM, adds each offset to the supplied load address, and prints `offset -> absolute address -> symbol`.

## Code changed

- Added `SceneTransitionCoordinator` as the central nonblocking lifecycle path. It records scene state, cancels scene-sensitive tasks through `AppCancellationBus`, stops voice and active model loads synchronously, and logs/asserts if transition handling exceeds 100 ms.
- Added `AppCancellationBus` to synchronously cancel registered task handles by category without performing cleanup.
- Added `DeferredMaintenanceQueue` to deduplicate maintenance jobs and run them only after foreground reactivation, active scene state, resource budget approval, and no chat/voice activity.
- Added `CPUWatchdogGuard` and integrated it into chat, voice, memory extraction, RAG indexing, and diagnostics collection so long loops can stop early under sustained work.
- Added `DiskWriteBudget` and integrated it into conversation saves, memory/RAG persistence, runtime metrics, and diagnostics export gating to defer repeated large writes.
- Hardened voice scene transitions to cancel recognition/TTS/generation immediately, stop high-frequency transcript/waveform updates, and defer persistence.
- Made diagnostics passive by default: opening Diagnostics displays a cached lightweight snapshot; explicit Refresh is required for active collection.
- Throttled chat/voice UI streaming updates to at most 10 Hz and reduced voice waveform invalidation to 1 Hz.

## Physical device validation

1. Build an archive with debug symbols and upload to TestFlight.
2. On a physical device, start voice mode, begin listening/speaking, then background the app repeatedly using the side button and app switcher. Confirm no watchdog termination occurs.
3. Start a long chat generation, background immediately, and confirm the app suspends without a `scene-update` watchdog.
4. Leave diagnostics open for several minutes; confirm it does not start model loads, RAG scans, or filesystem scans unless Refresh/export is tapped.
5. Use Instruments Energy Log and File Activity to verify CPU drops after backgrounding and routine writes remain far below 25 KB/s sustained.
6. Use Organizer crash reports and the symbolication script above if any new TestFlight incident arrives.

## Remaining risks

- SwiftData save size is estimated conservatively because the framework does not expose exact bytes dirtied before saving.
- Some OS frameworks may perform internal writes after Lumen calls APIs such as speech recognition; Lumen now cancels those paths promptly but cannot budget internal framework writes directly.
- CPU timing uses wall-clock task duration rather than private per-thread CPU counters, so it errs on the side of early degradation.
- Deferred maintenance is intentionally opportunistic and may drop stale jobs; user-visible state is preserved first, while derived memory/RAG maintenance may be skipped under pressure.
