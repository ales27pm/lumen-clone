# Runtime Resource Kill Analysis

## TestFlight evidence

- CPU resource report: Lumen consumed about 90 seconds of CPU over 170 seconds of wall time, averaging roughly 53% CPU and crossing the iOS CPU resource threshold. The same report showed resident footprint around 849 MB and Apple Foundation Models tokenizer assets loaded for `instruct_3b` and `instruct_300m`.
- Jetsam report: Lumen was the `largestProcess` with about 103858 resident pages. At a 16 KB page size, that is roughly 1.58 GB of resident pages.
- Crash/watchdog report: FrontBoard killed Lumen with `0x8BADF00D` after a scene-update watchdog transgression while background-visible, exhausting the 10 second scene update wall-clock allowance.

## Interpreted failure mode

The reports point to a runtime resource/watchdog kill, not a normal Swift exception. The likely dangerous path was a combination of heavy model/tokenizer residency, background/scene-update work, and insufficient cancellation as the app moved out of active foreground execution.

## Mitigation strategy

- No heavy model work at app startup. Model runtime initialization is deferred until a user explicitly starts chat or voice work.
- Model/tokenizer loading is intent-gated. Only `userChat` and `userVoice` may attempt actual local runtime loading; `appStartup`, `diagnostics`, and `background` degrade without loading assets.
- Scene phase transitions to inactive/background perform immediate nonblocking cancellation of generation/model-load/voice work and do not run memory compaction, RAG indexing, diagnostics, triggers, or model probing in the scene-update closure.
- Diagnostics are passive: capability snapshots and diagnostics read cheap cached/metadata status rather than instantiating FoundationModels tokenizer/model assets.
- Background tasks are bounded to a sub-5-second wall-clock budget and return skipped/degraded results rather than loading a model that is not already safe to use.
- Memory warnings cancel in-flight model loads and optional slot residency before recording metrics.

## Validation commands

```sh
find ios/Lumen -name "*.swift" -print | sed 's#.*/##' | sort | uniq -d
find ios/Lumen -name "ToolDefinition.swift" -o -name "DeviceCapabilitySnapshot.swift"
rm -rf /tmp/rork-publish-derived-data
cd ios
xcodebuild -project Lumen.xcodeproj -scheme Lumen -destination 'generic/platform=iOS Simulator' clean build
xcodebuild -project Lumen.xcodeproj -scheme Lumen -destination 'generic/platform=iOS Simulator' test
```

## Remaining work

- Symbolicate the original TestFlight reports with the matching dSYM to map offsets to precise functions.
- Re-run on physical devices because Jetsam and FoundationModels tokenizer residency vary by OS version, memory class, and device page size.
- Continue measuring CPU and resident size during long voice/chat sessions; conservative degradation should be preferred over keeping background work alive.
