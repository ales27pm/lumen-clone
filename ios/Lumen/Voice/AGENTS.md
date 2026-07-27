# AGENTS.md

## Scope

Governs `ios/Lumen/Voice/`, recognition/synthesis adapters, voice session state/controller, interruption handling, kernel routing, and event reduction. Audio engine ownership in `Services/VoiceService.swift` is governed by the Services parent. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

This subtree owns the user-facing push-to-talk voice state machine. It turns authorized audio into a transcript, routes the transcript through the same production kernel as chat, reduces events, and synthesizes the final answer.

## Key Files And Entry Points

- `VoiceSessionController.swift`: main-actor session lifecycle and permissions/cancellation.
- `VoiceSessionState.swift`: explicit state machine.
- `VoiceCommandRouter.swift`: transcript -> `AssistantKernel` with valid model context.
- `VoiceKernelEventReducer.swift`: kernel event -> voice state.
- `SpeechRecognitionService.swift`, `SpeechSynthesisService.swift`: abstractions.
- `VoiceInterruptionHandler.swift`: interruption/route handling.
- `ios/Lumen/Services/VoiceService.swift`: AVAudioEngine/Speech/TTS implementation.

## Public Interfaces

Session states/transitions, start/stop/cancel actions, transcript/final result, recognition/synthesis protocols, and event-reducer semantics are consumed by views, CarPlay, scene coordination, and tests.

## Internal Structure

Explicit user action -> permission policy -> audio session/tap -> recognition -> transcript -> `VoiceCommandRouter` -> kernel events -> reducer -> speech synthesis. Interruption/route/scene/cancellation tears down all active resources.

## Incoming Dependencies

`VoiceModeView`, CarPlay scene, app scene transitions, and user gestures call the controller.

## Outgoing Dependencies

Permissions, `Services/VoiceService`, AssistantKernel, SwiftData shared context, cancellation bus, and AVFoundation/Speech through the service.

## Data And Control Flow

Start -> authorize -> listen -> transcript -> kernel -> final/error/cancel -> speak or report -> idle. Every transition is main-actor-owned and terminal cancellation prevents late synthesis/finalization.

## Local Invariants

- Voice uses the production kernel; no legacy/direct model path.
- Microphone/speech permission is explicit and user-initiated.
- Install/remove audio taps and activate/deactivate audio session exactly once per lifecycle.
- Interruption, route change, scene transition, and cancellation stop recognition, generation, and synthesis.
- Do not retain/log raw audio or transcript in diagnostics.
- Voice and Chat reducer semantics remain compatible for shared kernel events.

## Coordinated Changes

Kernel event changes require both reducers and tests. Audio changes require `Services/VoiceService.swift`, permission usage/configuration, CarPlay, scene/cancellation policy, and audio tests. State changes require view/controller tests.

## Safe Editing Rules

Keep state mutation on the main actor and audio callbacks bounded. Do not block callbacks or start untracked detached tasks. Route transcript through `VoiceCommandRouter`; keep platform audio implementation in the service.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/VoiceCommandRouterTests -only-testing:LumenTests/VoiceKernelEventReducerTests -only-testing:LumenTests/VoiceModeLifecycleTests -only-testing:LumenTests/VoiceServiceAudioStartupTests
```

Also inspect permission-denied, coexistence, model-context routing, scene-transition, interruption, and CarPlay policy tests.

## Common Failure Modes

- An audio tap survives stop/interruption and crashes on restart.
- Voice routes around the kernel or loses `ModelContext`.
- Permission denial appears as an empty transcript.
- A cancelled generation is synthesized after teardown.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). Service ownership: [`../Services/AGENTS.md`](../Services/AGENTS.md). CarPlay adapter: [`../CarPlay/AGENTS.md`](../CarPlay/AGENTS.md).
