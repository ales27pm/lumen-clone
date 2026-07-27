# AGENTS.md

## Scope

Governs `ios/Lumen/CarPlay/`. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

This subtree adapts the existing voice/kernel lifecycle to a CarPlay scene. It is a platform surface, not a separate assistant or tool-policy implementation.

## Key Files And Entry Points

- `CarPlayVoiceSceneDelegate.swift`: `canImport(CarPlay)` scene lifecycle and voice-session integration.
- `CarPlayVoiceSessionPolicy.swift`: conditions under which a CarPlay voice session may begin/continue.
- `ios/Lumen/Voice/VoiceSessionController.swift`: owning voice session.
- `ios/Lumen/System/AppCancellationBus.swift`: global cancellation propagation.

## Public Interfaces

CarPlay scene delegate callbacks and policy outcomes are consumed by iOS/CarPlay and the app scene configuration. Session activation/deactivation behavior is user-visible and lifecycle-sensitive.

## Internal Structure

The delegate translates CarPlay connection/scene events into policy-controlled voice-session operations. Recognized commands continue through `VoiceCommandRouter` and `AssistantKernel`; tool permissions and approvals remain outside this subtree.

## Incoming Dependencies

CarPlay scene lifecycle and app cancellation events call this code.

## Outgoing Dependencies

The code depends on the voice controller/state machine, cancellation bus, and kernel route indirectly through voice. Framework use remains conditionally compiled.

## Data And Control Flow

CarPlay scene event -> session policy -> voice activation/listening -> normal voice/kernel path -> teardown on disconnect/cancel.

## Local Invariants

- A disconnect, interruption, or app cancellation ends active listening/generation and releases audio resources.
- Do not execute a sensitive tool merely because the request originated in CarPlay.
- Do not duplicate voice or kernel state; use the owning controllers.
- Keep all CarPlay references behind the existing availability/import boundary.

## Coordinated Changes

Scene configuration changes require Xcode project/Info configuration review. Voice lifecycle changes require `Voice/`, `Services/VoiceService.swift`, cancellation/scene logic, and CarPlay policy tests.

## Safe Editing Rules

Keep the delegate thin and main-actor lifecycle-safe. Add policy to `CarPlayVoiceSessionPolicy`, not ad hoc conditionals spread across callbacks. Never block scene callbacks on model work.

## Validation

From the repository root:

```bash
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/CarPlayVoiceSessionPolicyTests -only-testing:LumenTests/VoiceModeLifecycleTests -only-testing:LumenTests/VoiceModeSceneTransitionTests
```

A simulator test does not prove behavior in a CarPlay-capable vehicle/head unit; record that manual gap.

## Common Failure Modes

- Scene disconnect leaves an audio tap or generation task active.
- CarPlay policy diverges from voice permission/cancellation policy.
- Conditional compilation hides missing Release linkage.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). Also apply [`../Voice/AGENTS.md`](../Voice/AGENTS.md) when changing session behavior.
