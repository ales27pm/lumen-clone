# AGENTS.md

## Scope

Governs `ios/Lumen/Services/`, including direct shared runtime, storage, platform, network, diagnostics, and scheduling services. Child files refine AgentGrounding, Diagnostics, LLM abstractions, and Microsoft Graph. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

Services implement the kernel's platform/runtime dependencies. Important direct owners include native llama inference, lazy model loading, SwiftData memory/RAG stores, vector index, tool route guarding, voice audio, BGTask scheduling, E2E execution, and network resilience.

## Key Files And Entry Points

- `LlamaService.swift`: actor-owned SwiftLlama/llama.cpp context, streaming, embeddings, adapter activation, and cancellation.
- `ModelLoader.swift`: on-demand model preparation/loading policy.
- `MemoryStore.swift`, `RAGStore.swift`, `VectorIndex.swift`: SwiftData/vector persistence boundaries.
- `ToolExecutor.swift`: `ToolRouteGuard`, canonical route/alias normalization, and approval classification.
- `VoiceService.swift`: AVAudioEngine/Speech/TTS resource lifecycle.
- `TriggerScheduler.swift`: BGTask identifiers, registration, submission, and callback handoff.
- `E2ETestRunner.swift`: live scenario execution/evidence producer through `AssistantKernel`.
- `ToolNetworkResilience.swift`: bounded retry, circuit, and telemetry policy.

## Public Interfaces

Actors/services and their typed results are consumed by Assistant, Memory, RAG, Voice, Background, Tools, Views, Developer, and tests. Model state, store diagnostics, route canonicalization, audio lifecycle, BGTask identifiers, and E2E evidence semantics are contracts.

## Internal Structure

Direct service files are ownership boundaries rather than a generic utility layer. Native model work is actor-serialized; stores own `ModelContext`; voice owns audio taps/recognizer/synthesizer; scheduler delegates policy to `Background/`; E2E delegates inference to `AssistantKernel`.

## Incoming Dependencies

Kernel/context builders, secure tools, controllers/views, app startup, developer framework, and tests call these services.

## Outgoing Dependencies

Services use Apple frameworks, SwiftData, SwiftLlama/llama.cpp/Metal, child service modules, models, system resource/cancellation policy, and optional network APIs. Production services must not depend on tests or generated audit outcomes.

## Data And Control Flow

Kernel or controller -> typed service API -> actor/platform/store operation -> typed event/result/diagnostic. Platform callbacks return to the owning actor/coordinator. Errors retain category and do not become empty success.

## Local Invariants

- One shared chat base model context and at most one active role adapter; embeddings use their separate model path.
- Model load/inference is lazy, cancellable, budgeted, and off the main actor while actor state remains serialized.
- Stores preserve empty-vs-failure distinctions and run through valid main-actor `ModelContext` ownership.
- Voice installs/removes audio taps exactly once per lifecycle and handles interruptions/routes/cancellation.
- BGTask scheduler IDs remain exact and callbacks do not run inference.
- E2E runs through the production kernel and cannot manufacture correlated model/tool evidence.
- Network retries are bounded and do not duplicate non-idempotent side effects.

## Coordinated Changes

Model runtime changes require `Assistant/`, `Services/LLM/`, package pin/build linkage, hardening checks, and model tests. Store changes require Models, Memory/RAG, app schema, and persistence tests. Voice changes require `Voice/`, permissions, scene/CarPlay, and audio tests. BGTask changes require `Background/`, app startup/config, and registration tests.

## Safe Editing Rules

Keep ownership explicit: do not create a second model context, vector store, audio engine, task scheduler, or E2E agent loop for convenience. Prefer typed service errors. Use detached work only with cancellation and immutable/copy-safe input; return to owning actor for state mutation.

## Validation

From the repository root:

```bash
python3 tools/check_adapter_runtime_invariants.py
python3 tools/check_ios_lora_hardening_invariants.py
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' build-for-testing CODE_SIGNING_ALLOWED=NO
```

Run affected focused tests, especially `LlamaGenerationCancellationTests.swift`, `ModelLoaderPolicyTests.swift`, `PersistenceAuditTests.swift`, `VoiceServiceAudioStartupTests.swift`, `BackgroundTaskRegistrationTests.swift`, `ToolNetworkResilienceTests.swift`, and `E2ETestRunnerHygieneTests.swift`.

## Common Failure Modes

- A second service instance violates actor/resource singleton assumptions.
- A store catches and returns empty data.
- Retry logic repeats a side effect.
- Model work blocks launch/main actor or ignores cancellation.
- E2E reports fallback text as live model evidence.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). Children: [`AgentGrounding/AGENTS.md`](AgentGrounding/AGENTS.md), [`Diagnostics/AGENTS.md`](Diagnostics/AGENTS.md), [`LLM/AGENTS.md`](LLM/AGENTS.md), and [`MicrosoftGraph/AGENTS.md`](MicrosoftGraph/AGENTS.md).
