# AGENTS.md

## Scope

Governs `ios/Lumen/Assistant/`, the production agent-kernel contracts, orchestration, runtime routing, structured execution, context budgeting, cancellation, and headless adapter. Parent rules: [`../../AGENTS.md`](../../AGENTS.md).

## Role In The System

This subtree is the single production control plane between user/headless requests, model inference, grounding, native tool execution, and final response events. It owns orchestration, not platform implementations or UI rendering.

## Key Files And Entry Points

- `AgentKernelContracts.swift`: `AgentKernelRequest`, kernel events, options, and `AgentKernelRunning`.
- `AssistantKernel.swift` and `AssistantKernel+Streaming.swift`: shared main-actor kernel and run/stream path.
- `StructuredAgentKernelExecutor.swift`: constrained action/final schemas, repair, approval, observations, and finalization.
- `AssistantRuntimeRouter.swift`: Release-safe runtime selection.
- `HeadlessAgentKernelRunner.swift`: thin non-UI adapter and cancellation.
- `ContextBudgetAllocator.swift`: deterministic prompt/context allocation.

## Public Interfaces

`AssistantKernel.shared.run`, `AgentKernelRunning`, request/options/event types, cancellation handles, and event semantics are consumed by chat, voice, App Intents, CarPlay/headless flows, diagnostics, and tests. Event order, cancellation finality, tool-step status, and final-text completeness are caller contracts.

## Internal Structure

Request normalization/context construction precedes runtime selection. Structured mode exposes only allowed tool definitions to the model, parses constrained JSON, validates through the schema bridge, crosses the approval boundary, executes through the secure registry, records trusted observations, and finalizes. Surface-specific reducers remain outside this subtree.

## Incoming Dependencies

Major callers are `Views/ChatView.swift`, `Voice/VoiceCommandRouter.swift`, App Intents, `Services/E2ETestRunner.swift`, and `HeadlessAgentKernelRunner` users.

## Outgoing Dependencies

The kernel uses `Services/LlamaService.swift`, grounding/manifest services, memory/RAG stores and context builders, `Tools/SecureToolRegistry`, tool schema/approval policy, diagnostics/metrics, and system cancellation/resource policy.

## Data And Control Flow

Request -> bounded memory/RAG/manifest context -> runtime route -> token/structured events -> validated action -> approval/permission -> secure tool result -> trusted observation -> validated/sanitized final event. Cancellation must stop generation and prevent late tool/final events.

## Local Invariants

- All production agent entry points use this kernel; do not add a parallel tool loop.
- Release selection cannot route to deterministic, unavailable, experimental, or uncompiled backends.
- Model output is never an executable tool call until exact schema and availability validation succeeds.
- Repair is bounded; exhaustion produces a typed failure, not fabricated text or silent success.
- Tool observations are trusted only after registry execution; model-authored claims do not become observations.
- Final text must be complete and non-dangling. Synthesis from observations is allowed only when observations support it.
- Cancellation is terminal and must suppress late completion.
- Constructing the headless runner must not load a model.

## Coordinated Changes

Contract/event changes require updates to chat and voice reducers, E2E trace/report handling, headless callers, and kernel tests. Tool-step changes require `Tools/`, `Models/ToolDefinition.swift`, `Services/ToolExecutor.swift`, generated manifests, and schema tests. Runtime changes require `Services/`, hardening scripts, and status documentation.

## Safe Editing Rules

Keep orchestration deterministic around untrusted model output. Add behavior to the owning context builder, runtime adapter, or tool implementation rather than growing hidden state in the kernel. Preserve actor ownership; do not block the main actor during inference. Never use prompt wording as the only tool-safety mechanism.

## Validation

From the repository root:

```bash
python3 tools/check_agent_kernel_boundary.py --strict
python3 tools/check_release_hardening.py
xcodebuild -project ios/Lumen.xcodeproj -scheme Lumen -destination 'platform=iOS Simulator,name=Lumen Focused Test iPhone' test CODE_SIGNING_ALLOWED=NO -only-testing:LumenTests/AssistantKernelRunContractTests -only-testing:LumenTests/AssistantKernelStructuredAgentTests -only-testing:LumenTests/AssistantKernelToolExecutionTests
```

Also inspect `AgentKernelBoundaryGuardTests.swift`, `RuntimeRouterTests.swift`, `LlamaGenerationCancellationTests.swift`, reducer tests, and `E2ETestRunnerHygieneTests.swift` for affected contracts.

## Common Failure Modes

- A new surface bypasses `AssistantKernel`.
- An alias or extra argument reaches a native tool without canonical validation.
- A repair loop becomes unbounded or masks missing required arguments.
- A cancellation races with a final/tool event.
- A fallback-looking final answer is scored as model-backed evidence.

## Parent And Child Guidance

Parent: [`../../AGENTS.md`](../../AGENTS.md). No child file is needed; this directory is one orchestration boundary.
