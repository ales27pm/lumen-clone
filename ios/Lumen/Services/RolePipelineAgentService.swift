import Foundation
import OSLog

nonisolated enum RolePipelineSmokeTestTraceMode: String, Sendable, Equatable {
    case compatibilityDiagnosticsOnly
    case realModelTrace
}

nonisolated struct RolePipelineSmokeTestTraceExpectation: Sendable, Equatable {
    let mode: RolePipelineSmokeTestTraceMode
    let compatibilityModeNotice: String?

    var recordsAgentBehaviorTrace: Bool {
        mode == .realModelTrace
    }

    var buttonTitle: String {
        switch mode {
        case .compatibilityDiagnosticsOnly:
            return "Run Compatibility Diagnostics Smoke Test"
        case .realModelTrace:
            return "Run Real Model Trace Smoke Test"
        }
    }

    var runningTitle: String {
        switch mode {
        case .compatibilityDiagnosticsOnly:
            return "Running Diagnostics Smoke Test…"
        case .realModelTrace:
            return "Running Real Model Trace Smoke Test…"
        }
    }

    var startingSummary: String {
        switch mode {
        case .compatibilityDiagnosticsOnly:
            return "Starting compatibility diagnostics smoke test…"
        case .realModelTrace:
            return "Starting real-model trace smoke test…"
        }
    }

    var footerSentence: String {
        switch mode {
        case .compatibilityDiagnosticsOnly:
            return "The compatibility diagnostics smoke test exercises the deterministic role-pipeline grounding path only; it does not run AppLlamaService and does not record AgentBehaviorTrace entries."
        case .realModelTrace:
            return "The real-model trace smoke test must run through the production model generation path and record AgentBehaviorTrace entries before export."
        }
    }

    var promptPrefix: String {
        switch mode {
        case .compatibilityDiagnosticsOnly:
            return "Compatibility diagnostics smoke test"
        case .realModelTrace:
            return "Real-model trace smoke test"
        }
    }

    func completedSummary(recordedTraceCount: Int, tailChanged: Bool, producedOutput: Bool) -> String {
        let traceObserved = recordedTraceCount > 0 || tailChanged
        let countDescription = recordedTraceCount > 0 ? "\(recordedTraceCount)" : "new"

        if recordsAgentBehaviorTrace {
            if traceObserved {
                return "Real-model trace smoke test recorded \(countDescription) trace(s). Export the runtime audit package again."
            }
            if producedOutput {
                return "Real-model smoke test produced output but no AgentBehaviorTrace was recorded; check trace recorder wiring."
            }
            return "Real-model smoke test completed without output and no AgentBehaviorTrace was recorded. Confirm that a chat model is downloaded and that trace recorder wiring is active."
        }

        if traceObserved {
            return "Diagnostics smoke test unexpectedly recorded \(countDescription) AgentBehaviorTrace trace(s); review smoke-test mode and trace recorder wiring."
        }
        if producedOutput {
            if let compatibilityModeNotice {
                return "Diagnostics smoke test completed. \(compatibilityModeNotice)"
            }
            return "Diagnostics smoke test generated deterministic output; no AgentBehaviorTrace is expected for the compatibility-mode role pipeline."
        }
        if let compatibilityModeNotice {
            return "Diagnostics smoke test completed without model output. \(compatibilityModeNotice)"
        }
        return "Diagnostics smoke test completed without model output; no AgentBehaviorTrace is expected for the compatibility-mode role pipeline."
    }
}

@MainActor
final class RolePipelineAgentService {
    static let shared = RolePipelineAgentService()

    nonisolated static let smokeTestTraceExpectation = RolePipelineSmokeTestTraceExpectation(
        mode: .compatibilityDiagnosticsOnly,
        compatibilityModeNotice: "Compatibility-mode only: this smoke test exercises deterministic grounding diagnostics and does not run AppLlamaService or record AgentBehaviorTrace entries."
    )
    private let logger = Logger(subsystem: "ai.lumen.app", category: "role-pipeline")

    private init() {}

    func run(_ req: AgentRequest) -> AsyncStream<AgentEvent> {
        run(req, options: .default)
    }

    func run(_ req: AgentRequest, options: LegacyAgentRunOptions) -> AsyncStream<AgentEvent> {
        return AsyncStream { continuation in
            let task = Task { @MainActor in
                let grounded = await prepareGroundedRequest(req, options: options)
                let effectiveRequest = makeEffectiveRequest(original: req, grounded: grounded, options: options)
                let text = Self.deterministicAnswer(for: effectiveRequest)
                continuation.yield(.finalDelta(text))
                continuation.yield(.done(finalText: text, steps: []))
                continuation.finish()
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func prepareGroundedRequest(_ req: AgentRequest, options: LegacyAgentRunOptions) async -> LegacyGroundingResult {
        let provider = LegacyGroundingContextProvider(directContext: options.modelContext, allowSharedFallback: options.allowDegradedGrounding)
        let request = LegacyGroundingRequest(
            userMessage: req.userMessage,
            conversationID: options.conversationID ?? req.conversationID,
            turnID: options.turnID ?? req.turnID,
            history: req.history,
            mode: options.groundingMode == .headlessTrigger ? .headless : .foreground,
            task: .chat,
            roleOrSlot: "role-pipeline",
            externalRelevantMemories: req.relevantMemories,
            externalAvailableTools: req.availableTools,
            policy: .rolePipeline,
            baseSystemPrompt: req.systemPrompt,
            preventDoubleGrounding: options.preventDoubleGrounding
        )
        return await LegacyTurnGroundingCoordinator.shared.prepareGroundedRequest(request, provider: provider)
    }

    private func makeEffectiveRequest(original: AgentRequest, grounded: LegacyGroundingResult, options: LegacyAgentRunOptions) -> AgentRequest {
        let useGrounded = options.allowDegradedGrounding || grounded.grounding != nil
        return AgentRequest(
            systemPrompt: useGrounded ? grounded.systemPrompt : original.systemPrompt,
            history: original.history,
            userMessage: useGrounded ? grounded.userMessage : original.userMessage,
            temperature: original.temperature,
            topP: original.topP,
            repetitionPenalty: original.repetitionPenalty,
            maxTokens: original.maxTokens,
            maxSteps: original.maxSteps,
            availableTools: useGrounded ? grounded.bridgedTools : original.availableTools,
            relevantMemories: original.relevantMemories,
            attachments: original.attachments,
            conversationID: options.conversationID ?? original.conversationID,
            turnID: options.turnID ?? original.turnID,
            scenarioID: options.scenarioID ?? original.scenarioID,
            e2eRunID: options.e2eRunID ?? original.e2eRunID,
            agentRunID: options.agentRunID ?? original.agentRunID
        )
    }

    private nonisolated static func deterministicAnswer(for req: AgentRequest) -> String {
        let visible = req.userMessage
            .replacingOccurrences(of: "<!-- LUMEN_GROUNDING_V1 -->", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if visible.isEmpty { return "I need a message to answer." }
        return "I received your request. The role pipeline is temporarily running in compatibility mode while the native build is hardened."
    }
}
