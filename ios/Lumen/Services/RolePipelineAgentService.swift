import Foundation
import OSLog

@MainActor
final class RolePipelineAgentService {
    static let shared = RolePipelineAgentService()
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
            turnID: options.turnID ?? original.turnID
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
