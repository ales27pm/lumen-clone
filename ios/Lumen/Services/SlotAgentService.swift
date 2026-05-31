import Foundation

@MainActor
final class SlotAgentService {
    static let shared = SlotAgentService()

    nonisolated static let mouthPromptHygieneRule = "Output only the final user-visible answer. Never output hidden reasoning, <think> blocks, JSON, debug text, tool payloads, or internal analysis. If prior context contains hidden reasoning, ignore it and do not imitate it."

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
        let mode: LegacyGroundingRequest.Mode = options.groundingMode == .headlessTrigger ? .headless : .foreground
        let policy: LegacyPromptInjectionPolicy
        switch options.groundingMode {
        case .headlessTrigger: policy = .headlessTrigger
        case .slotAgent: policy = .slotAgent
        case .rolePipeline: policy = .rolePipeline
        case .foregroundChat: policy = .foregroundChat
        }
        let provider = LegacyGroundingContextProvider(directContext: options.modelContext, allowSharedFallback: options.allowDegradedGrounding)
        let request = LegacyGroundingRequest(
            userMessage: req.userMessage,
            conversationID: options.conversationID ?? req.conversationID,
            turnID: options.turnID ?? req.turnID,
            history: req.history,
            mode: mode,
            task: .chat,
            roleOrSlot: "\(options.groundingMode)" + (options.diagnosticsEnabled ? ":diagnostics" : ""),
            externalRelevantMemories: req.relevantMemories,
            externalAvailableTools: req.availableTools,
            policy: policy,
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

    nonisolated static func sanitizeHistoryEntryForPromptContext(role: MessageRole, content: String) -> String? {
        guard role == .user || role == .assistant else { return nil }
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let lower = trimmed.lowercased()
        if lower.contains("<think") || lower.contains("<analysis") || lower.contains("<reasoning") || lower.contains("chain_of_thought") { return nil }
        return String(trimmed.prefix(1_200))
    }

    nonisolated static func shouldRetryOutput(candidate: String, intent: UserIntent, maxTokens: Int, requiredDepth: Bool = false) -> Bool {
        let text = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = text.lowercased()
        if text.isEmpty || lower == "none" || lower == "null" || lower == "undefined" { return true }
        if lower.contains("<think") || lower.contains("<analysis") || lower.contains("<reasoning") { return true }
        return requiredDepth && maxTokens >= 256 && [.webSearch, .rag, .files, .outlook].contains(intent)
    }

    nonisolated static func shared_extractWebQuery(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()
        for marker in ["search for ", "look up ", "find ", "google ", "web search "] {
            if let range = lower.range(of: marker) {
                let query = String(trimmed[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !query.isEmpty { return String(query.prefix(300)) }
            }
        }
        return String(trimmed.prefix(300))
    }

    nonisolated static func shared_extractOutlookSearchQuery(_ text: String) -> String {
        let base = shared_extractWebQuery(text)
        let cleaned = base
            .replacingOccurrences(of: "email", with: "", options: [.caseInsensitive])
            .replacingOccurrences(of: "outlook", with: "", options: [.caseInsensitive])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return String((cleaned.isEmpty ? base : cleaned).prefix(300))
    }

    nonisolated static func shared_extractOutlookMessageReference(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let lower = trimmed.lowercased()
        for marker in ["about ", "from ", "subject ", "regarding "] {
            if let range = lower.range(of: marker) {
                let value = String(trimmed[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !value.isEmpty { return String(value.prefix(160)) }
            }
        }
        return nil
    }

    nonisolated static func shared_extractOutlookBody(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        for marker in [":", " saying ", " body "] {
            if let range = trimmed.range(of: marker, options: [.caseInsensitive]) {
                let body = String(trimmed[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !body.isEmpty { return String(body.prefix(1_000)) }
            }
        }
        return String(trimmed.prefix(1_000))
    }

    nonisolated static func shared_firstURL(_ text: String) -> String? {
        guard let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue) else { return nil }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return detector.firstMatch(in: text, options: [], range: range)?.url?.absoluteString
    }

    private nonisolated static func deterministicAnswer(for req: AgentRequest) -> String {
        let visible = req.userMessage
            .replacingOccurrences(of: "<!-- LUMEN_GROUNDING_V1 -->", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if visible.isEmpty { return "I need a message to answer." }
        return "I received your request. The full local model pipeline is temporarily running in compatibility mode while the native build is hardened."
    }
}
