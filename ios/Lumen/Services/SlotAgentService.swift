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
        let cancellationToken = AgentGroundingCancellationToken()
        return AsyncStream { continuation in
            let task = Task.detached(priority: .userInitiated) {
                do {
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentStart, values: ["promptChars": String(req.userMessage.count), "toolCount": String(req.availableTools.count), "memoryCount": String(req.relevantMemories.count)]))
                    try cancellationToken.checkCancellation()
                    let budgetDecision = await MainActor.run { Self.agentBudgetDecision() }
                    switch budgetDecision {
                    case .cancel:
                        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentCancel, values: ["reason": "resource-scene-inactive"]))
                        continuation.finish()
                        return
                    case .fallback:
                        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentFallback, values: ["reason": "resource-budget-fallback"]))
                        let text = Self.deterministicCompatibilityFallback()
                        continuation.yield(.finalDelta(text))
                        continuation.yield(.done(finalText: text, steps: []))
                        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentEnd, values: ["path": "fallback"]))
                        continuation.finish()
                        return
                    case .allow:
                        break
                    }

                    if Self.shouldUseFastAgentPath(req) {
                        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentPath, values: ["path": "fast-agent"]))
                        let grounded = Self.fastGroundingResult(for: req, options: options)
                        try cancellationToken.checkCancellation()
                        let effectiveRequest = await MainActor.run { self.makeEffectiveRequest(original: req, grounded: grounded, options: options) }
                        let text = Self.deterministicAnswer(for: effectiveRequest)
                        continuation.yield(.finalDelta(text))
                        continuation.yield(.done(finalText: text, steps: []))
                        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentEnd, values: ["path": "fast-agent", "groundingChars": String(grounded.userMessage.count + grounded.systemPrompt.count), "sectionCount": String(grounded.sections.count), "toolCount": String(grounded.bridgedTools.count)]))
                        continuation.finish()
                        return
                    }

                    try cancellationToken.checkCancellation()
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentPath, values: ["path": "normal-agent"]))
                    let grounded = await self.prepareGroundedRequest(req, options: options, cancellationToken: cancellationToken)
                    try cancellationToken.checkCancellation()
                    let effectiveRequest = await MainActor.run { self.makeEffectiveRequest(original: req, grounded: grounded, options: options) }
                    let text = Self.deterministicAnswer(for: effectiveRequest)
                    continuation.yield(.finalDelta(text))
                    continuation.yield(.done(finalText: text, steps: []))
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentEnd, values: ["path": "normal-agent", "groundingChars": String(grounded.userMessage.count + grounded.systemPrompt.count), "sectionCount": String(grounded.sections.count), "toolCount": String(grounded.bridgedTools.count)]))
                    continuation.finish()
                } catch is CancellationError {
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentCancel, values: ["reason": AppCancellationBus.shared.lastCancellationReason ?? "task-cancelled"]))
                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
            AppCancellationBus.shared.register(task, category: .chatGeneration)
            continuation.onTermination = { @Sendable _ in
                cancellationToken.cancel()
                task.cancel()
            }
        }
    }

    private func prepareGroundedRequest(_ req: AgentRequest, options: LegacyAgentRunOptions, cancellationToken: AgentGroundingCancellationToken? = nil) async -> LegacyGroundingResult {
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
        return await LegacyTurnGroundingCoordinator.shared.prepareGroundedRequest(request, provider: provider, cancellationToken: cancellationToken)
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

    enum AgentBudgetDecision: Sendable, Equatable { case allow, cancel, fallback }

    @MainActor
    static func agentBudgetDecision() -> AgentBudgetDecision {
        let snapshot = ResourceBudgetGate.diagnosticSnapshot()
        if snapshot.scenePhase == .inactive || snapshot.scenePhase == .background { return .cancel }
        if snapshot.thermalState == .serious || snapshot.thermalState == .critical { return .fallback }
        if CPUWatchdogGuard.shared.shouldDegrade(category: .chatGeneration) { return .fallback }
        guard ResourceBudgetGate.allowsHeavyModelWork(reason: "userChat.agentGrounding") else { return .fallback }
        return .allow
    }

    nonisolated static func shouldUseFastAgentPath(_ req: AgentRequest) -> Bool {
        let prompt = req.userMessage.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, prompt.count <= 80 else { return false }
        guard req.attachments.isEmpty else { return false }
        let lower = prompt.lowercased()
        let explicitToolTerms = ["search", "web", "file", "pdf", "document", "email", "outlook", "calendar", "remind", "map", "weather", "photo", "camera", "call", "message", "note", "memory", "rag", "tool"]
        guard !explicitToolTerms.contains(where: { lower.contains($0) }) else { return false }
        let wordCount = prompt.split { $0.isWhitespace || $0.isNewline }.count
        guard wordCount <= 8 else { return false }
        return true
    }

    nonisolated static func fastGroundingResult(for req: AgentRequest, options: LegacyAgentRunOptions) -> LegacyGroundingResult {
        let tinyMemories = req.relevantMemories.prefix(2).map { item in
            let content = String(item.content.prefix(96))
            return PromptGroundingSection(title: "Relevant memories", content: "- \(content)", estimatedChars: content.count + 2, sourceIDs: [item.id.uuidString], privacyLevel: .moderate)
        }
        let sections = Array(tinyMemories)
        let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: req.systemPrompt, baseUserMessage: req.userMessage, sections: sections, policy: .slotAgent, roleMetadata: nil, preventDoubleGrounding: options.preventDoubleGrounding)
        return .init(systemPrompt: assembled.systemPrompt, userMessage: assembled.userMessage, grounding: AssistantGroundingContext(memoryCount: sections.count, ragCount: 0, toolCount: 0, estimatedChars: assembled.estimatedChars), sections: sections, bridgedTools: [], degradedReasons: [], metricsSummary: "fast-agent", truncationOccurred: assembled.truncationOccurred)
    }

    nonisolated static func deterministicCompatibilityFallback() -> String {
        "I can’t safely start the full agent pipeline right now. Please try again when the app is active and the device has cooled down."
    }

    private nonisolated static func deterministicAnswer(for req: AgentRequest) -> String {
        let visible = req.userMessage
            .replacingOccurrences(of: "<!-- LUMEN_GROUNDING_V1 -->", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if visible.isEmpty { return "I need a message to answer." }
        return "I received your request. The full local model pipeline is temporarily running in compatibility mode while the native build is hardened."
    }
}
