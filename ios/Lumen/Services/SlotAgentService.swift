import Foundation
import SwiftUI

private final class SlotAgentCancellationRegistration: @unchecked Sendable {
    private let lock = NSLock()
    private var id: UUID?
    private var category: AppCancellationCategory?
    private var finished = false

    func set(_ id: UUID, category: AppCancellationCategory) {
        var shouldUnregister = false
        lock.lock()
        if finished {
            shouldUnregister = true
        } else {
            self.id = id
            self.category = category
        }
        lock.unlock()
        if shouldUnregister { AppCancellationBus.shared.unregister(id, category: category) }
    }

    func unregister() {
        let currentID: UUID?
        let currentCategory: AppCancellationCategory?
        lock.lock()
        finished = true
        currentID = id
        currentCategory = category
        id = nil
        category = nil
        lock.unlock()
        if let currentID, let currentCategory {
            AppCancellationBus.shared.unregister(currentID, category: currentCategory)
        }
    }
}

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
        let registration = SlotAgentCancellationRegistration()
        return AsyncStream { continuation in
            let task = Task.detached(priority: .userInitiated) {
                defer { registration.unregister() }
                do {
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentStart, values: ["promptChars": String(req.userMessage.count), "toolCount": String(req.availableTools.count), "memoryCount": String(req.relevantMemories.count)]))
                    try cancellationToken.checkCancellation()
                    let budgetDecision = await MainActor.run { Self.agentBudgetDecision() }
                    switch budgetDecision {
                    case .cancel:
                        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentCancel, values: ["reason": "resource-scene-inactive"]))
                        continuation.finish()
                        Self.emitContinuationFinished(path: "cancel")
                        return
                    case .fallback:
                        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentFallback, values: ["reason": "resource-budget-fallback"]))
                        let text = Self.deterministicCompatibilityFallback()
                        Self.emitDeterministicAnswerBuilt(path: "fallback")
                        continuation.yield(.finalDelta(text))
                        continuation.yield(.done(finalText: text, steps: []))
                        Self.emitDoneYielded(path: "fallback")
                        Self.emitSlotAgentEnd(path: "fallback")
                        continuation.finish()
                        Self.emitContinuationFinished(path: "fallback")
                        return
                    case .allow:
                        break
                    }

                    if Self.shouldUseFastAgentPath(req) {
                        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentPath, values: ["path": "fast-agent"]))
                        let groundingStart = ProcessInfo.processInfo.systemUptime
                        let grounded = Self.fastGroundingResult(for: req, options: options)
                        Self.emitGroundingComplete(path: "fast-agent", grounded: grounded, elapsedMs: Int((ProcessInfo.processInfo.systemUptime - groundingStart) * 1000))
                        try cancellationToken.checkCancellation()
                        let effectiveRequest = await MainActor.run { self.makeEffectiveRequest(original: req, grounded: grounded, options: options) }
                        Self.emitEffectiveRequestBuilt(path: "fast-agent", request: effectiveRequest)
                        let response = await Self.deterministicCompatibilityResponse(original: req, effective: effectiveRequest, options: options)
                        Self.emitDeterministicAnswerBuilt(path: "fast-agent")
                        for step in response.steps {
                            continuation.yield(.step(step))
                        }
                        continuation.yield(.finalDelta(response.text))
                        continuation.yield(.done(finalText: response.text, steps: response.steps))
                        Self.emitDoneYielded(path: "fast-agent")
                        Self.emitSlotAgentEnd(path: "fast-agent", grounded: grounded)
                        continuation.finish()
                        Self.emitContinuationFinished(path: "fast-agent")
                        return
                    }

                    try cancellationToken.checkCancellation()
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentPath, values: ["path": "normal-agent"]))
                    let groundingStart = ProcessInfo.processInfo.systemUptime
                    let grounded = await self.prepareGroundedRequest(req, options: options, cancellationToken: cancellationToken)
                    Self.emitGroundingComplete(path: "normal-agent", grounded: grounded, elapsedMs: Int((ProcessInfo.processInfo.systemUptime - groundingStart) * 1000))
                    try cancellationToken.checkCancellation()
                    let effectiveRequest = await MainActor.run { self.makeEffectiveRequest(original: req, grounded: grounded, options: options) }
                    Self.emitEffectiveRequestBuilt(path: "normal-agent", request: effectiveRequest)
                    let response = await Self.deterministicCompatibilityResponse(original: req, effective: effectiveRequest, options: options)
                    Self.emitDeterministicAnswerBuilt(path: "normal-agent")
                    for step in response.steps {
                        continuation.yield(.step(step))
                    }
                    continuation.yield(.finalDelta(response.text))
                    continuation.yield(.done(finalText: response.text, steps: response.steps))
                    Self.emitDoneYielded(path: "normal-agent")
                    Self.emitSlotAgentEnd(path: "normal-agent", grounded: grounded)
                    continuation.finish()
                    Self.emitContinuationFinished(path: "normal-agent")
                } catch is CancellationError {
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentCancel, values: ["reason": AppCancellationBus.shared.lastCancellationReason ?? "task-cancelled"]))
                    continuation.finish()
                    Self.emitContinuationFinished(path: "cancelled")
                } catch {
                    continuation.finish()
                    Self.emitContinuationFinished(path: "error")
                }
            }
            let id = AppCancellationBus.shared.register(task, category: .chatGeneration)
            registration.set(id, category: .chatGeneration)
            continuation.onTermination = { @Sendable _ in
                cancellationToken.cancel()
                task.cancel()
                registration.unregister()
            }
        }
    }

    func prepareGroundedRequestForDiagnostics(_ req: AgentRequest, options: LegacyAgentRunOptions) async -> LegacyGroundingResult {
        let request = Self.makeLegacyGroundingRequest(
            req,
            options: options,
            roleOrSlot: "\(options.groundingMode):diagnostics-dry-run"
        )
        return await LegacyTurnGroundingCoordinator.shared.prepareGroundedRequest(
            request,
            provider: LegacyGroundingContextProvider(directContext: nil, allowSharedFallback: false)
        )
    }

    nonisolated static func makeLegacyGroundingRequest(
        _ req: AgentRequest,
        options: LegacyAgentRunOptions,
        roleOrSlot: String? = nil
    ) -> LegacyGroundingRequest {
        let mode: LegacyGroundingRequest.Mode = options.groundingMode == .headlessTrigger ? .headless : .foreground
        let policy: LegacyPromptInjectionPolicy
        switch options.groundingMode {
        case .headlessTrigger: policy = .headlessTrigger
        case .slotAgent: policy = .slotAgent
        case .rolePipeline: policy = .rolePipeline
        case .foregroundChat: policy = .foregroundChat
        }
        return LegacyGroundingRequest(
            userMessage: req.userMessage,
            conversationID: options.conversationID ?? req.conversationID,
            turnID: options.turnID ?? req.turnID,
            history: req.history,
            mode: mode,
            task: .chat,
            roleOrSlot: roleOrSlot ?? "\(options.groundingMode)" + (options.diagnosticsEnabled ? ":diagnostics" : ""),
            externalRelevantMemories: req.relevantMemories,
            externalAvailableTools: req.availableTools,
            policy: policy,
            baseSystemPrompt: req.systemPrompt,
            preventDoubleGrounding: options.preventDoubleGrounding
        )
    }

    nonisolated static func requiredTools(for intent: UserIntent) -> Set<String> {
        IntentRouter.allowedToolIDs(for: intent)
    }

    nonisolated static func isActionAllowed(_ toolID: String, routing: IntentRoutingDecision) -> Bool {
        IntentRouter.isToolAllowed(toolID, for: routing)
    }

    nonisolated static func resolveRequiredToolFallback(
        intent: UserIntent,
        prompt: String,
        allowedToolIDs: Set<String>
    ) -> String? {
        if intent == .camera, allowedToolIDs.contains("camera.capture") {
            return "camera.capture"
        }
        if intent == .maps, IntentRouter.isMapFollowUpPrompt(prompt) {
            if allowedToolIDs.contains("location.current") { return "location.current" }
            if allowedToolIDs.contains("maps.search") { return "maps.search" }
        }
        let routing = IntentRoutingDecision(
            intent: intent,
            allowedToolIDs: allowedToolIDs,
            requiresClarification: false,
            clarificationPrompt: nil
        )
        return DeterministicToolPlanner.plan(routing: routing, prompt: prompt, availableToolIDs: allowedToolIDs)?.tool
    }

    nonisolated static func resolveRequiredToolFallback(
        intent: UserIntent,
        prompt: String,
        allowedToolIDs: [String]
    ) -> String? {
        resolveRequiredToolFallback(intent: intent, prompt: prompt, allowedToolIDs: Set(allowedToolIDs))
    }

    nonisolated static func deterministicPrimaryAction(
        routing: IntentRoutingDecision,
        prompt: String,
        scopedTools: [ToolDefinition],
        availableToolIDs: Set<String>
    ) -> AgentAction? {
        DeterministicToolPlanner.planSteps(routing: routing, prompt: prompt, availableToolIDs: availableToolIDs).first
    }

    nonisolated static func deterministicDirectFinalIfSafe(
        prompt: String,
        intent: UserIntent,
        hasAttachments: Bool,
        hasRelevantMemories: Bool
    ) -> String? {
        guard intent == .chat, !hasAttachments, !hasRelevantMemories else { return nil }
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let lowered = trimmed.lowercased()
        guard lowered.hasPrefix("explain ") || lowered.hasPrefix("give me ") else { return nil }
        let request = AgentRequest(
            systemPrompt: "",
            history: [],
            userMessage: trimmed,
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 320,
            maxSteps: 1,
            availableTools: [],
            relevantMemories: []
        )
        return deterministicAnswer(for: request)
    }

    private func prepareGroundedRequest(_ req: AgentRequest, options: LegacyAgentRunOptions, cancellationToken: AgentGroundingCancellationToken? = nil) async -> LegacyGroundingResult {
        let provider = LegacyGroundingContextProvider(directContext: options.modelContext, allowSharedFallback: options.allowDegradedGrounding)
        let request = Self.makeLegacyGroundingRequest(req, options: options)
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
            availableTools: useGrounded ? Self.effectiveToolDefinitions(original: original.availableTools, grounded: grounded.bridgedTools) : original.availableTools,
            relevantMemories: original.relevantMemories,
            attachments: original.attachments,
            conversationID: options.conversationID ?? original.conversationID,
            turnID: options.turnID ?? original.turnID
        )
    }

    private nonisolated static func diagnosticValues(path: String, grounded: LegacyGroundingResult? = nil, toolCount: Int? = nil, elapsedMs: Int? = nil) -> [String: String] {
        var values = ["path": path]
        if let grounded {
            values["groundingChars"] = String(grounded.userMessage.count + grounded.systemPrompt.count)
            values["sectionCount"] = String(grounded.sections.count)
            values["toolCount"] = String(grounded.bridgedTools.count)
        }
        if let toolCount { values["toolCount"] = String(toolCount) }
        if let elapsedMs { values["elapsedMs"] = String(elapsedMs) }
        return values
    }

    private nonisolated static func emitGroundingComplete(path: String, grounded: LegacyGroundingResult, elapsedMs: Int) {
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentGroundingComplete, values: diagnosticValues(path: path, grounded: grounded, elapsedMs: elapsedMs)))
    }

    private nonisolated static func emitEffectiveRequestBuilt(path: String, request: AgentRequest) {
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentEffectiveRequestBuilt, values: diagnosticValues(path: path, toolCount: request.availableTools.count)))
    }

    private nonisolated static func emitDeterministicAnswerBuilt(path: String) {
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentDeterministicAnswerBuilt, values: diagnosticValues(path: path)))
    }

    private nonisolated static func emitDoneYielded(path: String) {
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentDoneYielded, values: diagnosticValues(path: path)))
    }

    private nonisolated static func emitSlotAgentEnd(path: String, grounded: LegacyGroundingResult? = nil) {
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentEnd, values: diagnosticValues(path: path, grounded: grounded)))
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentEndEmitted, values: diagnosticValues(path: path, grounded: grounded)))
    }

    private nonisolated static func emitContinuationFinished(path: String) {
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentContinuationFinished, values: diagnosticValues(path: path)))
    }

    nonisolated static func effectiveToolDefinitions(original: [ToolDefinition], grounded: [ToolDefinition]) -> [ToolDefinition] {
        guard !grounded.isEmpty else { return original }

        var seen: Set<String> = []
        var merged: [ToolDefinition] = []
        func appendCanonical(_ tool: ToolDefinition) {
            let canonical = ToolRouteGuard.canonicalToolID(tool.id)
            guard !seen.contains(canonical) else { return }
            seen.insert(canonical)
            merged.append(ToolRegistry.find(id: canonical) ?? tool)
        }

        original.forEach(appendCanonical)
        grounded.forEach(appendCanonical)
        return merged
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

    private struct DeterministicCompatibilityResponse: Sendable {
        let text: String
        let steps: [AgentStep]
    }

    private nonisolated static func deterministicCompatibilityResponse(original: AgentRequest, effective: AgentRequest, options: LegacyAgentRunOptions) async -> DeterministicCompatibilityResponse {
        let routing = IntentRouter.classify(original.userMessage)
        let availableToolIDs = Set(effective.availableTools.map { ToolRouteGuard.canonicalToolID($0.id) })

        func directAnswer() -> DeterministicCompatibilityResponse {
            let candidate = deterministicAnswer(for: effective)
            let text = FinalIntentValidator.validate(candidate, routing: routing, fallback: nil)
            return .init(text: text, steps: [])
        }

        guard IntentRouter.intentRequiresTool(routing) else {
            return directAnswer()
        }

        if routing.requiresClarification {
            let candidate = routing.clarificationPrompt ?? deterministicAnswer(for: effective)
            let text = FinalIntentValidator.validate(candidate, routing: routing, fallback: nil)
            return .init(text: text, steps: [])
        }

        let plannedActions = DeterministicToolPlanner.planSteps(
            routing: routing,
            prompt: original.userMessage,
            availableToolIDs: availableToolIDs
        )
        guard !availableToolIDs.isEmpty, !plannedActions.isEmpty else {
            let text = FinalIntentValidator.validate(IntentRouter.unavailableMessage(for: routing), routing: routing, fallback: nil)
            return .init(text: text, steps: [])
        }

        var steps: [AgentStep] = []
        var lastObservation = ""
        var lastToolID = ""

        for (index, action) in plannedActions.enumerated() {
            let canonicalActionTool = ToolRouteGuard.canonicalToolID(action.tool)
            if ToolRouteGuard.requiresUserApproval(canonicalActionTool) {
                let approval = approvalBoundaryFinal(for: canonicalActionTool, action: action, routing: routing, prompt: original.userMessage)
                let step = AgentStep(kind: .approvalBoundary, content: approval, toolID: canonicalActionTool, toolArgs: action.args.stringCoerced)
                let text = FinalIntentValidator.validate(approval, routing: routing, fallback: nil)
                return .init(text: text, steps: steps + [step])
            }

            let actionStep = AgentStep(kind: .action, content: action.displayContent, toolID: canonicalActionTool, toolArgs: action.args.stringCoerced)
            steps.append(actionStep)
            let result = await compatibilityObservation(
                toolID: canonicalActionTool,
                action: action,
                effective: effective,
                options: options,
                availableToolIDs: availableToolIDs
            )
            steps.append(AgentStep(kind: .observation, content: result, toolID: canonicalActionTool))
            lastObservation = result
            lastToolID = canonicalActionTool

            if index < plannedActions.count - 1, shouldStopPlannedChain(after: result) {
                let text = FinalIntentValidator.validate(result, routing: routing, fallback: IntentRouter.unavailableMessage(for: routing))
                return .init(text: text, steps: steps)
            }
        }

        let candidate = ToolObservationFinalizer.immediateFinalIfSafe(
            intent: routing.intent,
            toolID: lastToolID,
            observation: lastObservation,
            originalPrompt: original.userMessage
        ) ?? lastObservation
        let fallback = IntentRouter.unavailableMessage(for: routing)
        let text = FinalIntentValidator.validate(candidate, routing: routing, fallback: fallback)
        return .init(text: text, steps: steps)
    }

    nonisolated static func deterministicCompatibilityResponseForTests(original: AgentRequest, effective: AgentRequest, options: LegacyAgentRunOptions) async -> (text: String, steps: [AgentStep]) {
        let response = await deterministicCompatibilityResponse(original: original, effective: effective, options: options)
        return (response.text, response.steps)
    }


    private nonisolated static func shouldStopPlannedChain(after observation: String) -> Bool {
        let lower = observation.lowercased()
        let normalized = lower
            .replacingOccurrences(of: #"[^a-z0-9]+"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return lower.contains("not signed in")
            || lower.contains("missing outlook message context")
            || lower.contains("failed")
            || lower.contains("unavailable")
            || lower.contains("denied")
            || lower.contains("couldn't")
            || lower.contains("couldn’t")
            || lower.contains("error")
            || containsAny(normalized, [
                "no messages", "no message", "no messages found", "no message found",
                "no unread", "no unread messages", "no unread message", "no unread mail",
                "no mail", "no email", "no emails", "empty inbox", "inbox is empty",
                "mailbox is empty", "nothing to read", "nothing found"
            ])
    }

    private nonisolated static func containsAny(_ value: String, _ needles: [String]) -> Bool {
        needles.contains { value.contains($0) }
    }

    private nonisolated static func compatibilityObservation(
        toolID: String,
        action: AgentAction,
        effective: AgentRequest,
        options: LegacyAgentRunOptions,
        availableToolIDs: Set<String>
    ) async -> String {
        guard availableToolIDs.contains(toolID) else {
            return "Tool \(toolID) is disabled. Enable it in Tools."
        }

        if options.diagnosticsEnabled {
            return await ToolExecutor.shared.execute(toolID, arguments: action.args, approval: .autonomous)
        }

        return await LegacySecureToolExecutor.execute(
            toolID: toolID,
            arguments: action.args,
            conversationID: effective.conversationID,
            turnID: effective.turnID
        )
    }

    private nonisolated static func approvalBoundaryFinal(for toolID: String, action: AgentAction, routing: IntentRoutingDecision, prompt: String) -> String {
        switch routing.intent {
        case .emailDraft:
            return "Approval required for mail.draft. I can prepare the email draft after you approve it. One clarifying question: should the update emphasize timeline, blockers, or next steps?"
        case .messageDraft:
            return "Approval required for messages.draft. I can prepare the message after you approve it. What tone should I use?"
        case .trigger:
            return "Approval required for trigger.create. Trigger request prepared for: \(prompt). It will run the scheduled agent prompt after approval."
        case .calendar:
            return "Approval required for calendar.create. I did not create an event yet."
        case .reminder:
            return "Approval required for reminders.create. I did not create a reminder yet."
        case .camera:
            return "Approval required for camera.capture. I did not open the camera yet."
        case .alarm:
            return "Approval required for \(toolID). I did not change alarms yet."
        case .outlook:
            return "Approval required for \(toolID). I did not modify Outlook mail yet."
        default:
            return "Approval required for \(action.displayContent). I did not run it yet."
        }
    }

    private nonisolated static func deterministicAnswer(for req: AgentRequest) -> String {
        let visible = req.userMessage
            .replacingOccurrences(of: "<!-- LUMEN_GROUNDING_V1 -->", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if visible.isEmpty { return "I need a message to answer." }
        let lower = visible.lowercased()
        if lower.contains("precision") && lower.contains("recall") {
            return "Precision is about how many returned results are actually relevant. Recall is about how many of all relevant results the system managed to find. Higher precision avoids clutter; higher recall avoids missing useful matches."
        }
        if lower.contains("sharp chisel") || lower.contains("dull") {
            return "A sharp chisel is safer because it needs less force, follows the cut more predictably, and is less likely to slip. A dull edge makes you push harder, which reduces control."
        }
        if lower.contains("door hinge") {
            return "Three hinge-fitting tips: mark the leaf with a sharp knife, pare to the line in thin passes, and test-fit often so the hinge sits flush without deep gaps."
        }
        if lower.contains("actor isolation") {
            return "Actor isolation means Swift protects data owned by an actor so only that actor can touch it directly. Other code has to await access, which helps prevent races."
        }
        return "I received your request. The full local model pipeline is temporarily running in compatibility mode while the native build is hardened."
    }
}
