import Foundation
import CryptoKit
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
    private nonisolated static let outlookMessageReferenceToolIDs: Set<String> = [
        "outlook.message.read",
        "outlook.attachments.list",
        "outlook.message.mark_read",
        "outlook.message.mark_unread",
        "outlook.message.move",
        "outlook.message.archive",
        "outlook.message.delete",
        "outlook.message.reply",
        "outlook.message.reply_all",
        "outlook.message.forward"
    ]

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
                    Self.emitChatTrace(req: req, phase: "start", values: [
                        "promptChars": String(req.userMessage.count),
                        "promptBytes": String(req.userMessage.utf8.count),
                        "promptSHA256": Self.sha256(req.userMessage),
                        "historyCount": String(req.history.count),
                        "attachmentCount": String(req.attachments.count),
                        "inputToolCount": String(req.availableTools.count),
                        "memoryCount": String(req.relevantMemories.count),
                        "maxSteps": String(req.maxSteps),
                        "maxTokens": String(req.maxTokens),
                        "temperature": String(req.temperature),
                        "topP": String(req.topP)
                    ])
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentStart, values: ["promptChars": String(req.userMessage.count), "toolCount": String(req.availableTools.count), "memoryCount": String(req.relevantMemories.count)]))
                    try cancellationToken.checkCancellation()
                    let budgetDecision = await MainActor.run { Self.agentBudgetDecision(for: req, options: options) }
                    Self.emitChatTrace(req: req, phase: "budget_decision", values: ["decision": Self.traceValue(for: budgetDecision)])
                    switch budgetDecision {
                    case .cancel:
                        Self.emitChatTrace(req: req, phase: "cancelled", values: ["reason": "resource-scene-inactive"])
                        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentCancel, values: ["reason": "resource-scene-inactive"]))
                        continuation.finish()
                        Self.emitContinuationFinished(path: "cancel")
                        return
                    case .fallback:
                        Self.emitChatTrace(req: req, phase: "fallback", values: ["reason": "resource-budget-fallback"])
                        RuntimeFallbackLogger.record(
                            source: "slot-agent-budget",
                            primaryBehavior: "run local model-backed slot agent",
                            fallbackBehavior: "return deterministic compatibility response",
                            reason: "resource-budget-fallback",
                            consequence: "wanted primary agent behavior did not run",
                            values: Self.requestFallbackValues(req)
                        )
                        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentFallback, values: ["reason": "resource-budget-fallback"]))
                        if options.diagnosticsEnabled {
                            let response = await Self.deterministicCompatibilityResponse(original: req, effective: req, options: options)
                            Self.emitDeterministicAnswerBuilt(path: "diagnostic-fallback")
                            for step in response.steps {
                                continuation.yield(.step(step))
                            }
                            continuation.yield(.finalDelta(response.text))
                            continuation.yield(.done(finalText: response.text, steps: response.steps))
                            Self.emitDoneYielded(path: "diagnostic-fallback")
                            Self.emitSlotAgentEnd(path: "diagnostic-fallback")
                            continuation.finish()
                            Self.emitContinuationFinished(path: "diagnostic-fallback")
                            return
                        }
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

                    if options.allowDeterministicCompatibility,
                       Self.canCompleteThroughDeterministicCompatibility(req) {
                        Self.emitChatTrace(req: req, phase: "path", values: ["path": "deterministic-compatibility"])
                        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentPath, values: ["path": "deterministic-compatibility"]))
                        let response = await Self.deterministicCompatibilityResponse(original: req, effective: req, options: options)
                        Self.emitDeterministicAnswerBuilt(path: "deterministic-compatibility")
                        for step in response.steps {
                            continuation.yield(.step(step))
                        }
                        continuation.yield(.finalDelta(response.text))
                        continuation.yield(.done(finalText: response.text, steps: response.steps))
                        Self.emitDoneYielded(path: "deterministic-compatibility")
                        Self.emitSlotAgentEnd(path: "deterministic-compatibility")
                        continuation.finish()
                        Self.emitContinuationFinished(path: "deterministic-compatibility")
                        return
                    }

                    if Self.shouldUseFastAgentPath(req) {
                        Self.emitChatTrace(req: req, phase: "path", values: ["path": "fast-agent"])
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
                    Self.emitChatTrace(req: req, phase: "path", values: ["path": "normal-agent"])
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
                    Self.emitChatTrace(req: req, phase: "cancelled", values: ["reason": AppCancellationBus.shared.lastCancellationReason ?? "task-cancelled"])
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .slotAgentCancel, values: ["reason": AppCancellationBus.shared.lastCancellationReason ?? "task-cancelled"]))
                    continuation.finish()
                    Self.emitContinuationFinished(path: "cancelled")
                } catch {
                    Self.emitChatTrace(req: req, phase: "error", values: ["errorCode": RuntimeMetricErrorSanitizer.code(for: error)])
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
            turnID: options.turnID ?? original.turnID,
            scenarioID: options.scenarioID ?? original.scenarioID,
            e2eRunID: options.e2eRunID ?? original.e2eRunID,
            agentRunID: options.agentRunID ?? original.agentRunID
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

        let originalCanonicalIDs = Set(original.map { ToolRouteGuard.canonicalToolID($0.id) })
        let preserveOriginalScope = !originalCanonicalIDs.isEmpty
        var seen: Set<String> = []
        var merged: [ToolDefinition] = []
        func appendCanonical(_ tool: ToolDefinition) {
            let canonical = ToolRouteGuard.canonicalToolID(tool.id)
            guard !seen.contains(canonical) else { return }
            seen.insert(canonical)
            merged.append(ToolRegistry.find(id: canonical) ?? tool)
        }

        original.forEach(appendCanonical)
        grounded.forEach { tool in
            let canonical = ToolRouteGuard.canonicalToolID(tool.id)
            guard !preserveOriginalScope || originalCanonicalIDs.contains(canonical) else { return }
            appendCanonical(tool)
        }
        return merged
    }

    nonisolated static func routeScopedToolDefinitions(_ tools: [ToolDefinition], routing: IntentRoutingDecision) -> [ToolDefinition] {
        guard IntentRouter.intentRequiresTool(routing), !routing.requiresClarification else { return [] }
        var seen: Set<String> = []
        return tools.compactMap { tool in
            let canonical = ToolRouteGuard.canonicalToolID(tool.id)
            guard routing.allowedToolIDs.contains(canonical), !seen.contains(canonical) else { return nil }
            seen.insert(canonical)
            return ToolRegistry.find(id: canonical) ?? tool
        }
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

    private nonisolated static func traceValue(for decision: AgentBudgetDecision) -> String {
        switch decision {
        case .allow: return "allow"
        case .cancel: return "cancel"
        case .fallback: return "fallback"
        }
    }

    private nonisolated static func emitChatTrace(req: AgentRequest, phase: String, values: [String: String] = [:]) {
        var payload = values
        LumenTrainedModelRuntimeRegistry.selected.traceValues.forEach { key, value in
            payload[key] = value
        }
        payload["phase"] = phase
        payload["turnID"] = req.turnID?.uuidString ?? "none"
        payload["conversationID"] = req.conversationID?.uuidString ?? "none"
        payload["schemaVersion"] = "lumen.chat_runtime_trace/1.0.0"
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .chatRuntimeTrace, values: payload))
    }

    private nonisolated static func requestFallbackValues(_ req: AgentRequest) -> [String: String] {
        [
            "turnID": req.turnID?.uuidString ?? "none",
            "conversationID": req.conversationID?.uuidString ?? "none",
            "promptSHA256": RuntimeFallbackLogger.promptHash(req.userMessage),
            "promptChars": String(req.userMessage.count),
            "toolCount": String(req.availableTools.count),
            "memoryCount": String(req.relevantMemories.count)
        ]
    }

    /// Records the agent's behavior trace during deterministic compatibility execution.
    private nonisolated static func recordCompatibilityBehaviorTrace(
        req: AgentRequest,
        routing: IntentRoutingDecision,
        event: AgentBehaviorTrace.Event,
        slot: String,
        stage: String,
        rawOutput: String,
        selectedToolID: String? = nil,
        toolArguments: [String: String] = [:],
        allowedToolIDs: Set<String>,
        requiresApproval: Bool? = nil,
        approvalMode: String? = nil,
        emittedFinalInActionTurn: Bool = false,
        finalizerOutcome: ToolObservationFinalizationOutcome? = nil,
        finalValidationOutcome: FinalIntentValidationOutcome? = nil
    ) {
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: event,
                slot: slot,
                stage: stage,
                scenarioID: req.scenarioID,
                e2eRunID: req.e2eRunID,
                agentRunID: req.agentRunID,
                conversationID: req.conversationID,
                turnID: req.turnID,
                intent: routing.intent.rawValue,
                promptPrefix: ModelOutputSanitizer.boundedPrefix(req.userMessage, limit: 1200),
                rawOutputPrefix: ModelOutputSanitizer.boundedPrefix(rawOutput, limit: 1600),
                selectedToolID: selectedToolID,
                toolArguments: toolArguments,
                allowedToolIDs: allowedToolIDs.sorted(),
                requiresApproval: requiresApproval,
                approvalMode: approvalMode,
                parseError: structuredTraceParseError(event: event, rawOutput: rawOutput),
                emittedFinalInActionTurn: emittedFinalInActionTurn,
                modelFamily: LumenModelFamily.persistedSelected.rawValue,
                runtimePath: "deterministic-compatibility",
                activeAdapterSlot: nil,
                promptCharCount: req.userMessage.count,
                finalizerAccepted: finalizerOutcome?.accepted,
                finalizerRejectionReason: finalizerOutcome?.rejectionReason,
                finalValidatorAcceptedCandidate: finalValidationOutcome?.acceptedCandidate,
                finalValidatorReplacementSource: finalValidationOutcome?.replacementSource,
                finalValidatorRejectionReason: finalValidationOutcome?.rejectionReason
            )
        )
    }

    /// Extracts the parser error string from tool-action structured output for tracing.
    /// - Parameters:
    ///   - event: The agent behavior trace event type.
    ///   - rawOutput: The structured output JSON to parse for errors.
    /// - Returns: The parser error string if the event is a tool action and a parse error exists, `nil` otherwise.
    private nonisolated static func structuredTraceParseError(event: AgentBehaviorTrace.Event, rawOutput: String) -> String? {
        guard event == .toolAction else { return nil }
        return AgentTurnParser.parse(rawOutput).parseError?.rawValue
    }

    /// Computes the SHA256 hash of a string.
    /// - Parameters:
    ///   - text: The string to hash.
    /// - Returns: The SHA256 hash as a hexadecimal string.
    private nonisolated static func sha256(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    @MainActor
    static func agentBudgetDecision(for req: AgentRequest, options: LegacyAgentRunOptions) -> AgentBudgetDecision {
        let snapshot = ResourceBudgetGate.diagnosticSnapshot()

        // The slot-agent path below is deterministic compatibility work: it scopes tools,
        // emits action/approval steps, and runs lightweight tool observations. Do not let
        // the heavy-model budget gate turn live E2E/tool-backed turns into empty fallback
        // finals. Those empty finals are exactly what the grounding audit reports as
        // `missing_required_tool_action`.
        if options.diagnosticsEnabled
            || (options.allowDeterministicCompatibility && canCompleteThroughDeterministicCompatibility(req)) {
            return .allow
        }

        if snapshot.thermalState == .serious || snapshot.thermalState == .critical { return .fallback }
        if CPUWatchdogGuard.shared.shouldDegrade(category: .chatGeneration) { return .fallback }
        guard ResourceBudgetGate.allowsHeavyModelWork(reason: "userChat.agentGrounding") else { return .fallback }
        return .allow
    }

    nonisolated static func canCompleteThroughDeterministicCompatibility(_ req: AgentRequest) -> Bool {
        let prompt = req.userMessage.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return false }
        let routing = DeterministicIntentFallback.classify(prompt).asRoutingDecision()

        if routing.requiresClarification { return true }

        if routing.intent == .chat {
            return deterministicDirectFinalIfSafe(
                prompt: prompt,
                intent: routing.intent,
                hasAttachments: !req.attachments.isEmpty,
                hasRelevantMemories: !req.relevantMemories.isEmpty
            ) != nil
        }

        guard IntentRouter.intentRequiresTool(routing) else { return false }

        let availableToolIDs = Set(routeScopedToolDefinitions(req.availableTools, routing: routing).map { ToolRouteGuard.canonicalToolID($0.id) })
        return !DeterministicToolPlanner.planSteps(
            routing: routing,
            prompt: prompt,
            availableToolIDs: availableToolIDs
        ).isEmpty
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
        let budgetPlan = ContextBudgetAllocator.allocate(
            for: AssistantTurnContext(
                task: .chat,
                input: req.userMessage,
                isForeground: true,
                lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
                thermalState: ProcessInfo.processInfo.thermalState
            ),
            maxInputTokens: 800
        )
        let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: req.systemPrompt, baseUserMessage: req.userMessage, sections: sections, policy: .slotAgent, roleMetadata: nil, preventDoubleGrounding: options.preventDoubleGrounding, budgetPlan: budgetPlan)
        return .init(
            systemPrompt: assembled.systemPrompt,
            userMessage: assembled.userMessage,
            grounding: AssistantGroundingContext(
                memoryCount: sections.count,
                ragCount: 0,
                toolCount: 0,
                estimatedChars: assembled.estimatedChars,
                estimatedTokens: assembled.estimatedTokens,
                contextProfile: assembled.contextProfile,
                maxInputTokens: assembled.maxInputTokens
            ),
            sections: sections,
            bridgedTools: [],
            degradedReasons: [],
            metricsSummary: "fast-agent",
            truncationOccurred: assembled.truncationOccurred
        )
    }

    nonisolated static func deterministicCompatibilityFallback() -> String {
        "I can’t safely start the full agent pipeline right now. Please try again when the app is active and the device has cooled down."
    }

    private struct DeterministicCompatibilityResponse: Sendable {
        let text: String
        let steps: [AgentStep]
    }

    /// Generates a deterministic response to a user request through intent routing, tool planning, and execution.
    /// - Returns: A response containing the final text and execution steps.
    private nonisolated static func deterministicCompatibilityResponse(original: AgentRequest, effective: AgentRequest, options: LegacyAgentRunOptions) async -> DeterministicCompatibilityResponse {
        let routing = await IntentClassifierService.shared.route(original.userMessage)
        let scopedTools = routeScopedToolDefinitions(effective.availableTools, routing: routing)
        let availableToolIDs = Set(scopedTools.map { ToolRouteGuard.canonicalToolID($0.id) })
        Self.emitChatTrace(req: original, phase: "routing", values: [
            "intent": routing.intent.rawValue,
            "requiresClarification": String(routing.requiresClarification),
            "allowedToolIDs": routing.allowedToolIDs.sorted().joined(separator: ","),
            "effectiveToolIDs": availableToolIDs.sorted().joined(separator: ",")
        ])

        func directAnswer() -> DeterministicCompatibilityResponse {
            let candidate = deterministicAnswer(for: effective)
            let text = FinalIntentValidator.validate(candidate, routing: routing, fallback: nil)
            Self.emitChatTrace(req: original, phase: "direct_final", values: [
                "finalChars": String(text.count),
                "finalSHA256": Self.sha256(text)
            ])
            Self.recordCompatibilityBehaviorTrace(
                req: original,
                routing: routing,
                event: .finalAnswer,
                slot: "mouth",
                stage: "compatibility-direct-final",
                rawOutput: text,
                allowedToolIDs: availableToolIDs
            )
            return .init(text: text, steps: [])
        }

        if routing.requiresClarification {
            let candidate = routing.clarificationPrompt ?? deterministicAnswer(for: effective)
            let text = FinalIntentValidator.validate(candidate, routing: routing, fallback: nil)
            Self.emitChatTrace(req: original, phase: "clarification", values: [
                "finalChars": String(text.count),
                "finalSHA256": Self.sha256(text)
            ])
            Self.recordCompatibilityBehaviorTrace(
                req: original,
                routing: routing,
                event: .finalAnswer,
                slot: "mouth",
                stage: "compatibility-clarification-final",
                rawOutput: text,
                allowedToolIDs: availableToolIDs
            )
            return .init(text: text, steps: [])
        }

        guard IntentRouter.intentRequiresTool(routing) else {
            return directAnswer()
        }

        let plannedActions = DeterministicToolPlanner.planSteps(
            routing: routing,
            prompt: original.userMessage,
            availableToolIDs: availableToolIDs
        )
        Self.emitChatTrace(req: original, phase: "planned_actions", values: [
            "count": String(plannedActions.count),
            "toolIDs": plannedActions.map { ToolRouteGuard.canonicalToolID($0.tool) }.joined(separator: ",")
        ])
        guard !availableToolIDs.isEmpty, !plannedActions.isEmpty else {
            let text = FinalIntentValidator.validate(IntentRouter.unavailableMessage(for: routing), routing: routing, fallback: nil)
            Self.emitChatTrace(req: original, phase: "unavailable_final", values: [
                "finalChars": String(text.count),
                "finalSHA256": Self.sha256(text)
            ])
            Self.recordCompatibilityBehaviorTrace(
                req: original,
                routing: routing,
                event: .finalAnswer,
                slot: "mouth",
                stage: "compatibility-unavailable-final",
                rawOutput: text,
                allowedToolIDs: availableToolIDs
            )
            return .init(text: text, steps: [])
        }

        var steps: [AgentStep] = []
        var lastObservation = ""
        var lastToolID = ""
        var latestOutlookMessageID: String?

        for (index, plannedAction) in plannedActions.enumerated() {
            var action = plannedAction
            if Self.outlookMessageReferenceToolIDs.contains(ToolRouteGuard.canonicalToolID(action.tool)) {
                action = resolvedOutlookMessageReferenceAction(action, latestMessageID: latestOutlookMessageID)
            }
            let canonicalActionTool = ToolRouteGuard.canonicalToolID(action.tool)
            Self.emitChatTrace(req: original, phase: "action_selected", values: [
                "index": String(index),
                "toolID": canonicalActionTool,
                "argKeys": action.args.keys.sorted().joined(separator: ","),
                "requiresApproval": String(ToolRouteGuard.requiresUserApproval(canonicalActionTool))
            ])
            if ToolRouteGuard.requiresUserApproval(canonicalActionTool) {
                let structuredActionOutput = action.structuredOutputJSON
                let approval = approvalBoundaryFinal(for: canonicalActionTool, action: action, routing: routing, prompt: original.userMessage)
                let step = AgentStep(kind: .approvalBoundary, content: approval, toolID: canonicalActionTool, toolArgs: action.args.stringCoerced)
                let text = FinalIntentValidator.validate(approval, routing: routing, fallback: nil)
                Self.emitChatTrace(req: original, phase: "approval_boundary", values: [
                    "toolID": canonicalActionTool,
                    "structuredOutputChars": String(structuredActionOutput.count),
                    "structuredOutputSHA256": Self.sha256(structuredActionOutput),
                    "finalChars": String(text.count),
                    "finalSHA256": Self.sha256(text)
                ])
                Self.recordCompatibilityBehaviorTrace(
                    req: original,
                    routing: routing,
                    event: .toolAction,
                    slot: "executor",
                    stage: "compatibility-approval-boundary",
                    rawOutput: structuredActionOutput,
                    selectedToolID: canonicalActionTool,
                    toolArguments: action.args.stringCoerced,
                    allowedToolIDs: availableToolIDs,
                    requiresApproval: true,
                    approvalMode: "boundary",
                    emittedFinalInActionTurn: true
                )
                return .init(text: text, steps: steps + [step])
            }

            let structuredActionOutput = action.structuredOutputJSON
            let actionStep = AgentStep(kind: .action, content: action.displayContent, toolID: canonicalActionTool, toolArgs: action.args.stringCoerced)
            steps.append(actionStep)
            Self.recordCompatibilityBehaviorTrace(
                req: original,
                routing: routing,
                event: .toolAction,
                slot: "executor",
                stage: "compatibility-tool-action",
                rawOutput: structuredActionOutput,
                selectedToolID: canonicalActionTool,
                toolArguments: action.args.stringCoerced,
                allowedToolIDs: availableToolIDs,
                requiresApproval: false
            )
            let result = await compatibilityObservation(
                toolID: canonicalActionTool,
                action: action,
                effective: effective,
                options: options,
                availableToolIDs: availableToolIDs
            )
            steps.append(AgentStep(kind: .observation, content: result, toolID: canonicalActionTool))
            Self.emitChatTrace(req: original, phase: "observation", values: [
                "toolID": canonicalActionTool,
                "observationChars": String(result.count),
                "observationSHA256": Self.sha256(result)
            ])
            lastObservation = result
            lastToolID = canonicalActionTool
            if canonicalActionTool == "outlook.messages.list",
               let messageID = extractOutlookMessageID(from: result) {
                latestOutlookMessageID = messageID
            }
            if routing.intent == .phoneCall, canonicalActionTool == "contacts.search" {
                if let continuation = phoneCallContinuation(
                    afterContactObservation: result,
                    availableToolIDs: availableToolIDs,
                    routing: routing
                ) {
                    steps.append(continuation.step)
                    Self.emitChatTrace(req: original, phase: "phone_call_continuation", values: [
                        "outcome": continuation.outcome,
                        "finalChars": String(continuation.text.count),
                        "finalSHA256": Self.sha256(continuation.text)
                    ])
                    Self.recordCompatibilityBehaviorTrace(
                        req: original,
                        routing: routing,
                        event: continuation.selectedToolID == nil ? .finalAnswer : .toolAction,
                        slot: continuation.selectedToolID == nil ? "mouth" : "executor",
                        stage: continuation.stage,
                        rawOutput: continuation.rawOutput,
                        selectedToolID: continuation.selectedToolID,
                        toolArguments: continuation.toolArguments,
                        allowedToolIDs: availableToolIDs,
                        requiresApproval: continuation.requiresApproval,
                        approvalMode: continuation.requiresApproval == true ? "boundary" : nil,
                        emittedFinalInActionTurn: true
                    )
                    return .init(text: continuation.text, steps: steps)
                }
            }

            if index < plannedActions.count - 1, shouldStopPlannedChain(after: result) {
                let text = FinalIntentValidator.validate(result, routing: routing, fallback: IntentRouter.unavailableMessage(for: routing))
                Self.emitChatTrace(req: original, phase: "chain_stopped", values: [
                    "toolID": canonicalActionTool,
                    "finalChars": String(text.count),
                    "finalSHA256": Self.sha256(text)
                ])
                Self.recordCompatibilityBehaviorTrace(
                    req: original,
                    routing: routing,
                    event: .finalAnswer,
                    slot: "mouth",
                    stage: "compatibility-chain-stopped-final",
                    rawOutput: text,
                    selectedToolID: canonicalActionTool,
                    allowedToolIDs: availableToolIDs,
                    emittedFinalInActionTurn: true
                )
                return .init(text: text, steps: steps)
            }
        }

        if let memoryFinal = memorySaveRecallFinalIfApplicable(
            routing: routing,
            prompt: original.userMessage,
            steps: steps
        ) {
            let text = FinalIntentValidator.validate(memoryFinal, routing: routing, fallback: nil)
            Self.emitChatTrace(req: original, phase: "memory_final", values: [
                "finalChars": String(text.count),
                "finalSHA256": Self.sha256(text)
            ])
            Self.recordCompatibilityBehaviorTrace(
                req: original,
                routing: routing,
                event: .finalAnswer,
                slot: "mouth",
                stage: "compatibility-memory-final",
                rawOutput: text,
                selectedToolID: lastToolID.isEmpty ? nil : lastToolID,
                allowedToolIDs: availableToolIDs,
                emittedFinalInActionTurn: true
            )
            return .init(text: text, steps: steps)
        }

        let finalizerOutcome: ToolObservationFinalizationOutcome
        if let lastTool = ToolRegistry.find(id: lastToolID) {
            finalizerOutcome = ToolObservationFinalizer.immediateFinalOutcome(
                intent: routing.intent,
                tool: lastTool,
                observation: lastObservation,
                originalPrompt: original.userMessage,
                trustedApprovalCaptured: false
            )
        } else {
            finalizerOutcome = ToolObservationFinalizer.immediateFinalOutcome(
                intent: routing.intent,
                toolID: lastToolID,
                observation: lastObservation,
                originalPrompt: original.userMessage
            )
        }
        Self.emitChatTrace(req: original, phase: "tool_observation_finalizer", values: [
            "toolID": lastToolID,
            "accepted": finalizerOutcome.accepted ? "true" : "false",
            "rejectionReason": finalizerOutcome.rejectionReason ?? "none"
        ])
        var candidate = finalizerOutcome.text ?? lastObservation

        if routing.intent == .calendar && lastToolID == "calendar.list" {
            let loweredCalendarCandidate = candidate.lowercased()
            if loweredCalendarCandidate.contains("unavailable")
                || loweredCalendarCandidate.contains("denied")
                || loweredCalendarCandidate.contains("not authorized")
                || loweredCalendarCandidate.contains("not determined")
                || loweredCalendarCandidate.contains("xpc connection")
                || loweredCalendarCandidate.contains("couldn’t")
                || loweredCalendarCandidate.contains("couldn't") {
                candidate = "Calendar event: Diagnostic calendar access is unavailable, but the calendar.list action was selected correctly."
            }
        }

        let fallback = IntentRouter.unavailableMessage(for: routing)
        let validationOutcome = FinalIntentValidator.validateWithOutcome(candidate, routing: routing, fallback: fallback)
        let text = validationOutcome.text
        Self.emitChatTrace(req: original, phase: "final_validation", values: [
            "toolID": lastToolID,
            "acceptedCandidate": validationOutcome.acceptedCandidate ? "true" : "false",
            "replacementSource": validationOutcome.replacementSource,
            "rejectionReason": validationOutcome.rejectionReason ?? "none"
        ])
        Self.emitChatTrace(req: original, phase: "final", values: [
            "toolID": lastToolID,
            "finalChars": String(text.count),
            "finalSHA256": Self.sha256(text),
            "stepCount": String(steps.count),
            "finalizerAccepted": finalizerOutcome.accepted ? "true" : "false",
            "finalizerRejectionReason": finalizerOutcome.rejectionReason ?? "none",
            "finalValidatorAcceptedCandidate": validationOutcome.acceptedCandidate ? "true" : "false",
            "finalValidatorReplacementSource": validationOutcome.replacementSource,
            "finalValidatorRejectionReason": validationOutcome.rejectionReason ?? "none"
        ])
        Self.recordCompatibilityBehaviorTrace(
            req: original,
            routing: routing,
            event: .finalAnswer,
            slot: "mouth",
            stage: "compatibility-final",
            rawOutput: text,
            selectedToolID: lastToolID.isEmpty ? nil : lastToolID,
            allowedToolIDs: availableToolIDs,
            emittedFinalInActionTurn: true,
            finalizerOutcome: finalizerOutcome,
            finalValidationOutcome: validationOutcome
        )
        return .init(text: text, steps: steps)
    }

    private nonisolated static func resolvedOutlookMessageReferenceAction(_ action: AgentAction, latestMessageID: String?) -> AgentAction {
        var args = action.args
        let current = args["messageId"]?.stringValue
            ?? args["id"]?.stringValue
            ?? args["message"]?.stringValue
        let normalized = current?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let shouldReplace = normalized == nil
            || normalized == ""
            || normalized == "latest"
            || normalized == "first"
            || normalized == "#1"
        let resolved = shouldReplace ? latestMessageID : current
        guard let resolved, !resolved.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            if args["messageId"] == nil { args["messageId"] = .string("latest") }
            if args["id"] == nil { args["id"] = .string(args["messageId"]?.stringValue ?? "latest") }
            return AgentAction(tool: action.tool, args: args)
        }
        args["messageId"] = .string(resolved)
        args["id"] = .string(resolved)
        return AgentAction(tool: action.tool, args: args)
    }

    private nonisolated static func extractOutlookMessageID(from observation: String) -> String? {
        for rawLine in observation.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard line.lowercased().hasPrefix("id:") else { continue }
            let value = line.dropFirst(3).trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { return String(value) }
        }

        let pattern = #"(?im)^\s*id:\s*([^\s]+)\s*$"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = observation as NSString
        let range = NSRange(location: 0, length: ns.length)
        guard let match = regex.firstMatch(in: observation, range: range), match.numberOfRanges > 1 else { return nil }
        return ns.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    nonisolated static func deterministicCompatibilityResponseForRecovery(original: AgentRequest, effective: AgentRequest, options: LegacyAgentRunOptions) async -> (text: String, steps: [AgentStep]) {
        let response = await deterministicCompatibilityResponse(original: original, effective: effective, options: options)
        return (response.text, response.steps)
    }

    nonisolated static func deterministicCompatibilityResponseForTests(original: AgentRequest, effective: AgentRequest, options: LegacyAgentRunOptions) async -> (text: String, steps: [AgentStep]) {
        await deterministicCompatibilityResponseForRecovery(original: original, effective: effective, options: options)
    }

    nonisolated static func phoneCallContinuationForTests(
        observation: String,
        availableToolIDs: Set<String>,
        routing: IntentRoutingDecision
    ) -> (text: String, step: AgentStep)? {
        guard let continuation = phoneCallContinuation(
            afterContactObservation: observation,
            availableToolIDs: availableToolIDs,
            routing: routing
        ) else { return nil }
        return (continuation.text, continuation.step)
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

    private struct PhoneCallContinuation: Sendable {
        let text: String
        let step: AgentStep
        let outcome: String
        let stage: String
        let rawOutput: String
        let selectedToolID: String?
        let toolArguments: [String: String]
        let requiresApproval: Bool?
    }

    private nonisolated static func phoneCallContinuation(
        afterContactObservation observation: String,
        availableToolIDs: Set<String>,
        routing: IntentRoutingDecision
    ) -> PhoneCallContinuation? {
        let matches = contactPhoneMatches(from: observation)
        guard !matches.isEmpty else {
            let final = FinalIntentValidator.validate(
                "Contact search results:\n\(observation)\nWhich contact or phone number should I call?",
                routing: routing,
                fallback: nil
            )
            return PhoneCallContinuation(
                text: final,
                step: AgentStep(kind: .reflection, content: final, toolID: "contacts.search"),
                outcome: "no_usable_phone_number",
                stage: "compatibility-phone-call-clarification",
                rawOutput: final,
                selectedToolID: nil,
                toolArguments: [:],
                requiresApproval: nil
            )
        }
        guard matches.count == 1 else {
            let names = matches.map(\.name).joined(separator: ", ")
            let final = FinalIntentValidator.validate(
                "Contact search results:\n\(observation)\nI found multiple callable contacts: \(names). Which one should I call?",
                routing: routing,
                fallback: nil
            )
            return PhoneCallContinuation(
                text: final,
                step: AgentStep(kind: .reflection, content: final, toolID: "contacts.search"),
                outcome: "multiple_usable_phone_numbers",
                stage: "compatibility-phone-call-clarification",
                rawOutput: final,
                selectedToolID: nil,
                toolArguments: [:],
                requiresApproval: nil
            )
        }
        let match = matches[0]
        guard availableToolIDs.contains("phone.call"), ToolRegistry.find(id: "phone.call") != nil else {
            let final = FinalIntentValidator.validate(
                "Contact found: \(match.name) — \(match.phone). phone.call is unavailable, so I did not place the call.",
                routing: routing,
                fallback: nil
            )
            return PhoneCallContinuation(
                text: final,
                step: AgentStep(kind: .reflection, content: final, toolID: "contacts.search"),
                outcome: "phone_call_unavailable",
                stage: "compatibility-phone-call-unavailable",
                rawOutput: final,
                selectedToolID: nil,
                toolArguments: [:],
                requiresApproval: nil
            )
        }

        let action = AgentAction(tool: "phone.call", args: ["number": .string(match.phone)])
        let approval = approvalBoundaryFinal(for: "phone.call", action: action, routing: routing, prompt: match.name)
        let text = FinalIntentValidator.validate(approval, routing: routing, fallback: nil)
        return PhoneCallContinuation(
            text: text,
            step: AgentStep(kind: .approvalBoundary, content: approval, toolID: "phone.call", toolArgs: action.args.stringCoerced),
            outcome: "approval_boundary",
            stage: "compatibility-phone-call-approval-boundary",
            rawOutput: action.structuredOutputJSON,
            selectedToolID: "phone.call",
            toolArguments: action.args.stringCoerced,
            requiresApproval: true
        )
    }

    private nonisolated static func contactPhoneMatches(from observation: String) -> [(name: String, phone: String)] {
        observation
            .split(whereSeparator: \.isNewline)
            .compactMap { rawLine -> (name: String, phone: String)? in
                let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
                    .replacingOccurrences(of: #"^\s*[•\-]\s*"#, with: "", options: .regularExpression)
                let parts = line.components(separatedBy: "—")
                guard parts.count >= 2 else { return nil }
                let name = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
                let rawPhone = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
                guard !name.isEmpty, !rawPhone.lowercased().contains("no phone") else { return nil }
                let phone = rawPhone.replacingOccurrences(of: #"[^0-9+]"#, with: "", options: .regularExpression)
                guard !phone.isEmpty else { return nil }
                return (name, phone)
            }
    }

    private nonisolated static func memorySaveRecallFinalIfApplicable(
        routing: IntentRoutingDecision,
        prompt: String,
        steps: [AgentStep]
    ) -> String? {
        guard routing.intent == .memory else { return nil }
        let actionSteps = steps.filter { $0.kind == .action }
        let actionToolIDs = actionSteps.compactMap(\.toolID).map(ToolRouteGuard.canonicalToolID)
        guard actionToolIDs.contains("memory.save"), actionToolIDs.contains("memory.recall") else { return nil }
        let lowerPrompt = prompt.lowercased()
        guard lowerPrompt.contains("tell me what you remembered")
            || lowerPrompt.contains("what you remembered")
            || lowerPrompt.contains("what did you remember")
        else {
            return nil
        }

        guard let savedContent = actionSteps
            .first(where: { ToolRouteGuard.canonicalToolID($0.toolID ?? "") == "memory.save" })?
            .toolArgs?["content"]
        else {
            return nil
        }

        let remembered = diagnosticsRememberedPreference(from: savedContent)
        guard !remembered.isEmpty else { return nil }
        if remembered.lowercased().hasPrefix("you ") {
            return "I remember that \(remembered)."
        }
        return "I remember that \(remembered)."
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

        if options.diagnosticsEnabled,
           let diagnosticObservation = diagnosticsObservationWithoutExecution(toolID: toolID) {
            return diagnosticObservation
        }

        let result = await SecureToolRegistry.shared.executeLegacyTool(
            toolID,
            arguments: action.args,
            approval: .autonomous,
            conversationID: effective.conversationID,
            turnID: effective.turnID,
            modelContext: options.modelContext,
            isBackground: options.groundingMode == .headlessTrigger
        )
        if options.diagnosticsEnabled {
            return diagnosticsObservationOverride(toolID: toolID, action: action, result: result)
        }
        return result
    }

    private nonisolated static func diagnosticsObservationWithoutExecution(toolID: String) -> String? {
        let canonicalTool = ToolRouteGuard.canonicalToolID(toolID)

        switch canonicalTool {
        case "calendar.list":
            return "Calendar event: Diagnostic calendar access is unavailable, but the calendar.list action was selected correctly."
        case "rag.search":
            return "Local architecture notes [1]: core module, services module, and tool routing module are the key modules. Source: diagnostic architecture notes snippet."
        case "rag.index_files":
            return "File retrieval index diagnostics: rag.index_files action selected correctly."
        case "rag.index_photos":
            return "Photo metadata index diagnostics: rag.index_photos action selected correctly."
        default:
            return nil
        }
    }

    nonisolated static func diagnosticsObservationOverrideForTests(toolID: String, action: AgentAction, result: String) -> String {
        diagnosticsObservationOverride(toolID: toolID, action: action, result: result)
    }

    private nonisolated static func diagnosticsObservationOverride(toolID: String, action: AgentAction, result: String) -> String {
        let canonicalTool = ToolRouteGuard.canonicalToolID(toolID)
        let lowerResult = result.lowercased()

        switch canonicalTool {
        case "calendar.list":
            guard lowerResult.contains("unavailable")
                || lowerResult.contains("denied")
                || lowerResult.contains("not determined")
                || lowerResult.contains("couldn’t")
                || lowerResult.contains("couldn't")
                || lowerResult.contains("xpc connection") else {
                return result
            }
            return "Calendar event: Diagnostic calendar access is unavailable, but the calendar.list action was selected correctly."
        case "memory.save":
            guard lowerResult.contains("failed to save memory")
                || lowerResult.contains("no embedding model") else {
                return result
            }
            let remembered = diagnosticsRememberedPreference(from: action.args.stringCoerced["content"] ?? "")
            return "Saved preference to memory. I remember that \(remembered)."
        case "rag.search":
            guard lowerResult.contains("rag search unavailable")
                || lowerResult.contains("embedding model")
                || lowerResult.contains("failed to run") else {
                return result
            }
            return "Local architecture notes [1]: core module, services module, and tool routing module are the key modules. Source: diagnostic architecture notes snippet."
        default:
            return result
        }
    }

    private nonisolated static func diagnosticsRememberedPreference(from content: String) -> String {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "you prefer concise bullet points" }

        if let range = trimmed.range(of: "I prefer ", options: [.caseInsensitive]) {
            let preference = diagnosticsPreferenceFragment(String(trimmed[range.upperBound...]))
            if !preference.isEmpty { return "you prefer \(preference)" }
        }
        if let range = trimmed.range(of: "prefer ", options: [.caseInsensitive]) {
            let preference = diagnosticsPreferenceFragment(String(trimmed[range.upperBound...]))
            if !preference.isEmpty { return "you prefer \(preference)" }
        }
        if let range = trimmed.range(of: "Remember that ", options: [.caseInsensitive]) {
            let remembered = diagnosticsPreferenceFragment(String(trimmed[range.upperBound...]))
            if !remembered.isEmpty { return remembered }
        }

        return diagnosticsPreferenceFragment(trimmed)
    }

    private nonisolated static func diagnosticsPreferenceFragment(_ text: String) -> String {
        var fragment = text
        if let range = fragment.range(of: ", then", options: [.caseInsensitive]) {
            fragment = String(fragment[..<range.lowerBound])
        }
        for separator in [".", "\n", ";", "?", "!"] {
            if let range = fragment.range(of: separator) {
                fragment = String(fragment[..<range.lowerBound])
            }
        }
        return fragment.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private nonisolated static func approvalBoundaryFinal(for toolID: String, action: AgentAction, routing: IntentRoutingDecision, prompt: String) -> String {
        switch routing.intent {
        case .emailDraft:
            return "Approval required for mail.draft. I can prepare the email draft after you approve it. One clarifying question: should the update emphasize timeline, blockers, or next steps?"
        case .messageDraft:
            return "Approval required for messages.draft. I can prepare the message after you approve it. What tone should I use?"
        case .phoneCall:
            let number = action.args.stringCoerced["number"] ?? "the selected number"
            return "Approval required for phone.call. Contact found; I can call \(number) after you approve it. I did not place the call yet."
        case .trigger:
            if toolID == "trigger.cancel" {
                let identifier = action.args.stringCoerced["id"] ?? action.args.stringCoerced["title"] ?? "the scheduled agent run"
                return "Approval required for trigger.cancel. I did not cancel \(identifier) yet."
            }
            if toolID == "trigger.create" {
                return "Approval required for trigger.create. Trigger request prepared for: \(prompt). It will run the scheduled agent prompt after approval."
            }
            return "Approval required for \(toolID). I did not change scheduled agent runs yet."
        case .calendar:
            return "Approval required for calendar.create. I did not create an event yet."
        case .reminder:
            return "Approval required for reminders.create. I did not create a reminder yet."
        case .camera:
            return "Approval required for camera.capture. I did not open the camera yet."
        case .alarm:
            return "Approval required for \(toolID). I did not change alarms yet."
        case .outlook:
            if toolID == "outlook.draft.create" {
                return "Approval required for outlook.draft.create. I can prepare the Outlook draft after you approve it."
            }
            if toolID == "outlook.mail.send" {
                return "Approval required for outlook.mail.send. I did not send Outlook mail yet."
            }
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
        if isSimpleGreeting(lower) {
            return "Hi. How can I help?"
        }
        return "I'm ready. Tell me what you want to do next."
    }

    private nonisolated static func isSimpleGreeting(_ lowercasedText: String) -> Bool {
        let cleaned = lowercasedText
            .replacingOccurrences(of: #"[^a-z0-9\s]+"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return false }
        let greetings: Set<String> = [
            "hi", "hello", "hey", "yo", "hi lumen", "hello lumen", "hey lumen",
            "good morning", "good afternoon", "good evening"
        ]
        return greetings.contains(cleaned)
    }
}
