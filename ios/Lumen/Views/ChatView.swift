import SwiftUI
import SwiftData
import UniformTypeIdentifiers

enum SchemaPlaceholderDetector {
    private static let repairFallback = "I couldn't produce a valid answer. Try rephrasing, or switch off Agent Mode for this prompt."
    private static let normalizedLiteralSentinelVariants: Set<String> = ["<user_final_text>", "<private_reasoning>"]
    private static let exactPlaceholderVariants: Set<String> = [
        "answershowntotheuser", "youranswertotheuser", "shortprivateroutingnote",
        "shortreasoning", "toolid", "key", "value", "privatereasoning", "userfinaltext"
    ]
    private static let sentinelPrefixVariants: [String] = [
        "answershowntotheuser", "youranswertotheuser", "shortprivateroutingnote",
        "shortreasoning", "privatereasoning", "userfinaltext"
    ]

    static func isSchemaPlaceholderPrefix(_ text: String) -> Bool {
        let normalized = normalizedLiteral(text)
        guard !normalized.isEmpty else { return false }
        if normalizedLiteralSentinelVariants.contains(where: { $0.hasPrefix(normalized) }) { return true }
        let compact = compacted(text)
        guard !compact.isEmpty else { return false }
        return sentinelPrefixVariants.contains { $0.hasPrefix(compact) }
    }

    static func isSchemaPlaceholderFinal(_ text: String) -> Bool {
        let normalized = normalizedLiteral(text)
        guard !normalized.isEmpty else { return false }
        if normalizedLiteralSentinelVariants.contains(normalized) { return true }
        if normalized.count >= 6, normalizedLiteralSentinelVariants.contains(where: { $0.hasPrefix(normalized) }) { return true }
        let compact = compacted(text)
        guard !compact.isEmpty else { return false }
        if exactPlaceholderVariants.contains(compact) { return true }
        if compact.count >= 6, sentinelPrefixVariants.contains(where: { $0.hasPrefix(compact) }) { return true }
        return false
    }

    static func isPlaceholderPrefix(_ text: String) -> Bool { isSchemaPlaceholderPrefix(text) }
    static func isPlaceholderFinal(_ text: String) -> Bool { isSchemaPlaceholderFinal(text) }

    static func repairOrFallback(_ text: String) -> String {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return clean.isEmpty || isSchemaPlaceholderFinal(clean) ? repairFallback : clean
    }

    private static func normalizedLiteral(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: #"\s+"#, with: "", options: .regularExpression)
    }

    private static func compacted(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: #"[^a-z0-9]+"#, with: "", options: .regularExpression)
    }
}

struct ChatView: View {
    @Bindable var conversation: Conversation
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @Query private var storedModels: [StoredModel]

    @State private var draft: String = ""
    @State private var streamingText: String = ""
    @State private var streamingSteps: [AgentStep] = []
    @State private var streamingTask: Task<Void, Never>?
    @State private var activeTurnID: UUID?
    @State private var generationController = GenerationTaskController<UUID>()
    @State private var showVoiceMode = false
    @State private var showFilePicker = false
    @State private var attachments: [ChatAttachment] = []
    @State private var attachmentPreview: [UUID: AttachmentRenderState] = [:]
    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 14) {
                        ForEach(displayedMessages) { message in
                            MessageBubble(message: message).id(message.id)
                        }
                        if !streamingSteps.isEmpty {
                            AgentStepsPanel(steps: streamingSteps, expanded: true).id("steps")
                        }
                        if !streamingText.isEmpty {
                            MessageBubble.streaming(text: streamingText).id("streaming")
                        }
                        Color.clear.frame(height: 8).id("bottom")
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                }
                .scrollDismissesKeyboard(.immediately)
                .contentShape(Rectangle())
                .onTapGesture { isFocused = false }
                .onChange(of: conversation.messages.count) { _, _ in withAnimation(.spring) { proxy.scrollTo("bottom", anchor: .bottom) } }
                .onChange(of: streamingText) { _, _ in withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo("bottom", anchor: .bottom) } }
                .onChange(of: streamingSteps.count) { _, _ in withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo("bottom", anchor: .bottom) } }
            }

            Divider().background(Theme.border)

            if !attachments.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(attachments) { a in
                            AttachmentChip(attachment: a, state: attachmentPreview[a.id]) {
                                attachments.removeAll { $0.id == a.id }
                                recomputeAttachmentPreview()
                            }
                        }
                    }.padding(.horizontal, 12)
                }.padding(.top, 6)
            }

            ChatInputBar(
                draft: $draft,
                isFocused: _isFocused,
                isGenerating: appState.isGenerating,
                onSend: { send(text: nil) },
                onStop: stop,
                onVoice: { showVoiceMode = true },
                onAttach: { showFilePicker = true },
                onDismissKeyboard: { isFocused = false }
            )
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(Theme.background)
        }
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") { isFocused = false }
            }
        }
        .fullScreenCover(isPresented: $showVoiceMode) {
            VoiceModeView(onTranscript: { text in showVoiceMode = false; send(text: text) })
        }
        .fileImporter(
            isPresented: $showFilePicker,
            allowedContentTypes: [.plainText, .pdf, .text, .utf8PlainText, .rtf, .commaSeparatedText, .json, .xml, .html, .sourceCode, UTType(filenameExtension: "md") ?? .plainText],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                for url in urls {
                    if let dest = FileStore.importFile(from: url),
                       let attachment = AttachmentResolver.make(from: dest),
                       !attachments.contains(where: { $0.path == attachment.path }) {
                        attachments.append(attachment)
                    }
                }
                if !urls.isEmpty { UIImpactFeedbackGenerator(style: .soft).impactOccurred() }
                recomputeAttachmentPreview()
            }
        }
        .onChange(of: draft) { _, _ in recomputeAttachmentPreview() }
    }

    private var displayedMessages: [ChatMessage] {
        let sorted = conversation.sortedMessages
        let renderLimit = 300
        if sorted.count > renderLimit {
            return Array(sorted.suffix(renderLimit))
        }
        return sorted
    }

    private func recomputeAttachmentPreview() {
        guard !attachments.isEmpty else { attachmentPreview = [:]; return }
        let states = PromptAssembler.previewAttachmentStates(
            attachments: attachments,
            contextSize: appState.contextSize,
            maxTokens: appState.maxTokens,
            systemPromptChars: (conversation.systemPrompt ?? appState.systemPrompt).count,
            userMessageChars: draft.count,
            hasMemories: appState.autoMemory
        )
        var map: [UUID: AttachmentRenderState] = [:]
        for s in states { map[s.id] = s }
        attachmentPreview = map
    }

    private func send(text overrideText: String?) {
        let source = overrideText ?? draft
        var text = source.trimmingCharacters(in: .whitespacesAndNewlines)
        let turnAttachments = attachments
        if text.isEmpty && !turnAttachments.isEmpty {
            text = turnAttachments.count == 1 ? "Please review the attached file." : "Please review the attached files."
        }
        guard !text.isEmpty, !appState.isGenerating else { return }
        if overrideText == nil { draft = ""; attachments = []; attachmentPreview = [:] }

        let displayContent: String
        if turnAttachments.isEmpty {
            displayContent = text
        } else {
            displayContent = "\(text)\n\nAttached:\n\(turnAttachments.map { "• \($0.name)" }.joined(separator: "\n"))"
        }
        let userMsg = ChatMessage(role: .user, content: displayContent)
        conversation.messages.append(userMsg)
        conversation.updatedAt = Date()
        if conversation.title == "New Chat" { conversation.title = String(displayContent.prefix(36)) }
        try? modelContext.save()

        UIImpactFeedbackGenerator(style: .soft).impactOccurred()
        appState.isGenerating = true
        streamingText = ""
        streamingSteps = []
        let turnID = UUID()
        let controllerRequestID = UUID()
        activeTurnID = turnID

        let task = Task {
            if !(await ensureChatModelLoaded()) {
                guard activeTurnID == turnID else { return }
                let msg = ChatMessage(role: .assistant, content: "No chat model is loaded. Open the Models tab, download a chat model, and tap Use to activate it.")
                conversation.messages.append(msg)
                try? modelContext.save()
                appState.isGenerating = false
                return
            }
            let routing = await IntentClassifierService.shared.route(text)
            let memories = await safeRecalledMemories(query: text, routing: routing)
            let recentContext = safeShortTermContext(excludingCurrentUserMessageID: userMsg.id)
            if appState.agentModeEnabled {
                await runAgent(turnID: turnID, requestID: controllerRequestID, text: text, routing: routing, memories: memories, attachments: turnAttachments, recentContext: recentContext)
            } else {
                await runPlain(turnID: turnID, requestID: controllerRequestID, text: text, memories: memories, attachments: turnAttachments)
            }
        }
        _ = generationController.begin(for: conversation.id, task: task, requestID: controllerRequestID)
        streamingTask = task
    }

    private func runAgent(turnID: UUID, requestID: UUID, text: String, routing: IntentRoutingDecision, memories: [MemoryContextItem], attachments: [ChatAttachment], recentContext: [(role: MessageRole, content: String)]) async {
        let enabledTools = ToolRegistry.all.filter { appState.enabledToolIDs.contains($0.id) }
        let routedTools = enabledTools.filter { IntentRouter.isToolAllowed($0.id, for: routing) }
        let baseSystemPrompt = conversation.systemPrompt ?? appState.systemPrompt
        let gatedMemories = MemoryGate.filter(intent: routing.intent, items: memories, userMessage: text)
        let req = AgentRequest(
            systemPrompt: baseSystemPrompt,
            history: recentContext,
            userMessage: text,
            temperature: appState.temperature,
            topP: appState.topP,
            repetitionPenalty: appState.repetitionPenalty,
            maxTokens: appState.maxTokens,
            maxSteps: appState.maxAgentSteps,
            availableTools: routedTools,
            relevantMemories: gatedMemories,
            attachments: attachments,
            conversationID: conversation.id,
            turnID: turnID
        )

        var steps: [AgentStep] = []
        var finalText = ""

        for await event in SlotAgentService.shared.run(req, options: .init(modelContext: modelContext, conversationID: conversation.id, turnID: turnID, groundingMode: .slotAgent, allowDegradedGrounding: true, preventDoubleGrounding: true, diagnosticsEnabled: false)) {
            if Task.isCancelled || activeTurnID != turnID || !generationController.isCurrent(requestID, for: conversation.id) { break }
            switch event {
            case .step(let step):
                if let idx = steps.firstIndex(where: { $0.id == step.id }) { steps[idx] = step } else { steps.append(step) }
                streamingSteps = steps
                UIImpactFeedbackGenerator(style: .soft).impactOccurred()
            case .stepDelta(let id, let text):
                if let idx = steps.firstIndex(where: { $0.id == id }) {
                    steps[idx].content = text
                    streamingSteps = steps
                }
            case .finalDelta(let chunk):
                finalText += chunk
                let sanitized = AssistantOutputSanitizer.sanitize(finalText, lastUserMessage: text)
                streamingText = SchemaPlaceholderDetector.isPlaceholderPrefix(sanitized) ? "" : sanitized
            case .done(let final, let allSteps):
                finalText = final.isEmpty ? finalText : final
                steps = allSteps
            case .error(let msg):
                finalText = msg
            }
        }

        guard !Task.isCancelled, activeTurnID == turnID, generationController.isCurrent(requestID, for: conversation.id) else { return }
        finalText = await repairSchemaPlaceholderFinalIfNeeded(finalText, userText: text, routing: routing, memories: memories, attachments: attachments)
        finalText = AssistantOutputSanitizer.sanitize(finalText, lastUserMessage: text)
        finalText = FinalIntentValidator.validate(finalText, routing: routing, fallback: nil)
        let sanitizedSteps = AgentVisibleContentSanitizer.sanitizedSteps(steps)

        let persistedFinal = FinalOutputSanitizer.sanitizeUserVisibleText(finalText).text
        let assistantMsg = ChatMessage(role: .assistant, content: persistedFinal, agentSteps: sanitizedSteps, visibleContent: persistedFinal)
        if appState.developerTraceModeEnabled {
            let trace = makeAgentDeveloperTrace(
                systemPrompt: baseSystemPrompt,
                userPrompt: text,
                modelName: "slot-agent",
                memories: gatedMemories,
                steps: sanitizedSteps,
                visibleAnswer: persistedFinal,
                messageID: assistantMsg.id
            )
            assistantMsg.developerTraceID = trace.id
            assistantMsg.developerTraceJSON = DeveloperTraceCodec.encode(trace)
        }
        conversation.messages.append(assistantMsg)
        if let approvalStep = sanitizedSteps.first(where: { $0.kind == .approvalBoundary }),
           let toolID = approvalStep.toolID {
            let pendingToolMessage = ChatMessage(
                role: .tool,
                content: serializedToolArgs(approvalStep.toolArgs ?? [:]),
                toolName: toolID,
                toolStatus: .pendingApproval,
                toolResult: nil
            )
            conversation.messages.append(pendingToolMessage)
        }
        streamingText = ""
        streamingSteps = []
        activeTurnID = nil
        generationController.clearIfCurrent(requestID, for: conversation.id)

        if appState.autoMemory, persistedFinal.count > 60, isSafeToStoreMemory(userText: text, assistantText: persistedFinal, routing: routing) {
            try? await MemoryStore.remember("User asked: \(text). Assistant: \(String(persistedFinal.prefix(160)))", kind: .conversation, source: "chat", context: modelContext)
            let transient = sanitizedSteps.filter { $0.kind == .observation || $0.kind == .action }.map(\.content)
            await MemoryStore.extractAndStore(userText: text, assistantText: persistedFinal, transientTexts: transient, context: modelContext)
        }

        conversation.updatedAt = Date()
        try? modelContext.save()
        appState.isGenerating = false
    }

    private func serializedToolArgs(_ args: [String: String]) -> String {
        args.keys.sorted().map { key in "\(key): \(args[key] ?? "")" }.joined(separator: ", ")
    }

    private func runPlain(turnID: UUID, requestID: UUID, text: String, memories: [MemoryContextItem], attachments: [ChatAttachment]) async {
        let request = GenerateRequest(
            id: requestID,
            sessionID: conversation.id.uuidString,
            systemPrompt: conversation.systemPrompt ?? appState.systemPrompt,
            history: safeShortTermContext(excludingCurrentUserMessageID: conversation.sortedMessages.last?.id, maxTurns: 8),
            userMessage: text,
            temperature: appState.temperature,
            topP: appState.topP,
            repetitionPenalty: appState.repetitionPenalty,
            maxTokens: appState.maxTokens,
            modelName: conversation.modelName ?? "default",
            relevantMemories: memories,
            attachments: attachments,
            developerTraceModeEnabled: appState.developerTraceModeEnabled,
            reasoningCaptureEnabled: appState.developerReasoningCaptureEnabled
        )

        var accumulated = ""
        for await token in await AppLlamaService.shared.stream(request) {
            if Task.isCancelled || activeTurnID != turnID || !generationController.isCurrent(requestID, for: conversation.id) { break }
            switch token {
            case .text(let s):
                accumulated += s
                streamingText = AssistantOutputSanitizer.sanitize(accumulated, lastUserMessage: text)
            case .done:
                break
            }
        }

        guard !Task.isCancelled, activeTurnID == turnID, generationController.isCurrent(requestID, for: conversation.id) else { return }
        let completedPayload = await AppLlamaService.shared.takeCompletedTracePayload(requestID: request.id)
        let modelVisibleAnswer = completedPayload?.visibleAnswer ?? accumulated
        let sanitized = AssistantOutputSanitizer.sanitize(modelVisibleAnswer, lastUserMessage: text)
        let finalized = FinalOutputSanitizer.sanitizeUserVisibleText(sanitized).text
        let assistantMsg = ChatMessage(
            role: .assistant,
            content: finalized,
            visibleContent: finalized,
            reasoningTrace: appState.developerTraceModeEnabled ? completedPayload?.reasoningText : nil,
            rawModelOutput: appState.developerTraceModeEnabled ? completedPayload?.rawModelOutput : nil
        )
        if appState.developerTraceModeEnabled {
            let trace = makeDeveloperTrace(
                request: request,
                messageID: assistantMsg.id,
                userPrompt: text,
                memories: memories,
                attachments: attachments,
                payload: completedPayload,
                visibleAnswer: finalized,
                error: nil
            )
            assistantMsg.developerTraceID = trace.id
            assistantMsg.developerTraceJSON = DeveloperTraceCodec.encode(trace)
        }
        conversation.messages.append(assistantMsg)
        streamingText = ""
        activeTurnID = nil
        generationController.clearIfCurrent(requestID, for: conversation.id)

        if appState.autoMemory, finalized.count > 60 {
            try? await MemoryStore.remember("User asked: \(text). Assistant said: \(finalized.prefix(140))", kind: .conversation, source: "chat", context: modelContext)
        }

        conversation.updatedAt = Date()
        try? modelContext.save()
        appState.isGenerating = false
    }

    private func safeShortTermContext(excludingCurrentUserMessageID currentID: UUID? = nil, maxTurns: Int = 4) -> [(role: MessageRole, content: String)] {
        conversation.sortedMessages
            .filter { message in
                guard let currentID else { return true }
                return message.id != currentID
            }
            .suffix(maxTurns)
            .compactMap { message in
                guard message.messageRole == .user || message.messageRole == .assistant else { return nil }
                guard let clean = SlotAgentService.sanitizeHistoryEntryForPromptContext(role: message.messageRole, content: message.content) else { return nil }
                return (message.messageRole, clean)
            }
    }

    private func safeRecalledMemories(query: String, routing: IntentRoutingDecision) async -> [MemoryContextItem] {
        await MemoryRecall.recallAndNormalize(query: query, routing: routing, context: modelContext, limit: 8)
    }

    private func makeDeveloperTrace(
        request: GenerateRequest,
        messageID: UUID?,
        userPrompt: String,
        memories: [MemoryContextItem],
        attachments: [ChatAttachment],
        payload: CompletedGenerationTracePayload?,
        visibleAnswer: String,
        error: String?
    ) -> DeveloperTrace {
        DeveloperTrace(
            conversationID: conversation.id,
            messageID: messageID,
            modelName: request.modelName,
            systemPrompt: request.systemPrompt,
            developerPrompt: ModelThinkingControl.developerInstruction(reasoningCaptureEnabled: request.reasoningCaptureEnabled),
            userPrompt: userPrompt,
            resolvedContext: traceContextItems(history: request.history, attachments: attachments),
            retrievedMemory: memories.map(traceMemoryItem),
            toolPlan: [],
            toolCalls: [],
            agentMessages: [],
            rawModelOutput: payload?.rawModelOutput ?? "",
            reasoningText: payload?.reasoningText,
            visibleAnswer: visibleAnswer,
            parserWarnings: payload?.parserWarnings ?? [],
            tokenUsage: payload?.tokenUsage,
            finishReason: payload?.finishReason,
            error: error ?? payload?.error
        )
    }

    private func makeAgentDeveloperTrace(
        systemPrompt: String,
        userPrompt: String,
        modelName: String,
        memories: [MemoryContextItem],
        steps: [AgentStep],
        visibleAnswer: String,
        messageID: UUID?
    ) -> DeveloperTrace {
        let toolPlan = steps.compactMap { step -> TraceToolPlanItem? in
            guard step.kind == .action || step.kind == .approvalBoundary, let toolID = step.toolID else { return nil }
            return TraceToolPlanItem(
                toolID: toolID,
                reason: step.content,
                requiresApproval: step.kind == .approvalBoundary,
                arguments: step.toolArgs ?? [:]
            )
        }
        let toolCalls = steps.compactMap { step -> TraceToolCall? in
            guard step.kind == .action || step.kind == .observation || step.kind == .approvalBoundary,
                  let toolID = step.toolID else { return nil }
            let status: String
            switch step.kind {
            case .action: status = "planned"
            case .approvalBoundary: status = "pendingApproval"
            case .observation: status = "completed"
            case .thought, .reflection: status = "message"
            }
            return TraceToolCall(
                toolID: toolID,
                arguments: step.toolArgs ?? [:],
                status: status,
                result: step.kind == .observation ? step.content : nil
            )
        }
        let agentMessages = steps.map { step in
            TraceAgentMessage(
                id: step.id,
                role: step.kind.rawValue,
                content: step.content,
                toolID: step.toolID,
                metadata: step.toolArgs ?? [:]
            )
        }
        return DeveloperTrace(
            conversationID: conversation.id,
            messageID: messageID,
            modelName: modelName,
            systemPrompt: systemPrompt,
            developerPrompt: SlotAgentService.mouthPromptHygieneRule,
            userPrompt: userPrompt,
            resolvedContext: [],
            retrievedMemory: memories.map(traceMemoryItem),
            toolPlan: toolPlan,
            toolCalls: toolCalls,
            agentMessages: agentMessages,
            rawModelOutput: visibleAnswer,
            reasoningText: nil,
            visibleAnswer: visibleAnswer,
            parserWarnings: [],
            tokenUsage: nil,
            finishReason: "stop",
            error: nil
        )
    }

    private func traceContextItems(
        history: [(role: MessageRole, content: String)],
        attachments: [ChatAttachment]
    ) -> [TraceContextItem] {
        let historyItems = history.map { item in
            TraceContextItem(
                role: item.role.rawValue,
                title: "History",
                content: item.content,
                source: "conversation"
            )
        }
        let attachmentItems = attachments.map { attachment in
            TraceContextItem(
                role: "attachment",
                title: attachment.name,
                content: "Attachment included in prompt assembly.",
                source: attachment.path,
                metadata: [
                    "kind": attachment.kind.rawValue,
                    "byteSize": String(attachment.byteSize)
                ]
            )
        }
        return historyItems + attachmentItems
    }

    private func traceMemoryItem(_ item: MemoryContextItem) -> TraceMemoryItem {
        TraceMemoryItem(
            content: item.content,
            scope: item.scope.rawValue,
            authority: item.authority.rawValue,
            createdAt: item.createdAt,
            expiresAt: item.expiresAt,
            source: item.source,
            topic: item.topic
        )
    }

    private func isSafeToStoreMemory(userText: String, assistantText: String, routing: IntentRoutingDecision) -> Bool {
        FinalIntentValidator.validate(assistantText, routing: routing, fallback: nil) == assistantText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func repairSchemaPlaceholderFinalIfNeeded(_ finalText: String, userText: String, routing: IntentRoutingDecision, memories: [MemoryContextItem], attachments: [ChatAttachment]) async -> String {
        let trimmed = finalText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard SchemaPlaceholderDetector.isPlaceholderFinal(trimmed) else { return finalText }

        if appState.enabledToolIDs.contains("web.search"), routing.intent == .webSearch, shouldUseWebRepair(for: userText) {
            let query = cleanedSearchQuery(userText)
            let result = await WebTools.webSearch(query: query)
            if !isWeakSearchResult(result) { return result }
        }

        return FinalIntentValidator.validate(trimmed, routing: routing, fallback: nil)
    }

    private func shouldUseWebRepair(for userText: String) -> Bool {
        let normalized = userText.lowercased()
        let webMarkers = ["search for", "look up", "research", "web", "internet", "diy", "tutorial", "guide", "how to", "plans", "blueprint", "documentation"]
        return webMarkers.contains { normalized.contains($0) }
    }

    private func cleanedSearchQuery(_ userText: String) -> String {
        var query = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        let prefixes = ["search for ", "search ", "look up ", "research "]
        let lower = query.lowercased()
        for prefix in prefixes where lower.hasPrefix(prefix) {
            query = String(query.dropFirst(prefix.count))
            break
        }
        return query.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func isWeakSearchResult(_ result: String) -> Bool {
        let normalized = result.lowercased()
        if result.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return true }
        return normalized.contains("no direct answer") || normalized.contains("no results") || normalized.contains("search failed") || normalized.contains("need a query")
    }

    private func ensureChatModelLoaded() async -> Bool { await ModelLoader.ensureChatLoaded(appState: appState, stored: storedModels) }

    private func stop() {
        let task = streamingTask
        streamingTask = nil
        let stoppedTurnID = activeTurnID
        activeTurnID = nil
        generationController.cancel(for: conversation.id)
        task?.cancel()
        let captured = AssistantOutputSanitizer.sanitize(streamingText)
        let finalizedCaptured = FinalOutputSanitizer.sanitizeUserVisibleText(captured).text
        let capturedSteps = AgentVisibleContentSanitizer.sanitizedSteps(streamingSteps)
        streamingText = ""
        streamingSteps = []
        Task {
            _ = await task?.value
            await MainActor.run {
                appState.isGenerating = false
                if stoppedTurnID != nil, !finalizedCaptured.isEmpty {
                    let msg = ChatMessage(role: .assistant, content: finalizedCaptured, agentSteps: capturedSteps, wasStopped: true)
                    conversation.messages.append(msg)
                    conversation.updatedAt = Date()
                    try? modelContext.save()
                }
            }
        }
    }
}

struct AttachmentChip: View {
    let attachment: ChatAttachment
    var state: AttachmentRenderState?
    var onRemove: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: attachment.kind.icon).font(.caption).foregroundStyle(Theme.textSecondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(attachment.name).font(.caption).foregroundStyle(Theme.textPrimary).lineLimit(1).truncationMode(.middle)
                if let state, state.truncated { Text(truncationLabel(state)).font(.caption2).foregroundStyle(.orange) }
            }
            Button(action: onRemove) { Image(systemName: "xmark.circle.fill").font(.caption).foregroundStyle(Theme.textTertiary) }.buttonStyle(.plain)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 8))
        .overlay { RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder((state?.truncated ?? false) ? Color.orange.opacity(0.6) : Theme.border, lineWidth: 1) }
        .frame(maxWidth: 240)
    }

    private func truncationLabel(_ s: AttachmentRenderState) -> String {
        guard s.totalChars > 0 else { return "Truncated" }
        let pct = Int((Double(s.includedChars) / Double(s.totalChars)) * 100)
        return "Truncated — \(max(1, pct))% included"
    }
}

struct ChatInputBar: View {
    @Binding var draft: String
    @FocusState var isFocused: Bool
    var isGenerating: Bool
    var onSend: () -> Void
    var onStop: () -> Void
    var onVoice: () -> Void
    var onAttach: () -> Void
    var onDismissKeyboard: () -> Void

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if isFocused {
                Button(action: onDismissKeyboard) { Image(systemName: "keyboard.chevron.compact.down").font(.body).foregroundStyle(Theme.textSecondary).frame(width: 36, height: 36) }.buttonStyle(.plain)
            } else {
                Button(action: onAttach) { Image(systemName: "paperclip").font(.body).foregroundStyle(Theme.textSecondary).frame(width: 36, height: 36) }.buttonStyle(.plain)
            }

            HStack(alignment: .bottom, spacing: 4) {
                TextField("Message Lumen", text: $draft, axis: .vertical)
                    .lineLimit(1...6)
                    .focused($isFocused)
                    .font(.body)
                    .foregroundStyle(Theme.textPrimary)
                    .tint(Theme.accent)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 9)
                if !draft.isEmpty {
                    Button { draft = "" } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(Theme.textTertiary) }.padding(.trailing, 8)
                }
            }
            .background(Theme.surface)
            .clipShape(.rect(cornerRadius: 10))
            .overlay { RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Theme.border, lineWidth: 1) }

            if draft.trimmingCharacters(in: .whitespaces).isEmpty && !isGenerating {
                Button(action: onVoice) {
                    Image(systemName: "waveform")
                        .font(.body)
                        .foregroundStyle(Theme.textPrimary)
                        .frame(width: 36, height: 36)
                        .background(Theme.surface)
                        .clipShape(.rect(cornerRadius: 10))
                        .overlay { RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Theme.border, lineWidth: 1) }
                }.buttonStyle(.plain)
            } else {
                Button { isGenerating ? onStop() : onSend() } label: {
                    Image(systemName: isGenerating ? "stop.fill" : "arrow.up")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                        .frame(width: 36, height: 36)
                        .background(isGenerating ? Color.red.opacity(0.85) : Theme.accent)
                        .clipShape(.rect(cornerRadius: 10))
                }.buttonStyle(.plain)
            }
        }
    }
}
