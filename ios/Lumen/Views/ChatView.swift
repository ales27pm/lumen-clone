import SwiftUI
import SwiftData
import UniformTypeIdentifiers
import CryptoKit

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
    var initialDraft: String? = nil
    var onInitialDraftConsumed: (() -> Void)? = nil
    @Environment(AppState.self) private var appState
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.modelContext) private var modelContext
    @Query private var storedModels: [StoredModel]

    @State private var draft: String = ""
    @State private var streamingText: String = ""
    @State private var streamingSteps: [AgentStep] = []
    @State private var streamingTask: Task<Void, Never>?
    @State private var streamingCancellationID: UUID?
    @State private var activeTurnID: UUID?
    @State private var generationController = GenerationTaskController<UUID>()
    @State private var didApplyInitialDraft = false
    @State private var showVoiceMode = false
    @State private var showFilePicker = false
    @State private var fileImportMessage: String?
    @State private var attachments: [ChatAttachment] = []
    @State private var attachmentPreview: [UUID: AttachmentRenderState] = [:]
    @FocusState private var isFocused: Bool

    var body: some View {
        ZStack {
            LumenBrandAsset(kind: .mark)
                .frame(maxWidth: 560)
                .opacity(displayedMessages.isEmpty ? 0.18 : 0.075)
                .offset(y: -28)
                .allowsHitTesting(false)
                .accessibilityHidden(true)

            VStack(spacing: 0) {
                conversationHeader

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
                .background {
                    Rectangle()
                        .fill(.ultraThinMaterial)
                        .overlay(Theme.background.opacity(0.70))
                        .ignoresSafeArea(edges: .bottom)
                }
            }
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
            switch result {
            case .success(let urls):
                for url in urls {
                    let imported = FileStore.importFileWithDiagnostics(from: url)
                    guard let dest = imported.url else {
                        fileImportMessage = importFailureMessage(diagnostic: imported.diagnostic)
                        continue
                    }
                    guard let attachment = AttachmentResolver.make(from: dest) else {
                        fileImportMessage = "The file was imported, but attachment metadata could not be read."
                        continue
                    }
                    if !attachments.contains(where: { $0.path == attachment.path }) {
                        attachments.append(attachment)
                    }
                }
                if !urls.isEmpty { UIImpactFeedbackGenerator(style: .soft).impactOccurred() }
                recomputeAttachmentPreview()
            case .failure(let error):
                fileImportMessage = "The file picker failed. Diagnostic: picker_failed:\(RuntimeMetricErrorSanitizer.code(for: error))."
            }
        }
        .alert("File import failed", isPresented: Binding(
            get: { fileImportMessage != nil },
            set: { if !$0 { fileImportMessage = nil } }
        )) {
            Button("OK", role: .cancel) { fileImportMessage = nil }
        } message: {
            Text(fileImportMessage ?? "")
        }
        .onChange(of: draft) { _, _ in recomputeAttachmentPreview() }
        .onChange(of: scenePhase) { _, phase in
            SceneTransitionCoordinator.shared.handleScenePhaseChange(phase)
        }
        .onAppear(perform: applyInitialDraftIfNeeded)
        .onDisappear { stopForSceneTransition() }
    }

    private func importFailureMessage(diagnostic: String?) -> String {
        "Could not import the selected file. Diagnostic: \(diagnostic ?? "import_failed")."
    }

    private var conversationHeader: some View {
        HStack(spacing: 12) {
            LumenBrandAsset(kind: .assistantMark, accessibilityLabel: "Lumen")
                .frame(width: 42, height: 42)
                .clipShape(.rect(cornerRadius: 12))
                .shadow(color: LumenBrand.lumen.opacity(0.22), radius: 12, y: 6)

            VStack(alignment: .leading, spacing: 3) {
                Text(conversation.title)
                    .font(.headline.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                    .lineLimit(1)
                Text(activeModelSummary)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 8)

            if horizontalSizeClass != .compact {
                HStack(spacing: 6) {
                    LumenStatusChip(
                        title: appState.agentModeEnabled ? "Agent" : "Chat",
                        systemImage: appState.agentModeEnabled ? "wand.and.stars" : "text.bubble",
                        tint: appState.agentModeEnabled ? Theme.accent : LumenBrand.corona
                    )
                    if appState.autoMemory {
                        LumenStatusChip(title: "Memory", systemImage: "brain", tint: LumenBrand.violet)
                    }
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 10)
        .padding(.bottom, 8)
        .background {
            Rectangle()
                .fill(.ultraThinMaterial)
                .overlay(Theme.background.opacity(0.48))
                .ignoresSafeArea(edges: .top)
        }
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(Theme.border)
                .frame(height: 1)
        }
    }

    private var activeModelSummary: String {
        if appState.isGenerating {
            return "Streaming from \(activeModelName)"
        }
        return activeModelName
    }

    private var activeModelName: String {
        storedModels.first { $0.id.uuidString == appState.activeChatModelID }?.name
            ?? conversation.modelName
            ?? "No active model"
    }

    private func applyInitialDraftIfNeeded() {
        guard !didApplyInitialDraft, draft.isEmpty, let initialDraft else { return }
        didApplyInitialDraft = true
        draft = initialDraft
        isFocused = true
        onInitialDraftConsumed?()
    }

    private var displayedMessages: [ChatMessage] {
        let sorted = conversation.sortedMessages
        let renderLimit = 120
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
        saveConversationIfBudgetAllows(estimatedBytes: displayContent.utf8.count + 2048)

        UIImpactFeedbackGenerator(style: .soft).impactOccurred()
        appState.isGenerating = true
        streamingText = ""
        streamingSteps = []
        let turnID = UUID()
        let controllerRequestID = UUID()
        activeTurnID = turnID
        DeferredMaintenanceQueue.shared.setChatOrVoiceActive(true)

        let task = Task {
            let cpuToken = CPUWatchdogGuard.shared.begin(category: .chatGeneration)
            defer {
                CPUWatchdogGuard.shared.end(token: cpuToken)
                DeferredMaintenanceQueue.shared.setChatOrVoiceActive(false)
            }
            if appState.agentModeEnabled {
                streamingSteps = [AgentStep(kind: .thought, content: "Loading local model")]
            }
            if !(await ensureChatModelLoaded()) {
                guard activeTurnID == turnID else { return }
                let msg = ChatMessage(role: .assistant, content: "No chat model is loaded. Open the Models tab, download a chat model, and tap Use to activate it.")
                conversation.messages.append(msg)
                saveConversationIfBudgetAllows(estimatedBytes: 4096)
                streamingSteps = []
                activeTurnID = nil
                generationController.clearIfCurrent(controllerRequestID, for: conversation.id)
                appState.isGenerating = false
                return
            }
            if appState.agentModeEnabled {
                streamingSteps = [AgentStep(kind: .thought, content: "Preparing agent context")]
            }
            AgentGroundingInstrumentation.mark("before IntentClassifierService.route", metrics: .init(promptChars: text.count))
            let routeStart = ProcessInfo.processInfo.systemUptime
            let routing = await IntentClassifierService.shared.route(text)
            AgentGroundingInstrumentation.mark("after IntentClassifierService.route", metrics: .init(promptChars: text.count), elapsedMs: AgentGroundingInstrumentation.elapsedMs(since: routeStart))
            let memories = await safeRecalledMemories(query: text, routing: routing)
            let recentContext = safeShortTermContext(excludingCurrentUserMessageID: userMsg.id)
            if appState.agentModeEnabled {
                await runAgent(turnID: turnID, requestID: controllerRequestID, text: text, routing: routing, memories: memories, attachments: turnAttachments, recentContext: recentContext)
            } else {
                await runPlain(turnID: turnID, requestID: controllerRequestID, text: text, memories: memories, attachments: turnAttachments)
            }
        }
        _ = generationController.begin(for: conversation.id, task: task, requestID: controllerRequestID)
        streamingCancellationID = AppCancellationBus.shared.register(task, category: .chatGeneration)
        streamingTask = task
    }

    private func runAgent(turnID: UUID, requestID: UUID, text: String, routing: IntentRoutingDecision, memories: [MemoryContextItem], attachments: [ChatAttachment], recentContext: [(role: MessageRole, content: String)]) async {
        emitChatViewTrace(turnID: turnID, phase: "chat_agent_start", text: text, values: [
            "path": "chat-view-agent",
            "intent": routing.intent.rawValue,
            "attachmentCount": String(attachments.count),
            "memoryCount": String(memories.count)
        ])
        let baseSystemPrompt = conversation.systemPrompt ?? appState.systemPrompt
        let gatedMemories = MemoryGate.filter(intent: routing.intent, items: memories, userMessage: text)
        let kernelHistory = recentContext.map { item in
            AgentKernelMessage(messageRole: item.role, content: item.content)
        }
        let kernelRequest = AgentKernelRequest(
            conversationID: conversation.id,
            turnID: turnID,
            userMessage: text,
            history: kernelHistory,
            systemPrompt: baseSystemPrompt,
            relevantMemories: gatedMemories,
            attachments: attachments,
            task: .chat,
            source: .chat,
            options: AgentKernelOptions(
                allowHeavyRuntime: true,
                allowDegradedMode: true,
                requireUserVisibleFinal: true,
                diagnosticsEnabled: false,
                maxSteps: appState.maxAgentSteps,
                prefersFoundationModels: true,
                temperature: appState.temperature,
                topP: appState.topP,
                repetitionPenalty: appState.repetitionPenalty,
                maxTokens: appState.maxTokens
            )
        )

        var kernelEventState = ChatKernelEventState()

        var lastUIUpdate = Date.distantPast
        let kernel = AssistantKernel.shared
        for await kernelEvent in kernel.run(kernelRequest, modelContext: modelContext) {
            if Task.isCancelled || activeTurnID != turnID || !generationController.isCurrent(requestID, for: conversation.id) || CPUWatchdogGuard.shared.shouldDegrade(category: .chatGeneration) || !ResourceBudgetGate.allowsHeavyModelWork(reason: "userChat.stream") { break }
            let workStartedAt = ProcessInfo.processInfo.systemUptime
            defer { CPUWatchdogGuard.shared.recordWork(category: .chatGeneration, duration: ProcessInfo.processInfo.systemUptime - workStartedAt) }
            let mutation = ChatKernelEventReducer.reduce(kernelEvent, state: &kernelEventState, lastUserMessage: text)
            if mutation.stepsChanged, Date().timeIntervalSince(lastUIUpdate) >= 0.1 {
                streamingSteps = AgentStepContentBudget.boundedSanitizedSteps(kernelEventState.steps)
                lastUIUpdate = Date()
                if mutation.shouldEmitStepFeedback {
                    UIImpactFeedbackGenerator(style: .soft).impactOccurred()
                }
            }
            if mutation.textChanged, Date().timeIntervalSince(lastUIUpdate) >= 0.1 {
                streamingText = kernelEventState.streamingText
                if mutation.shouldEmitUIUpdateDiagnostic {
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .uiUpdate, values: ["surface": "chat", "targetHz": "10"]))
                }
                lastUIUpdate = Date()
            }
        }

        guard !Task.isCancelled, activeTurnID == turnID, generationController.isCurrent(requestID, for: conversation.id) else {
            emitChatViewTrace(turnID: turnID, phase: "cancelled", text: text, values: ["path": "chat-view-agent", "reason": chatCancellationReason(turnID: turnID, requestID: requestID)])
            return
        }
        var finalText = kernelEventState.finalText
        let steps = kernelEventState.steps
        finalText = await repairSchemaPlaceholderFinalIfNeeded(finalText, userText: text, routing: routing, memories: memories, attachments: attachments)
        finalText = AssistantOutputSanitizer.sanitize(finalText, lastUserMessage: text)
        finalText = FinalIntentValidator.validate(finalText, routing: routing, fallback: nil)
        let sanitizedSteps = AgentStepContentBudget.boundedSanitizedSteps(steps)

        let persistedFinal = FinalOutputSanitizer.sanitizeUserVisibleText(finalText).text
        emitChatViewTrace(turnID: turnID, phase: "chat_agent_final", text: text, values: [
            "path": "chat-view-agent",
            "stepCount": String(sanitizedSteps.count),
            "finalChars": String(persistedFinal.count),
            "finalSHA256": chatTraceSHA256(persistedFinal)
        ])
        let assistantMsg = ChatMessage(role: .assistant, content: persistedFinal, agentSteps: sanitizedSteps, visibleContent: persistedFinal)
        #if DEBUG
        if appState.developerTraceModeEnabled {
            let trace = makeAgentDeveloperTrace(
                systemPrompt: baseSystemPrompt,
                userPrompt: text,
                modelName: "agent-kernel",
                memories: gatedMemories,
                steps: sanitizedSteps,
                visibleAnswer: persistedFinal,
                messageID: assistantMsg.id
            )
            assistantMsg.developerTraceID = trace.id
            assistantMsg.developerTraceJSON = DeveloperTraceCodec.encode(trace)
        }
        #endif
        conversation.messages.append(assistantMsg)
        if let approvalStep = sanitizedSteps.first(where: { $0.kind == .approvalBoundary }),
           let pendingToolMessage = ChatApprovalBoundaryMapper.pendingToolMessage(for: approvalStep) {
            conversation.messages.append(pendingToolMessage)
        }
        streamingText = ""
        streamingSteps = []
        activeTurnID = nil
        generationController.clearIfCurrent(requestID, for: conversation.id)

        if appState.autoMemory, persistedFinal.count > 60, isSafeToStoreMemory(userText: text, assistantText: persistedFinal, routing: routing) {
            let memoryResult = await MemoryStore.rememberWithDiagnostics("User asked: \(text). Assistant: \(String(persistedFinal.prefix(160)))", kind: .conversation, source: "chat", context: modelContext)
            recordAutoMemoryResult(memoryResult, surface: "chat", turnID: turnID, userText: text, path: "chat-view-kernel")
            let transient = sanitizedSteps.filter { $0.kind == .observation || $0.kind == .action }.map(\.content)
            await MemoryStore.extractAndStore(userText: text, assistantText: persistedFinal, transientTexts: transient, context: modelContext)
        }

        conversation.updatedAt = Date()
        saveConversationIfBudgetAllows(estimatedBytes: persistedFinal.utf8.count + 4096)
        appState.isGenerating = false
    }


    private func recordAutoMemoryResult(
        _ result: MemoryStore.RememberResult,
        surface: String,
        turnID: UUID,
        userText: String,
        path: String
    ) {
        guard result.mode != "stored" else { return }
        emitChatViewTrace(turnID: turnID, phase: "auto_memory_\(result.mode)", text: userText, values: [
            "path": path,
            "surface": surface,
            "memoryMode": result.mode,
            "memoryDiagnostic": result.diagnostic ?? "none"
        ])
    }


    private var debugDeveloperTraceEnabled: Bool {
        #if DEBUG
        return appState.developerTraceModeEnabled
        #else
        return false
        #endif
    }

    private var debugDeveloperReasoningCaptureEnabled: Bool {
        #if DEBUG
        return appState.developerReasoningCaptureEnabled
        #else
        return false
        #endif
    }

    private func runPlain(turnID: UUID, requestID: UUID, text: String, memories: [MemoryContextItem], attachments: [ChatAttachment]) async {
        emitChatViewTrace(turnID: turnID, phase: "plain_start", text: text, values: [
            "path": "chat-view-plain",
            "attachmentCount": String(attachments.count),
            "memoryCount": String(memories.count)
        ])
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
            developerTraceModeEnabled: debugDeveloperTraceEnabled,
            reasoningCaptureEnabled: debugDeveloperReasoningCaptureEnabled
        )

        var accumulated = ""
        var lastUIUpdate = Date.distantPast
        for await token in await AppLlamaService.shared.stream(request) {
            if Task.isCancelled || activeTurnID != turnID || !generationController.isCurrent(requestID, for: conversation.id) || CPUWatchdogGuard.shared.shouldDegrade(category: .chatGeneration) || !ResourceBudgetGate.allowsHeavyModelWork(reason: "userChat.stream") { break }
            let workStartedAt = ProcessInfo.processInfo.systemUptime
            defer { CPUWatchdogGuard.shared.recordWork(category: .chatGeneration, duration: ProcessInfo.processInfo.systemUptime - workStartedAt) }
            switch token {
            case .text(let s):
                accumulated += s
                if Date().timeIntervalSince(lastUIUpdate) >= 0.1 {
                    streamingText = AssistantOutputSanitizer.sanitize(accumulated, lastUserMessage: text)
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .uiUpdate, values: ["surface": "chat", "targetHz": "10"]))
                    lastUIUpdate = Date()
                }
            case .done:
                break
            }
        }

        guard !Task.isCancelled, activeTurnID == turnID, generationController.isCurrent(requestID, for: conversation.id) else {
            emitChatViewTrace(turnID: turnID, phase: "cancelled", text: text, values: ["path": "chat-view-plain", "reason": chatCancellationReason(turnID: turnID, requestID: requestID)])
            return
        }
        let completedPayload = await AppLlamaService.shared.takeCompletedTracePayload(requestID: request.id)
        let modelVisibleAnswer = completedPayload?.visibleAnswer ?? accumulated
        let sanitized = AssistantOutputSanitizer.sanitize(modelVisibleAnswer, lastUserMessage: text)
        let finalized = FinalOutputSanitizer.sanitizeUserVisibleText(sanitized).text
        emitChatViewTrace(turnID: turnID, phase: "plain_final", text: text, values: [
            "path": "chat-view-plain",
            "modelName": request.modelName,
            "finalChars": String(finalized.count),
            "finalSHA256": chatTraceSHA256(finalized),
            "finishReason": completedPayload?.finishReason ?? "unknown",
            "parserWarningCount": String(completedPayload?.parserWarnings.count ?? 0)
        ])
        let assistantMsg = ChatMessage(
            role: .assistant,
            content: finalized,
            visibleContent: finalized,
            reasoningTrace: debugDeveloperTraceEnabled ? completedPayload?.reasoningText : nil,
            rawModelOutput: debugDeveloperTraceEnabled ? completedPayload?.rawModelOutput : nil
        )
        #if DEBUG
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
        #endif
        conversation.messages.append(assistantMsg)
        streamingText = ""
        activeTurnID = nil
        generationController.clearIfCurrent(requestID, for: conversation.id)

        if appState.autoMemory, finalized.count > 60 {
            let memoryResult = await MemoryStore.rememberWithDiagnostics("User asked: \(text). Assistant said: \(finalized.prefix(140))", kind: .conversation, source: "chat", context: modelContext)
            recordAutoMemoryResult(memoryResult, surface: "chat", turnID: turnID, userText: text, path: "chat-view-plain")
        }

        conversation.updatedAt = Date()
        saveConversationIfBudgetAllows(estimatedBytes: finalized.utf8.count + 4096)
        appState.isGenerating = false
    }

    private func emitChatViewTrace(turnID: UUID, phase: String, text: String, values: [String: String] = [:]) {
        var payload = values
        LumenTrainedModelRuntimeRegistry.selected.traceValues.forEach { key, value in
            payload[key] = value
        }
        payload["phase"] = phase
        payload["schemaVersion"] = "lumen.chat_runtime_trace/1.0.0"
        payload["conversationID"] = conversation.id.uuidString
        payload["turnID"] = turnID.uuidString
        payload["promptChars"] = String(text.count)
        payload["promptBytes"] = String(text.utf8.count)
        payload["promptSHA256"] = chatTraceSHA256(text)
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .chatRuntimeTrace, values: payload))
    }

    private func chatTraceSHA256(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private func chatCancellationReason(turnID: UUID, requestID: UUID) -> String {
        if Task.isCancelled { return "task-cancelled" }
        if activeTurnID != turnID { return "turn-replaced" }
        if !generationController.isCurrent(requestID, for: conversation.id) { return "request-replaced" }
        if let reason = AppCancellationBus.shared.lastCancellationReason { return reason }
        return "generation-stopped"
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
        AgentGroundingInstrumentation.mark("before safeRecalledMemories", metrics: .init(promptChars: query.count))
        let start = ProcessInfo.processInfo.systemUptime
        let memories = await MemoryRecall.recallAndNormalize(query: query, routing: routing, context: modelContext, limit: 8)
        AgentGroundingInstrumentation.mark("after safeRecalledMemories", metrics: .init(memoryCount: memories.count, promptChars: query.count), elapsedMs: AgentGroundingInstrumentation.elapsedMs(since: start))
        return memories
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
            developerPrompt: "Agent Kernel chat stream",
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
                content: "history_chars=\(item.content.count);sha256=\(String(chatTraceSHA256(item.content).prefix(16)))",
                source: "conversation"
            )
        }
        let attachmentItems = attachments.map { attachment in
            TraceContextItem(
                role: "attachment",
                title: "Attachment",
                content: "Attachment included in prompt assembly.",
                source: "attachment_path_sha256=\(String(chatTraceSHA256(attachment.path).prefix(16)))",
                metadata: [
                    "kind": attachment.kind.rawValue,
                    "byteSize": String(attachment.byteSize),
                    "nameSHA256": String(chatTraceSHA256(attachment.name).prefix(16))
                ]
            )
        }
        return historyItems + attachmentItems
    }

    private func traceMemoryItem(_ item: MemoryContextItem) -> TraceMemoryItem {
        TraceMemoryItem(
            content: AgentDiagnosticFileRedactor.summary(label: "memory", text: item.content),
            scope: item.scope.rawValue,
            authority: item.authority.rawValue,
            createdAt: item.createdAt,
            expiresAt: item.expiresAt,
            source: item.source.map { AgentDiagnosticFileRedactor.summary(label: "source", text: $0) },
            topic: item.topic.map { AgentDiagnosticFileRedactor.summary(label: "topic", text: $0) }
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

    private func ensureChatModelLoaded() async -> Bool { await ModelLoader.ensureChatLoaded(appState: appState, stored: storedModels, intent: .userChat) }

    private func stopForSceneTransition() {
        let task = streamingTask
        streamingTask = nil
        activeTurnID = nil
        generationController.cancel(for: conversation.id)
        task?.cancel()
        if let streamingCancellationID {
            AppCancellationBus.shared.unregister(streamingCancellationID, category: .chatGeneration)
            self.streamingCancellationID = nil
        }
        appState.isGenerating = false
        DeferredMaintenanceQueue.shared.setChatOrVoiceActive(false)
        RuntimeLifecycleCanceller.cancelForSceneTransition(reason: "chat-scene")
    }

    private func saveConversationIfBudgetAllows(estimatedBytes: Int) {
        guard !DiskWriteBudget.shared.shouldDefer(bytes: estimatedBytes, category: .conversation) else {
            DeferredMaintenanceQueue.shared.enqueue(DeferredMaintenanceJob(key: "conversation-save-\(conversation.id)", category: .conversation, staleAfter: 10 * 60, maxRuntime: 2) { @MainActor in
                try? modelContext.save()
            })
            return
        }
        try? modelContext.save()
        DiskWriteBudget.shared.recordWrite(bytes: estimatedBytes, category: .conversation)
    }

    private func stop() {
        let task = streamingTask
        streamingTask = nil
        let stoppedTurnID = activeTurnID
        activeTurnID = nil
        generationController.cancel(for: conversation.id)
        task?.cancel()
        if let streamingCancellationID {
            AppCancellationBus.shared.unregister(streamingCancellationID, category: .chatGeneration)
            self.streamingCancellationID = nil
        }
        DeferredMaintenanceQueue.shared.setChatOrVoiceActive(false)
        let captured = AssistantOutputSanitizer.sanitize(streamingText)
        let finalizedCaptured = FinalOutputSanitizer.sanitizeUserVisibleText(captured).text
        let capturedSteps = AgentStepContentBudget.boundedSanitizedSteps(streamingSteps)
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
                    saveConversationIfBudgetAllows(estimatedBytes: finalizedCaptured.utf8.count + 4096)
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
                if let state, let label = stateLabel(state) {
                    Text(label)
                        .font(.caption2)
                        .foregroundStyle(state.extractionFailed ? Color.red : Color.orange)
                }
            }
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(Theme.textTertiary)
                    .frame(width: 30, height: 30)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Remove \(attachment.name)")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .frame(minHeight: 44)
        .background(Theme.surfaceHigh)
        .clipShape(.rect(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(borderColor, lineWidth: 1) }
        .frame(maxWidth: 240)
    }

    private var borderColor: Color {
        if state?.extractionFailed == true { return Color.red.opacity(0.65) }
        if state?.truncated == true || state?.emptyExtractedText == true { return Color.orange.opacity(0.6) }
        return Theme.border
    }

    private func stateLabel(_ s: AttachmentRenderState) -> String? {
        if s.extractionFailed { return "Unreadable" }
        if s.emptyExtractedText { return "No extractable text" }
        if s.truncated { return truncationLabel(s) }
        return nil
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
            LumenIconControl(
                systemImage: isFocused ? "keyboard.chevron.compact.down" : "paperclip",
                accessibilityLabel: isFocused ? "Dismiss keyboard" : "Attach file",
                action: isFocused ? onDismissKeyboard : onAttach
            )

            HStack(alignment: .bottom, spacing: 4) {
                TextField("Ask Lumen", text: $draft, axis: .vertical)
                    .lineLimit(1...6)
                    .focused($isFocused)
                    .font(.body)
                    .foregroundStyle(Theme.textPrimary)
                    .tint(Theme.accent)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 11)
                    .frame(minHeight: 44)
                if !draft.isEmpty {
                    Button { draft = "" } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(Theme.textTertiary)
                            .frame(width: 34, height: 44)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Clear message")
                    .padding(.trailing, 6)
                }
            }
            .background(Theme.surfaceHigh)
            .clipShape(.rect(cornerRadius: 14))
            .overlay { RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(Theme.border, lineWidth: 1) }

            if draft.trimmingCharacters(in: .whitespaces).isEmpty && !isGenerating {
                LumenIconControl(
                    systemImage: "waveform",
                    accessibilityLabel: "Start voice mode",
                    tint: Theme.textPrimary,
                    action: onVoice
                )
            } else {
                Button { isGenerating ? onStop() : onSend() } label: {
                    Image(systemName: isGenerating ? "stop.fill" : "arrow.up")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(isGenerating ? .white : LumenBrand.midnight)
                        .frame(width: 44, height: 44)
                        .background(isGenerating ? Color.red.opacity(0.85) : Theme.accent)
                        .clipShape(.rect(cornerRadius: 12))
                        .overlay {
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .strokeBorder(isGenerating ? Color.red.opacity(0.32) : Theme.accent.opacity(0.35), lineWidth: 1)
                        }
                }
                .buttonStyle(.plain)
                .accessibilityLabel(isGenerating ? "Stop generating" : "Send message")
            }
        }
        .padding(8)
        .background {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(.ultraThinMaterial)
                .overlay(Theme.surface)
        }
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        }
    }
}
