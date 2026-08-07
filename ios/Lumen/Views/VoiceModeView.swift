import SwiftUI
import SwiftData

struct VoiceModeView: View {
    var onTranscript: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @Environment(AppState.self) private var appState
    @Environment(\.scenePhase) private var scenePhase
    @State private var session = VoiceSessionController()
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Conversation.updatedAt, order: .reverse) private var conversations: [Conversation]
    @Query private var storedModels: [StoredModel]

    @State private var phase: Phase = .idle
    @State private var responseText: String = ""
    @State private var responseTask: Task<Void, Never>?
    @State private var responseCancellationID: UUID?
    @State private var spokenPrefix: Int = 0
    @State private var finishedStreaming = false
    @State private var stepsBuffer: [AgentStep] = []
    @State private var activeVoiceTurnID: UUID?
    @State private var activeSpeechTurnID: UUID?
    @State private var speechEndObserverTask: Task<Void, Never>?
    @State private var generationController = GenerationTaskController<String>()
    @State private var conversationPersistenceFailure: ConversationPersistenceFailure?

    enum Phase { case idle, listening, thinking, speaking }

    var body: some View {
        ZStack {
            AppBackground()

            VStack(spacing: 24) {
                HStack {
                    Button { close() } label: {
                        Image(systemName: "xmark")
                            .font(.body)
                            .foregroundStyle(Theme.textPrimary)
                            .frame(width: 36, height: 36)
                            .background(Theme.surface)
                            .clipShape(.rect(cornerRadius: 10))
                            .overlay {
                                RoundedRectangle(cornerRadius: 10, style: .continuous)
                                    .strokeBorder(Theme.border, lineWidth: 1)
                            }
                    }
                    .buttonStyle(.plain)
                    Spacer()
                    statusPill
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)

                Spacer()

                VoiceWaveform(level: VoiceService.shared.inputLevel, phase: phase)
                    .frame(height: 120)
                    .padding(.horizontal, 40)

                VStack(spacing: 10) {
                    Text(statusTitle)
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(Theme.textPrimary)
                    Text(transcriptText)
                        .font(.body)
                        .multilineTextAlignment(.center)
                        .foregroundStyle(Theme.textSecondary)
                        .padding(.horizontal, 24)
                        .frame(minHeight: 72)
                        .animation(.easeInOut(duration: 0.2), value: transcriptText)
                }

                Spacer()

                HStack(spacing: 32) {
                    Button { toggleHandsFree() } label: {
                        VStack(spacing: 4) {
                            Image(systemName: appState.handsFree ? "infinity.circle.fill" : "infinity")
                                .font(.title3)
                                .foregroundStyle(appState.handsFree ? Theme.accent : Theme.textSecondary)
                            Text("Hands-free").font(.caption2).foregroundStyle(Theme.textTertiary)
                        }
                    }
                    .buttonStyle(.plain)

                    Button { mainAction() } label: {
                        Image(systemName: mainIcon)
                            .font(.system(size: 26, weight: .semibold))
                            .foregroundStyle(.white)
                            .frame(width: 72, height: 72)
                            .background(mainButtonColor)
                            .clipShape(Circle())
                    }
                    .buttonStyle(.plain)

                    Button { interrupt() } label: {
                        VStack(spacing: 4) {
                            Image(systemName: "hand.raised")
                                .font(.title3)
                                .foregroundStyle(phase == .speaking ? Color(red: 0.95, green: 0.5, blue: 0.5) : Theme.textTertiary)
                            Text("Interrupt").font(.caption2).foregroundStyle(Theme.textTertiary)
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(phase != .speaking)
                }
                .padding(.bottom, 40)
            }
        }
        .preferredColorScheme(.dark)
        .onAppear { generationController.startupIfNeeded(for: "voice") { } }
        .onChange(of: scenePhase) { _, phase in
            SceneTransitionCoordinator.shared.handleScenePhaseChange(phase)
        }
        .onDisappear { cleanup() }
        .alert(item: $conversationPersistenceFailure) { failure in
            Alert(
                title: Text("Conversation not saved"),
                message: Text(failure.userMessage),
                primaryButton: .default(Text("Retry")) {
                    retryVoiceConversationSave(failure)
                },
                secondaryButton: .cancel(Text("Dismiss"))
            )
        }
    }

    private var statusTitle: String {
        if case .denied = session.state { return "Permission denied" }
        if case .interrupted = session.state { return "Interrupted" }
        if case .failed(let reason) = session.state { return reason }
        return switch phase {
        case .idle: "Tap to speak"
        case .listening: "Listening"
        case .thinking: "Thinking"
        case .speaking: "Speaking"
        }
    }

    private var transcriptText: String {
        switch phase {
        case .listening: session.partialTranscript.isEmpty ? "Say something — Lumen is listening." : session.partialTranscript
        case .thinking: session.finalTranscript.isEmpty ? session.partialTranscript : session.finalTranscript
        case .speaking: responseText
        case .idle:
            switch session.state {
            case .denied(let reason): "Permission denied: \(reason). Check Settings and try again."
            case .interrupted: "Voice session interrupted. Tap the mic to start again."
            case .failed(let reason): "Voice error: \(reason)"
            default: "Tap the mic to start."
            }
        }
    }

    private var statusPill: some View {
        HStack(spacing: 6) {
            StatusDot(color: phaseColor, size: 6)
            Text(statusTitle)
                .font(.caption.weight(.medium))
                .foregroundStyle(Theme.textSecondary)
        }
        .padding(.horizontal, 10).padding(.vertical, 5)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8).strokeBorder(Theme.border, lineWidth: 1)
        }
    }

    private var phaseColor: Color {
        switch phase {
        case .idle: Theme.textTertiary
        case .listening: Color(red: 0.5, green: 0.85, blue: 0.6)
        case .thinking: Color(red: 0.95, green: 0.75, blue: 0.4)
        case .speaking: Theme.accent
        }
    }

    private var mainIcon: String {
        switch phase {
        case .idle: "mic.fill"
        case .listening: "checkmark"
        case .thinking: "stop.fill"
        case .speaking: "stop.fill"
        }
    }

    private var mainButtonColor: Color {
        switch phase {
        case .idle: Theme.accent
        case .listening: Color(red: 0.4, green: 0.75, blue: 0.55)
        case .thinking: Color(red: 0.85, green: 0.65, blue: 0.35)
        case .speaking: Color(red: 0.9, green: 0.45, blue: 0.45)
        }
    }

    private func mainAction() {
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        switch phase {
        case .idle: startListening()
        case .listening: finishListening()
        case .thinking, .speaking: interrupt()
        }
    }

    private func startListening() {
        cancelSpeechEndObservation()
        session.stopSpeaking()
        responseText = ""
        activeVoiceTurnID = nil
        activeSpeechTurnID = nil
        Task {
            await session.startPushToTalk { text in
                Task { @MainActor in handleTranscript(text) }
            }
            syncPhaseFromSession()
        }
    }

    private func finishListening() {
        session.finishListening()
        syncPhaseFromSession()
    }

    private func syncPhaseFromSession() {
        switch session.state {
        case .idle, .denied, .interrupted, .failed:
            phase = .idle
        case .requestingPermissions, .listening:
            phase = .listening
        case .processing:
            phase = .thinking
        case .speaking:
            phase = .speaking
        }
    }

    private func handleTranscript(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            if appState.handsFree { startListening() } else { phase = .idle }
            return
        }
        phase = .thinking
        runAgent(text: trimmed)
    }

    private func runAgent(text: String) {
        responseText = ""
        spokenPrefix = 0
        finishedStreaming = false
        stepsBuffer = []
        let turnID = UUID()
        let controllerRequestID = UUID()
        activeVoiceTurnID = turnID
        activeSpeechTurnID = turnID
        DeferredMaintenanceQueue.shared.setChatOrVoiceActive(true)

        let task = Task {
            let cpuToken = CPUWatchdogGuard.shared.begin(category: .voice)
            defer {
                CPUWatchdogGuard.shared.end(token: cpuToken)
                DeferredMaintenanceQueue.shared.setChatOrVoiceActive(false)
            }
            let convo = conversations.first ?? {
                let c = Conversation(title: String(text.prefix(36)), systemPrompt: appState.systemPrompt)
                modelContext.insert(c)
                return c
            }()
            let userMsg = ChatMessage(role: .user, content: text)
            convo.messages.append(userMsg)
            guard await ensureChatModelLoaded() else {
                guard activeVoiceTurnID == turnID else { return }
                let fallback = "No chat model is loaded. Open the Models tab, download a chat model, and tap Use to activate it."
                responseText = fallback
                finishedStreaming = true
                speakPending(turnID: turnID)
                let assistantMsg = ChatMessage(role: .assistant, content: fallback)
                convo.messages.append(assistantMsg)
                convo.updatedAt = Date()
                saveVoiceConversationIfBudgetAllows(estimatedBytes: fallback.utf8.count + text.utf8.count + 4096)
                activeVoiceTurnID = nil
                generationController.clearIfCurrent(controllerRequestID, for: "voice")
                return
            }

            AgentGroundingInstrumentation.mark("before IntentClassifierService.route", metrics: .init(promptChars: text.count))
            let routeStart = ProcessInfo.processInfo.systemUptime
            let routing = await IntentClassifierService.shared.route(text)
            AgentGroundingInstrumentation.mark("after IntentClassifierService.route", metrics: .init(promptChars: text.count), elapsedMs: AgentGroundingInstrumentation.elapsedMs(since: routeStart))
            let memories = await safeRecalledMemories(query: text, routing: routing)
            let history = safeShortTermContext(in: convo, excludingCurrentUserMessageID: userMsg.id)

            var voiceEventState = VoiceKernelEventState()
            var lastUIUpdate = Date.distantPast
            let eventStream = VoiceAgentRuntimeBridge.streamVoiceTurn(
                text: text,
                appState: appState,
                routing: routing,
                memories: memories,
                history: history,
                conversationID: convo.id,
                turnID: turnID,
                modelContext: modelContext
            )
            for await kernelEvent in eventStream {
                if let cancellationReason = voiceCancellationReason(turnID: turnID, requestID: controllerRequestID) {
                    _ = VoiceKernelEventReducer.cancel(state: &voiceEventState, reason: cancellationReason)
                    break
                }
                let workStartedAt = ProcessInfo.processInfo.systemUptime
                defer { CPUWatchdogGuard.shared.recordWork(category: .voice, duration: ProcessInfo.processInfo.systemUptime - workStartedAt) }
                let mutation = VoiceKernelEventReducer.reduce(kernelEvent, state: &voiceEventState, lastUserMessage: text, routing: routing)
                if mutation.stepsChanged {
                    stepsBuffer = AgentStepContentBudget.boundedSanitizedSteps(voiceEventState.steps)
                    if mutation.shouldEmitStepFeedback {
                        UIImpactFeedbackGenerator(style: .soft).impactOccurred()
                    }
                }
                if mutation.textChanged, Date().timeIntervalSince(lastUIUpdate) >= 0.1 {
                    responseText = mutation.shouldUseFinalResponseText
                        ? voiceEventState.responseText
                        : VoiceKernelEventReducer.streamingResponseText(from: voiceEventState.finalText, lastUserMessage: text)
                    if mutation.shouldEmitUIUpdateDiagnostic {
                        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .uiUpdate, values: ["surface": "voice", "targetHz": "1"]))
                    }
                    lastUIUpdate = Date()
                    if mutation.shouldStartSpeaking {
                        if phase != .speaking { phase = .speaking }
                        session.startSpeaking()
                    }
                    if mutation.shouldSpeakPending {
                        speakPending(turnID: turnID)
                    }
                }
            }

            guard !Task.isCancelled, !voiceEventState.isCancelled, activeVoiceTurnID == turnID, generationController.isCurrent(controllerRequestID, for: "voice") else { return }
            finishedStreaming = true
            let finalText = VoiceKernelEventReducer.finalResponseText(from: voiceEventState.finalText, lastUserMessage: text, routing: routing)
            let persistedFinal = FinalOutputSanitizer.sanitizeUserVisibleText(finalText).text
            responseText = persistedFinal
            speakPending(turnID: turnID)
            let assistantMsg = ChatMessage(role: .assistant, content: persistedFinal, agentSteps: AgentStepContentBudget.boundedSanitizedSteps(stepsBuffer))
            convo.messages.append(assistantMsg)
            convo.updatedAt = Date()
            saveVoiceConversationIfBudgetAllows(estimatedBytes: persistedFinal.utf8.count + text.utf8.count + 4096)

            if appState.autoMemory, persistedFinal.count > 60, isSafeToStoreMemory(userText: text, assistantText: persistedFinal, routing: routing) {
                let memoryResult = await MemoryStore.rememberWithDiagnostics(
                    "User asked: \(text). Assistant: \(String(persistedFinal.prefix(160)))",
                    kind: .conversation, source: "voice", context: modelContext
                )
                if memoryResult.mode != "stored" {
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .fallbackUsed, values: [
                        "surface": "voice",
                        "operation": "auto_memory",
                        "memoryMode": memoryResult.mode,
                        "memoryDiagnostic": memoryResult.diagnostic ?? "none",
                        "promptSHA256": RuntimeFallbackLogger.promptHash(text),
                        "promptChars": String(text.count)
                    ]))
                }
                let transient = stepsBuffer.filter { $0.kind == .observation || $0.kind == .action }.map(\.content)
                await MemoryStore.extractAndStore(userText: text, assistantText: persistedFinal, transientTexts: transient, context: modelContext)
            }

            activeVoiceTurnID = nil
            generationController.clearIfCurrent(controllerRequestID, for: "voice")
        }
        _ = generationController.begin(for: "voice", task: task, requestID: controllerRequestID)
        responseCancellationID = AppCancellationBus.shared.register(task, category: .chatGeneration)
        responseTask = task
    }

    private func safeShortTermContext(in conversation: Conversation, excludingCurrentUserMessageID currentID: UUID) -> [(role: MessageRole, content: String)] {
        conversation.sortedMessages
            .filter { $0.id != currentID }
            .suffix(4)
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

    private func isSafeToStoreMemory(userText: String, assistantText: String, routing: IntentRoutingDecision) -> Bool {
        FinalIntentValidator.validate(assistantText, routing: routing, fallback: nil) == assistantText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func ensureChatModelLoaded() async -> Bool {
        await ModelLoader.ensureChatLoaded(appState: appState, stored: storedModels, intent: .userVoice)
    }

    private func voiceCancellationReason(turnID: UUID, requestID: UUID) -> String? {
        if Task.isCancelled { return AppCancellationBus.shared.lastCancellationReason ?? "task-cancelled" }
        if activeVoiceTurnID != turnID { return "voice-turn-superseded" }
        if !generationController.isCurrent(requestID, for: "voice") { return "voice-generation-superseded" }
        if CPUWatchdogGuard.shared.shouldDegrade(category: .voice) { return "voice-cpu-budget" }
        if !ResourceBudgetGate.allowsHeavyModelWork(reason: "userVoice.stream") { return "resource-budget" }
        return nil
    }

    private func speakPending(turnID: UUID) {
        guard activeSpeechTurnID == turnID else { return }
        if phase != .speaking { phase = .speaking }; session.startSpeaking()
        guard spokenPrefix < responseText.count else {
            if finishedStreaming && !VoiceService.shared.isSpeaking { onFinishedSpeaking(turnID: turnID) }
            return
        }
        guard let next = VoiceStreamingChunker.nextChunk(
            in: responseText,
            startingAt: spokenPrefix,
            finishedStreaming: finishedStreaming
        ) else { return }
        spokenPrefix = next.nextOffset
        let chunk = next.text
        guard !chunk.isEmpty else {
            speakPending(turnID: turnID)
            return
        }
        if !VoiceService.shared.isSpeaking {
            session.speakChunk(chunk, voiceID: appState.voiceID, rate: appState.speakingRate)
            observeSpeechEnd(turnID: turnID)
        } else {
            session.speakChunk(chunk, voiceID: appState.voiceID, rate: appState.speakingRate)
        }
    }

    private func observeSpeechEnd(turnID: UUID) {
        cancelSpeechEndObservation()
        speechEndObserverTask = Task { @MainActor in
            while VoiceService.shared.isSpeaking { try? await Task.sleep(for: .milliseconds(150)); if Task.isCancelled { return } }
            guard !Task.isCancelled, activeSpeechTurnID == turnID else { return }
            if finishedStreaming && spokenPrefix >= responseText.count {
                onFinishedSpeaking(turnID: turnID)
            } else {
                speakPending(turnID: turnID)
            }
        }
    }

    private func onFinishedSpeaking(turnID: UUID) {
        guard VoiceTurnCompletionPolicy.acceptsSpeechCompletion(
            turnID: turnID,
            activeSpeechTurnID: activeSpeechTurnID
        ) else { return }
        let shouldResume = VoiceTurnCompletionPolicy.shouldResumeHandsFree(
            handsFree: appState.handsFree,
            turnID: turnID,
            activeSpeechTurnID: activeSpeechTurnID
        )
        activeSpeechTurnID = nil
        activeVoiceTurnID = nil
        session.stopSpeaking()
        if shouldResume {
            startListening()
        } else {
            phase = .idle
        }
    }

    private func unregisterResponseCancellation() {
        if let responseCancellationID {
            AppCancellationBus.shared.unregister(responseCancellationID, category: .chatGeneration)
            self.responseCancellationID = nil
        }
    }

    private func interrupt() {
        UINotificationFeedbackGenerator().notificationOccurred(.warning)
        let turnID = activeVoiceTurnID ?? activeSpeechTurnID
        activeVoiceTurnID = nil
        activeSpeechTurnID = nil
        cancelSpeechEndObservation()
        generationController.cancel(for: "voice")
        responseTask?.cancel()
        unregisterResponseCancellation()
        DeferredMaintenanceQueue.shared.setChatOrVoiceActive(false)
        session.stopSpeaking()
        session.cancel()
        phase = .idle
        if turnID != nil, appState.handsFree { startListening() }
    }

    private func toggleHandsFree() {
        appState.handsFree.toggle()
        UIImpactFeedbackGenerator(style: .soft).impactOccurred()
    }

    private func close() {
        cleanup()
        dismiss()
    }

    private func cancelForSceneTransition() {
        activeVoiceTurnID = nil
        activeSpeechTurnID = nil
        cancelSpeechEndObservation()
        generationController.cancel(for: "voice")
        responseTask?.cancel()
        unregisterResponseCancellation()
        session.handleAppDidEnterBackground()
        session.stopSpeaking()
        ConversationPersistenceCoordinator.enqueueDeferredSave(
            context: modelContext,
            estimatedBytes: 4096,
            operation: "voice.scene-transition.save",
            deferredKey: "voice-scene-transition-save",
            deferredCategory: .voice
        ) { failure in
            conversationPersistenceFailure = failure
        }
        DeferredMaintenanceQueue.shared.setChatOrVoiceActive(false)
        RuntimeLifecycleCanceller.cancelForSceneTransition(reason: "voice-scene")
        syncPhaseFromSession()
    }

    private func saveVoiceConversationIfBudgetAllows(estimatedBytes: Int) {
        _ = ConversationPersistenceCoordinator.saveOrDefer(
            context: modelContext,
            estimatedBytes: estimatedBytes,
            operation: "voice.conversation.save",
            deferredKey: "voice-conversation-save"
        ) { failure in
            conversationPersistenceFailure = failure
        }
    }

    private func retryVoiceConversationSave(_ previousFailure: ConversationPersistenceFailure) {
        conversationPersistenceFailure = nil
        _ = ConversationPersistenceCoordinator.saveOrDefer(
            context: modelContext,
            estimatedBytes: previousFailure.estimatedBytes,
            operation: "voice.conversation.retry",
            deferredKey: "voice-conversation-save"
        ) { failure in
            conversationPersistenceFailure = failure
        }
    }

    private func cleanup() {
        activeVoiceTurnID = nil
        activeSpeechTurnID = nil
        cancelSpeechEndObservation()
        generationController.cancel(for: "voice")
        responseTask?.cancel()
        unregisterResponseCancellation()
        session.cancel()
        session.stopSpeaking()
        DeferredMaintenanceQueue.shared.setChatOrVoiceActive(false)
    }

    private func cancelSpeechEndObservation() {
        speechEndObserverTask?.cancel()
        speechEndObserverTask = nil
    }
}

struct VoiceWaveform: View {
    var level: Double
    var phase: VoiceModeView.Phase
    @State private var animate = false

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0, paused: phase == .idle)) { ctx in
            let t = ctx.date.timeIntervalSinceReferenceDate
            GeometryReader { geo in
                HStack(spacing: 4) {
                    let count = 32
                    ForEach(0..<count, id: \.self) { i in
                        let norm = Double(i) / Double(count - 1)
                        let wave = sin(t * 3.0 + norm * 6.0) * 0.5 + 0.5
                        let reactive = max(0.15, level * 1.8)
                        let active = phase == .idle ? 0.2 : 1.0
                        let h = geo.size.height * (0.15 + wave * reactive * active * 0.85)
                        RoundedRectangle(cornerRadius: 2)
                            .fill(color(for: phase))
                            .frame(maxWidth: .infinity)
                            .frame(height: h)
                    }
                }
                .frame(maxHeight: .infinity, alignment: .center)
            }
        }
    }

    private func color(for phase: VoiceModeView.Phase) -> Color {
        switch phase {
        case .idle: Theme.textTertiary
        case .listening: Color(red: 0.5, green: 0.85, blue: 0.6)
        case .thinking: Color(red: 0.95, green: 0.75, blue: 0.4)
        case .speaking: Theme.accent
        }
    }
}
