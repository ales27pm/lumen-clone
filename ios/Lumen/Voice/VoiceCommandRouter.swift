import Foundation
import SwiftData

@MainActor
struct VoiceCommandRouter {
    static func routeFinalTranscript(_ text: String, appState: AppState, conversation: Conversation, modelContext: ModelContext) async -> AsyncStream<AgentKernelEvent> {
        let routing = await IntentClassifierService.shared.route(text)
        let memories = await MemoryRecall.recallAndNormalize(query: text, routing: routing, context: modelContext, limit: 8)
        let turnID = UUID()
        let maxTokens = min(appState.maxTokens, ProcessInfo.processInfo.isLowPowerModeEnabled ? 384 : appState.maxTokens)
        return VoiceAgentRuntimeBridge.streamVoiceTurn(
            text: text,
            appState: appState,
            routing: routing,
            memories: memories,
            history: [],
            conversationID: conversation.id,
            turnID: turnID,
            modelContext: modelContext,
            maxTokens: maxTokens
        )
    }
}

@MainActor
enum VoiceAgentRuntimeBridge {
    static func streamVoiceTurn(
        text: String,
        appState: AppState,
        routing: IntentRoutingDecision,
        memories: [MemoryContextItem],
        history: [(role: MessageRole, content: String)],
        conversationID: UUID,
        turnID: UUID,
        modelContext: ModelContext,
        maxTokens: Int? = nil
    ) -> AsyncStream<AgentKernelEvent> {
        let gatedMemories = MemoryGate.filter(intent: routing.intent, items: memories, userMessage: text)
        let effectiveMaxTokens = maxTokens ?? appState.maxTokens
        let availableTools = enabledTools(for: routing, appState: appState)

        if shouldUseLegacyToolPath(routing: routing, availableTools: availableTools) {
            let request = makeLegacyAgentRequest(
                text: text,
                appState: appState,
                history: history,
                relevantMemories: gatedMemories,
                availableTools: availableTools,
                conversationID: conversationID,
                turnID: turnID,
                maxTokens: effectiveMaxTokens
            )
            let options = LegacyAgentRunOptions(
                modelContext: modelContext,
                conversationID: conversationID,
                turnID: turnID,
                groundingMode: .foregroundChat,
                allowDegradedGrounding: true,
                preventDoubleGrounding: true,
                diagnosticsEnabled: false
            )
            return AssistantKernel.shared.runLegacyAgentBridge(request, options: options)
        }

        let request = makeKernelRequest(
            text: text,
            appState: appState,
            history: history,
            relevantMemories: gatedMemories,
            conversationID: conversationID,
            turnID: turnID,
            maxTokens: effectiveMaxTokens
        )
        return AssistantKernel.shared.run(request, modelContext: modelContext)
    }

    private static func makeKernelRequest(
        text: String,
        appState: AppState,
        history: [(role: MessageRole, content: String)],
        relevantMemories: [MemoryContextItem],
        conversationID: UUID,
        turnID: UUID,
        maxTokens: Int
    ) -> AgentKernelRequest {
        AgentKernelRequest(
            conversationID: conversationID,
            turnID: turnID,
            userMessage: text,
            history: history.map { AgentKernelMessage(messageRole: $0.role, content: $0.content) },
            systemPrompt: appState.systemPrompt,
            relevantMemories: relevantMemories,
            attachments: [],
            task: .chat,
            source: .voice,
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
                maxTokens: maxTokens
            )
        )
    }

    private static func makeLegacyAgentRequest(
        text: String,
        appState: AppState,
        history: [(role: MessageRole, content: String)],
        relevantMemories: [MemoryContextItem],
        availableTools: [ToolDefinition],
        conversationID: UUID,
        turnID: UUID,
        maxTokens: Int
    ) -> AgentRequest {
        AgentRequest(
            systemPrompt: appState.systemPrompt,
            history: history,
            userMessage: text,
            temperature: appState.temperature,
            topP: appState.topP,
            repetitionPenalty: appState.repetitionPenalty,
            maxTokens: maxTokens,
            maxSteps: appState.maxAgentSteps,
            availableTools: availableTools,
            relevantMemories: relevantMemories,
            attachments: [],
            conversationID: conversationID,
            turnID: turnID
        )
    }

    private static func enabledTools(for routing: IntentRoutingDecision, appState: AppState) -> [ToolDefinition] {
        ToolRegistry.all.filter { tool in
            appState.enabledToolIDs.contains(tool.id) && IntentRouter.isToolAllowed(tool.id, for: routing)
        }
    }

    private static func shouldUseLegacyToolPath(routing: IntentRoutingDecision, availableTools: [ToolDefinition]) -> Bool {
        IntentRouter.intentRequiresTool(routing) && !availableTools.isEmpty
    }
}
