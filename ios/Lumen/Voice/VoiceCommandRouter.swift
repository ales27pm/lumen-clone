import Foundation
import SwiftData

@MainActor
struct VoiceCommandRouter {
    static func routeFinalTranscript(_ text: String, appState: AppState, conversation: Conversation, modelContext: ModelContext) async -> AsyncStream<AgentEvent> {
        let routing = await IntentClassifierService.shared.route(text)
        let memories = await MemoryRecall.recallAndNormalize(query: text, routing: routing, context: modelContext, limit: 8)
        let tools = ToolRegistry.all.filter { appState.enabledToolIDs.contains($0.id) }.filter { IntentRouter.isToolAllowed($0.id, for: routing) }
        let req = AgentRequest(systemPrompt: appState.systemPrompt, history: [], userMessage: text, temperature: appState.temperature, topP: appState.topP, repetitionPenalty: appState.repetitionPenalty, maxTokens: min(appState.maxTokens, ProcessInfo.processInfo.isLowPowerModeEnabled ? 384 : appState.maxTokens), maxSteps: appState.maxAgentSteps, availableTools: tools, relevantMemories: memories, conversationID: conversation.id, turnID: UUID())
        return SlotAgentService.shared.run(req, options: .init(modelContext: modelContext, conversationID: conversation.id, turnID: req.turnID, groundingMode: .slotAgent, allowDegradedGrounding: true, preventDoubleGrounding: true, diagnosticsEnabled: false))
    }
}
