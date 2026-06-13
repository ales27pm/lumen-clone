import Foundation
import SwiftData

@MainActor
struct VoiceCommandRouter {
    static func routeFinalTranscript(_ text: String, appState: AppState, conversation: Conversation, modelContext: ModelContext) async -> AsyncStream<AgentEvent> {
        let routing = await IntentClassifierService.shared.route(text)
        let memories = await MemoryRecall.recallAndNormalize(query: text, routing: routing, context: modelContext, limit: 8)
        let gatedMemories = MemoryGate.filter(intent: routing.intent, items: memories, userMessage: text)
        let turnID = UUID()
        let kernelRequest = AgentKernelRequest(
            conversationID: conversation.id,
            turnID: turnID,
            userMessage: text,
            history: [],
            systemPrompt: appState.systemPrompt,
            relevantMemories: gatedMemories,
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
                maxTokens: min(appState.maxTokens, ProcessInfo.processInfo.isLowPowerModeEnabled ? 384 : appState.maxTokens)
            )
        )

        let kernel = AssistantKernel.shared
        return AsyncStream { continuation in
            let task = Task { @MainActor in
                for await kernelEvent in kernel.run(kernelRequest, modelContext: modelContext) {
                    guard let legacyEvent = kernelEvent.legacyAgentEvent else { continue }
                    continuation.yield(legacyEvent)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
