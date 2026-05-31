import Foundation

struct LegacyGroundingRequest: Sendable {
    enum Mode: String, Codable, Sendable, Equatable { case foreground, background, headless }

    let userMessage: String
    let conversationID: UUID?
    let turnID: UUID?
    let history: [(role: MessageRole, content: String)]
    let mode: Mode
    let task: AssistantTaskKind
    let roleOrSlot: String?
    let externalRelevantMemories: [MemoryContextItem]
    let externalAvailableTools: [ToolDefinition]
    let policy: LegacyPromptInjectionPolicy
    let baseSystemPrompt: String
    let preventDoubleGrounding: Bool
}
