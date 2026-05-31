import Foundation

struct LLMRequest: Sendable, Codable, Equatable {
    let id: UUID
    let conversationID: UUID?
    let messages: [LLMChatMessage]
    let systemPrompt: String?
    let sampling: LLMSamplingConfig
    let context: [LLMContextItem]
    let tools: [LLMToolDefinition]
    let responseFormat: LLMResponseFormat
    let budget: InferenceBudget
    let metadata: [String: String]

    init(
        id: UUID = UUID(),
        conversationID: UUID? = nil,
        messages: [LLMChatMessage],
        systemPrompt: String? = nil,
        sampling: LLMSamplingConfig = .balanced,
        context: [LLMContextItem] = [],
        tools: [LLMToolDefinition] = [],
        responseFormat: LLMResponseFormat = .plainText,
        budget: InferenceBudget = .standard,
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.conversationID = conversationID
        self.messages = messages
        self.systemPrompt = systemPrompt
        self.sampling = sampling
        self.context = context
        self.tools = tools
        self.responseFormat = responseFormat
        self.budget = budget
        self.metadata = metadata
    }
}
