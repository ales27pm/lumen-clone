import Foundation

nonisolated struct DeveloperTrace: Codable, Identifiable, Sendable, Equatable {
    let id: UUID
    let createdAt: Date
    let conversationID: UUID?
    let messageID: UUID?
    let modelName: String
    let systemPrompt: String?
    let developerPrompt: String?
    let userPrompt: String
    let resolvedContext: [TraceContextItem]
    let retrievedMemory: [TraceMemoryItem]
    let toolPlan: [TraceToolPlanItem]
    let toolCalls: [TraceToolCall]
    let agentMessages: [TraceAgentMessage]
    let rawModelOutput: String
    let reasoningText: String?
    let visibleAnswer: String
    let parserWarnings: [String]
    let tokenUsage: TraceTokenUsage?
    let finishReason: String?
    let error: String?

    init(
        id: UUID = UUID(),
        createdAt: Date = Date(),
        conversationID: UUID?,
        messageID: UUID?,
        modelName: String,
        systemPrompt: String?,
        developerPrompt: String?,
        userPrompt: String,
        resolvedContext: [TraceContextItem],
        retrievedMemory: [TraceMemoryItem],
        toolPlan: [TraceToolPlanItem],
        toolCalls: [TraceToolCall],
        agentMessages: [TraceAgentMessage],
        rawModelOutput: String,
        reasoningText: String?,
        visibleAnswer: String,
        parserWarnings: [String],
        tokenUsage: TraceTokenUsage?,
        finishReason: String?,
        error: String?
    ) {
        self.id = id
        self.createdAt = createdAt
        self.conversationID = conversationID
        self.messageID = messageID
        self.modelName = modelName
        self.systemPrompt = systemPrompt
        self.developerPrompt = developerPrompt
        self.userPrompt = userPrompt
        self.resolvedContext = resolvedContext
        self.retrievedMemory = retrievedMemory
        self.toolPlan = toolPlan
        self.toolCalls = toolCalls
        self.agentMessages = agentMessages
        self.rawModelOutput = rawModelOutput
        self.reasoningText = reasoningText
        self.visibleAnswer = visibleAnswer
        self.parserWarnings = parserWarnings
        self.tokenUsage = tokenUsage
        self.finishReason = finishReason
        self.error = error
    }
}

nonisolated struct TraceContextItem: Codable, Identifiable, Sendable, Equatable, Hashable {
    let id: UUID
    let role: String?
    let title: String?
    let content: String
    let source: String?
    let metadata: [String: String]

    init(
        id: UUID = UUID(),
        role: String? = nil,
        title: String? = nil,
        content: String,
        source: String? = nil,
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.role = role
        self.title = title
        self.content = content
        self.source = source
        self.metadata = metadata
    }
}

nonisolated struct TraceMemoryItem: Codable, Identifiable, Sendable, Equatable, Hashable {
    let id: UUID
    let content: String
    let scope: String
    let authority: String
    let createdAt: Date?
    let expiresAt: Date?
    let source: String?
    let topic: String?

    init(
        id: UUID = UUID(),
        content: String,
        scope: String,
        authority: String,
        createdAt: Date?,
        expiresAt: Date?,
        source: String?,
        topic: String?
    ) {
        self.id = id
        self.content = content
        self.scope = scope
        self.authority = authority
        self.createdAt = createdAt
        self.expiresAt = expiresAt
        self.source = source
        self.topic = topic
    }
}

nonisolated struct TraceToolPlanItem: Codable, Identifiable, Sendable, Equatable, Hashable {
    let id: UUID
    let toolID: String
    let reason: String?
    let requiresApproval: Bool?
    let arguments: [String: String]

    init(
        id: UUID = UUID(),
        toolID: String,
        reason: String? = nil,
        requiresApproval: Bool? = nil,
        arguments: [String: String] = [:]
    ) {
        self.id = id
        self.toolID = toolID
        self.reason = reason
        self.requiresApproval = requiresApproval
        self.arguments = arguments
    }
}

nonisolated struct TraceToolCall: Codable, Identifiable, Sendable, Equatable, Hashable {
    let id: UUID
    let toolID: String
    let arguments: [String: String]
    let status: String
    let result: String?
    let startedAt: Date?
    let completedAt: Date?
    let error: String?

    init(
        id: UUID = UUID(),
        toolID: String,
        arguments: [String: String] = [:],
        status: String,
        result: String? = nil,
        startedAt: Date? = nil,
        completedAt: Date? = nil,
        error: String? = nil
    ) {
        self.id = id
        self.toolID = toolID
        self.arguments = arguments
        self.status = status
        self.result = result
        self.startedAt = startedAt
        self.completedAt = completedAt
        self.error = error
    }
}

nonisolated struct TraceAgentMessage: Codable, Identifiable, Sendable, Equatable, Hashable {
    let id: UUID
    let role: String
    let content: String
    let toolID: String?
    let metadata: [String: String]
    let createdAt: Date?

    init(
        id: UUID = UUID(),
        role: String,
        content: String,
        toolID: String? = nil,
        metadata: [String: String] = [:],
        createdAt: Date? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.toolID = toolID
        self.metadata = metadata
        self.createdAt = createdAt
    }
}

nonisolated struct TraceTokenUsage: Codable, Sendable, Equatable, Hashable {
    let promptTokens: Int?
    let completionTokens: Int?
    let reasoningTokens: Int?
    let visibleTokens: Int?
    let totalTokens: Int?

    init(
        promptTokens: Int? = nil,
        completionTokens: Int? = nil,
        reasoningTokens: Int? = nil,
        visibleTokens: Int? = nil,
        totalTokens: Int? = nil
    ) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.reasoningTokens = reasoningTokens
        self.visibleTokens = visibleTokens
        self.totalTokens = totalTokens
    }
}

nonisolated enum DeveloperTraceCodec {
    static func encode(_ trace: DeveloperTrace) -> String? {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(trace) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func decode(_ string: String?) -> DeveloperTrace? {
        guard let string, let data = string.data(using: .utf8) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(DeveloperTrace.self, from: data)
    }
}

nonisolated struct CompletedGenerationTracePayload: Codable, Sendable, Equatable {
    let requestID: UUID
    let rawModelOutput: String
    let reasoningText: String?
    let visibleAnswer: String
    let parserWarnings: [String]
    let tokenUsage: TraceTokenUsage?
    let finishReason: String?
    let error: String?
}
