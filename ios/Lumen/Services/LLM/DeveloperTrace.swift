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

nonisolated extension DeveloperTrace {
    func redactedForPersistence() -> DeveloperTrace {
        DeveloperTrace(
            id: id,
            createdAt: createdAt,
            conversationID: conversationID,
            messageID: messageID,
            modelName: modelName,
            systemPrompt: systemPrompt.map { AgentDiagnosticFileRedactor.summary(label: "systemPrompt", text: $0) },
            developerPrompt: developerPrompt.map { AgentDiagnosticFileRedactor.summary(label: "developerPrompt", text: $0) },
            userPrompt: AgentDiagnosticFileRedactor.summary(label: "userPrompt", text: userPrompt),
            resolvedContext: resolvedContext.map(\.redactedForPersistence),
            retrievedMemory: retrievedMemory.map(\.redactedForPersistence),
            toolPlan: toolPlan.map(\.redactedForPersistence),
            toolCalls: toolCalls.map(\.redactedForPersistence),
            agentMessages: agentMessages.map(\.redactedForPersistence),
            rawModelOutput: AgentDiagnosticFileRedactor.summary(label: "rawModelOutput", text: rawModelOutput),
            reasoningText: reasoningText.map { AgentDiagnosticFileRedactor.summary(label: "reasoningText", text: $0) },
            visibleAnswer: AgentDiagnosticFileRedactor.summary(label: "visibleAnswer", text: visibleAnswer),
            parserWarnings: parserWarnings.map { AgentDiagnosticFileRedactor.summary(label: "parserWarning", text: $0) },
            tokenUsage: tokenUsage,
            finishReason: finishReason,
            error: error.map { AgentDiagnosticFileRedactor.summary(label: "error", text: $0) }
        )
    }
}

private nonisolated extension TraceContextItem {
    var redactedForPersistence: TraceContextItem {
        TraceContextItem(
            id: id,
            role: role,
            title: title.map { AgentDiagnosticFileRedactor.summary(label: "title", text: $0) },
            content: AgentDiagnosticFileRedactor.summary(label: "content", text: content),
            source: source.map { AgentDiagnosticFileRedactor.summary(label: "source", text: $0) },
            metadata: AgentDiagnosticFileRedactor.redactedMap(metadata)
        )
    }
}

private nonisolated extension TraceMemoryItem {
    var redactedForPersistence: TraceMemoryItem {
        TraceMemoryItem(
            id: id,
            content: AgentDiagnosticFileRedactor.summary(label: "memory", text: content),
            scope: scope,
            authority: authority,
            createdAt: createdAt,
            expiresAt: expiresAt,
            source: source.map { AgentDiagnosticFileRedactor.summary(label: "source", text: $0) },
            topic: topic.map { AgentDiagnosticFileRedactor.summary(label: "topic", text: $0) }
        )
    }
}

private nonisolated extension TraceToolPlanItem {
    var redactedForPersistence: TraceToolPlanItem {
        TraceToolPlanItem(
            id: id,
            toolID: toolID,
            reason: reason.map { AgentDiagnosticFileRedactor.summary(label: "reason", text: $0) },
            requiresApproval: requiresApproval,
            arguments: AgentDiagnosticFileRedactor.redactedMap(arguments)
        )
    }
}

private nonisolated extension TraceToolCall {
    var redactedForPersistence: TraceToolCall {
        TraceToolCall(
            id: id,
            toolID: toolID,
            arguments: AgentDiagnosticFileRedactor.redactedMap(arguments),
            status: status,
            result: result.map { AgentDiagnosticFileRedactor.summary(label: "result", text: $0) },
            startedAt: startedAt,
            completedAt: completedAt,
            error: error.map { AgentDiagnosticFileRedactor.summary(label: "error", text: $0) }
        )
    }
}

private nonisolated extension TraceAgentMessage {
    var redactedForPersistence: TraceAgentMessage {
        TraceAgentMessage(
            id: id,
            role: role,
            content: AgentDiagnosticFileRedactor.summary(label: "message", text: content),
            toolID: toolID,
            metadata: AgentDiagnosticFileRedactor.redactedMap(metadata),
            createdAt: createdAt
        )
    }
}

nonisolated enum DeveloperTraceCodec {
    static func encode(_ trace: DeveloperTrace) -> String? {
        #if DEBUG
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(trace.redactedForPersistence()) else { return nil }
        return String(data: data, encoding: .utf8)
        #else
        return nil
        #endif
    }

    static func decode(_ string: String?) -> DeveloperTrace? {
        #if DEBUG
        guard let string, let data = string.data(using: .utf8) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(DeveloperTrace.self, from: data)
        #else
        return nil
        #endif
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
    let streamStarted: Bool?
    let selectedRuntime: String?
    let selectedAdapter: String?
    let modelIdentifier: String?
    let modelLoaded: Bool?
    let maxTokensRequested: Int?
    let maxTokensEffective: Int?
    let stopSequences: [String]?
    let temperature: Double?
    let topP: Double?
    let promptCharCount: Int?
    let estimatedPromptTokenCount: Int?
    let cancellationStateBeforeStream: String?
    let firstChunkReceived: Bool?
    let textChunkCount: Int?
    let finalChunkReceived: Bool?
    let streamTerminationReason: String?
    let elapsedMs: Int?
    let outputTokenCount: Int?
    let emptyOutputReason: String?

    init(
        requestID: UUID,
        rawModelOutput: String,
        reasoningText: String?,
        visibleAnswer: String,
        parserWarnings: [String],
        tokenUsage: TraceTokenUsage?,
        finishReason: String?,
        error: String?,
        streamStarted: Bool? = nil,
        selectedRuntime: String? = nil,
        selectedAdapter: String? = nil,
        modelIdentifier: String? = nil,
        modelLoaded: Bool? = nil,
        maxTokensRequested: Int? = nil,
        maxTokensEffective: Int? = nil,
        stopSequences: [String]? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        promptCharCount: Int? = nil,
        estimatedPromptTokenCount: Int? = nil,
        cancellationStateBeforeStream: String? = nil,
        firstChunkReceived: Bool? = nil,
        textChunkCount: Int? = nil,
        finalChunkReceived: Bool? = nil,
        streamTerminationReason: String? = nil,
        elapsedMs: Int? = nil,
        outputTokenCount: Int? = nil,
        emptyOutputReason: String? = nil
    ) {
        self.requestID = requestID
        self.rawModelOutput = rawModelOutput
        self.reasoningText = reasoningText
        self.visibleAnswer = visibleAnswer
        self.parserWarnings = parserWarnings
        self.tokenUsage = tokenUsage
        self.finishReason = finishReason
        self.error = error
        self.streamStarted = streamStarted
        self.selectedRuntime = selectedRuntime
        self.selectedAdapter = selectedAdapter
        self.modelIdentifier = modelIdentifier
        self.modelLoaded = modelLoaded
        self.maxTokensRequested = maxTokensRequested
        self.maxTokensEffective = maxTokensEffective
        self.stopSequences = stopSequences
        self.temperature = temperature
        self.topP = topP
        self.promptCharCount = promptCharCount
        self.estimatedPromptTokenCount = estimatedPromptTokenCount
        self.cancellationStateBeforeStream = cancellationStateBeforeStream
        self.firstChunkReceived = firstChunkReceived
        self.textChunkCount = textChunkCount
        self.finalChunkReceived = finalChunkReceived
        self.streamTerminationReason = streamTerminationReason
        self.elapsedMs = elapsedMs
        self.outputTokenCount = outputTokenCount
        self.emptyOutputReason = emptyOutputReason
    }
}
