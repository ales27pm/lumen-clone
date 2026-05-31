import Foundation

nonisolated enum LLMFinishReason: String, Sendable, Codable, Equatable {
    case stop
    case length
    case cancelled
    case toolCall
    case error
    case unknown
}

nonisolated struct LLMCompletionSummary: Sendable, Codable, Equatable {
    let requestID: UUID
    let modelID: String?
    let promptTokens: Int?
    let completionTokens: Int?
    let totalTokens: Int?
    let durationSeconds: Double?
    let finishReason: LLMFinishReason

    init(
        requestID: UUID,
        modelID: String?,
        promptTokens: Int? = nil,
        completionTokens: Int? = nil,
        totalTokens: Int? = nil,
        durationSeconds: Double? = nil,
        finishReason: LLMFinishReason
    ) {
        self.requestID = requestID
        self.modelID = modelID
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.durationSeconds = durationSeconds
        self.finishReason = finishReason
    }
}
