import Foundation

struct AssistantGroundingContext: Codable, Sendable {
    let memoryCount: Int
    let ragCount: Int
    let toolCount: Int
    let estimatedChars: Int
    let estimatedTokens: Int
    let contextProfile: String?
    let maxInputTokens: Int?
    let ragConfidence: Double?
    let memoryTierCounts: [String: Int]?
    let contextQueryExpanded: Bool?

    init(
        memoryCount: Int,
        ragCount: Int,
        toolCount: Int,
        estimatedChars: Int,
        estimatedTokens: Int? = nil,
        contextProfile: String? = nil,
        maxInputTokens: Int? = nil,
        ragConfidence: Double? = nil,
        memoryTierCounts: [String: Int]? = nil,
        contextQueryExpanded: Bool? = nil
    ) {
        self.memoryCount = memoryCount
        self.ragCount = ragCount
        self.toolCount = toolCount
        self.estimatedChars = max(0, estimatedChars)
        self.estimatedTokens = estimatedTokens ?? ContextBudgetAllocator.estimateTokens(forCharacterCount: max(0, estimatedChars))
        self.contextProfile = contextProfile
        self.maxInputTokens = maxInputTokens
        self.ragConfidence = ragConfidence
        self.memoryTierCounts = memoryTierCounts
        self.contextQueryExpanded = contextQueryExpanded
    }

    private static let zeroCount = 0
    static let empty = AssistantGroundingContext(memoryCount: zeroCount, ragCount: zeroCount, toolCount: zeroCount, estimatedChars: zeroCount)
}
