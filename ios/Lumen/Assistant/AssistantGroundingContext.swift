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
    let selfModelIncluded: Bool?
    let selfModelSchemaVersion: String?
    let selfModelEstimatedChars: Int?
    let selfModelSourceIDs: [String]?
    let selfModelMode: String?

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
        contextQueryExpanded: Bool? = nil,
        selfModelIncluded: Bool? = nil,
        selfModelSchemaVersion: String? = nil,
        selfModelEstimatedChars: Int? = nil,
        selfModelSourceIDs: [String]? = nil,
        selfModelMode: String? = nil
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
        self.selfModelIncluded = selfModelIncluded
        self.selfModelSchemaVersion = selfModelSchemaVersion
        self.selfModelEstimatedChars = selfModelEstimatedChars
        self.selfModelSourceIDs = selfModelSourceIDs
        self.selfModelMode = selfModelMode
    }

    private static let zeroCount = 0
    static let empty = AssistantGroundingContext(memoryCount: zeroCount, ragCount: zeroCount, toolCount: zeroCount, estimatedChars: zeroCount)
}
