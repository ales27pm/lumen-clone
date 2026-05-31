import Foundation

struct AssistantGroundingContext: Codable, Sendable {
    let memoryCount: Int
    let ragCount: Int
    let toolCount: Int
    let estimatedChars: Int

    private static let zeroCount = 0
    static let empty = AssistantGroundingContext(memoryCount: zeroCount, ragCount: zeroCount, toolCount: zeroCount, estimatedChars: zeroCount)
}
