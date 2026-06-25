import XCTest
@testable import Lumen

final class AssistantGroundingContextTests: XCTestCase {
    func testCodable() throws {
        let g = AssistantGroundingContext(
            memoryCount: 1,
            ragCount: 2,
            toolCount: 3,
            estimatedChars: 1200,
            estimatedTokens: 300,
            contextProfile: ContextPolicyProfile.rag.rawValue,
            maxInputTokens: 2_048,
            ragConfidence: 0.82
        )
        let d = try JSONEncoder().encode(g)
        let decoded = try JSONDecoder().decode(AssistantGroundingContext.self, from: d)

        XCTAssertEqual(decoded.estimatedTokens, 300)
        XCTAssertEqual(decoded.contextProfile, ContextPolicyProfile.rag.rawValue)
        XCTAssertEqual(decoded.maxInputTokens, 2_048)
        XCTAssertEqual(decoded.ragConfidence, 0.82)
    }
}
