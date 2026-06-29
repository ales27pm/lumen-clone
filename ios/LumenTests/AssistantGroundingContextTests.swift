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
            ragConfidence: 0.82,
            memoryTierCounts: ["semantic": 1, "working": 2],
            contextQueryExpanded: true,
            selfModelIncluded: true,
            selfModelSchemaVersion: "0.1.0",
            selfModelEstimatedChars: 512,
            selfModelSourceIDs: ["selfModelSnapshot/0.1.0"],
            selfModelMode: "foreground"
        )
        let d = try JSONEncoder().encode(g)
        let decoded = try JSONDecoder().decode(AssistantGroundingContext.self, from: d)

        XCTAssertEqual(decoded.estimatedTokens, 300)
        XCTAssertEqual(decoded.contextProfile, ContextPolicyProfile.rag.rawValue)
        XCTAssertEqual(decoded.maxInputTokens, 2_048)
        XCTAssertEqual(decoded.ragConfidence, 0.82)
        XCTAssertEqual(decoded.memoryTierCounts?["semantic"], 1)
        XCTAssertEqual(decoded.memoryTierCounts?["working"], 2)
        XCTAssertEqual(decoded.contextQueryExpanded, true)
        XCTAssertEqual(decoded.selfModelIncluded, true)
        XCTAssertEqual(decoded.selfModelSchemaVersion, "0.1.0")
        XCTAssertEqual(decoded.selfModelEstimatedChars, 512)
        XCTAssertEqual(decoded.selfModelSourceIDs, ["selfModelSnapshot/0.1.0"])
        XCTAssertEqual(decoded.selfModelMode, "foreground")
    }
}
