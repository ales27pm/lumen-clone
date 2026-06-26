import XCTest
@testable import Lumen

final class ContextQueryRewriterTests: XCTestCase {
    func testAddsRecentHistoryTermsWithoutRepeatingUserTerms() {
        let result = ContextQueryRewriter.rewrite(
            userInput: "what about that adapter?",
            history: [
                (.assistant, "We were debugging executor preflight adapter path failures."),
                (.user, "The Qwen3 role adapter was missing.")
            ]
        )

        XCTAssertTrue(result.expansionApplied)
        XCTAssertTrue(result.query.contains("executor"))
        XCTAssertTrue(result.query.contains("preflight"))
        XCTAssertEqual(result.addedTerms.filter { $0 == "adapter" }.count, 0)
        XCTAssertLessThanOrEqual(result.query.count, 320)
    }

    func testUsesRelevantMemoryTopics() {
        let memory = MemoryContextItem(
            content: "Core ML path should be separate from GGUF runtime.",
            scope: .project,
            authority: .referenceOnly,
            createdAt: nil,
            expiresAt: nil,
            source: "test",
            topic: "ane routing"
        )

        let result = ContextQueryRewriter.rewrite(userInput: "continue", relevantMemories: [memory])

        XCTAssertTrue(result.query.contains("ane"))
        XCTAssertTrue(result.query.contains("routing"))
        XCTAssertTrue(result.query.contains("core"))
    }
}
