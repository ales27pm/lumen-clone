import XCTest
@testable import Lumen

final class LegacyPromptAssemblerTests: XCTestCase {
    func testDeterministicAndBounded() {
        let secs = [PromptGroundingSection(title: "Relevant memories", content: String(repeating: "m", count: 2000), estimatedChars: 2000, sourceIDs: [], privacyLevel: .moderate)]
        let a = LegacyPromptAssembler.assemble(baseSystemPrompt: "sys", baseUserMessage: "hi", sections: secs, policy: .foregroundChat)
        let b = LegacyPromptAssembler.assemble(baseSystemPrompt: "sys", baseUserMessage: "hi", sections: secs, policy: .foregroundChat)
        XCTAssertEqual(a.userMessage, b.userMessage)
        XCTAssertTrue(a.truncationOccurred)
        XCTAssertTrue(a.userMessage.contains("LOCAL MEMORY"))
    }
}
