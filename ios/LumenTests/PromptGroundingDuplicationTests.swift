import XCTest
@testable import Lumen

final class PromptGroundingDuplicationTests: XCTestCase {
    func testNoDuplicateSectionHeaders() {
        let secs = [PromptGroundingSection(title: "Relevant memories", content: "a", estimatedChars: 1, sourceIDs: [], privacyLevel: .moderate)]
        let out = LegacyPromptAssembler.assemble(baseSystemPrompt: "s", baseUserMessage: "u", sections: secs, policy: .foregroundChat)
        XCTAssertEqual(out.userMessage.components(separatedBy: "LOCAL MEMORY").count - 1, 1)
    }
}
