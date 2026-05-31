import XCTest
@testable import Lumen

final class SlotAgentGroundingTests: XCTestCase {
    func testRendererUsesRealEstimatedChars() {
        let content = "- slot memory"
        let sections = [PromptGroundingSection(title: "Relevant memories", content: content, estimatedChars: content.count, sourceIDs: ["m"], privacyLevel: .moderate)]
        let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: "sys", baseUserMessage: "hi", sections: sections, policy: .slotAgent)
        XCTAssertEqual(assembled.memorySectionChars, content.count)
        XCTAssertGreaterThan(assembled.estimatedChars, content.count)
    }
}
