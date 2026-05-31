import XCTest
@testable import Lumen

final class LegacyExternalMemoryMergeTests: XCTestCase {
    func testSectionCap() {
        let secs = [PromptGroundingSection(title: "Relevant memories", content: String(repeating: "x", count: 9999), estimatedChars: 9999, sourceIDs: [], privacyLevel: .moderate)]
        let out = LegacyPromptAssembler.assemble(baseSystemPrompt: "s", baseUserMessage: "u", sections: secs, policy: .headlessTrigger)
        XCTAssertLessThanOrEqual(out.memorySectionChars, LegacyPromptInjectionPolicy.headlessTrigger.memoryMax)
    }
}
