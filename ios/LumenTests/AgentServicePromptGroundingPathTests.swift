import XCTest
@testable import Lumen

final class AgentServicePromptGroundingPathTests: XCTestCase {
    func testAssemblerIncludesRealMemoryAndToolContent() {
        let sections = [
            PromptGroundingSection(title: "Relevant memories", content: "- prefers concise answers", estimatedChars: 25, sourceIDs: ["m1"], privacyLevel: .moderate),
            PromptGroundingSection(title: "Available tools", content: "- memory.search: Search memory", estimatedChars: 30, sourceIDs: ["memory.search"], privacyLevel: .low)
        ]
        let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: "sys", baseUserMessage: "hi", sections: sections, policy: .rolePipeline)
        XCTAssertTrue(assembled.userMessage.contains("prefers concise answers"))
        XCTAssertTrue(assembled.userMessage.contains("memory.search: Search memory"))
    }
}
