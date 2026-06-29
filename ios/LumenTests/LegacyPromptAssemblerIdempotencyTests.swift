import XCTest
@testable import Lumen

final class LegacyPromptAssemblerIdempotencyTests: XCTestCase {
    func testAssemblerInjectsOnceWhenExistingGroundingPresent() {
        let sections = [PromptGroundingSection(title: "Relevant memories", content: "- x", estimatedChars: 0, sourceIDs: [], privacyLevel: .moderate)]
        let base = "Hello\n\n" + PromptGroundingIdempotencyGuard.marker + "\n[LOCAL MEMORY]\nold"
        let out = LegacyPromptAssembler.assemble(baseSystemPrompt: "sys", baseUserMessage: base, sections: sections, policy: .foregroundChat)
        let counts = PromptGroundingIdempotencyGuard.sectionOccurrenceCounts(out.userMessage)
        XCTAssertEqual(counts["[LOCAL MEMORY]"], 1)
    }
}


extension LegacyPromptAssemblerIdempotencyTests {
    func testSingleUserAuthoredHeaderDoesNotSuppressGeneratedGrounding() {
        let sections = [PromptGroundingSection(title: "Relevant memories", content: "- generated", estimatedChars: 11, sourceIDs: [], privacyLevel: .moderate)]
        let base = "User wrote [LOCAL MEMORY] as prose"
        let out = LegacyPromptAssembler.assemble(baseSystemPrompt: "sys", baseUserMessage: base, sections: sections, policy: .foregroundChat)
        XCTAssertTrue(out.userMessage.contains(base))
        XCTAssertTrue(out.userMessage.contains("- generated"))
    }

    func testAssemblerInjectsSelfModelWithinRuntimeBudget() {
        let sections = [
            PromptGroundingSection(title: "Runtime policy", content: "lowPower=false", estimatedChars: 14, sourceIDs: [], privacyLevel: .low),
            PromptGroundingSection(title: "Self model", content: "schemaVersion=0.1.0\nactiveSlot=cortex", estimatedChars: 38, sourceIDs: ["selfModelSnapshot/0.1.0"], privacyLevel: .low)
        ]
        let turn = AssistantTurnContext(
            task: .chat,
            input: "hello",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal
        )
        let budget = ContextBudgetAllocator.allocate(for: turn, maxInputTokens: 1_200)
        let out = LegacyPromptAssembler.assemble(
            baseSystemPrompt: "sys",
            baseUserMessage: "hello",
            sections: sections,
            policy: .foregroundChat,
            budgetPlan: budget
        )

        XCTAssertTrue(out.userMessage.contains("[SELF MODEL]"))
        XCTAssertTrue(out.userMessage.contains("schemaVersion=0.1.0"))
        XCTAssertLessThanOrEqual(out.runtimeSectionChars, budget.charSections.runtime)
    }
}
