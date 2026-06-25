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

    func testBudgetPlanCapsSectionsAndReportsTokens() {
        let secs = [
            PromptGroundingSection(title: "Relevant memories", content: String(repeating: "m", count: 1_000), estimatedChars: 1_000, sourceIDs: [], privacyLevel: .moderate),
            PromptGroundingSection(title: "Retrieved sources", content: String(repeating: "r", count: 1_000), estimatedChars: 1_000, sourceIDs: [], privacyLevel: .moderate)
        ]
        let plan = ContextBudgetAllocator.allocate(profile: .chat, maxInputTokens: 100)

        let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: "sys", baseUserMessage: "hi", sections: secs, policy: .foregroundChat, budgetPlan: plan)

        XCTAssertEqual(assembled.contextProfile, "chat")
        XCTAssertEqual(assembled.maxInputTokens, 100)
        XCTAssertEqual(assembled.memorySectionChars, plan.charSections.memories)
        XCTAssertEqual(assembled.ragSectionChars, plan.charSections.rag)
        XCTAssertGreaterThan(assembled.estimatedTokens, 0)
        XCTAssertTrue(assembled.truncationOccurred)
    }

    func testToolProfileCanAllocateMoreToolContractSpaceThanLegacyCap() {
        let toolContent = String(repeating: "t", count: 1_500)
        let secs = [
            PromptGroundingSection(title: "Available tools", content: toolContent, estimatedChars: toolContent.count, sourceIDs: [], privacyLevel: .low)
        ]
        let plan = ContextBudgetAllocator.allocate(profile: .tool, maxInputTokens: 1_200)

        let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: "sys", baseUserMessage: "Call Alexis", sections: secs, policy: .slotAgent, budgetPlan: plan)

        XCTAssertGreaterThan(plan.charSections.tools, LegacyPromptInjectionPolicy.slotAgent.toolMax)
        XCTAssertEqual(assembled.toolSectionChars, toolContent.count)
        XCTAssertFalse(assembled.truncationOccurred)
    }
}
