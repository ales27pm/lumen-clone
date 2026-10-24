import XCTest
@testable import Lumen

final class LegacyPromptAssemblerTests: XCTestCase {
    @MainActor
    func testOversizedSelfModelIsDroppedWholeAndGroundingCountsActualInclusion() {
        let content = "schemaVersion=0.2.0\nselectedRuntime=llama\nrequiresApproval=open.url"
        let section = PromptGroundingSection(title: "Self model", content: content, estimatedChars: content.count, sourceIDs: ["selfModelSnapshot/0.2.0"], privacyLevel: .low)
        let grounding = AssistantGroundingContext(memoryCount: 0, ragCount: 0, toolCount: 0, estimatedChars: 0, selfModelIncluded: true, selfModelSchemaVersion: "0.2.0", selfModelEstimatedChars: content.count, selfModelSourceIDs: section.sourceIDs, selfModelMode: "foreground")
        let small = LegacyPromptInjectionPolicy(memoryMax: 0, ragMax: 0, toolMax: 0, runtimeMax: content.count - 1, allowSensitiveSections: false, backgroundSafeToolsOnly: false)
        let dropped = LegacyPromptAssembler.assemble(baseSystemPrompt: "sys", baseUserMessage: "hello", sections: [section], policy: small)
        XCTAssertFalse(dropped.userMessage.contains("[SELF MODEL]"))
        XCTAssertEqual(dropped.selfModelSectionChars, 0)
        XCTAssertTrue(dropped.truncationOccurred)
        let reported = LegacyTurnGroundingCoordinator.groundingAfterAssembly(grounding, assembled: dropped)
        XCTAssertEqual(reported.selfModelIncluded, false)
        XCTAssertEqual(reported.selfModelEstimatedChars, 0)
        XCTAssertEqual(reported.selfModelSourceIDs, [])
        XCTAssertNil(reported.selfModelSchemaVersion)

        let included = LegacyPromptAssembler.assemble(baseSystemPrompt: "sys", baseUserMessage: "hello", sections: [section], policy: .foregroundChat)
        XCTAssertTrue(included.userMessage.contains(content))
        XCTAssertEqual(included.selfModelSectionChars, content.count)
        XCTAssertEqual(LegacyTurnGroundingCoordinator.groundingAfterAssembly(grounding, assembled: included).selfModelEstimatedChars, content.count)
    }

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
