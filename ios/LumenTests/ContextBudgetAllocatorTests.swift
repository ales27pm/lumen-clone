import XCTest
@testable import Lumen

final class ContextBudgetAllocatorTests: XCTestCase {
    func testAllocationWithinBudget() {
        let s = ContextBudgetAllocator.allocate(maxChars: 4000)
        XCTAssertLessThanOrEqual(s.total, 4000)
    }

    func testTokenAwareRAGProfilePrioritizesRetrievedContext() {
        let plan = ContextBudgetAllocator.allocate(
            profile: .rag,
            maxInputTokens: 1_000,
            input: "Summarize the retrieved source with citations."
        )

        XCTAssertEqual(plan.profile, .rag)
        XCTAssertLessThanOrEqual(plan.tokenSections.total, 1_000)
        XCTAssertGreaterThan(plan.tokenSections.rag, plan.tokenSections.history)
        XCTAssertGreaterThan(plan.charSections.rag, plan.charSections.history)
        XCTAssertGreaterThan(plan.estimatedInputTokens, 0)
    }

    func testToolProfilePrioritizesToolContracts() {
        let chat = ContextBudgetAllocator.allocate(profile: .chat, maxInputTokens: 1_000)
        let tool = ContextBudgetAllocator.allocate(profile: .tool, maxInputTokens: 1_000)

        XCTAssertGreaterThan(tool.tokenSections.tools, chat.tokenSections.tools)
        XCTAssertLessThanOrEqual(tool.tokenSections.total, 1_000)
    }

    func testProfileSelectionUsesTurnShape() {
        let ragTurn = AssistantTurnContext(
            task: .chat,
            input: "Summarize this document with citations",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal
        )
        let toolTurn = AssistantTurnContext(
            task: .toolDecision,
            input: "Call Alexis",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal
        )
        let backgroundTurn = AssistantTurnContext(
            task: .backgroundTrigger,
            input: "status",
            isForeground: false,
            lowPowerMode: false,
            thermalState: .nominal
        )

        XCTAssertEqual(ContextBudgetAllocator.profile(for: ragTurn), .rag)
        XCTAssertEqual(ContextBudgetAllocator.profile(for: toolTurn), .tool)
        XCTAssertEqual(ContextBudgetAllocator.profile(for: backgroundTurn), .background)
    }
}
