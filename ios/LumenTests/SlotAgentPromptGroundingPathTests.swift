import XCTest
@testable import Lumen

final class SlotAgentPromptGroundingPathTests: XCTestCase {
    func testRoleMetadataAffectsAssembledPrompt() {
        let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: "sys", baseUserMessage: "hi", sections: [], policy: .slotAgent, roleMetadata: "planner")
        XCTAssertTrue(assembled.userMessage.contains("[ROLE STAGE]"))
        XCTAssertTrue(assembled.userMessage.contains("planner"))
    }

    func testOptionsAffectGroundingRequest() {
        let conversationID = UUID()
        let turnID = UUID()
        let req = AgentRequest(systemPrompt: "sys", history: [], userMessage: "hi", temperature: 0.1, topP: 0.9, repetitionPenalty: 1.0, maxTokens: 64, maxSteps: 2, availableTools: [], relevantMemories: [])
        let options = LegacyAgentRunOptions(modelContext: nil, conversationID: conversationID, turnID: turnID, groundingMode: .headlessTrigger, allowDegradedGrounding: false, preventDoubleGrounding: false, diagnosticsEnabled: true)
        let grounding = SlotAgentService.makeLegacyGroundingRequest(req, options: options)
        XCTAssertEqual(grounding.conversationID, conversationID)
        XCTAssertEqual(grounding.turnID, turnID)
        XCTAssertEqual(grounding.mode, .headless)
        XCTAssertEqual(grounding.policy.memoryMax, LegacyPromptInjectionPolicy.headlessTrigger.memoryMax)
        XCTAssertEqual(grounding.roleOrSlot, "headlessTrigger:diagnostics")
        XCTAssertFalse(grounding.preventDoubleGrounding)
    }

}
