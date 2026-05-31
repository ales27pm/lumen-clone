import XCTest
@testable import Lumen

final class LegacyGroundingRequestTests: XCTestCase {
    func testConstructs() {
        let req = LegacyGroundingRequest(userMessage: "hi", conversationID: nil, turnID: nil, history: [], mode: .foreground, task: .chat, roleOrSlot: nil, externalRelevantMemories: [], externalAvailableTools: [], policy: .foregroundChat, baseSystemPrompt: "sys", preventDoubleGrounding: true)
        XCTAssertEqual(req.userMessage, "hi")
    }
}
