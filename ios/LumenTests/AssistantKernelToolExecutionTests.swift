import XCTest
@testable import Lumen

@MainActor
final class AssistantKernelToolExecutionTests: XCTestCase {
    func testOpenURLRequiresApprovalForModelProposed() async {
        let kernel = AssistantKernel()
        let inv = ToolInvocation(id: UUID(), toolID: "open.url", arguments: ["url":"https://example.com"], source: .modelProposed, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await kernel.executeTool(inv)
        XCTAssertEqual(res.status, .requiresApproval)
    }
}
