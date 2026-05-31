import XCTest
@testable import Lumen

final class CalendarReadToolPolicyTests: XCTestCase {
    func testBackgroundDenied() async {
        let tool = CalendarReadTool(provider: CalendarReadTool.EventKitProvider())
        let inv = ToolInvocation(id: UUID(), toolID: "calendar.read", arguments: [:], source: .backgroundTrigger, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await tool.execute(invocation: inv, context: .init(isForeground: false, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: .shared))
        XCTAssertEqual(res.status, .denied)
    }
}
