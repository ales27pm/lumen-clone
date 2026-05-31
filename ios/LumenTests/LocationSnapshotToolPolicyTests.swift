import XCTest
@testable import Lumen

final class LocationSnapshotToolPolicyTests: XCTestCase {
    func testBackgroundDenied() async {
        let tool = LocationSnapshotTool()
        let inv = ToolInvocation(id: UUID(), toolID: "location.snapshot", arguments: [:], source: .backgroundTrigger, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await tool.execute(invocation: inv, context: .init(isForeground: false, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: .shared))
        XCTAssertEqual(res.status, .denied)
    }
}
