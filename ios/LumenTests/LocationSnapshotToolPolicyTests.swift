import XCTest
@testable import Lumen

final class LocationSnapshotToolPolicyTests: XCTestCase {
    func testBackgroundDenied() async {
        let inv = ToolInvocation(id: UUID(), toolID: "location.snapshot", arguments: [:], source: .backgroundTrigger, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await SecureToolRegistry(tools: [LocationSnapshotTool()])
            .execute(inv, context: .init(isForeground: false, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: .shared))
        XCTAssertEqual(res.status, .denied)
    }
}
