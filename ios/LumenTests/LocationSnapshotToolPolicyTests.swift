import XCTest
@testable import Lumen

final class LocationSnapshotToolPolicyTests: XCTestCase {
    func testBackgroundDenied() async {
        let inv = ToolInvocation(id: UUID(), toolID: "position.snapshot", arguments: [:], source: .backgroundTrigger, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await SecureToolRegistry(tools: [LocationSnapshotTool()])
            .execute(inv, context: .init(isForeground: false, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: .shared))
        XCTAssertEqual(res.status, .denied)
        XCTAssertEqual(res.errorCode, "denied")
        XCTAssertEqual(res.metricsSummary, "denied")
    }

    func testCanonicalLocationCurrentBackgroundDenied() async {
        guard let catalogTool = ToolRegistry.find(id: "location.current") else {
            XCTFail("Expected canonical location.current catalog definition")
            return
        }
        let inv = ToolInvocation(id: UUID(), toolID: "location.current", arguments: [:], source: .backgroundTrigger, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await SecureToolRegistry(tools: [LocationMediaHealthLocalTool(catalogTool)])
            .execute(inv, context: .init(isForeground: false, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: .shared))

        XCTAssertEqual(res.status, .denied)
        XCTAssertEqual(res.errorCode, "denied")
        XCTAssertEqual(res.metricsSummary, "denied")
    }
}
