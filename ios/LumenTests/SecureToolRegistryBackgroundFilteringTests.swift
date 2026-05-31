import XCTest
@testable import Lumen

final class SecureToolRegistryBackgroundFilteringTests: XCTestCase {
    func testBackgroundVisibleTools() async {
        let ctx = ToolExecutionContext(isForeground: false, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: .shared)
        let defs = await SecureToolRegistry.shared.availableDefinitions(context: ctx, source: .backgroundTrigger)
        XCTAssertTrue(defs.contains(where: {$0.id == "device.status"}))
        XCTAssertTrue(defs.contains(where: {$0.id == "memory.search"}))
        XCTAssertTrue(defs.contains(where: {$0.id == "rag.search.secure"}))
        XCTAssertFalse(defs.contains(where: {$0.id == "open.url"}))
    }
}
