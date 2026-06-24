import XCTest
@testable import Lumen

final class SecureToolRegistryBackgroundFilteringTests: XCTestCase {
    @MainActor
    func testBackgroundVisibleTools() async {
        let ctx = ToolExecutionContext(isForeground: false, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: .shared)
        let defs = await SecureToolRegistry.shared.availableDefinitions(context: ctx, source: .backgroundTrigger)
        XCTAssertTrue(defs.contains(where: {$0.id == "device.status"}))
        XCTAssertTrue(defs.contains(where: {$0.id == "memory.search"}))
        XCTAssertTrue(defs.contains(where: {$0.id == "rag.search.secure"}))
        XCTAssertTrue(defs.allSatisfy(\.supportsBackgroundExecution))
        XCTAssertFalse(defs.contains(where: {$0.id == "position.snapshot"}))
        XCTAssertFalse(defs.contains(where: {$0.id == "open.url"}))
    }
}
