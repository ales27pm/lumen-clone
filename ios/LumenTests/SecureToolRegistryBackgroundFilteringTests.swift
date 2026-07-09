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

    @MainActor
    func testCatalogMemoryRecallExecutesSecureMemorySearchImplementation() async {
        let ctx = ToolExecutionContext(isForeground: false, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: .shared)
        let invocation = ToolInvocation(
            id: UUID(),
            toolID: "memory.recall",
            arguments: ["query": "workshop preferences"],
            source: .backgroundTrigger,
            conversationID: nil,
            turnID: nil,
            createdAt: Date()
        )

        let result = await SecureToolRegistry.shared.execute(invocation, context: ctx)

        XCTAssertEqual(result.status, .unavailable)
        XCTAssertEqual(result.metricsSummary, "no_model_context")
        XCTAssertEqual(result.errorCode, "unavailable")
    }
}
