import XCTest
@testable import Lumen

final class SecureToolRegistryTests: XCTestCase {
    func testDefaultDefinitionsHaveUniqueIDs() async {
        let ids = await SecureToolRegistry.shared.definitions().map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count)
    }

    func testBackgroundHidesSensitive() async {
        let ctx = ToolExecutionContext(isForeground: false, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: RuntimeMetricsStore.shared)
        let defs = await SecureToolRegistry.shared.availableDefinitions(context: ctx, source: .backgroundTrigger)
        XCTAssertFalse(defs.contains(where: { $0.category == .sensitiveAction }))
    }
}

private struct DuplicateToolForRegistryTest: LocalTool {
    let definition = SecureToolDefinition(id: "duplicate.test", displayName: "Duplicate", description: "", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "", resultPrivacyLevel: .low, maxOutputCharacters: 10)
    func validateArguments(_ arguments: [String : String]) throws {}
    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        ToolResult(invocationID: invocation.id, status: .success, displayText: "ok", modelText: "ok", structuredPayload: nil, privacyLevel: .low, metricsSummary: "ok", errorCode: nil)
    }
}
