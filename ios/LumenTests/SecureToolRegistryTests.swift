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

    func testProductivityToolsAreRegisteredNatively() async {
        let ids = await Set(SecureToolRegistry.shared.definitions().map(\.id))
        XCTAssertTrue(ProductivityLocalTool.nativeToolIDs.isSubset(of: ids))
    }


    func testCommunicationToolsAreRegisteredNatively() async {
        let ids = await Set(SecureToolRegistry.shared.definitions().map(\.id))
        XCTAssertTrue(CommunicationLocalTool.nativeToolIDs.isSubset(of: ids))
    }

    func testAllCatalogToolsAreRegisteredAsNativeLocalTools() async {
        let defaultIDs = await Set(SecureToolRegistry.shared.definitions().map(\.id))
        let catalogIDs = Set(ToolRegistry.all.map { ToolRouteGuard.canonicalToolID($0.id) })
        XCTAssertTrue(catalogIDs.isSubset(of: defaultIDs))
    }

    func testNativeToolGroupsCoverEveryCatalogTool() async {
        let nativeIDs = ProductivityLocalTool.nativeToolIDs
            .union(CommunicationLocalTool.nativeToolIDs)
            .union(LocationMediaHealthLocalTool.nativeToolIDs)
            .union(KnowledgeLocalTool.nativeToolIDs)
        let catalogIDs = Set(ToolRegistry.all.map { ToolRouteGuard.canonicalToolID($0.id) })
        XCTAssertEqual(nativeIDs, catalogIDs)
    }

    func testToolResultStatusClassifiesApprovalRequiredResponses() {
        XCTAssertEqual(
            ToolResultStatusClassifier.status(from: "Calendar event creation requires explicit user approval. I did not create an event."),
            .requiresApproval
        )
        XCTAssertEqual(
            ToolResultStatusClassifier.status(from: "This tool requires explicit user approval before it can run: outlook.mail.send."),
            .requiresApproval
        )
    }

    func testToolResultStatusClassifiesPermissionFailuresAsDenied() {
        XCTAssertEqual(
            ToolResultStatusClassifier.status(from: "I need calendar access to do that. Please enable it in Settings or provide an alternative."),
            .denied
        )
        XCTAssertEqual(
            ToolResultStatusClassifier.status(from: "Missing required permission: contacts."),
            .denied
        )
    }
}

private struct DuplicateToolForRegistryTest: LocalTool {
    let definition = SecureToolDefinition(id: "duplicate.test", displayName: "Duplicate", description: "", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "", resultPrivacyLevel: .low, maxOutputCharacters: 10)
    func validateArguments(_ arguments: [String : String]) throws {}
    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        ToolResult(invocationID: invocation.id, status: .success, displayText: "ok", modelText: "ok", structuredPayload: nil, privacyLevel: .low, metricsSummary: "ok", errorCode: nil)
    }
}
