import XCTest
@testable import Lumen

final class SecureToolRegistryBackgroundFilteringTests: XCTestCase {
    private struct StubTool: LocalTool {
        let definition: SecureToolDefinition
        let output: String

        init(id: String, output: String) {
            self.definition = SecureToolDefinition(
                id: id,
                displayName: id,
                description: id,
                category: .readOnly,
                requiredPermissions: [],
                supportsBackgroundExecution: true,
                requiresUserApproval: false,
                argumentSchemaDescription: "{}",
                resultPrivacyLevel: .low,
                maxOutputCharacters: 200
            )
            self.output = output
        }

        func validateArguments(_ arguments: [String: String]) throws {}

        func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
            ToolResult(
                invocationID: invocation.id,
                status: .success,
                displayText: output,
                modelText: output,
                structuredPayload: nil,
                privacyLevel: .low,
                metricsSummary: "stub",
                errorCode: nil
            )
        }
    }

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

    @MainActor
    func testCatalogCurrentLocationExecutesCanonicalImplementation() async {
        let registry = SecureToolRegistry(tools: [
            StubTool(id: "location.current", output: "canonical-current-location"),
            StubTool(id: "position.snapshot", output: "disabled-snapshot")
        ])
        let invocation = ToolInvocation(
            id: UUID(),
            toolID: "location.current",
            arguments: [:],
            source: .modelProposed,
            conversationID: nil,
            turnID: nil,
            createdAt: Date()
        )
        let context = ToolExecutionContext(
            isForeground: true,
            appState: nil,
            modelContext: nil,
            permissionRegistry: .shared,
            metricsStore: .shared
        )

        let result = await registry.execute(invocation, context: context)

        XCTAssertEqual(result.status, .success)
        XCTAssertEqual(result.modelText, "canonical-current-location")
    }
}
