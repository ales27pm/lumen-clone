import XCTest
@testable import Lumen

final class SecureToolRegistryTests: XCTestCase {
    func testDefaultDefinitionsHaveUniqueIDs() async {
        let ids = await SecureToolRegistry.shared.definitions().map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count)
    }

    @MainActor
    func testBackgroundHidesSensitive() async {
        let ctx = ToolExecutionContext(isForeground: false, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: RuntimeMetricsStore.shared)
        let defs = await SecureToolRegistry.shared.availableDefinitions(context: ctx, source: .backgroundTrigger)
        XCTAssertFalse(defs.contains(where: { $0.category == .sensitiveAction }))
    }

    func testProductivityToolsAreRegisteredNatively() async {
        let ids = await Set(SecureToolRegistry.shared.definitions().map(\.id))
        XCTAssertTrue(ProductivityLocalTool.nativeToolIDs.isSubset(of: ids))
    }

    func testAlarmPermissionBridgeUsesAlarmKitUsageDescription() {
        let bridged = ToolSchemaBridge.toCatalogToolDefinitions([
            SecureToolDefinition(
                id: "example.alarm",
                displayName: "Example Alarm",
                description: "Example",
                category: .userVisibleAction,
                requiredPermissions: [.alarms],
                supportsBackgroundExecution: false,
                requiresUserApproval: false,
                argumentSchemaDescription: "{}",
                resultPrivacyLevel: .low,
                maxOutputCharacters: 100
            )
        ])

        XCTAssertEqual(bridged.first?.permissionKey, "NSAlarmKitUsageDescription")
        XCTAssertEqual(bridged.first?.permissionKind, .alarms)
    }

    @MainActor
    func testAlarmToolsBypassGenericPermissionGateForRuntimeEvidence() async {
        let statusFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: "alarm.authorization_status", arguments: [:])
        let requestFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: "alarm.request_authorization", arguments: [:])
        let countdownFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: "alarm.countdown", arguments: ["durationSeconds": "300"])

        XCTAssertNil(statusFailure)
        XCTAssertNil(requestFailure)
        XCTAssertNil(countdownFailure)
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


    func testUnsupportedNativeToolOutputClassifiesAsUnavailable() {
        XCTAssertEqual(
            ToolResultStatusClassifier.status(from: "Unsupported native productivity tool: example.tool."),
            .unavailable
        )
        XCTAssertEqual(
            ToolResultStatusClassifier.status(from: AlarmTools.unavailableMessage),
            .unavailable
        )
    }

    func testAlarmUnavailablePayloadRequiresDeviceRuntimeEvidence() {
        let payload = ProductivityLocalTool.alarmStructuredPayload(
            toolID: "alarm.authorization_status",
            text: AlarmTools.unavailableMessage,
            status: .unavailable
        )

        XCTAssertEqual(payload["availability"], "unavailable")
        XCTAssertEqual(payload["runtimeEvidence"], "device-runtime-required")
        XCTAssertEqual(payload["alarmKitStatus"], "unavailable")
        XCTAssertEqual(
            ProductivityLocalTool.alarmErrorCode(text: AlarmTools.unavailableMessage, status: .unavailable),
            "alarmkit_runtime_unavailable"
        )
    }


    func testMediaToolsAreNotBackgroundReadOnly() async {
        let definitions = await SecureToolRegistry.shared.definitions()
        XCTAssertEqual(definitions.first(where: { $0.id == "photos.search" })?.category, .userVisibleAction)
        XCTAssertEqual(definitions.first(where: { $0.id == "camera.capture" })?.category, .sensitiveAction)
    }

    @MainActor
    func testBackgroundAvailabilityExcludesMediaTools() async {
        let ctx = ToolExecutionContext(isForeground: false, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: RuntimeMetricsStore.shared)
        let defs = await SecureToolRegistry.shared.availableDefinitions(context: ctx, source: .backgroundTrigger)
        let ids = Set(defs.map(\.id))
        XCTAssertFalse(ids.contains("photos.search"))
        XCTAssertFalse(ids.contains("camera.capture"))
    }

    func testWebToolsAreExternalNetworkCategory() async {
        let definitions = await SecureToolRegistry.shared.definitions()
        XCTAssertEqual(definitions.first(where: { $0.id == "web.search" })?.category, .externalNetwork)
        XCTAssertEqual(definitions.first(where: { $0.id == "web.fetch" })?.category, .externalNetwork)
    }

    @MainActor func testMapsSearchWebFallbackDeniesForNetworkBeforeLocationPermission() async {
        PermissionRegistry.shared.setNetworkAccessEnabled(false)
        let invocation = ToolInvocation(
            id: UUID(),
            toolID: "maps.search",
            arguments: ["query": "how to build a bookshelf"],
            source: .modelProposed,
            conversationID: nil,
            turnID: nil,
            createdAt: Date()
        )
        let context = ToolExecutionContext(isForeground: true, appState: nil, modelContext: nil, permissionRegistry: .shared, metricsStore: .shared)
        let result = await SecureToolRegistry.shared.execute(invocation, context: context)
        XCTAssertEqual(result.status, .denied)
        XCTAssertEqual(result.errorCode, "denied")
        XCTAssertTrue(result.modelText.localizedCaseInsensitiveContains("Network tools are disabled"))
        XCTAssertFalse(result.modelText.localizedCaseInsensitiveContains("location"))
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

    @MainActor
    func testLegacyBridgeMapsConfirmedApprovalToUserApprovedSource() async {
        let registry = SecureToolRegistry(tools: [CapturingApprovalTool()])

        let result = await registry.executeLegacyTool(
            "capture.approval",
            arguments: AgentJSONArguments(stringDictionary: [:]),
            approval: .userApproved
        )

        XCTAssertEqual(result, "source=userApproved")
    }

    @MainActor
    func testLegacyBridgeDoesNotTreatAutonomousExecutionAsApproval() async {
        let registry = SecureToolRegistry(tools: [CapturingApprovalTool()])

        let result = await registry.executeLegacyTool(
            "capture.approval",
            arguments: AgentJSONArguments(stringDictionary: [:]),
            approval: .autonomous
        )

        XCTAssertEqual(result, "User approval required")
    }

    @MainActor
    func testBackgroundAvailabilityExcludesApprovalRequiredToolEvenWhenBackgroundSupported() async {
        let registry = SecureToolRegistry(tools: [BackgroundApprovalRequiredTool()])
        let context = ToolExecutionContext(
            isForeground: false,
            appState: nil,
            modelContext: nil,
            permissionRegistry: .shared,
            metricsStore: .shared
        )

        let definitions = await registry.availableDefinitions(context: context, source: .backgroundTrigger)
        let result = await registry.execute(
            ToolInvocation(
                id: UUID(),
                toolID: "memory.export",
                arguments: [:],
                source: .backgroundTrigger,
                conversationID: nil,
                turnID: nil,
                createdAt: Date()
            ),
            context: context
        )

        XCTAssertFalse(definitions.contains(where: { $0.id == "memory.export" }))
        XCTAssertEqual(result.status, .denied)
        XCTAssertEqual(result.displayText, "Tool requires foreground approval")
    }
}

private struct DuplicateToolForRegistryTest: LocalTool {
    let definition = SecureToolDefinition(id: "duplicate.test", displayName: "Duplicate", description: "", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "", resultPrivacyLevel: .low, maxOutputCharacters: 10)
    func validateArguments(_ arguments: [String : String]) throws {}
    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        ToolResult(invocationID: invocation.id, status: .success, displayText: "ok", modelText: "ok", structuredPayload: nil, privacyLevel: .low, metricsSummary: "ok", errorCode: nil)
    }
}

private struct CapturingApprovalTool: LocalTool {
    let definition = SecureToolDefinition(
        id: "capture.approval",
        displayName: "Capture Approval",
        description: "",
        category: .sensitiveAction,
        requiredPermissions: [],
        supportsBackgroundExecution: false,
        requiresUserApproval: true,
        argumentSchemaDescription: "",
        resultPrivacyLevel: .low,
        maxOutputCharacters: 100
    )

    func validateArguments(_ arguments: [String : String]) throws {}

    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        let text = "source=\(invocation.source.rawValue)"
        return ToolResult(
            invocationID: invocation.id,
            status: .success,
            displayText: text,
            modelText: text,
            structuredPayload: nil,
            privacyLevel: .low,
            metricsSummary: "ok",
            errorCode: nil
        )
    }
}

private struct BackgroundApprovalRequiredTool: LocalTool {
    let definition = SecureToolDefinition(
        id: "memory.export",
        displayName: "Export Memory",
        description: "",
        category: .readOnly,
        requiredPermissions: [],
        supportsBackgroundExecution: true,
        requiresUserApproval: true,
        argumentSchemaDescription: "",
        resultPrivacyLevel: .sensitive,
        maxOutputCharacters: 100
    )

    func validateArguments(_ arguments: [String : String]) throws {}

    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        ToolResult(
            invocationID: invocation.id,
            status: .success,
            displayText: "exported",
            modelText: "exported",
            structuredPayload: nil,
            privacyLevel: .sensitive,
            metricsSummary: "ok",
            errorCode: nil
        )
    }
}
