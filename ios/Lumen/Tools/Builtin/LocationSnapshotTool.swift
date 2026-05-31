import Foundation

struct LocationSnapshotTool: LocalTool {
    let definition = SecureToolDefinition(
        id: "position.snapshot",
        displayName: "Position Snapshot",
        description: "Position snapshot is disabled in this build",
        category: .readOnly,
        requiredPermissions: [],
        supportsBackgroundExecution: false,
        requiresUserApproval: false,
        argumentSchemaDescription: "{}",
        resultPrivacyLevel: .low,
        maxOutputCharacters: 240
    )

    init() {}

    func validateArguments(_ arguments: [String: String]) throws {}

    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        ToolResult(
            invocationID: invocation.id,
            status: .unavailable,
            displayText: "Position snapshot is disabled in this build.",
            modelText: "Position snapshot unavailable.",
            structuredPayload: nil,
            privacyLevel: .low,
            metricsSummary: "unavailable_disabled",
            errorCode: "disabled"
        )
    }
}
