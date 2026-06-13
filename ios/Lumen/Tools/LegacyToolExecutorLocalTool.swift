import Foundation
import SwiftData

struct LegacyToolExecutorLocalTool: LocalTool {
    let definition: SecureToolDefinition
    private let legacyToolID: String

    init(_ legacy: ToolDefinition) {
        let canonical = ToolRouteGuard.canonicalToolID(legacy.id)
        self.legacyToolID = canonical
        self.definition = SecureToolDefinition(
            id: canonical,
            displayName: legacy.name,
            description: legacy.description,
            category: Self.secureCategory(for: legacy),
            requiredPermissions: [],
            supportsBackgroundExecution: !legacy.requiresApproval,
            requiresUserApproval: legacy.requiresApproval,
            argumentSchemaDescription: Self.argumentSchemaDescription(from: legacy.description),
            resultPrivacyLevel: Self.privacyLevel(for: legacy),
            maxOutputCharacters: 2_400
        )
    }

    static var all: [LegacyToolExecutorLocalTool] {
        ToolRegistry.all
            .filter { !ProductivityLocalTool.nativeToolIDs.contains(ToolRouteGuard.canonicalToolID($0.id)) }
            .map(LegacyToolExecutorLocalTool.init)
    }

    func validateArguments(_ arguments: [String: String]) throws {}

    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        let approval: ToolExecutionApproval = invocation.source == .userInitiated ? .userApproved : .autonomous
        let text = await ToolExecutor.shared.execute(
            legacyToolID,
            arguments: AgentJSONArguments(stringDictionary: invocation.arguments),
            approval: approval
        )
        let status = Self.status(from: text)
        return ToolResult(
            invocationID: invocation.id,
            status: status,
            displayText: text,
            modelText: text,
            structuredPayload: ["legacyToolID": legacyToolID],
            privacyLevel: definition.resultPrivacyLevel,
            metricsSummary: status == .success ? "legacy_tool_executor_bridge" : "legacy_tool_executor_bridge_\(status.rawValue)",
            errorCode: status == .success ? nil : status.rawValue
        )
    }

    private static func secureCategory(for legacy: ToolDefinition) -> SecureToolCategory {
        if legacy.requiresApproval {
            if legacy.id.contains("delete") || legacy.id.contains("cancel") || legacy.id.contains("stop") {
                return .destructiveAction
            }
            return .sensitiveAction
        }
        switch legacy.category {
        case .knowledge, .location, .health:
            return .readOnly
        case .media, .productivity, .communication:
            return .userVisibleAction
        }
    }

    private static func privacyLevel(for legacy: ToolDefinition) -> ToolResultPrivacyLevel {
        switch legacy.category {
        case .communication, .health, .media:
            return .sensitive
        case .location, .productivity:
            return .moderate
        case .knowledge:
            return .low
        }
    }

    private static func argumentSchemaDescription(from description: String) -> String {
        guard let range = description.range(of: "Args:") else { return "{}" }
        return String(description[range.lowerBound...])
    }

    static func status(from text: String) -> ToolResultStatus {
        let lower = text.lowercased()

        if containsAny(lower, [
            "approval required",
            "requires approval",
            "requires explicit user approval",
            "requires user approval",
            "before it can run",
            "i did not create",
            "not create an event"
        ]) {
            return .requiresApproval
        }

        if containsAny(lower, [
            "permission denied",
            "missing permission",
            "missing required permission",
            "access denied",
            "please enable it in settings",
            "enable it in settings",
            "i need calendar access",
            "i need reminders access",
            "i need contacts access",
            "i need location access",
            "i need photo library access",
            "i need camera access",
            "i need health access",
            "i need motion access",
            "tool denied",
            "denied by",
            " is disabled",
            "tool disabled"
        ]) {
            return .denied
        }

        if containsAny(lower, [
            "unknown tool",
            "tool unavailable",
            "unavailable pending",
            "not available",
            "not configured",
            "not signed in"
        ]) {
            return .unavailable
        }

        if containsAny(lower, [
            "failed",
            "error",
            "couldn't",
            "could not",
            "unable to"
        ]) {
            return .failed
        }

        return .success
    }

    private static func containsAny(_ text: String, _ needles: [String]) -> Bool {
        needles.contains { text.contains($0) }
    }
}

extension SecureToolRegistry {
    func executeLegacyTool(
        _ toolID: String,
        arguments: AgentJSONArguments,
        approval: ToolExecutionApproval = .autonomous,
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
        modelContext: ModelContext? = nil,
        isBackground: Bool = false,
        appState: AppState? = nil
    ) async -> String {
        let source: ToolInvocationSource
        switch approval {
        case .userApproved:
            source = .userInitiated
        case .pending, .autonomous:
            source = isBackground ? .backgroundTrigger : .modelProposed
        }

        let canonical = ToolRouteGuard.canonicalToolID(toolID)
        let invocation = ToolInvocation(
            id: UUID(),
            toolID: canonical,
            arguments: arguments.stringCoerced,
            source: source,
            conversationID: conversationID,
            turnID: turnID,
            createdAt: Date()
        )
        let context = ToolExecutionContext(
            isForeground: !isBackground,
            appState: appState,
            modelContext: modelContext,
            permissionRegistry: .shared,
            metricsStore: .shared
        )
        let result = await execute(invocation, context: context)
        return result.modelText
    }
}
