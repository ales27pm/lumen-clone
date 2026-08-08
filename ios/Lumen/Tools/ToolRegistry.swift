import Foundation
import SwiftData

struct SecureToolDefinitionAvailability: Sendable, Equatable {
    let definition: SecureToolDefinition
    let decision: ToolApprovalDecision
}

@MainActor
final class SecureToolRegistry {
    static let shared = SecureToolRegistry()
    private let tools: [ToolID: any LocalTool]
    private let duplicateToolIDs: [ToolID]

    private static var defaultTools: [any LocalTool] {
        [
            DeviceStatusTool(),
            MemorySearchTool(),
            RAGSearchTool(),
            CalendarReadTool(),
            ContactsLookupTool(),
            LocationSnapshotTool(),
            OpenURLTool(),
            NotificationTool()
        ] + ProductivityLocalTool.all
          + CommunicationLocalTool.all
          + LocationMediaHealthLocalTool.all
          + KnowledgeLocalTool.all
    }

    init(tools: [any LocalTool]? = nil) {
        var map: [ToolID: any LocalTool] = [:]
        var duplicates: [ToolID] = []
        for tool in tools ?? Self.defaultTools {
            guard map[tool.definition.id] == nil else {
                duplicates.append(tool.definition.id)
                continue
            }
            map[tool.definition.id] = tool
        }
        self.tools = map
        self.duplicateToolIDs = duplicates.sorted()
    }

    func definitions() -> [SecureToolDefinition] { tools.values.map(\.definition).sorted { $0.id < $1.id } }

    func duplicateDefinitionIDs() -> [ToolID] { duplicateToolIDs }

    func availableDefinitions(context: ToolExecutionContext, source: ToolInvocationSource) async -> [SecureToolDefinition] {
        await definitionAvailability(context: context, source: source).compactMap { assessment in
            if case .deny = assessment.decision { return nil }
            return assessment.definition
        }
    }

    func definitionAvailability(context: ToolExecutionContext, source: ToolInvocationSource) async -> [SecureToolDefinitionAvailability] {
        let states = await context.permissionRegistry.diagnostics()
        return definitions().map { def in
            let inv = ToolInvocation(id: UUID(), toolID: def.id, arguments: [:], source: source, conversationID: nil, turnID: nil, createdAt: Date())
            let decision = ToolApprovalPolicy.decide(definition: def, invocation: inv, isForeground: context.isForeground, permissionStates: states, settings: .init(networkAccessEnabled: states[.networkAccess] == .granted, userAllowlist: []))
            return SecureToolDefinitionAvailability(definition: def, decision: decision)
        }
    }

    func execute(_ invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        let executionToolID = Self.preferredExecutionToolID(for: invocation.toolID)
        guard let tool = tools[executionToolID] ?? tools[invocation.toolID] else {
            let result = ToolResult(invocationID: invocation.id, status: .unavailable, displayText: "Tool unavailable.", modelText: "Tool unavailable.", structuredPayload: nil, privacyLevel: .low, metricsSummary: "missing_tool", errorCode: "missing_tool")
            _ = await ToolMetricsRecorder(store: context.metricsStore).record(toolID: invocation.toolID, status: result.status, success: false, errorCode: result.errorCode, memoryWarningCount: MemoryPressureMonitor.shared.warningCount)
            return result
        }
        let states = await context.permissionRegistry.diagnostics()
        let policy = ToolApprovalPolicy.decide(definition: tool.definition, invocation: invocation, isForeground: context.isForeground, permissionStates: states, settings: .init(networkAccessEnabled: states[.networkAccess] == .granted, userAllowlist: []))
        switch policy {
        case .deny(let reason):
            let result = ToolResult(invocationID: invocation.id, status: .denied, displayText: reason, modelText: "Tool denied: \(reason)", structuredPayload: nil, privacyLevel: tool.definition.resultPrivacyLevel, metricsSummary: "denied", errorCode: "denied")
            _ = await ToolMetricsRecorder(store: context.metricsStore).record(toolID: invocation.toolID, status: result.status, success: false, errorCode: result.errorCode, memoryWarningCount: MemoryPressureMonitor.shared.warningCount)
            return result
        case .requiresApproval(let reason):
            let result = ToolResult(invocationID: invocation.id, status: .requiresApproval, displayText: reason, modelText: "Approval required.", structuredPayload: nil, privacyLevel: tool.definition.resultPrivacyLevel, metricsSummary: "requires_approval", errorCode: nil)
            _ = await ToolMetricsRecorder(store: context.metricsStore).record(toolID: invocation.toolID, status: result.status, success: false, memoryWarningCount: MemoryPressureMonitor.shared.warningCount)
            return result
        case .allow:
            let raw = await tool.execute(invocation: invocation, context: context)
            let bounded = SafeToolOutputLimiter.limit(result: raw, maxOutput: tool.definition.maxOutputCharacters)
            _ = await ToolMetricsRecorder(store: context.metricsStore).record(toolID: invocation.toolID, status: bounded.status, success: bounded.status == .success, errorCode: bounded.errorCode, memoryWarningCount: MemoryPressureMonitor.shared.warningCount)
            return bounded
        }
    }

    private nonisolated static func preferredExecutionToolID(for toolID: ToolID) -> ToolID {
        switch ToolRouteGuard.canonicalToolID(toolID) {
        case "calendar.list":
            return "calendar.read"
        case "contacts.search":
            return "contacts.lookup"
        case "memory.recall":
            return "memory.search"
        case "rag.search":
            return "rag.search.secure"
        default:
            return toolID
        }
    }

    func executeToolCommand(
        _ rawToolID: String,
        arguments: AgentJSONArguments,
        approval: ToolExecutionApproval,
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
        modelContext: ModelContext? = nil,
        isBackground: Bool = false
    ) async -> String {
        let result = await executeToolCommandResult(
            rawToolID,
            arguments: arguments,
            approval: approval,
            conversationID: conversationID,
            turnID: turnID,
            modelContext: modelContext,
            isBackground: isBackground
        )
        return Self.commandText(from: result)
    }

    func executeToolCommandResult(
        _ rawToolID: String,
        arguments: AgentJSONArguments,
        approval: ToolExecutionApproval,
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
        modelContext: ModelContext? = nil,
        isBackground: Bool = false
    ) async -> ToolResult {
        let canonicalToolID = ToolRouteGuard.canonicalToolID(rawToolID)
        let normalizedArguments = ToolRouteGuard.normalizedArguments(
            for: canonicalToolID,
            rawToolID: rawToolID,
            arguments: arguments.stringCoerced
        )
        let source: ToolInvocationSource
        if approval == .userApproved {
            source = .userApproved
        } else if isBackground {
            source = .backgroundTrigger
        } else {
            source = .modelProposed
        }
        let invocation = ToolInvocation(
            id: UUID(),
            toolID: canonicalToolID,
            arguments: normalizedArguments,
            source: source,
            conversationID: conversationID,
            turnID: turnID,
            createdAt: Date()
        )
        let context = ToolExecutionContext(
            isForeground: !isBackground,
            appState: nil,
            modelContext: modelContext,
            permissionRegistry: .shared,
            metricsStore: .shared
        )
        return await execute(invocation, context: context)
    }

    nonisolated static func commandText(from result: ToolResult) -> String {
        if result.modelText == "Approval required." || result.modelText.isEmpty {
            return result.displayText
        }
        return result.modelText
    }
}
