import Foundation
import SwiftData

@MainActor
final class SecureToolRegistry {
    static let shared = SecureToolRegistry()
    private let tools: [ToolID: any LocalTool]

    init(tools: [any LocalTool] = [DeviceStatusTool(), MemorySearchTool(), RAGSearchTool(), CalendarReadTool(), ContactsLookupTool(), LocationSnapshotTool(), OpenURLTool(), NotificationTool()]) {
        var map: [ToolID: any LocalTool] = [:]
        for tool in tools {
            precondition(map[tool.definition.id] == nil, "Duplicate secure tool id: \(tool.definition.id)")
            map[tool.definition.id] = tool
        }
        self.tools = map
    }

    func definitions() -> [SecureToolDefinition] { tools.values.map(\.definition).sorted { $0.id < $1.id } }

    func availableDefinitions(context: ToolExecutionContext, source: ToolInvocationSource) async -> [SecureToolDefinition] {
        let states = await context.permissionRegistry.diagnostics()
        return definitions().filter { def in
            let inv = ToolInvocation(id: UUID(), toolID: def.id, arguments: [:], source: source, conversationID: nil, turnID: nil, createdAt: Date())
            let decision = ToolApprovalPolicy.decide(definition: def, invocation: inv, isForeground: context.isForeground, permissionStates: states, settings: .init(networkAccessEnabled: states[.networkAccess] == .granted, userAllowlist: []))
            if case .deny = decision { return false }
            return true
        }
    }

    func execute(_ invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        guard let tool = tools[invocation.toolID] else {
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
}
