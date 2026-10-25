import Foundation
import OSLog
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
    private let predictionLedger: SelfPredictionLedger
    private let predictionLogger = Logger(subsystem: "com.lumen.runtime", category: "self-prediction")

    struct PredictionRecordingFailure: Sendable, Equatable {
        enum Stage: String, Sendable { case prediction, outcome }
        let invocationID: UUID
        let stage: Stage
    }

    private(set) var lastPredictionRecordingFailure: PredictionRecordingFailure?

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

    init(tools: [any LocalTool]? = nil, predictionLedger: SelfPredictionLedger = .shared) {
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
        self.predictionLedger = predictionLedger
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

    func execute(
        _ invocation: ToolInvocation,
        context: ToolExecutionContext,
        predictionContext: SelfPredictionLedger.Context? = nil
    ) async -> ToolResult {
        var predictionRecorded = false
        if let predictionContext {
            let boundContext = SelfPredictionLedger.Context(
                runID: predictionContext.runID,
                toolID: ToolRouteGuard.canonicalToolID(invocation.toolID),
                runtime: predictionContext.runtime,
                probabilityOrigin: predictionContext.probabilityOrigin,
                probability: predictionContext.probability
            )
            do {
                try await predictionLedger.begin(context: boundContext, id: invocation.id)
                predictionRecorded = true
            } catch {
                recordPredictionFailure(invocationID: invocation.id, stage: .prediction)
            }
        }
        var toolBodyStarted = false
        let result = await executeRecordedInvocation(invocation, context: context, toolBodyStarted: &toolBodyStarted)
        if predictionRecorded {
            do {
                try await predictionLedger.complete(
                    id: invocation.id,
                    status: Self.predictionOutcomeStatus(result.status),
                    toolBodyStarted: toolBodyStarted,
                    cancelled: Task.isCancelled || result.errorCode == "cancelled"
                )
            } catch {
                recordPredictionFailure(invocationID: invocation.id, stage: .outcome)
            }
        }
        return result
    }

    private func executeRecordedInvocation(
        _ invocation: ToolInvocation,
        context: ToolExecutionContext,
        toolBodyStarted: inout Bool
    ) async -> ToolResult {
        guard !Task.isCancelled else { return Self.cancelledResult(invocation.id) }
        let executionToolID = Self.preferredExecutionToolID(for: invocation.toolID)
        guard let tool = tools[executionToolID] ?? tools[invocation.toolID] else {
            let result = ToolResult(invocationID: invocation.id, status: .unavailable, displayText: "Tool unavailable.", modelText: "Tool unavailable.", structuredPayload: nil, privacyLevel: .low, metricsSummary: "missing_tool", errorCode: "missing_tool")
            _ = await ToolMetricsRecorder(store: context.metricsStore).record(toolID: invocation.toolID, status: result.status, success: false, errorCode: result.errorCode, memoryWarningCount: MemoryPressureMonitor.shared.warningCount)
            return result
        }
        let states = await context.permissionRegistry.diagnostics()
        guard !Task.isCancelled else { return Self.cancelledResult(invocation.id) }
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
            guard !Task.isCancelled else { return Self.cancelledResult(invocation.id) }
            toolBodyStarted = true
            let raw = await tool.execute(invocation: invocation, context: context)
            guard raw.invocationID == invocation.id else {
                let result = ToolResult(invocationID: invocation.id, status: .failed,
                    displayText: "The tool returned an invalid result.", modelText: "The tool returned an invalid result.",
                    structuredPayload: nil, privacyLevel: tool.definition.resultPrivacyLevel,
                    metricsSummary: "result_invocation_mismatch", errorCode: "result_invocation_mismatch")
                _ = await ToolMetricsRecorder(store: context.metricsStore).record(toolID: invocation.toolID, status: result.status, success: false, errorCode: result.errorCode, memoryWarningCount: MemoryPressureMonitor.shared.warningCount)
                return result
            }
            let bounded = SafeToolOutputLimiter.limit(result: raw, maxOutput: tool.definition.maxOutputCharacters)
            _ = await ToolMetricsRecorder(store: context.metricsStore).record(toolID: invocation.toolID, status: bounded.status, success: bounded.status == .success, errorCode: bounded.errorCode, memoryWarningCount: MemoryPressureMonitor.shared.warningCount)
            return bounded
        }
    }

    private func recordPredictionFailure(invocationID: UUID, stage: PredictionRecordingFailure.Stage) {
        lastPredictionRecordingFailure = .init(invocationID: invocationID, stage: stage)
        predictionLogger.error("event=self_prediction_recording_failed stage=\(stage.rawValue, privacy: .public) invocation=\(invocationID.uuidString, privacy: .public)")
    }

    private nonisolated static func cancelledResult(_ id: UUID) -> ToolResult {
        ToolResult(invocationID: id, status: .failed, displayText: "Tool execution cancelled.",
                   modelText: "Tool execution cancelled.", structuredPayload: nil, privacyLevel: .low,
                   metricsSummary: "cancelled", errorCode: "cancelled")
    }

    private nonisolated static func predictionOutcomeStatus(_ status: ToolResultStatus) -> SelfPredictionLedger.OutcomeStatus {
        switch status {
        case .success: return .success
        case .failed: return .failed
        case .denied: return .denied
        case .requiresApproval: return .requiresApproval
        case .unavailable: return .unavailable
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
