import Foundation
import SwiftData

nonisolated struct BackgroundToolExecutionAssessment: Sendable, Equatable {
    enum Status: String, Sendable, Equatable {
        case runnable
        case notToolBacked
        case clarificationRequired
        case noRoutedTools
        case noBackgroundSafeRoutedTools
        case blockedByCurrentPolicy
        case toolMappingUnavailable
    }

    let status: Status
    let intent: UserIntent
    let routedToolIDs: [String]
    let backgroundCapableToolIDs: [String]
    let policyAllowedToolIDs: [String]
    let availableTools: [ToolDefinition]

    var canRunWithoutLoadedTextRuntime: Bool {
        status == .runnable
    }

    var availableToolIDs: [String] {
        Self.sortedCanonicalIDs(availableTools.map(\.id))
    }

    var skipMessage: String {
        switch status {
        case .runnable:
            return "Background-safe local tool-only path is available."
        case .notToolBacked:
            return "Background trigger needs a loaded local model because this request is not tool-backed."
        case .clarificationRequired:
            return "Background trigger skipped: the request needs foreground clarification."
        case .noRoutedTools:
            return "Background trigger skipped: no routed tools were selected for this request."
        case .noBackgroundSafeRoutedTools:
            return "Background trigger skipped: no routed tool is allowed to run in background."
        case .blockedByCurrentPolicy:
            return "Background trigger skipped: routed background tools are blocked by current permissions or runtime policy."
        case .toolMappingUnavailable:
            return "Background trigger skipped: routed background tools are not available to the background tool runner."
        }
    }

    var diagnosticMetadata: [String: String] {
        [
            "status": status.rawValue,
            "intent": intent.rawValue,
            "routedToolIDs": routedToolIDs.joined(separator: ","),
            "backgroundCapableToolIDs": backgroundCapableToolIDs.joined(separator: ","),
            "policyAllowedToolIDs": policyAllowedToolIDs.joined(separator: ","),
            "availableToolIDs": availableToolIDs.joined(separator: ",")
        ]
    }

    static func sortedCanonicalIDs(_ ids: [String]) -> [String] {
        Array(Set(ids.map { ToolRouteGuard.canonicalToolID($0) })).sorted()
    }
}

@MainActor
enum BackgroundToolExecutionPolicy {
    static func assess(
        prompt: String,
        routing providedRouting: IntentRoutingDecision? = nil,
        modelContext: ModelContext?,
        toolRegistry: SecureToolRegistry? = nil,
        metricsStore: RuntimeMetricsStore? = nil
    ) async -> BackgroundToolExecutionAssessment {
        let routing = providedRouting ?? IntentRouter.classify(prompt)
        let routedIDs = BackgroundToolExecutionAssessment.sortedCanonicalIDs(Array(routing.allowedToolIDs))
        guard IntentRouter.intentRequiresTool(routing) else {
            return .init(
                status: .notToolBacked,
                intent: routing.intent,
                routedToolIDs: routedIDs,
                backgroundCapableToolIDs: [],
                policyAllowedToolIDs: [],
                availableTools: []
            )
        }
        guard !routing.requiresClarification else {
            return .init(
                status: .clarificationRequired,
                intent: routing.intent,
                routedToolIDs: routedIDs,
                backgroundCapableToolIDs: [],
                policyAllowedToolIDs: [],
                availableTools: []
            )
        }
        guard !routedIDs.isEmpty else {
            return .init(
                status: .noRoutedTools,
                intent: routing.intent,
                routedToolIDs: routedIDs,
                backgroundCapableToolIDs: [],
                policyAllowedToolIDs: [],
                availableTools: []
            )
        }

        let routedIDSet = Set(routedIDs)
        let registry = toolRegistry ?? .shared
        let backgroundCapableIDs = BackgroundToolExecutionAssessment.sortedCanonicalIDs(
            registry.definitions()
                .filter { definition in
                    definition.supportsBackgroundExecution
                        && !definition.requiresUserApproval
                        && (definition.category == .readOnly || definition.category == .permissionRead)
                }
                .map(\.id)
        )
        let context = ToolExecutionContext(
            isForeground: false,
            appState: nil,
            modelContext: modelContext,
            permissionRegistry: .shared,
            metricsStore: metricsStore ?? .shared
        )
        let backgroundDefinitions = await registry.availableDefinitions(
            context: context,
            source: .backgroundTrigger
        )
        let policyAllowedIDs = BackgroundToolExecutionAssessment.sortedCanonicalIDs(backgroundDefinitions.map(\.id))
        let catalogDefinitions = ToolSchemaBridge.toCatalogToolDefinitions(backgroundDefinitions)
        let availableTools = catalogDefinitions
            .filter { routedIDSet.contains(ToolRouteGuard.canonicalToolID($0.id)) }
            .reduce(into: [ToolDefinition]()) { output, tool in
                let canonical = ToolRouteGuard.canonicalToolID(tool.id)
                guard !output.contains(where: { ToolRouteGuard.canonicalToolID($0.id) == canonical }) else { return }
                output.append(tool)
            }
            .sorted { $0.id < $1.id }

        let status: BackgroundToolExecutionAssessment.Status
        if !availableTools.isEmpty {
            status = .runnable
        } else if Set(backgroundCapableIDs).intersection(routedIDSet).isEmpty {
            status = .noBackgroundSafeRoutedTools
        } else if Set(policyAllowedIDs).intersection(routedIDSet).isEmpty {
            status = .blockedByCurrentPolicy
        } else {
            status = .toolMappingUnavailable
        }

        return .init(
            status: status,
            intent: routing.intent,
            routedToolIDs: routedIDs,
            backgroundCapableToolIDs: backgroundCapableIDs,
            policyAllowedToolIDs: policyAllowedIDs,
            availableTools: availableTools
        )
    }

    static func availableTools(
        for prompt: String,
        routing providedRouting: IntentRoutingDecision? = nil,
        modelContext: ModelContext?,
        toolRegistry: SecureToolRegistry? = nil,
        metricsStore: RuntimeMetricsStore? = nil
    ) async -> [ToolDefinition] {
        let assessment = await assess(
            prompt: prompt,
            routing: providedRouting,
            modelContext: modelContext,
            toolRegistry: toolRegistry,
            metricsStore: metricsStore
        )
        return assessment.availableTools
    }

    static func canRunWithoutLoadedTextRuntime(
        prompt: String,
        routing: IntentRoutingDecision? = nil,
        modelContext: ModelContext?,
        toolRegistry: SecureToolRegistry? = nil,
        metricsStore: RuntimeMetricsStore? = nil
    ) async -> Bool {
        let assessment = await assess(
            prompt: prompt,
            routing: routing,
            modelContext: modelContext,
            toolRegistry: toolRegistry,
            metricsStore: metricsStore
        )
        return assessment.canRunWithoutLoadedTextRuntime
    }
}
