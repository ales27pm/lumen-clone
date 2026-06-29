import Foundation

nonisolated struct SelfModelSnapshot: Codable, Sendable, Equatable {
    static let schemaVersion = "0.1.0"

    let schemaVersion: String
    let generatedAt: Date
    let app: AppInfo
    let agent: AgentInfo
    let runtime: RuntimeInfo
    let contextBudget: BudgetInfo
    let tools: ToolInfo
    let evidence: EvidenceInfo
    let policy: PolicyInfo

    nonisolated struct AppInfo: Codable, Sendable, Equatable {
        let name: String
        let buildNumber: String
        let shortVersion: String
        let platform: String
        let mode: String
    }

    nonisolated struct AgentInfo: Codable, Sendable, Equatable {
        let logicalIdentity: String
        let activeSlot: String
        let availableSlots: [String]
        let manifestCommit: String
        let fleetContractVersion: String
    }

    nonisolated struct RuntimeInfo: Codable, Sendable, Equatable {
        let selectedRuntimePathKind: String
        let selectedRuntime: String
        let selectionReason: String
        let availableBackendKinds: [String]
        let embeddingAvailable: Bool
        let thermalState: String
        let powerState: String
        let networkState: String
    }

    nonisolated struct BudgetInfo: Codable, Sendable, Equatable {
        let profile: String
        let maxInputTokens: Int
        let sections: TokenSections
    }

    nonisolated struct TokenSections: Codable, Sendable, Equatable {
        let system: Int
        let history: Int
        let memories: Int
        let rag: Int
        let tools: Int
        let runtime: Int

        init(_ sections: ContextBudgetTokenSections) {
            system = sections.system
            history = sections.history
            memories = sections.memories
            rag = sections.rag
            tools = sections.tools
            runtime = sections.runtime
        }
    }

    nonisolated struct ToolInfo: Codable, Sendable, Equatable {
        let available: [String]
        let requiresApproval: [String]
        let backgroundSafe: [String]
    }

    nonisolated struct EvidenceInfo: Codable, Sendable, Equatable {
        let manifestFreshness: String
        let runtimeAuditPresent: Bool
        let exportPolicy: ExportPolicy
    }

    nonisolated struct ExportPolicy: Codable, Sendable, Equatable {
        let sourceLayer: String
        let ownsLiveE2EScenarios: Bool
        let includesDeterministicStaticScenarios: Bool
    }

    nonisolated struct PolicyInfo: Codable, Sendable, Equatable {
        let mustNotInventToolIDs: Bool
        let mustNotBypassApproval: Bool
        let mustCiteRuntimeSourceWhenClaimingRuntimeState: Bool
    }
}

nonisolated enum SelfModelSnapshotBuilder {
    static func build(
        turn: AssistantTurnContext,
        budget: ContextBudgetPlan,
        selectedRuntime: AssistantRuntimeRouter.Selection,
        tools: [SecureToolDefinition],
        availableBackendKinds: [String] = [],
        activeSlot: LumenModelSlot? = nil,
        manifestCommit: String = "unknown",
        runtimeAuditPresent: Bool = false,
        now: Date = Date(),
        bundle: Bundle = .main
    ) -> SelfModelSnapshot {
        let slot = activeSlot ?? inferredSlot(for: turn.task)
        let sortedTools = tools.sorted { $0.id < $1.id }
        let availableToolIDs = sortedTools.map { ToolRouteGuard.canonicalToolID($0.id) }
        let approvalToolIDs = sortedTools
            .filter { requiresApprovalSummary($0) }
            .map { ToolRouteGuard.canonicalToolID($0.id) }
        let backgroundSafeToolIDs = sortedTools
            .filter { isBackgroundSafeSummary($0) }
            .map { ToolRouteGuard.canonicalToolID($0.id) }

        return SelfModelSnapshot(
            schemaVersion: SelfModelSnapshot.schemaVersion,
            generatedAt: now,
            app: .init(
                name: bundle.object(forInfoDictionaryKey: "CFBundleName") as? String ?? "Lumen",
                buildNumber: bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown",
                shortVersion: bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown",
                platform: "ios",
                mode: turn.isForeground ? "foreground" : "background"
            ),
            agent: .init(
                logicalIdentity: "lumen",
                activeSlot: slot.rawValue,
                availableSlots: LumenModelSlot.allCases.map(\.rawValue).sorted(),
                manifestCommit: manifestCommit,
                fleetContractVersion: LumenModelSlotContract.fleetContractVersion
            ),
            runtime: .init(
                selectedRuntimePathKind: runtimePathKind(for: selectedRuntime.runtime).rawValue,
                selectedRuntime: selectedRuntime.runtime.rawValue,
                selectionReason: selectedRuntime.reason,
                availableBackendKinds: availableBackendKinds.sorted(),
                embeddingAvailable: selectedRuntime.runtime == .coreML || availableBackendKinds.contains(LLMBackendKind.coreML.rawValue),
                thermalState: DeviceThermalState.from(processThermalState: turn.thermalState).rawValue,
                powerState: turn.lowPowerMode ? "low_power" : "battery_or_unknown",
                networkState: "unknown"
            ),
            contextBudget: .init(
                profile: budget.profile.rawValue,
                maxInputTokens: budget.maxInputTokens,
                sections: .init(budget.tokenSections)
            ),
            tools: .init(
                available: availableToolIDs,
                requiresApproval: approvalToolIDs,
                backgroundSafe: backgroundSafeToolIDs
            ),
            evidence: .init(
                manifestFreshness: "bundled",
                runtimeAuditPresent: runtimeAuditPresent,
                exportPolicy: .init(
                    sourceLayer: "agentGroundingRuntimeAudit",
                    ownsLiveE2EScenarios: false,
                    includesDeterministicStaticScenarios: false
                )
            ),
            policy: .init(
                mustNotInventToolIDs: true,
                mustNotBypassApproval: true,
                mustCiteRuntimeSourceWhenClaimingRuntimeState: true
            )
        )
    }

    private static func inferredSlot(for task: AssistantTaskKind) -> LumenModelSlot {
        switch task {
        case .agentPlan:
            return .cortex
        case .toolDecision:
            return .executor
        case .embedding:
            return .embedding
        case .remConsolidation:
            return .rem
        case .chat, .summarization, .memoryExtraction, .safetyClassification, .speechCommandParsing, .backgroundTrigger:
            return .mouth
        }
    }

    private static func runtimePathKind(for runtime: AssistantRuntimeKind) -> LumenRuntimePathKind {
        switch runtime {
        case .foundationModels:
            return .foundationModels
        case .coreML:
            return .coreML
        case .llama:
            return .llamaGGUF
        case .deterministicFallback:
            return .deterministicFallback
        }
    }

    private static func requiresApprovalSummary(_ definition: SecureToolDefinition) -> Bool {
        definition.requiresUserApproval || definition.category == .sensitiveAction || definition.category == .destructiveAction
    }

    private static func isBackgroundSafeSummary(_ definition: SecureToolDefinition) -> Bool {
        definition.supportsBackgroundExecution
            && !requiresApprovalSummary(definition)
            && (definition.category == .readOnly || definition.category == .permissionRead)
    }
}
