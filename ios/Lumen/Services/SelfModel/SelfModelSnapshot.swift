import Foundation

nonisolated struct SelfModelSnapshot: Codable, Sendable, Equatable {
    static let schemaVersion = "0.2.0"

    nonisolated enum CapabilityAvailability: String, Codable, Sendable, Equatable {
        case unknown
        case available
        case unavailable

        init(from decoder: Decoder) throws {
            let value = try decoder.singleValueContainer().decode(String.self)
            self = Self(rawValue: value) ?? .unknown
        }

        func encode(to encoder: Encoder) throws {
            var value = encoder.singleValueContainer()
            try value.encode(rawValue)
        }
    }

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
        private enum CodingKeys: String, CodingKey { case name, buildNumber, shortVersion, platform, mode, sourceRevision }
        let name: String
        let buildNumber: String
        let shortVersion: String
        let platform: String
        let mode: String
        /// Build metadata only; never inferred from a developer checkout or the grounding manifest.
        let sourceRevision: String?
    }

    nonisolated struct AgentInfo: Codable, Sendable, Equatable {
        private enum CodingKeys: String, CodingKey { case logicalIdentity, activeSlot, availableSlots, declaredSlots, slotAvailability, manifestCommit, fleetContractVersion }
        let logicalIdentity: String
        let activeSlot: String
        let availableSlots: [String]
        let declaredSlots: [String]
        let slotAvailability: CapabilityAvailability
        let manifestCommit: String
        let fleetContractVersion: String
    }

    nonisolated struct RuntimeInfo: Codable, Sendable, Equatable {
        private enum CodingKeys: String, CodingKey { case selectedRuntimePathKind, selectedRuntime, selectionReason, readiness, availableBackendKinds, embeddingAvailable, embeddingReadiness, thermalState, powerState, networkState }
        let selectedRuntimePathKind: String
        let selectedRuntime: String
        let selectionReason: String
        let readiness: CapabilityAvailability
        let availableBackendKinds: [String]
        /// nil means not observed. Generative backend selection says nothing about embeddings.
        let embeddingAvailable: Bool?
        let embeddingReadiness: CapabilityAvailability
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
        private enum CodingKeys: String, CodingKey { case availabilityScope, available, requiresApproval, backgroundSafe }
        let availabilityScope: String
        let available: [String]
        let requiresApproval: [String]
        let backgroundSafe: [String]
    }

    nonisolated struct EvidenceInfo: Codable, Sendable, Equatable {
        private enum CodingKeys: String, CodingKey { case manifestFreshness, runtimeAuditPresent, exportPolicy }
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
        embeddingAvailable: Bool? = nil,
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
            .flatMap { approvalSummaryIDs($0) }
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
                mode: turn.isForeground ? "foreground" : "background",
                sourceRevision: validatedSourceRevision(bundle.object(forInfoDictionaryKey: "LumenGitSHA"))
            ),
            agent: .init(
                logicalIdentity: "lumen",
                activeSlot: selectedRuntime.runtime == .coreAI ? "assistant" : slot.rawValue,
                availableSlots: [],
                declaredSlots: selectedRuntime.runtime == .coreAI
                    ? ["assistant"]
                    : LumenModelSlot.allCases.map(\.rawValue).sorted(),
                slotAvailability: .unknown,
                manifestCommit: validatedSourceRevision(manifestCommit) ?? "unknown",
                fleetContractVersion: LumenModelSlotContract.fleetContractVersion
            ),
            runtime: .init(
                selectedRuntimePathKind: runtimePathKind(for: selectedRuntime.runtime).rawValue,
                selectedRuntime: selectedRuntime.runtime.rawValue,
                selectionReason: selectedRuntime.reason,
                readiness: .unknown,
                availableBackendKinds: availableBackendKinds.sorted(),
                embeddingAvailable: embeddingAvailable,
                embeddingReadiness: embeddingAvailable.map { $0 ? .available : .unavailable } ?? .unknown,
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
                availabilityScope: "policy_filtered_definitions_not_execution_guarantee",
                available: availableToolIDs,
                requiresApproval: approvalToolIDs,
                backgroundSafe: backgroundSafeToolIDs
            ),
            evidence: .init(
                // A bundled artifact is not evidence that it matches the running source.
                manifestFreshness: "unknown",
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

    static func validatedSourceRevision(_ value: Any?) -> String? {
        guard let raw = value as? String else { return nil }
        let revision = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard revision.count == 40 || revision.count == 64,
              revision.unicodeScalars.allSatisfy({ CharacterSet(charactersIn: "0123456789abcdef").contains($0) }) else {
            return nil
        }
        return revision
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
        case .coreAI:
            return .coreAI
        case .llama:
            return .llamaGGUF
        case .deterministicFallback:
            return .deterministicFallback
        case .unavailable:
            return .unavailable
        }
    }

    private static func requiresApprovalSummary(_ definition: SecureToolDefinition) -> Bool {
        definition.requiresUserApproval || definition.category == .sensitiveAction || definition.category == .destructiveAction
    }

    private static func approvalSummaryIDs(_ definition: SecureToolDefinition) -> [String] {
        let canonical = ToolRouteGuard.canonicalToolID(definition.id)
        if canonical == definition.id {
            return [canonical]
        }
        return [definition.id, canonical]
    }

    private static func isBackgroundSafeSummary(_ definition: SecureToolDefinition) -> Bool {
        definition.supportsBackgroundExecution
            && !requiresApprovalSummary(definition)
            && (definition.category == .readOnly || definition.category == .permissionRead)
    }
}

// Older snapshots remain readable, but their catalogue/backend-derived flags must
// not acquire an observed-readiness meaning when replayed by the new runtime.
extension SelfModelSnapshot.AppInfo {
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        name = try values.decode(String.self, forKey: .name)
        buildNumber = try values.decode(String.self, forKey: .buildNumber)
        shortVersion = try values.decode(String.self, forKey: .shortVersion)
        platform = try values.decode(String.self, forKey: .platform)
        mode = try values.decode(String.self, forKey: .mode)
        sourceRevision = SelfModelSnapshotBuilder.validatedSourceRevision(try values.decodeIfPresent(String.self, forKey: .sourceRevision))
    }
}

extension SelfModelSnapshot.AgentInfo {
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        logicalIdentity = try values.decode(String.self, forKey: .logicalIdentity)
        activeSlot = try values.decode(String.self, forKey: .activeSlot)
        let legacySlots = try values.decode([String].self, forKey: .availableSlots)
        declaredSlots = try values.decodeIfPresent([String].self, forKey: .declaredSlots) ?? legacySlots
        slotAvailability = try values.decodeIfPresent(SelfModelSnapshot.CapabilityAvailability.self, forKey: .slotAvailability) ?? .unknown
        availableSlots = slotAvailability == .available ? legacySlots : []
        manifestCommit = try values.decode(String.self, forKey: .manifestCommit)
        fleetContractVersion = try values.decode(String.self, forKey: .fleetContractVersion)
    }
}

extension SelfModelSnapshot.RuntimeInfo {
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        selectedRuntimePathKind = try values.decode(String.self, forKey: .selectedRuntimePathKind)
        selectedRuntime = try values.decode(String.self, forKey: .selectedRuntime)
        selectionReason = try values.decode(String.self, forKey: .selectionReason)
        readiness = try values.decodeIfPresent(SelfModelSnapshot.CapabilityAvailability.self, forKey: .readiness) ?? .unknown
        availableBackendKinds = try values.decode([String].self, forKey: .availableBackendKinds)
        embeddingReadiness = try values.decodeIfPresent(SelfModelSnapshot.CapabilityAvailability.self, forKey: .embeddingReadiness) ?? .unknown
        switch embeddingReadiness {
        case .available: embeddingAvailable = true
        case .unavailable: embeddingAvailable = false
        case .unknown: embeddingAvailable = nil
        }
        thermalState = try values.decode(String.self, forKey: .thermalState)
        powerState = try values.decode(String.self, forKey: .powerState)
        networkState = try values.decode(String.self, forKey: .networkState)
    }
}

extension SelfModelSnapshot.ToolInfo {
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        availabilityScope = try values.decodeIfPresent(String.self, forKey: .availabilityScope) ?? "policy_filtered_definitions_not_execution_guarantee"
        available = try values.decode([String].self, forKey: .available)
        requiresApproval = try values.decode([String].self, forKey: .requiresApproval)
        backgroundSafe = try values.decode([String].self, forKey: .backgroundSafe)
    }
}

extension SelfModelSnapshot.EvidenceInfo {
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let legacyFreshness = try values.decode(String.self, forKey: .manifestFreshness)
        manifestFreshness = legacyFreshness == "bundled" ? "unknown" : legacyFreshness
        runtimeAuditPresent = try values.decode(Bool.self, forKey: .runtimeAuditPresent)
        exportPolicy = try values.decode(SelfModelSnapshot.ExportPolicy.self, forKey: .exportPolicy)
    }
}
