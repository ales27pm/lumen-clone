import Foundation
import OSLog

nonisolated enum LumenModelSlot: String, Codable, CaseIterable, Sendable, Identifiable {
    case cortex
    case executor
    case mouth
    case mimicry
    case rem
    case embedding

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .cortex: return "Cortex"
        case .executor: return "Executor"
        case .mouth: return "Mouth"
        case .mimicry: return "Mimicry"
        case .rem: return "REM"
        case .embedding: return "Embedding"
        }
    }

    var isMemoryResidentCandidate: Bool {
        switch self {
        case .mimicry, .embedding: return true
        case .cortex, .executor, .mouth, .rem: return false
        }
    }

    var shouldRunOnlyWhenIdle: Bool { self == .rem }
}

nonisolated enum LumenFleetRuntimeMode: String, Codable, Sendable {
    case v1MultiResident
    case qwen3AdapterRuntime

    var displayName: String {
        switch self {
        case .v1MultiResident: return "v1 adapter-first compatible"
        case .qwen3AdapterRuntime: return "Qwen3 shared-base LoRA adapter runtime"
        }
    }
}

nonisolated enum LumenRuntimePathKind: String, Codable, Sendable, Hashable {
    case llamaGGUF
    case coreML
    case foundationModels
    case deterministicFallback
    case unavailable
    case embedding
    case unknown
}

nonisolated enum LumenSlotOutputContract: String, Codable, Sendable, Hashable {
    case decisionObject
    case structuredJSON
    case finalText
    case diagnosticSummary
    case embeddingVector
}

nonisolated enum LumenSlotBudgetPolicy: String, Codable, Sendable, Hashable {
    case foregroundInteractive
    case maintenanceIdle
    case embedding
}

nonisolated struct LumenModelSlotContract: Sendable, Hashable {
    nonisolated static let fleetContractVersion = "2026.05.03-adapter-first"
    private nonisolated static let logger = Logger(subsystem: "ai.lumen.app", category: "model-fleet")

    nonisolated enum ContractError: LocalizedError, Equatable {
        case missingContract(slot: LumenModelSlot, appVersion: String, modelConfigVersion: String)
        case incompleteMapping(missingSlots: [LumenModelSlot], appVersion: String, modelConfigVersion: String)

        var errorDescription: String? {
            switch self {
            case .missingContract(let slot, let appVersion, let modelConfigVersion):
                return "Missing LumenModelSlotContract for slot=\(slot.rawValue) appVersion=\(appVersion) modelConfigVersion=\(modelConfigVersion)"
            case .incompleteMapping(let missingSlots, let appVersion, let modelConfigVersion):
                let slots = missingSlots.map(\.rawValue).sorted().joined(separator: ",")
                return "Incomplete LumenModelSlotContract mapping missingSlots=[\(slots)] appVersion=\(appVersion) modelConfigVersion=\(modelConfigVersion)"
            }
        }
    }
    let slot: LumenModelSlot
    let systemContract: String
    let defaultTemperature: Double
    let defaultTopP: Double
    let maxOutputTokens: Int
    let outputContract: LumenSlotOutputContract
    let budgetPolicy: LumenSlotBudgetPolicy
    let acceptedRuntimePathKinds: Set<LumenRuntimePathKind>

    static let cortex = LumenModelSlotContract(slot: .cortex, systemContract: "You are Lumen Cortex. Read the user intent and app state. Return a compact decision object describing the next model slot, whether a native capability is required, and a short rationale. Do not write the final user-facing answer.", defaultTemperature: 0.15, defaultTopP: 0.85, maxOutputTokens: 220, outputContract: .decisionObject, budgetPolicy: .foregroundInteractive, acceptedRuntimePathKinds: [.llamaGGUF])
    static let executor = LumenModelSlotContract(slot: .executor, systemContract: "You are Lumen Executor. Convert a Cortex decision into one validated structured capability request. Return only valid JSON. Do not explain.", defaultTemperature: 0.0, defaultTopP: 0.1, maxOutputTokens: 180, outputContract: .structuredJSON, budgetPolicy: .foregroundInteractive, acceptedRuntimePathKinds: [.llamaGGUF])
    static let mouth = LumenModelSlotContract(slot: .mouth, systemContract: "You are Lumen Mouth. Write the final user-facing response from approved facts and results. Be concise and do not invent actions.", defaultTemperature: 0.55, defaultTopP: 0.9, maxOutputTokens: 420, outputContract: .finalText, budgetPolicy: .foregroundInteractive, acceptedRuntimePathKinds: [.llamaGGUF])
    static let mimicry = LumenModelSlotContract(slot: .mimicry, systemContract: "You are Lumen Mimicry. Summarize the user's tone preference and rewrite assistant text without changing meaning.", defaultTemperature: 0.2, defaultTopP: 0.8, maxOutputTokens: 160, outputContract: .finalText, budgetPolicy: .foregroundInteractive, acceptedRuntimePathKinds: [.llamaGGUF])
    static let rem = LumenModelSlotContract(slot: .rem, systemContract: "You are Lumen REM. During idle cycles, compress traces, find repeated failures, and produce training records for later review.", defaultTemperature: 0.35, defaultTopP: 0.9, maxOutputTokens: 900, outputContract: .diagnosticSummary, budgetPolicy: .maintenanceIdle, acceptedRuntimePathKinds: [.llamaGGUF])
    static let embedding = LumenModelSlotContract(slot: .embedding, systemContract: "Embedding model slot for semantic memory.", defaultTemperature: 0, defaultTopP: 1, maxOutputTokens: 0, outputContract: .embeddingVector, budgetPolicy: .embedding, acceptedRuntimePathKinds: [.embedding, .coreML])

    static let all: [LumenModelSlotContract] = [.cortex, .executor, .mouth, .mimicry, .rem, .embedding]

    private static let contractsBySlot: [LumenModelSlot: LumenModelSlotContract] = [.cortex: .cortex, .executor: .executor, .mouth: .mouth, .mimicry: .mimicry, .rem: .rem, .embedding: .embedding]

    static func contract(for slot: LumenModelSlot) -> LumenModelSlotContract? { contractsBySlot[slot] }

    func acceptsRuntimePath(_ runtimePath: String) -> Bool {
        acceptedRuntimePathKinds.contains(Self.runtimePathKind(for: runtimePath))
    }

    static func runtimePathKind(for runtimePath: String) -> LumenRuntimePathKind {
        switch runtimePath {
        case "sharedAdapter", "sharedAdapterLoadedContinuation", "legacySlot", "legacySlotLoadedContinuation", "agent-model":
            return .llamaGGUF
        case "coreML":
            return .coreML
        case "foundationModels":
            return .foundationModels
        case "deterministic-compatibility", "deterministicFallback":
            return .deterministicFallback
        case "unavailable":
            return .unavailable
        case "embedding":
            return .embedding
        default:
            return .unknown
        }
    }

    static func requiredContract(for slot: LumenModelSlot) throws -> LumenModelSlotContract {
        try requiredContract(for: slot, using: contractsBySlot)
    }

    static func requiredContract(for slot: LumenModelSlot, using mapping: [LumenModelSlot: LumenModelSlotContract]) throws -> LumenModelSlotContract {
        if let contract = mapping[slot] { return contract }
        let error = ContractError.missingContract(slot: slot, appVersion: appVersionString(), modelConfigVersion: fleetContractVersion)
        emitMissingContractTelemetry(for: error)
        throw error
    }

    static func validateCompletenessAtStartup() throws { try validateCompleteness(using: contractsBySlot) }

    static func validateCompleteness(using mapping: [LumenModelSlot: LumenModelSlotContract]) throws {
        let missing = LumenModelSlot.allCases.filter { mapping[$0] == nil }
        guard missing.isEmpty else {
            let error = ContractError.incompleteMapping(missingSlots: missing, appVersion: appVersionString(), modelConfigVersion: fleetContractVersion)
            missing.forEach { slot in emitMissingContractTelemetry(for: .missingContract(slot: slot, appVersion: appVersionString(), modelConfigVersion: fleetContractVersion)) }
            throw error
        }
    }

    private static func emitMissingContractTelemetry(for error: ContractError) {
        guard case .missingContract(let slot, let appVersion, let modelConfigVersion) = error else { return }
        logger.error("slot_contract_missing slot=\(slot.rawValue, privacy: .public) appVersion=\(appVersion, privacy: .public) modelConfigVersion=\(modelConfigVersion, privacy: .public)")
    }

    private static func appVersionString() -> String {
        let bundle = Bundle.main
        let short = bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
        let build = bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown"
        return "\(short) (\(build))"
    }
}

nonisolated struct LumenModelAssignment: Sendable, Hashable {
    let slot: LumenModelSlot
    let modelID: UUID
    let repoID: String
    let localPath: String
    let fileName: String
    let sizeBytes: Int64
    let expectedSHA256: String?
    let displayName: String
    let parameters: String
    let quantization: String
    let modelFamily: LumenModelFamily?
    let artifactKind: ModelRole
    let adapterID: UUID?
    let adapterRepoID: String?
    let adapterPath: String?
    let adapterFileName: String?
    let adapterSizeBytes: Int64?
    let adapterExpectedSHA256: String?
    let adapterScale: Float

    var expectedSharedBaseContract: LumenTrainedModelRuntimeContract? {
        guard modelFamily == .qwen3 else { return nil }
        return LumenTrainedModelRuntimeRegistry.contract(for: .qwen3)
    }
    var configuredSharedBaseMatchesContract: Bool {
        guard let expectedSharedBaseContract else { return false }
        return expectedSharedBaseContract.matchesSharedBase(
            repoID: repoID,
            fileName: fileName,
            sizeBytes: sizeBytes,
            expectedSHA256: expectedSHA256
        )
    }
    var hasConfiguredRoleAdapter: Bool { adapterPath?.isEmpty == false }
    var usesRoleAdapter: Bool { artifactKind == .chat && hasConfiguredRoleAdapter }
    var expectedRoleAdapterContract: LumenAdapterRoleContract? {
        guard artifactKind == .chat, let modelFamily else { return nil }
        let contract = LumenTrainedModelRuntimeRegistry.contract(for: modelFamily)
        guard contract.selectAdapterByAgentSlot else { return nil }
        return contract.adapterRole(for: slot)
    }
    var expectedRoleAdapterRepoID: String? { expectedRoleAdapterContract?.adapterRepoID }
    var expectedRoleAdapterFileName: String? { expectedRoleAdapterContract?.adapterFileName }
    var requiresRoleAdapterForRuntime: Bool {
        usesRoleAdapter || expectedRoleAdapterContract != nil
    }

    var configuredRoleAdapterMatchesContract: Bool {
        guard let expected = expectedRoleAdapterContract,
              let adapterRepoID,
              let adapterFileName,
              let adapterSizeBytes,
              let adapterExpectedSHA256,
              CatalogModel.isValidSHA256(expected.adapterExpectedSHA256)
        else { return false }
        return adapterRepoID == expected.adapterRepoID
            && adapterFileName == expected.adapterFileName
            && adapterSizeBytes == expected.adapterSizeBytes
            && adapterExpectedSHA256.caseInsensitiveCompare(expected.adapterExpectedSHA256) == .orderedSame
    }
}

nonisolated enum LumenModelSelectionPolicy {
    enum Failure: LocalizedError, Equatable, Sendable {
        case incompatibleChatModel(family: LumenModelFamily, expectedFileName: String)

        var errorDescription: String? {
            switch self {
            case .incompatibleChatModel(let family, let expectedFileName):
                return "\(family.shortLabel) adapter mode can only activate its verified shared chat base (\(expectedFileName)). Switch model families before selecting this chat model."
            }
        }
    }

    static func trustedExpectedSHA256(
        repoID: String,
        fileName: String,
        sizeBytes: Int64,
        family: LumenModelFamily
    ) -> String? {
        switch family {
        case .qwen25:
            let expectedSHA256 = ModelCatalog.catalogModel(repoId: repoID, fileName: fileName)?.expectedSHA256
            return expectedSHA256.flatMap { CatalogModel.isValidSHA256($0) ? $0 : nil }
        case .qwen3:
            let contract = LumenTrainedModelRuntimeRegistry.contract(for: family)
            guard let catalog = ModelCatalog.catalogModel(repoId: repoID, fileName: fileName),
                  catalog.role == .chat,
                  catalog.repoId == contract.sharedBaseRepoID,
                  catalog.fileName == contract.sharedBaseFileName,
                  catalog.sizeBytes == contract.sharedBaseSizeBytes,
                  contract.matchesSharedBase(
                      repoID: repoID,
                      fileName: fileName,
                      sizeBytes: sizeBytes,
                      expectedSHA256: catalog.expectedSHA256
                  )
            else { return nil }
            return contract.sharedBaseExpectedSHA256
        }
    }

    static func isPersistedChatModelCompatible(
        repoID: String,
        fileName: String,
        sizeBytes: Int64,
        family: LumenModelFamily
    ) -> Bool {
        isChatModelCompatible(
            repoID: repoID,
            fileName: fileName,
            sizeBytes: sizeBytes,
            expectedSHA256: trustedExpectedSHA256(
                repoID: repoID,
                fileName: fileName,
                sizeBytes: sizeBytes,
                family: family
            ),
            family: family
        )
    }

    static func isChatModelCompatible(
        repoID: String,
        fileName: String,
        sizeBytes: Int64,
        expectedSHA256: String?,
        family: LumenModelFamily
    ) -> Bool {
        switch family {
        case .qwen25:
            return true
        case .qwen3:
            return LumenTrainedModelRuntimeRegistry.contract(for: family).matchesSharedBase(
                repoID: repoID,
                fileName: fileName,
                sizeBytes: sizeBytes,
                expectedSHA256: expectedSHA256
            )
        }
    }

    static func validatePersistedChatModel(
        repoID: String,
        fileName: String,
        sizeBytes: Int64,
        family: LumenModelFamily
    ) throws {
        guard isPersistedChatModelCompatible(
            repoID: repoID,
            fileName: fileName,
            sizeBytes: sizeBytes,
            family: family
        ) else {
            let expectedFileName = LumenTrainedModelRuntimeRegistry.contract(for: family).sharedBaseFileName
            throw Failure.incompatibleChatModel(family: family, expectedFileName: expectedFileName)
        }
    }
}

nonisolated struct LumenModelFleetSnapshot: Sendable, Hashable {
    let mode: LumenFleetRuntimeMode
    let assignments: [LumenModelSlot: LumenModelAssignment]
    let missingSlots: [LumenModelSlot]
    let missingAdapterSlots: [LumenModelSlot]
    let targetResidentSlots: Set<LumenModelSlot>
    let runtimeResidentSlots: Set<LumenModelSlot>

    init(mode: LumenFleetRuntimeMode = .v1MultiResident, assignments: [LumenModelSlot: LumenModelAssignment], missingSlots: [LumenModelSlot], missingAdapterSlots: [LumenModelSlot] = [], targetResidentSlots: Set<LumenModelSlot> = [], runtimeResidentSlots: Set<LumenModelSlot> = []) {
        self.mode = mode
        self.assignments = assignments
        self.missingSlots = missingSlots
        self.missingAdapterSlots = missingAdapterSlots
        self.targetResidentSlots = targetResidentSlots
        self.runtimeResidentSlots = runtimeResidentSlots
    }

    func assignment(for slot: LumenModelSlot) -> LumenModelAssignment? { assignments[slot] }

    var isRunnableV1: Bool {
        assignment(for: .cortex) != nil && assignment(for: .executor) != nil && assignment(for: .mouth) != nil && assignment(for: .mimicry) != nil
    }

    var isFullyAdapted: Bool {
        missingAdapterSlots.isEmpty
    }
}

@MainActor
enum LumenModelFleetResolver {
    static func resolveV1(appState: AppState, storedModels: [StoredModel]) -> LumenModelFleetSnapshot {
        resolveV1(activeChatModelID: appState.activeChatModelID, activeEmbeddingModelID: appState.activeEmbeddingModelID, storedModels: storedModels)
    }

    static func resolveV1(settings: SettingsSnapshot, storedModels: [StoredModel]) -> LumenModelFleetSnapshot {
        resolveV1(activeChatModelID: settings.activeChatModelID, activeEmbeddingModelID: settings.activeEmbeddingModelID, storedModels: storedModels)
    }

    nonisolated static func resolveV1(snapshot: ModelLoadSnapshot) -> LumenModelFleetSnapshot {
        resolveV1(activeChatModelID: snapshot.activeChatModelID, activeEmbeddingModelID: snapshot.activeEmbeddingModelID, selectedFamily: snapshot.selectedModelFamily, storedModels: snapshot.storedModels)
    }

    static func resolveV1(activeChatModelID: String?, activeEmbeddingModelID: String?, storedModels: [StoredModel]) -> LumenModelFleetSnapshot {
        var assignments: [LumenModelSlot: LumenModelAssignment] = [:]
        let existingStoredModels = storedModels.filter { modelFileExists($0) }
        let textModels = existingStoredModels.filter { $0.modelRole == .chat && isStandaloneLoadableChatArtifact($0) }
        let adapterModels = existingStoredModels.filter { $0.modelRole == .roleAdapter }
        let activeText = activeChatModelID.flatMap { id in textModels.first { $0.id.uuidString == id } }
        let selectedFamily = LumenModelFamily.persistedSelected
        let runtimeContract = LumenTrainedModelRuntimeRegistry.contract(for: selectedFamily)
        let qwen3AdapterBase = selectedFamily == .qwen3
            ? activeText.flatMap { model in
                LumenModelSelectionPolicy.isPersistedChatModelCompatible(
                    repoID: model.repoId,
                    fileName: model.fileName,
                    sizeBytes: model.sizeBytes,
                    family: selectedFamily
                ) ? model : nil
            }
            : nil

        if selectedFamily == .qwen3, let sharedBase = qwen3AdapterBase {
            for slot in runtimeContract.runtimeSlots {
                let adapter = preferredAdapter(for: slot, storedModels: adapterModels, contract: runtimeContract)
                assignments[slot] = assignment(slot: slot, model: sharedBase, family: .qwen3, adapter: adapter)
            }
        } else if selectedFamily != .qwen3, let activeText {
            for slot in [LumenModelSlot.cortex, .executor, .mouth, .mimicry, .rem] {
                assignments[slot] = assignment(slot: slot, model: activeText, family: selectedFamily, adapter: nil)
            }
        }

        if let activeEmbeddingModelID,
           let embed = existingStoredModels.first(where: {
               $0.id.uuidString == activeEmbeddingModelID && $0.modelRole == .embedding
           }) {
            assignments[.embedding] = assignment(slot: .embedding, model: embed)
        }

        let missing = LumenModelSlot.allCases.filter { assignments[$0] == nil }
        let mode: LumenFleetRuntimeMode = selectedFamily == .qwen3 && qwen3AdapterBase != nil ? .qwen3AdapterRuntime : .v1MultiResident
        let missingAdapterSlots = mode == .qwen3AdapterRuntime ? runtimeContract.runtimeSlots.filter { assignments[$0]?.adapterPath == nil } : []
        return LumenModelFleetSnapshot(mode: mode, assignments: assignments, missingSlots: missing, missingAdapterSlots: missingAdapterSlots, targetResidentSlots: Set(assignments.keys), runtimeResidentSlots: mode == .qwen3AdapterRuntime ? [.cortex] : [])
    }

    nonisolated private static func resolveV1(activeChatModelID: String?, activeEmbeddingModelID: String?, selectedFamily: LumenModelFamily, storedModels: [StoredModelLoadItem]) -> LumenModelFleetSnapshot {
        var assignments: [LumenModelSlot: LumenModelAssignment] = [:]
        let existingStoredModels = storedModels.filter { FileManager.default.fileExists(atPath: $0.resolvedPath) }
        let textModels = existingStoredModels.filter { $0.modelRole == .chat && isStandaloneLoadableChatArtifact($0) }
        let adapterModels = existingStoredModels.filter { $0.modelRole == .roleAdapter }
        let activeText = activeChatModelID.flatMap { id in textModels.first { $0.id.uuidString == id } }
        let runtimeContract = LumenTrainedModelRuntimeRegistry.contract(for: selectedFamily)
        let qwen3AdapterBase = selectedFamily == .qwen3
            ? activeText.flatMap { model in
                LumenModelSelectionPolicy.isChatModelCompatible(
                    repoID: model.repoId,
                    fileName: model.fileName,
                    sizeBytes: model.sizeBytes,
                    expectedSHA256: model.expectedSHA256,
                    family: selectedFamily
                ) ? model : nil
            }
            : nil

        if selectedFamily == .qwen3, let sharedBase = qwen3AdapterBase {
            for slot in runtimeContract.runtimeSlots {
                let adapter = preferredAdapter(for: slot, storedModels: adapterModels, contract: runtimeContract)
                assignments[slot] = assignment(slot: slot, model: sharedBase, family: .qwen3, adapter: adapter)
            }
        } else if selectedFamily != .qwen3, let activeText {
            for slot in [LumenModelSlot.cortex, .executor, .mouth, .mimicry, .rem] {
                assignments[slot] = assignment(slot: slot, model: activeText, family: selectedFamily, adapter: nil)
            }
        }

        if let activeEmbeddingModelID,
           let embed = existingStoredModels.first(where: {
               $0.id.uuidString == activeEmbeddingModelID && $0.modelRole == .embedding
           }) {
            assignments[.embedding] = assignment(slot: .embedding, model: embed)
        }

        let missing = LumenModelSlot.allCases.filter { assignments[$0] == nil }
        let mode: LumenFleetRuntimeMode = selectedFamily == .qwen3 && qwen3AdapterBase != nil ? .qwen3AdapterRuntime : .v1MultiResident
        let missingAdapterSlots = mode == .qwen3AdapterRuntime ? runtimeContract.runtimeSlots.filter { assignments[$0]?.adapterPath == nil } : []
        return LumenModelFleetSnapshot(mode: mode, assignments: assignments, missingSlots: missing, missingAdapterSlots: missingAdapterSlots, targetResidentSlots: Set(assignments.keys), runtimeResidentSlots: mode == .qwen3AdapterRuntime ? [.cortex] : [])
    }

    nonisolated private static func preferredAdapter(for slot: LumenModelSlot, storedModels: [StoredModelLoadItem], contract: LumenTrainedModelRuntimeContract) -> StoredModelLoadItem? {
        guard let role = contract.adapterRole(for: slot) else { return nil }
        return storedModels
            .filter { $0.repoId == role.adapterRepoID && $0.fileName == role.adapterFileName }
            .sorted { $0.downloadedAt > $1.downloadedAt }
            .first
    }

    nonisolated private static func preferredEmbedding(activeEmbeddingModelID: String?, storedModels: [StoredModelLoadItem]) -> StoredModelLoadItem? {
        let embedModels = storedModels.filter { $0.modelRole == .embedding }
        let activeEmbed = activeEmbeddingModelID.flatMap { id in embedModels.first { $0.id.uuidString == id } }
        return activeEmbed ?? preferredModel(for: .embedding, storedModels: embedModels) ?? mostRecentModel(from: embedModels)
    }

    nonisolated private static func preferredModel(for slot: LumenModelSlot, storedModels: [StoredModelLoadItem]) -> StoredModelLoadItem? {
        let weights = hintWeights(for: slot)
        let slotTokens = slotHintTokens(for: slot)
        return storedModels.map { model in (model: model, score: score(model, weights: weights)) }.filter { $0.score > 0 }.sorted { lhs, rhs in
            if lhs.score != rhs.score { return lhs.score > rhs.score }
            return lhs.model.downloadedAt > rhs.model.downloadedAt
        }.first { slot == .embedding || matchesSlotHint($0.model, slotTokens: slotTokens) }?.model
    }

    nonisolated private static func preferredFineTunedModel(for slot: LumenModelSlot, storedModels: [StoredModelLoadItem]) -> StoredModelLoadItem? {
        let slotTokens = slotHintTokens(for: slot)
        return storedModels.map { model in (model: model, score: fineTunedScore(model, slotTokens: slotTokens)) }.filter { $0.score > 0 }.sorted { lhs, rhs in
            if lhs.score != rhs.score { return lhs.score > rhs.score }
            return lhs.model.downloadedAt > rhs.model.downloadedAt
        }.first?.model
    }

    nonisolated private static func preferredTextModel(from models: [StoredModelLoadItem]) -> StoredModelLoadItem? {
        preferredModel(for: .cortex, storedModels: models) ?? preferredModel(for: .mouth, storedModels: models) ?? mostRecentModel(from: models)
    }

    nonisolated private static func mostRecentModel(from models: [StoredModelLoadItem]) -> StoredModelLoadItem? { models.sorted { $0.downloadedAt > $1.downloadedAt }.first }

    nonisolated private static func score(_ model: StoredModelLoadItem, weights: [String: Int]) -> Int {
        let primary = [model.name, model.repoId, model.fileName].joined(separator: " ").lowercased()
        let secondary = [model.parameters, model.quantization, model.role].joined(separator: " ").lowercased()
        return weights.reduce(0) { partial, item in
            let hint = item.key.lowercased()
            let weight = item.value
            if primary.contains(hint) { return partial + weight }
            if secondary.contains(hint) { return partial + max(1, weight / 2) }
            return partial
        }
    }

    nonisolated private static func fineTunedScore(_ model: StoredModelLoadItem, slotTokens: [String]) -> Int {
        let primary = [model.name, model.repoId, model.fileName, model.localPath].joined(separator: " ").lowercased()
        let secondary = [model.parameters, model.quantization, model.role].joined(separator: " ").lowercased()
        let primaryTokens = tokenSet(primary)
        let secondaryTokens = tokenSet(secondary)
        let standaloneTunedMarkers = ["release", "bake", "merged", "gguf", "finetune", "finetuned", "sft", "dpo", "orpo", "agent"]
        let tunedPhrases = ["release-bake", "release_bake", "release baked", "fine-tune", "fine_tune", "fine tuned"]
        let slotMatchPrimary = slotTokens.contains { primaryTokens.contains($0) }
        let slotMatchSecondary = slotTokens.contains { secondaryTokens.contains($0) }
        guard slotMatchPrimary || slotMatchSecondary else { return 0 }
        guard isStandaloneLoadableChatArtifact(model) else { return 0 }
        let tunedPrimary = standaloneTunedMarkers.contains { primaryTokens.contains($0) } || tunedPhrases.contains { primary.contains($0) }
        let tunedSecondary = standaloneTunedMarkers.contains { secondaryTokens.contains($0) } || tunedPhrases.contains { secondary.contains($0) }
        let releaseBake = primary.contains("release-bake") || primary.contains("release_bake") || primaryTokens.contains("release") && primaryTokens.contains("bake")
        var score = 0
        score += slotMatchPrimary ? 120 : 70
        score += releaseBake ? 160 : 0
        score += tunedPrimary ? 80 : 0
        score += tunedSecondary ? 30 : 0
        score += (tunedPrimary || tunedSecondary) ? 30 : 0
        return score
    }

    nonisolated private static func isStandaloneLoadableChatArtifact(_ model: StoredModelLoadItem) -> Bool {
        let artifactText = [model.repoId, model.fileName, model.localPath, model.parameters, model.quantization, model.role].joined(separator: " ").lowercased()
        let fileName = model.fileName.lowercased()
        let artifactTokens = tokenSet(artifactText)
        let hasAdapterMarker = fileName.hasSuffix(".lora") || fileName.hasSuffix(".adapter") || artifactTokens.contains("adapter") || artifactTokens.contains("lora")
        if hasAdapterMarker { return false }
        return fileName.hasSuffix(".gguf") || fileName.hasSuffix(".bin") || fileName.hasSuffix(".safetensors") || fileName.hasSuffix(".mlmodelc")
    }

    nonisolated private static func assignment(slot: LumenModelSlot, model: StoredModelLoadItem, family: LumenModelFamily? = nil, adapter: StoredModelLoadItem? = nil) -> LumenModelAssignment {
        let sharedBaseContract = family == .qwen3
            ? LumenTrainedModelRuntimeRegistry.contract(for: .qwen3)
            : nil
        let adapterContract = family.flatMap {
            LumenTrainedModelRuntimeRegistry.contract(for: $0).adapterRole(for: slot)
        }
        return LumenModelAssignment(
            slot: slot,
            modelID: model.id,
            repoID: model.repoId,
            localPath: model.resolvedPath,
            fileName: model.fileName,
            sizeBytes: sharedBaseContract?.sharedBaseSizeBytes ?? model.sizeBytes,
            expectedSHA256: sharedBaseContract?.sharedBaseExpectedSHA256 ?? model.expectedSHA256,
            displayName: model.name,
            parameters: model.parameters,
            quantization: model.quantization,
            modelFamily: family,
            artifactKind: model.modelRole,
            adapterID: adapter?.id,
            adapterRepoID: adapter?.repoId,
            adapterPath: adapter?.resolvedPath,
            adapterFileName: adapter?.fileName,
            adapterSizeBytes: adapter.map { adapterContract?.adapterSizeBytes ?? $0.sizeBytes },
            adapterExpectedSHA256: adapter == nil ? nil : (adapterContract?.adapterExpectedSHA256 ?? adapter?.expectedSHA256),
            adapterScale: 1.0
        )
    }

    private static func preferredAdapter(for slot: LumenModelSlot, storedModels: [StoredModel], contract: LumenTrainedModelRuntimeContract) -> StoredModel? {
        guard let role = contract.adapterRole(for: slot) else { return nil }
        return storedModels
            .filter { $0.repoId == role.adapterRepoID && $0.fileName == role.adapterFileName }
            .sorted { $0.downloadedAt > $1.downloadedAt }
            .first
    }

    private static func preferredEmbedding(activeEmbeddingModelID: String?, storedModels: [StoredModel]) -> StoredModel? {
        let embedModels = storedModels.filter { $0.modelRole == .embedding }
        let activeEmbed = activeEmbeddingModelID.flatMap { id in embedModels.first { $0.id.uuidString == id } }
        return activeEmbed ?? preferredModel(for: .embedding, storedModels: embedModels) ?? mostRecentModel(from: embedModels)
    }

    private static func preferredModel(for slot: LumenModelSlot, storedModels: [StoredModel]) -> StoredModel? {
        let weights = hintWeights(for: slot)
        let slotTokens = slotHintTokens(for: slot)
        return storedModels.map { model in (model: model, score: score(model, weights: weights)) }.filter { $0.score > 0 }.sorted { lhs, rhs in
            if lhs.score != rhs.score { return lhs.score > rhs.score }
            return lhs.model.downloadedAt > rhs.model.downloadedAt
        }.first { slot == .embedding || matchesSlotHint($0.model, slotTokens: slotTokens) }?.model
    }

    private static func preferredFineTunedModel(for slot: LumenModelSlot, storedModels: [StoredModel]) -> StoredModel? {
        let slotTokens = slotHintTokens(for: slot)
        return storedModels.map { model in (model: model, score: fineTunedScore(model, slotTokens: slotTokens)) }.filter { $0.score > 0 }.sorted { lhs, rhs in
            if lhs.score != rhs.score { return lhs.score > rhs.score }
            return lhs.model.downloadedAt > rhs.model.downloadedAt
        }.first?.model
    }

    private static func preferredTextModel(from models: [StoredModel]) -> StoredModel? {
        preferredModel(for: .cortex, storedModels: models) ?? preferredModel(for: .mouth, storedModels: models) ?? mostRecentModel(from: models)
    }

    private static func mostRecentModel(from models: [StoredModel]) -> StoredModel? { models.sorted { $0.downloadedAt > $1.downloadedAt }.first }

    nonisolated private static func slotHintTokens(for slot: LumenModelSlot) -> [String] {
        switch slot {
        case .cortex: return ["cortex"]
        case .executor: return ["executor"]
        case .mouth: return ["mouth"]
        case .mimicry: return ["mimicry"]
        case .rem: return ["rem"]
        case .embedding: return ["embedding", "embed"]
        }
    }

    nonisolated private static func hintWeights(for slot: LumenModelSlot) -> [String: Int] {
        switch slot {
        case .cortex: return ["cortex": 120, "release": 90, "bake": 90, "1.7b": 60, "qwen3": 50, "1.5b": 25, "coder": 20]
        case .executor: return ["executor": 120, "release": 90, "bake": 90, "json": 60, "qwen3": 50, "coder": 20]
        case .mouth: return ["mouth": 120, "release": 90, "bake": 90, "qwen3": 50, "voice": 20]
        case .mimicry: return ["mimicry": 120, "release": 90, "bake": 90, "qwen3": 50, "voice": 20]
        case .rem: return ["rem": 120, "release": 90, "bake": 90, "qwen3": 50, "phi": 20]
        case .embedding: return ["qwen3": 70, "embed": 50, "embedding": 40, "nomic": 30, "memory": 15]
        }
    }

    private static func score(_ model: StoredModel, weights: [String: Int]) -> Int {
        let primary = [model.name, model.repoId, model.fileName].joined(separator: " ").lowercased()
        let secondary = [model.parameters, model.quantization, model.role].joined(separator: " ").lowercased()
        return weights.reduce(0) { partial, item in
            let hint = item.key.lowercased()
            let weight = item.value
            if primary.contains(hint) { return partial + weight }
            if secondary.contains(hint) { return partial + max(1, weight / 2) }
            return partial
        }
    }

    nonisolated private static func matchesSlotHint(_ model: StoredModelLoadItem, slotTokens: [String]) -> Bool {
        let text = [model.name, model.repoId, model.fileName, model.localPath, model.parameters, model.quantization, model.role].joined(separator: " ").lowercased()
        let tokens = tokenSet(text)
        return slotTokens.contains { tokens.contains($0) }
    }

    private static func matchesSlotHint(_ model: StoredModel, slotTokens: [String]) -> Bool {
        let text = [model.name, model.repoId, model.fileName, model.localPath, model.parameters, model.quantization, model.role].joined(separator: " ").lowercased()
        let tokens = tokenSet(text)
        return slotTokens.contains { tokens.contains($0) }
    }

    private static func fineTunedScore(_ model: StoredModel, slotTokens: [String]) -> Int {
        let primary = [model.name, model.repoId, model.fileName, model.localPath].joined(separator: " ").lowercased()
        let secondary = [model.parameters, model.quantization, model.role].joined(separator: " ").lowercased()
        let primaryTokens = tokenSet(primary)
        let secondaryTokens = tokenSet(secondary)
        let standaloneTunedMarkers = ["release", "bake", "merged", "gguf", "finetune", "finetuned", "sft", "dpo", "orpo", "agent"]
        let tunedPhrases = ["release-bake", "release_bake", "release baked", "fine-tune", "fine_tune", "fine tuned"]
        let slotMatchPrimary = slotTokens.contains { primaryTokens.contains($0) }
        let slotMatchSecondary = slotTokens.contains { secondaryTokens.contains($0) }
        guard slotMatchPrimary || slotMatchSecondary else { return 0 }
        guard isStandaloneLoadableChatArtifact(model) else { return 0 }
        let tunedPrimary = standaloneTunedMarkers.contains { primaryTokens.contains($0) } || tunedPhrases.contains { primary.contains($0) }
        let tunedSecondary = standaloneTunedMarkers.contains { secondaryTokens.contains($0) } || tunedPhrases.contains { secondary.contains($0) }
        let releaseBake = primary.contains("release-bake") || primary.contains("release_bake") || primaryTokens.contains("release") && primaryTokens.contains("bake")
        var score = 0
        score += slotMatchPrimary ? 120 : 70
        score += releaseBake ? 160 : 0
        score += tunedPrimary ? 80 : 0
        score += tunedSecondary ? 30 : 0
        score += (tunedPrimary || tunedSecondary) ? 30 : 0
        return score
    }

    private static func isStandaloneLoadableChatArtifact(_ model: StoredModel) -> Bool {
        let artifactText = [model.repoId, model.fileName, model.localPath, model.parameters, model.quantization, model.role].joined(separator: " ").lowercased()
        let fileName = model.fileName.lowercased()
        let artifactTokens = tokenSet(artifactText)
        let hasAdapterMarker = fileName.hasSuffix(".lora") || fileName.hasSuffix(".adapter") || artifactTokens.contains("adapter") || artifactTokens.contains("lora")
        if hasAdapterMarker { return false }
        return fileName.hasSuffix(".gguf") || fileName.hasSuffix(".bin") || fileName.hasSuffix(".safetensors") || fileName.hasSuffix(".mlmodelc")
    }

    private static func modelFileExists(_ model: StoredModel) -> Bool {
        FileManager.default.fileExists(atPath: ModelStorage.resolvedModelURL(from: model.localPath, fileName: model.fileName).path)
    }

    nonisolated private static func tokenSet(_ value: String) -> Set<String> {
        Set(value.split { !$0.isLetter && !$0.isNumber }.map(String.init))
    }

    private static func assignment(slot: LumenModelSlot, model: StoredModel, family: LumenModelFamily? = nil, adapter: StoredModel? = nil) -> LumenModelAssignment {
        let modelCatalog = ModelCatalog.catalogModel(repoId: model.repoId, fileName: model.fileName)
        let adapterCatalog = adapter.flatMap { ModelCatalog.catalogModel(repoId: $0.repoId, fileName: $0.fileName) }
        let sharedBaseContract = family == .qwen3
            ? LumenTrainedModelRuntimeRegistry.contract(for: .qwen3)
            : nil
        let adapterContract = family.flatMap {
            LumenTrainedModelRuntimeRegistry.contract(for: $0).adapterRole(for: slot)
        }
        return LumenModelAssignment(
            slot: slot,
            modelID: model.id,
            repoID: model.repoId,
            localPath: ModelStorage.resolvedModelURL(from: model.localPath, fileName: model.fileName).path,
            fileName: model.fileName,
            sizeBytes: sharedBaseContract?.sharedBaseSizeBytes ?? modelCatalog?.sizeBytes ?? model.sizeBytes,
            expectedSHA256: sharedBaseContract?.sharedBaseExpectedSHA256 ?? modelCatalog?.expectedSHA256,
            displayName: model.name,
            parameters: model.parameters,
            quantization: model.quantization,
            modelFamily: family,
            artifactKind: model.modelRole,
            adapterID: adapter?.id,
            adapterRepoID: adapter?.repoId,
            adapterPath: adapter.map { ModelStorage.resolvedModelURL(from: $0.localPath, fileName: $0.fileName).path },
            adapterFileName: adapter?.fileName,
            adapterSizeBytes: adapter.map { adapterContract?.adapterSizeBytes ?? adapterCatalog?.sizeBytes ?? $0.sizeBytes },
            adapterExpectedSHA256: adapter == nil ? nil : (adapterContract?.adapterExpectedSHA256 ?? adapterCatalog?.expectedSHA256),
            adapterScale: 1.0
        )
    }
}
