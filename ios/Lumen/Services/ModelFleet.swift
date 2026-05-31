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

    static let cortex = LumenModelSlotContract(slot: .cortex, systemContract: "You are Lumen Cortex. Read the user intent and app state. Return a compact decision object describing the next model slot, whether a native capability is required, and a short rationale. Do not write the final user-facing answer.", defaultTemperature: 0.15, defaultTopP: 0.85, maxOutputTokens: 220)
    static let executor = LumenModelSlotContract(slot: .executor, systemContract: "You are Lumen Executor. Convert a Cortex decision into one validated structured capability request. Return only valid JSON. Do not explain.", defaultTemperature: 0.0, defaultTopP: 0.1, maxOutputTokens: 180)
    static let mouth = LumenModelSlotContract(slot: .mouth, systemContract: "You are Lumen Mouth. Write the final user-facing response from approved facts and results. Be concise and do not invent actions.", defaultTemperature: 0.55, defaultTopP: 0.9, maxOutputTokens: 420)
    static let mimicry = LumenModelSlotContract(slot: .mimicry, systemContract: "You are Lumen Mimicry. Summarize the user's tone preference and rewrite assistant text without changing meaning.", defaultTemperature: 0.2, defaultTopP: 0.8, maxOutputTokens: 160)
    static let rem = LumenModelSlotContract(slot: .rem, systemContract: "You are Lumen REM. During idle cycles, compress traces, find repeated failures, and produce training records for later review.", defaultTemperature: 0.35, defaultTopP: 0.9, maxOutputTokens: 900)
    static let embedding = LumenModelSlotContract(slot: .embedding, systemContract: "Embedding model slot for semantic memory.", defaultTemperature: 0, defaultTopP: 1, maxOutputTokens: 0)

    static let all: [LumenModelSlotContract] = [.cortex, .executor, .mouth, .mimicry, .rem, .embedding]

    private static let contractsBySlot: [LumenModelSlot: LumenModelSlotContract] = [.cortex: .cortex, .executor: .executor, .mouth: .mouth, .mimicry: .mimicry, .rem: .rem, .embedding: .embedding]

    static func contract(for slot: LumenModelSlot) -> LumenModelSlotContract? { contractsBySlot[slot] }

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
    let localPath: String
    let fileName: String
    let displayName: String
    let parameters: String
    let quantization: String
    let modelFamily: LumenModelFamily?
    let artifactKind: ModelRole
    let adapterID: UUID?
    let adapterPath: String?
    let adapterFileName: String?
    let adapterScale: Float

    var usesRoleAdapter: Bool { artifactKind == .chat && adapterPath != nil }
}

nonisolated struct LumenModelFleetSnapshot: Sendable, Hashable {
    let mode: LumenFleetRuntimeMode
    let assignments: [LumenModelSlot: LumenModelAssignment]
    let missingSlots: [LumenModelSlot]
    let targetResidentSlots: Set<LumenModelSlot>
    let runtimeResidentSlots: Set<LumenModelSlot>

    init(mode: LumenFleetRuntimeMode = .v1MultiResident, assignments: [LumenModelSlot: LumenModelAssignment], missingSlots: [LumenModelSlot], targetResidentSlots: Set<LumenModelSlot> = [], runtimeResidentSlots: Set<LumenModelSlot> = []) {
        self.mode = mode
        self.assignments = assignments
        self.missingSlots = missingSlots
        self.targetResidentSlots = targetResidentSlots
        self.runtimeResidentSlots = runtimeResidentSlots
    }

    func assignment(for slot: LumenModelSlot) -> LumenModelAssignment? { assignments[slot] }

    var isRunnableV1: Bool {
        assignment(for: .cortex) != nil && assignment(for: .executor) != nil && assignment(for: .mouth) != nil && assignment(for: .mimicry) != nil
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

    static func resolveV1(activeChatModelID: String?, activeEmbeddingModelID: String?, storedModels: [StoredModel]) -> LumenModelFleetSnapshot {
        var assignments: [LumenModelSlot: LumenModelAssignment] = [:]
        let existingStoredModels = storedModels.filter { modelFileExists($0) }
        let textModels = existingStoredModels.filter { $0.modelRole == .chat && isStandaloneLoadableChatArtifact($0) }
        let adapterModels = existingStoredModels.filter { $0.modelRole == .roleAdapter }
        let activeText = activeChatModelID.flatMap { id in textModels.first { $0.id.uuidString == id } }
        let fallbackText = activeText ?? preferredTextModel(from: textModels)
        let selectedFamily = LumenModelFamily.persistedSelected
        let qwen3AdapterBase = (activeText?.isQwen3SharedAdapterBase == true ? activeText : nil)
            ?? textModels.filter(\.isQwen3SharedAdapterBase).sorted { $0.downloadedAt > $1.downloadedAt }.first

        if selectedFamily == .qwen3, let sharedBase = qwen3AdapterBase {
            for slot in [LumenModelSlot.cortex, .executor, .mouth, .mimicry, .rem] {
                let adapter = preferredAdapter(for: slot, storedModels: adapterModels)
                assignments[slot] = assignment(slot: slot, model: sharedBase, family: .qwen3, adapter: adapter)
            }
        } else {
            let fallbackFamily: LumenModelFamily? = selectedFamily == .qwen3 ? nil : selectedFamily
            for slot in [LumenModelSlot.cortex, .executor, .mouth, .mimicry, .rem] {
                if let model = preferredFineTunedModel(for: slot, storedModels: textModels)
                    ?? preferredModel(for: slot, storedModels: textModels)
                    ?? fallbackText {
                    assignments[slot] = assignment(slot: slot, model: model, family: fallbackFamily, adapter: nil)
                }
            }
        }

        if let embed = preferredEmbedding(activeEmbeddingModelID: activeEmbeddingModelID, storedModels: existingStoredModels) {
            assignments[.embedding] = assignment(slot: .embedding, model: embed)
        }

        let missing = LumenModelSlot.allCases.filter { assignments[$0] == nil }
        return LumenModelFleetSnapshot(mode: selectedFamily == .qwen3 && qwen3AdapterBase != nil ? .qwen3AdapterRuntime : .v1MultiResident, assignments: assignments, missingSlots: missing, targetResidentSlots: Set(assignments.keys), runtimeResidentSlots: selectedFamily == .qwen3 && qwen3AdapterBase != nil ? [.cortex] : [])
    }

    private static func preferredAdapter(for slot: LumenModelSlot, storedModels: [StoredModel]) -> StoredModel? {
        let slotToken = slot.rawValue
        let scored = storedModels.compactMap { model -> (model: StoredModel, rank: Int)? in
            let text = [model.name, model.repoId, model.fileName, model.localPath].joined(separator: " ").lowercased()
            if text.contains(slotToken) { return (model, 2) }
            if slot == .cortex, text.contains("fleet") { return (model, 1) }
            return nil
        }
        return scored.sorted { lhs, rhs in
            if lhs.rank != rhs.rank { return lhs.rank > rhs.rank }
            return lhs.model.downloadedAt > rhs.model.downloadedAt
        }.first?.model
    }

    private static func preferredEmbedding(activeEmbeddingModelID: String?, storedModels: [StoredModel]) -> StoredModel? {
        let embedModels = storedModels.filter { $0.modelRole == .embedding }
        let activeEmbed = activeEmbeddingModelID.flatMap { id in embedModels.first { $0.id.uuidString == id } }
        return activeEmbed ?? preferredModel(for: .embedding, storedModels: embedModels) ?? mostRecentModel(from: embedModels)
    }

    private static func preferredModel(for slot: LumenModelSlot, storedModels: [StoredModel]) -> StoredModel? {
        let weights = hintWeights(for: slot)
        return storedModels.map { model in (model: model, score: score(model, weights: weights)) }.filter { $0.score > 0 }.sorted { lhs, rhs in
            if lhs.score != rhs.score { return lhs.score > rhs.score }
            return lhs.model.downloadedAt > rhs.model.downloadedAt
        }.first?.model
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

    private static func slotHintTokens(for slot: LumenModelSlot) -> [String] {
        switch slot {
        case .cortex: return ["cortex"]
        case .executor: return ["executor"]
        case .mouth: return ["mouth"]
        case .mimicry: return ["mimicry"]
        case .rem: return ["rem"]
        case .embedding: return ["embedding", "embed"]
        }
    }

    private static func hintWeights(for slot: LumenModelSlot) -> [String: Int] {
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

    private static func tokenSet(_ value: String) -> Set<String> {
        Set(value.split { !$0.isLetter && !$0.isNumber }.map(String.init))
    }

    private static func assignment(slot: LumenModelSlot, model: StoredModel, family: LumenModelFamily? = nil, adapter: StoredModel? = nil) -> LumenModelAssignment {
        LumenModelAssignment(
            slot: slot,
            modelID: model.id,
            localPath: ModelStorage.resolvedModelURL(from: model.localPath, fileName: model.fileName).path,
            fileName: model.fileName,
            displayName: model.name,
            parameters: model.parameters,
            quantization: model.quantization,
            modelFamily: family,
            artifactKind: model.modelRole,
            adapterID: adapter?.id,
            adapterPath: adapter.map { ModelStorage.resolvedModelURL(from: $0.localPath, fileName: $0.fileName).path },
            adapterFileName: adapter?.fileName,
            adapterScale: 1.0
        )
    }
}


private extension StoredModel {
    var isQwen3SharedAdapterBase: Bool {
        repoId == "ales27pm/lumen-qwen3-bootstrap-gguf" && fileName == "lumen-qwen3-fast-shared-q4_k_m.gguf"
    }
}
