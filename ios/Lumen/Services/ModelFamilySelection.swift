import Foundation

nonisolated enum LumenModelFamily: String, CaseIterable, Identifiable, Codable, Hashable, Sendable {
    case qwen25 = "qwen2.5"
    case qwen3 = "qwen3"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .qwen25: return "Qwen 2.5 baseline"
        case .qwen3: return "Qwen3 fast adapter bootstrap"
        }
    }

    var shortLabel: String {
        switch self {
        case .qwen25: return "Qwen2.5"
        case .qwen3: return "Qwen3"
        }
    }

    var description: String {
        switch self {
        case .qwen25:
            return "Stable Qwen2.5 baseline fleet: Qwen2.5 chat base plus lightweight embedding support."
        case .qwen3:
            return "Fast Qwen3 adapter runtime: one shared chat base, role-specific LoRA GGUF adapters, and the Qwen3 embedding candidate."
        }
    }

    static let defaultFamily: LumenModelFamily = .qwen3

    static func fromStoredID(_ id: String?) -> LumenModelFamily {
        guard let id, let family = LumenModelFamily(rawValue: id) else { return defaultFamily }
        return family
    }

    static var persistedSelected: LumenModelFamily {
        get { PersistedModelSelectionStore.selectedFamily() }
        set {
            let current = PersistedModelSelectionStore.loadOrMigrate()
            _ = try? PersistedModelSelectionStore.commit(
                chatModelID: current.chatModelID,
                embeddingModelID: current.embeddingModelID,
                family: newValue,
                provisioningPlanID: current.familyID == newValue.rawValue ? current.provisioningPlanID : nil
            )
        }
    }
}

nonisolated extension LumenModelFleetCatalog {
    static var qwen25BootstrapModels: [CatalogModel] {
        let contract = LumenTrainedModelRuntimeRegistry.contract(for: .qwen25)
        return [
            CatalogModel(id: "fleet-bootstrap-qwen2.5-chat-base-q4", name: "Qwen2.5 Bootstrap Chat Base", repoId: contract.sharedBaseRepoID, fileName: contract.sharedBaseFileName, parameters: "1.5B", quantization: "Q4_K_M", sizeBytes: contract.sharedBaseSizeBytes, role: .chat, description: "Qwen2.5 baseline shared chat base. Use this family as the rollback/baseline candidate.", tags: ["bootstrap", "qwen2.5", "baseline", "shared-base"], sourceRevision: contract.sharedBaseSourceRevision, expectedSHA256: contract.sharedBaseExpectedSHA256),
            CatalogModel(id: "fleet-bootstrap-qwen2.5-embedding-nomic-q4", name: "Qwen2.5 Bootstrap Embedding — Nomic", repoId: contract.embeddingRepoID ?? "", fileName: contract.embeddingFileName ?? "", parameters: "137M", quantization: "Q4_K_M", sizeBytes: contract.embeddingSizeBytes ?? 0, role: .embedding, description: "Small embedding model for the Qwen2.5 baseline family.", tags: ["bootstrap", "qwen2.5", "embedding", "nomic"], sourceRevision: contract.embeddingSourceRevision ?? "", expectedSHA256: contract.embeddingExpectedSHA256 ?? ""),
        ]
    }

    static var qwen3BootstrapModels: [CatalogModel] {
        let contract = LumenTrainedModelRuntimeRegistry.qwen3AdapterBootstrapContract
        return [
            CatalogModel(id: "fleet-bootstrap-qwen3-fast-shared-q4", name: "Qwen3 Fast Shared Chat Base", repoId: contract.sharedBaseRepoID, fileName: contract.sharedBaseFileName, parameters: "1.7B", quantization: "Q4_K_M", sizeBytes: contract.sharedBaseSizeBytes, role: .chat, description: "Shared Qwen3 chat base loaded once for all Lumen role adapters.", tags: ["bootstrap", "qwen3", "adapter-runtime", "shared-base"], sourceRevision: contract.sharedBaseSourceRevision, expectedSHA256: contract.sharedBaseExpectedSHA256),
            CatalogModel(id: "fleet-bootstrap-qwen3-embedding-0.6b-q8", name: "Qwen3 Bootstrap Embedding 0.6B", repoId: contract.embeddingRepoID ?? "", fileName: contract.embeddingFileName ?? "", parameters: "0.6B", quantization: "Q8_0", sizeBytes: contract.embeddingSizeBytes ?? 0, role: .embedding, description: "Qwen3 embedding candidate for source-map, tool-schema, memory, RAG, and repair retrieval.", tags: ["bootstrap", "qwen3", "embedding", "current", "q8"], sourceRevision: contract.embeddingSourceRevision ?? "", expectedSHA256: contract.embeddingExpectedSHA256 ?? ""),
        ] + contract.adapterRoles.map { adapter in
            CatalogModel(id: "fleet-bootstrap-qwen3-\(adapter.roleID)-lora", name: "Qwen3 \(adapter.roleID.capitalized) LoRA Adapter", repoId: adapter.adapterRepoID, fileName: adapter.adapterFileName, parameters: "LoRA", quantization: "GGUF", sizeBytes: adapter.adapterSizeBytes, role: .roleAdapter, description: "Role-specific Qwen3 LoRA adapter for the \(adapter.roleID) runtime role.", tags: ["bootstrap", "qwen3", "adapter-runtime", "role-adapter", adapter.roleID], sourceRevision: adapter.adapterSourceRevision, expectedSHA256: adapter.adapterExpectedSHA256, sourcePath: adapter.adapterSourcePath)
        }
    }

    static func bootstrapModels(for family: LumenModelFamily) -> [CatalogModel] {
        switch family {
        case .qwen25: return qwen25BootstrapModels
        case .qwen3: return qwen3BootstrapModels
        }
    }

    static var selectableBootstrapModels: [CatalogModel] {
        qwen3BootstrapModels + qwen25BootstrapModels
    }
}
