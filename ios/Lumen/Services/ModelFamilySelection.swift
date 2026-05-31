import Foundation

nonisolated enum LumenModelFamily: String, CaseIterable, Identifiable, Codable, Hashable, Sendable {
    case qwen25 = "qwen2.5"
    case qwen3 = "qwen3"

    private static let defaultsKey = "selectedModelFamilyID"

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
            return "Stable Qwen2.5 baseline fleet: Qwen2.5 chat base plus lightweight embedding fallback."
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
        get { fromStoredID(UserDefaults.standard.string(forKey: defaultsKey)) }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: defaultsKey) }
    }
}

nonisolated extension LumenModelFleetCatalog {
    static var qwen25BootstrapModels: [CatalogModel] {
        [
            CatalogModel(id: "fleet-bootstrap-qwen2.5-chat-base-q4", name: "Qwen2.5 Bootstrap Chat Base", repoId: "Qwen/Qwen2.5-1.5B-Instruct-GGUF", fileName: "qwen2.5-1.5b-instruct-q4_k_m.gguf", parameters: "1.5B", quantization: "Q4_K_M", sizeBytes: 1_117_000_000, role: .chat, description: "Qwen2.5 baseline shared chat base. Use this family as the rollback/baseline candidate.", tags: ["bootstrap", "qwen2.5", "baseline", "shared-base"]),
            CatalogModel(id: "fleet-bootstrap-qwen2.5-embedding-fallback-nomic-q4", name: "Qwen2.5 Bootstrap Embedding Fallback — Nomic", repoId: "nomic-ai/nomic-embed-text-v1.5-GGUF", fileName: "nomic-embed-text-v1.5.Q4_K_M.gguf", parameters: "137M", quantization: "Q4_K_M", sizeBytes: 85_000_000, role: .embedding, description: "Small embedding fallback for the Qwen2.5 baseline family.", tags: ["bootstrap", "qwen2.5", "embedding", "fallback", "nomic"]),
        ]
    }

    static var qwen3BootstrapModels: [CatalogModel] {
        let adapterRepo = "ales27pm/lumen-qwen3-bootstrap-adapters-gguf"
        let adapters: [(roleID: String, fileName: String, sizeBytes: Int64)] = [
            ("cortex", "lumen-cortex-lora.gguf", 70_000_000),
            ("executor", "lumen-executor-lora.gguf", 70_000_000),
            ("mouth", "lumen-mouth-lora.gguf", 70_000_000),
            ("mimicry", "lumen-mimicry-lora.gguf", 70_000_000),
            ("rem", "lumen-rem-lora.gguf", 70_000_000),
            ("fleet", "lumen-fleet-lora.gguf", 70_000_000),
        ]
        return [
            CatalogModel(id: "fleet-bootstrap-qwen3-fast-shared-q4", name: "Qwen3 Fast Shared Chat Base", repoId: "ales27pm/lumen-qwen3-bootstrap-gguf", fileName: "lumen-qwen3-fast-shared-q4_k_m.gguf", parameters: "1.7B", quantization: "Q4_K_M", sizeBytes: 1_350_000_000, role: .chat, description: "Shared Qwen3 chat base loaded once for all Lumen role adapters.", tags: ["bootstrap", "qwen3", "adapter-runtime", "shared-base"]),
            CatalogModel(id: "fleet-bootstrap-qwen3-embedding-0.6b-q8", name: "Qwen3 Bootstrap Embedding 0.6B", repoId: "Qwen/Qwen3-Embedding-0.6B-GGUF", fileName: "Qwen3-Embedding-0.6B-Q8_0.gguf", parameters: "0.6B", quantization: "Q8_0", sizeBytes: 650_000_000, role: .embedding, description: "Qwen3 embedding candidate for source-map, tool-schema, memory, RAG, and repair retrieval.", tags: ["bootstrap", "qwen3", "embedding", "current", "q8"]),
        ] + adapters.map { adapter in
            CatalogModel(id: "fleet-bootstrap-qwen3-\(adapter.roleID)-lora", name: "Qwen3 \(adapter.roleID.capitalized) LoRA Adapter", repoId: adapterRepo, fileName: adapter.fileName, parameters: "LoRA", quantization: "GGUF", sizeBytes: adapter.sizeBytes, role: .roleAdapter, description: "Role-specific Qwen3 LoRA adapter for the \(adapter.roleID) runtime role.", tags: ["bootstrap", "qwen3", "adapter-runtime", "role-adapter", adapter.roleID])
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
