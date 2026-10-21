import Foundation

/// Presentation and model-management actions follow the selected engine, while
/// embedding selection remains independent of the one generative assistant.
nonisolated struct CoreAIGemmaPresentationPolicy: Sendable, Equatable {
    let usesGemma: Bool

    func allowsModelManagement(role: ModelRole) -> Bool {
        !usesGemma || role == .embedding
    }

    func activeAssistantName(legacyName: String?) -> String {
        usesGemma ? "Gemma 4 E2B" : (legacyName ?? "No active model")
    }

    func catalogModels(_ models: [CatalogModel]) -> [CatalogModel] {
        var seen: Set<String> = []
        return models.filter { model in
            guard allowsModelManagement(role: model.role) else { return false }
            let artifact = model.repoId.lowercased() + "/" + model.fileName.lowercased()
            return seen.insert(artifact).inserted
        }
    }

    static func embeddingStatusMessage(_ status: String) -> String {
        switch status {
        case "ready": return "Local embeddings are ready for memory and document search."
        case "not_selected": return "Choose an embedding model for semantic memory and document search."
        case "selected_model_missing", "selected_file_missing": return "The selected embedding model is missing. Select or install it below."
        case "resource_denied": return "Embedding preparation is paused while device resources are unavailable."
        case "not_checked": return "Embedding readiness is checked when memory or document search needs it."
        default: return "The selected embedding model could not load. Select or reload a local embedding model below."
        }
    }

    static func embeddingDisplayName(_ name: String) -> String {
        for prefix in ["Fleet v1 Memory — ", "Fleet Bootstrap Memory — ", "Qwen2.5 Bootstrap Embedding — "] where name.hasPrefix(prefix) {
            return String(name.dropFirst(prefix.count))
        }
        return name.replacingOccurrences(of: " Bootstrap Embedding ", with: " Embedding ")
    }
}
