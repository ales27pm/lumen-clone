import Foundation

enum BuiltInModelCatalog {
    static let all: [ModelCatalogEntry] = [
        ModelCatalogEntry(
            id: "builtin.tiny-intent",
            displayName: "Tiny Intent",
            backend: .tinyIntent,
            recommendedUse: .tinyIntent,
            source: .bundled,
            contextLength: 512,
            minimumRecommendedTier: .constrained,
            tags: ["builtin", "fallback", "intent"],
            notes: "Deterministic local rules engine for intent-style fallback routing."
        ),
        ModelCatalogEntry(
            id: "qwen2.5-1.5b-instruct-q4-k-m-gguf",
            displayName: "Qwen 2.5 1.5B Instruct Q4_K_M GGUF",
            backend: .gguf,
            recommendedUse: .standardChat,
            source: .unknown,
            parameterCountBillion: 1.5,
            quantization: "Q4_K_M",
            contextLength: 32_768,
            minimumRecommendedTier: .constrained,
            tags: ["gguf", "qwen", "instruct", "q4"],
            notes: "Descriptor only; Lumen policy may reduce active context based on device, memory, thermal, and power state."
        ),
        ModelCatalogEntry(
            id: "qwen2.5-3b-instruct-q4-k-m-gguf",
            displayName: "Qwen 2.5 3B Instruct Q4_K_M GGUF",
            backend: .gguf,
            recommendedUse: .deepThink,
            source: .unknown,
            parameterCountBillion: 3.0,
            quantization: "Q4_K_M",
            contextLength: 32_768,
            minimumRecommendedTier: .balanced,
            tags: ["gguf", "qwen", "instruct", "q4"],
            notes: "Descriptor only; active context and Deep Think availability are constrained by Lumen device policy."
        ),
        ModelCatalogEntry(
            id: "llama3.2-3b-instruct-q4-k-m-gguf",
            displayName: "Llama 3.2 3B Instruct Q4_K_M GGUF",
            backend: .gguf,
            recommendedUse: .standardChat,
            source: .unknown,
            parameterCountBillion: 3.0,
            quantization: "Q4_K_M",
            contextLength: 131_072,
            minimumRecommendedTier: .balanced,
            tags: ["gguf", "llama", "instruct", "q4"],
            notes: "Descriptor-level context metadata is high; active context is policy-limited before loading."
        ),
        ModelCatalogEntry(
            id: "nomic-embed-text-local",
            displayName: "Nomic Embed Text Local",
            backend: .gguf,
            recommendedUse: .testing,
            source: .unknown,
            contextLength: 8_192,
            minimumRecommendedTier: .balanced,
            tags: ["local"],
            notes: "Catalog descriptor for future local embedding support; embedding execution is not implemented yet."
        )
    ]

    static func entry(id: String) -> ModelCatalogEntry? {
        all.first { $0.id == id }
    }

    static func entries(for use: ModelRecommendedUse) -> [ModelCatalogEntry] {
        all.filter { $0.recommendedUse == use }
    }

    static func entries(backend: LLMBackendKind) -> [ModelCatalogEntry] {
        all.filter { $0.backend == backend }
    }
}
