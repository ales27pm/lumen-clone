import Foundation

nonisolated enum LumenModelFleetCatalog {
    static let v1AdapterFirstBase: [CatalogModel] = [
        CatalogModel(
            id: "fleet-v1-adapter-base-qwen2.5-1.5b-q4",
            name: "Fleet v1 Adapter Base — Qwen 2.5 1.5B",
            repoId: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            fileName: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            parameters: "1.5B",
            quantization: "Q4_K_M",
            sizeBytes: 1_117_000_000,
            role: .chat,
            description: "Default shared chat base for the adapter-first fleet. Role adapters stay separate unless an explicit release bake is produced.",
            tags: ["fleet-v1", "adapter-first", "shared-base", "qwen", "baseline"]
        ),
        CatalogModel(
            id: "fleet-v1-qwen3-embedding-0.6b",
            name: "Fleet v1 Memory — Qwen3 Embedding 0.6B",
            repoId: "Qwen/Qwen3-Embedding-0.6B-GGUF",
            fileName: "qwen3-embedding-0.6b-q4_k_m.gguf",
            parameters: "0.6B",
            quantization: "Q4_K_M",
            sizeBytes: 450_000_000,
            role: .embedding,
            description: "Qwen3 embedding candidate for source-map, memory, RAG, runtime repair, and tool-schema retrieval. Keep fallback enabled until eval gates pass.",
            tags: ["fleet-v1", "adapter-first", "embedding", "qwen3", "candidate"]
        )
    ]

    static let v1ReleaseBaked: [CatalogModel] = [
        CatalogModel(
            id: "fleet-v1-release-bake-cortex-qwen1.5b-q4",
            name: "Fleet v1 Release Bake Cortex — Qwen 1.5B",
            repoId: "ales27pm/lumen-fleet-gguf",
            fileName: "lumen-cortex-release-bake-q4_k_m.gguf",
            parameters: "1.5B",
            quantization: "Q4_K_M",
            sizeBytes: 1_150_000_000,
            role: .chat,
            description: "Optional release-baked Cortex artifact for runtimes that cannot load adapters dynamically. Not the default training artifact.",
            tags: ["fleet-v1", "release-bake", "optional", "gguf", "cortex"]
        ),
        CatalogModel(
            id: "fleet-v1-release-bake-executor-qwen1.5b-q4",
            name: "Fleet v1 Release Bake Executor — Qwen 1.5B",
            repoId: "ales27pm/lumen-fleet-gguf",
            fileName: "lumen-executor-release-bake-q4_k_m.gguf",
            parameters: "1.5B",
            quantization: "Q4_K_M",
            sizeBytes: 1_150_000_000,
            role: .chat,
            description: "Optional release-baked Executor artifact for strict tool JSON when adapter loading is unavailable.",
            tags: ["fleet-v1", "release-bake", "optional", "gguf", "executor", "structured"]
        ),
        CatalogModel(
            id: "fleet-v1-release-bake-mouth-qwen1.5b-q4",
            name: "Fleet v1 Release Bake Mouth — Qwen 1.5B",
            repoId: "ales27pm/lumen-fleet-gguf",
            fileName: "lumen-mouth-release-bake-q4_k_m.gguf",
            parameters: "1.5B",
            quantization: "Q4_K_M",
            sizeBytes: 1_150_000_000,
            role: .chat,
            description: "Optional release-baked Mouth artifact for user-facing responses when adapter loading is unavailable.",
            tags: ["fleet-v1", "release-bake", "optional", "gguf", "mouth"]
        ),
        CatalogModel(
            id: "fleet-v1-release-bake-mimicry-qwen1.5b-q4",
            name: "Fleet v1 Release Bake Mimicry — Qwen 1.5B",
            repoId: "ales27pm/lumen-fleet-gguf",
            fileName: "lumen-mimicry-release-bake-q4_k_m.gguf",
            parameters: "1.5B",
            quantization: "Q4_K_M",
            sizeBytes: 1_150_000_000,
            role: .chat,
            description: "Optional release-baked Mimicry artifact for style adaptation when adapter loading is unavailable.",
            tags: ["fleet-v1", "release-bake", "optional", "gguf", "mimicry"]
        ),
        CatalogModel(
            id: "fleet-v1-release-bake-rem-qwen1.5b-q4",
            name: "Fleet v1 Release Bake REM — Qwen 1.5B",
            repoId: "ales27pm/lumen-fleet-gguf",
            fileName: "lumen-rem-release-bake-q4_k_m.gguf",
            parameters: "1.5B",
            quantization: "Q4_K_M",
            sizeBytes: 1_150_000_000,
            role: .chat,
            description: "Optional release-baked REM artifact for reflection and repair when adapter loading is unavailable.",
            tags: ["fleet-v1", "release-bake", "optional", "gguf", "rem", "idle"]
        ),
    ]

    // Backward-compatible alias. These are optional release-baked full-model artifacts,
    // not the default output of the improvement loop.
    static var v1FineTunedMerged: [CatalogModel] { v1ReleaseBaked }

    static let v1Recommended: [CatalogModel] = [
        CatalogModel(
            id: "fleet-v1-core-qwen-coder-0.5b-q4",
            name: "Fleet v1 Core — Qwen Coder 0.5B",
            repoId: "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
            fileName: "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
            parameters: "0.5B",
            quantization: "Q4_K_M",
            sizeBytes: 397_000_000,
            role: .chat,
            description: "Tiny code-aware base for Cortex and structured coordination in v1.",
            tags: ["fleet-v1", "cortex", "structured", "tiny"]
        ),
        CatalogModel(
            id: "fleet-v1-voice-qwen-0.5b-q4",
            name: "Fleet v1 Voice — Qwen 0.5B",
            repoId: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            fileName: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            parameters: "0.5B",
            quantization: "Q4_K_M",
            sizeBytes: 397_000_000,
            role: .chat,
            description: "Small response and tone base for Mouth and Mimicry in v1.",
            tags: ["fleet-v1", "mouth", "mimicry", "fast"]
        ),
        CatalogModel(
            id: "fleet-v1-rem-smollm2-1.7b-q4",
            name: "Fleet v1 REM — SmolLM2 1.7B",
            repoId: "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
            fileName: "smollm2-1.7b-instruct-q4_k_m.gguf",
            parameters: "1.7B",
            quantization: "Q4_K_M",
            sizeBytes: 1_120_000_000,
            role: .chat,
            description: "Idle-cycle reflection base for summarizing traces and preparing training records.",
            tags: ["fleet-v1", "rem", "idle", "apache-2.0"]
        ),
        CatalogModel(
            id: "fleet-v1-nomic-embed-q4",
            name: "Fleet v1 Memory — Nomic Embed v1.5",
            repoId: "nomic-ai/nomic-embed-text-v1.5-GGUF",
            fileName: "nomic-embed-text-v1.5.Q4_K_M.gguf",
            parameters: "137M",
            quantization: "Q4_K_M",
            sizeBytes: 85_000_000,
            role: .embedding,
            description: "Semantic memory fallback model for recall and codebase knowledge chunks.",
            tags: ["fleet-v1", "memory", "embedding", "fallback", "tiny"]
        )
    ]

    static let v1Candidates: [CatalogModel] = [
        CatalogModel(
            id: "fleet-v1-qwen-coder-1.5b-q4",
            name: "Fleet v1 Cortex — Qwen Coder 1.5B",
            repoId: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            fileName: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
            parameters: "1.5B",
            quantization: "Q4_K_M",
            sizeBytes: 1_117_000_000,
            role: .chat,
            description: "Recommended dedicated v1 orchestrator fallback while adapter-first role artifacts are evaluated.",
            tags: ["fleet-v1", "cortex", "coder", "fallback"]
        ),
        CatalogModel(
            id: "fleet-v1-phi3.5-mini-q4",
            name: "Fleet v1 REM — Phi 3.5 Mini",
            repoId: "bartowski/Phi-3.5-mini-instruct-GGUF",
            fileName: "Phi-3.5-mini-instruct-Q4_K_M.gguf",
            parameters: "3.8B",
            quantization: "Q4_K_M",
            sizeBytes: 2_390_000_000,
            role: .chat,
            description: "Heavier idle-only reasoning model for advanced self-improvement cycles.",
            tags: ["fleet-v1", "rem", "idle-only", "fallback"]
        )
    ]

    static var defaultFleetModels: [CatalogModel] {
        v1AdapterFirstBase + v1Recommended + v1Candidates
    }

    static var allFleetModels: [CatalogModel] {
        defaultFleetModels + v1ReleaseBaked
    }
}
