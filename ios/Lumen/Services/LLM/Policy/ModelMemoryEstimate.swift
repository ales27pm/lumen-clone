import Foundation

enum EstimateConfidence: String, Sendable, Codable, Equatable, CaseIterable {
    case high
    case medium
    case low
}

struct ModelMemoryEstimate: Sendable, Codable, Equatable {
    let modelID: String
    let estimatedModelMemoryMB: Int
    let estimatedKVCacheMB: Int
    let estimatedRuntimeOverheadMB: Int
    let estimatedTotalMB: Int
    let confidence: EstimateConfidence
    let notes: [String]

    init(
        modelID: String,
        estimatedModelMemoryMB: Int,
        estimatedKVCacheMB: Int,
        estimatedRuntimeOverheadMB: Int,
        confidence: EstimateConfidence,
        notes: [String]
    ) {
        self.modelID = modelID
        self.estimatedModelMemoryMB = max(1, estimatedModelMemoryMB)
        self.estimatedKVCacheMB = max(1, estimatedKVCacheMB)
        self.estimatedRuntimeOverheadMB = max(1, estimatedRuntimeOverheadMB)
        self.estimatedTotalMB = self.estimatedModelMemoryMB + self.estimatedKVCacheMB + self.estimatedRuntimeOverheadMB
        self.confidence = confidence
        self.notes = notes
    }
}

enum ModelMemoryEstimator {
    static func estimate(
        model: LocalLLMModel,
        profile: InferenceProfile,
        budget: InferenceBudget
    ) -> ModelMemoryEstimate {
        var notes: [String] = []
        let modelMemory: Int
        let confidence: EstimateConfidence

        if let fileSizeBytes = model.fileSizeBytes, fileSizeBytes > 0 {
            modelMemory = megabytesRoundedUp(bytes: UInt64(fileSizeBytes))
            confidence = .high
            notes.append("Used local model file size as memory baseline.")
        } else if
            let parameterCountBillion = model.parameterCountBillion,
            parameterCountBillion > 0,
            let quantization = model.quantization
        {
            let bits = bitsPerWeight(for: quantization)
            let bytes = parameterCountBillion * 1_000_000_000.0 * bits / 8.0
            modelMemory = megabytesRoundedUp(bytes: bytes)
            confidence = .medium
            notes.append("Estimated model memory from parameter count and quantization.")
            notes.append("Quantization \(quantization) mapped to \(bits) bits per weight.")
        } else {
            modelMemory = fallbackModelMemoryMB(for: model.backend)
            confidence = .low
            notes.append("Used conservative backend fallback because model size metadata is incomplete.")
        }

        let kvCache = kvCacheEstimateMB(model: model, profile: profile, budget: budget)
        let overhead = runtimeOverheadMB(backend: model.backend, useMetal: profile.useMetal && budget.allowGPU)

        return ModelMemoryEstimate(
            modelID: model.id,
            estimatedModelMemoryMB: modelMemory,
            estimatedKVCacheMB: kvCache,
            estimatedRuntimeOverheadMB: overhead,
            confidence: confidence,
            notes: notes
        )
    }

    private static func bitsPerWeight(for quantization: String) -> Double {
        let normalized = quantization
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .uppercased()

        if normalized.contains("IQ2_XXS") { return 2.1 }
        if normalized.contains("Q2_K") { return 2.6 }
        if normalized.contains("Q3_K_M") { return 3.3 }
        if normalized.contains("IQ4_XS") { return 4.25 }
        if normalized.contains("Q4_K_S") { return 4.4 }
        if normalized.contains("Q4_K_M") { return 4.5 }
        if normalized.contains("Q5_K_S") { return 5.0 }
        if normalized.contains("Q5_K_M") { return 5.1 }
        if normalized.contains("Q6_K") { return 6.0 }
        if normalized.contains("Q8_0") { return 8.0 }
        return 4.8
    }

    private static func kvCacheEstimateMB(
        model: LocalLLMModel,
        profile: InferenceProfile,
        budget: InferenceBudget
    ) -> Int {
        switch model.backend {
        case .tinyIntent, .mock:
            return 16
        case .remote:
            return 16
        case .gguf, .coreML:
            let effectiveContextTokens = max(
                1,
                min(profile.contextTokens, min(model.contextLength, budget.maxPromptTokens))
            )
            if let parameterCountBillion = model.parameterCountBillion, parameterCountBillion > 0 {
                let contextScale = Double(effectiveContextTokens) / 4_096.0
                let modelScale = max(128.0, parameterCountBillion * 256.0)
                return max(64, Int(ceil(contextScale * modelScale)))
            }
            return max(128, Int(ceil(Double(effectiveContextTokens) / 16.0)))
        }
    }

    private static func runtimeOverheadMB(backend: LLMBackendKind, useMetal: Bool) -> Int {
        let base: Int
        switch backend {
        case .tinyIntent, .mock:
            base = 32
        case .remote:
            base = 64
        case .coreML:
            base = 256
        case .gguf:
            base = 384
        }
        return base + (useMetal ? 256 : 0)
    }

    private static func fallbackModelMemoryMB(for backend: LLMBackendKind) -> Int {
        switch backend {
        case .gguf:
            return 2_048
        case .coreML:
            return 1_024
        case .tinyIntent, .mock:
            return 64
        case .remote:
            return 128
        }
    }

    private static func megabytesRoundedUp(bytes: UInt64) -> Int {
        Int((bytes + 1_048_575) / 1_048_576)
    }

    private static func megabytesRoundedUp(bytes: Double) -> Int {
        max(1, Int(ceil(bytes / 1_048_576.0)))
    }
}
