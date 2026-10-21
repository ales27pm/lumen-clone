import Foundation

/// A separate sentence-vector runtime; never a second generative assistant.
struct CoreAITextEmbeddingRuntimeAdapter: LocalEmbeddingRuntime {
    let kind: AssistantRuntimeKind = .coreAI
    private let availabilityOverride: Bool?
    private let preferenceOverride: Bool?
    private let embedHandler: (@Sendable (EmbeddingRequest) async throws -> EmbeddingRuntimeResult)?

    init(
        isAvailable: Bool? = nil,
        isPreferred: Bool? = nil,
        embedHandler: (@Sendable (EmbeddingRequest) async throws -> EmbeddingRuntimeResult)? = nil
    ) {
        availabilityOverride = isAvailable
        preferenceOverride = isPreferred ?? isAvailable
        self.embedHandler = embedHandler
    }

    static var disabled: Self { .init(isAvailable: false, isPreferred: false) }
    var isPreferred: Bool { preferenceOverride ?? CoreAITextEmbeddingModelStore.isSelected }
    var isAvailable: Bool {
        availabilityOverride ?? (CoreAITextEmbeddingService.isSupported && CoreAITextEmbeddingModelStore.isConfigured)
    }
    var unavailableReason: String? {
        if !isPreferred { return "Core AI text embeddings are not selected." }
        if isAvailable { return nil }
        if !CoreAITextEmbeddingService.isSupported { return "Core AI text embeddings require a supported physical device with iOS 27.2 or later." }
        return "Import the pinned local EmbeddingGemma 2 text package in Settings."
    }

    func embed(request: EmbeddingRequest) async throws -> [Float] {
        try await embedWithIdentity(request: request).vector.map(Float.init)
    }

    func embedWithIdentity(request: EmbeddingRequest) async throws -> EmbeddingRuntimeResult {
        guard isPreferred, isAvailable else {
            throw LocalRuntimeError.unavailable(unavailableReason ?? "Core AI text embedding runtime unavailable")
        }
        try Task.checkCancellation()
        let result: EmbeddingRuntimeResult
        if let embedHandler { result = try await embedHandler(request) }
        else { result = try await CoreAITextEmbeddingService.shared.embed(request) }
        try Task.checkCancellation()
        let dimension = request.dimensions ?? 256
        let norm = sqrt(result.vector.reduce(0) { $0 + $1 * $1 })
        guard [128, 256, 512, 768].contains(dimension),
              norm.isFinite, abs(norm - 1) < 0.001,
              result.vector.count == dimension,
              result.vector.allSatisfy(\.isFinite),
              !result.modelIdentifier.isEmpty, result.formatVersion == 4 else {
            throw LocalRuntimeError.unavailable("Core AI text embedding runtime returned incompatible vector metadata")
        }
        return result
    }

    func handleMemoryPressure() async { await CoreAITextEmbeddingService.shared.unload() }
}
