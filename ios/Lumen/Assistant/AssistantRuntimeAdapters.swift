import Foundation
#if canImport(CoreML)
import CoreML
#endif

enum LocalRuntimeError: LocalizedError, Sendable, Equatable {
    case unavailable(String)

    var errorDescription: String? {
        switch self {
        case .unavailable(let reason):
            let trimmed = reason.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? "Local runtime unavailable: no reason provided." : "Local runtime unavailable: \(trimmed)"
        }
    }
}


protocol LlamaRuntimeStreamingService: Sendable {
    var isChatLoaded: Bool { get async }
    var isEmbedLoaded: Bool { get async }
    func stream(_ req: GenerateRequest, slot: LumenModelSlot) async -> AsyncStream<GenerationToken>
    func takeCompletedTracePayload(requestID: UUID) async -> CompletedGenerationTracePayload?
    func embed(_ text: String) async throws -> [Double]
}

extension LlamaRuntimeStreamingService {
    func takeCompletedTracePayload(requestID: UUID) async -> CompletedGenerationTracePayload? { nil }
}

extension AppLlamaService: LlamaRuntimeStreamingService {}

enum CoreMLRuntimeError: Error, Sendable, Equatable {
    case unsupportedOnPlatform
    case modelNotConfigured
    case modelNotFound
    case incompatibleModel(String)
    case shapeMismatch
    case experimentalRuntimeDisabled
    case computeFailure(String)
}

struct AssistantRuntimeCapabilityRow: Sendable, Equatable, Identifiable {
    let kind: AssistantRuntimeKind
    let generationSupported: Bool
    let generationSelectable: Bool
    let embeddingSupported: Bool
    let embeddingSelectable: Bool
    let status: String
    let unavailableReason: String?

    var id: String { kind.rawValue }
}

struct AssistantRuntimeCapabilityMatrix: Sendable, Equatable {
    let rows: [AssistantRuntimeCapabilityRow]

    static func current(
        foundation: FoundationModelsRuntimeAdapter = .init(),
        llama: LlamaRuntimeAdapter = .live(),
        fallback: DeterministicFallbackRuntime = .init(),
        coreML: CoreMLRuntimeAdapter = .init(modelURL: nil),
        llamaEmbeddingSelectableOverride: Bool? = nil
    ) -> AssistantRuntimeCapabilityMatrix {
        let llamaEmbeddingSelectable = llamaEmbeddingSelectableOverride ?? llama.hasKnownSelectableEmbeddingRuntime
        return AssistantRuntimeCapabilityMatrix(rows: [
            AssistantRuntimeCapabilityRow(
                kind: foundation.kind,
                generationSupported: foundation.supportsGeneration,
                generationSelectable: foundation.supportsGeneration && foundation.isAvailable,
                embeddingSupported: false,
                embeddingSelectable: false,
                status: foundation.availabilityStatus,
                unavailableReason: foundation.unavailableReason
            ),
            AssistantRuntimeCapabilityRow(
                kind: llama.kind,
                generationSupported: true,
                generationSelectable: llama.isAvailable,
                embeddingSupported: llama.supportsEmbeddings,
                embeddingSelectable: llamaEmbeddingSelectable,
                status: llama.capabilityStatus(embeddingSelectable: llamaEmbeddingSelectable),
                unavailableReason: llama.unavailableReason
            ),
            AssistantRuntimeCapabilityRow(
                kind: fallback.kind,
                generationSupported: fallback.isAvailable,
                generationSelectable: fallback.isAvailable,
                embeddingSupported: false,
                embeddingSelectable: false,
                status: fallback.availabilityStatus,
                unavailableReason: fallback.unavailableReason
            ),
            AssistantRuntimeCapabilityRow(
                kind: coreML.kind,
                generationSupported: false,
                generationSelectable: false,
                embeddingSupported: coreML.supportsEmbeddings,
                embeddingSelectable: coreML.supportsEmbeddings && coreML.isAvailable,
                status: coreML.availabilityStatus,
                unavailableReason: coreML.unavailableReason
            )
        ])
    }

    static func currentIncludingRuntimeState(
        foundation: FoundationModelsRuntimeAdapter = .init(),
        llama: LlamaRuntimeAdapter = .live(),
        fallback: DeterministicFallbackRuntime = .init(),
        coreML: CoreMLRuntimeAdapter = .init(modelURL: nil)
    ) async -> AssistantRuntimeCapabilityMatrix {
        let llamaEmbeddingSelectable = await llama.isEmbeddingSelectable()
        return current(
            foundation: foundation,
            llama: llama,
            fallback: fallback,
            coreML: coreML,
            llamaEmbeddingSelectableOverride: llamaEmbeddingSelectable
        )
    }

    func row(for kind: AssistantRuntimeKind) -> AssistantRuntimeCapabilityRow? {
        rows.first { $0.kind == kind }
    }

    var selectableGenerationRuntimes: [AssistantRuntimeKind] {
        rows.filter(\.generationSelectable).map(\.kind)
    }

    var selectableEmbeddingRuntimes: [AssistantRuntimeKind] {
        rows.filter(\.embeddingSelectable).map(\.kind)
    }
}

struct DeterministicFallbackRuntime: LocalTextGenerationRuntime {
    let kind: AssistantRuntimeKind = .deterministicFallback
    let isAvailable: Bool
    let unavailableReason: String?
    private let generateHandler: (@Sendable (TextGenerationRequest) async throws -> String)?

    init(generateHandler: (@Sendable (TextGenerationRequest) async throws -> String)? = nil) {
        self.generateHandler = generateHandler
        #if DEBUG
        self.isAvailable = true
        self.unavailableReason = nil
        #else
        self.isAvailable = false
        self.unavailableReason = "Diagnostic deterministic runtime is excluded from Release routing."
        #endif
    }

    var availabilityStatus: String {
        isAvailable ? "debug diagnostic only" : "excluded from Release routing"
    }

    func generate(request: TextGenerationRequest) async throws -> String {
        guard isAvailable else {
            throw LocalRuntimeError.unavailable(unavailableReason ?? "Diagnostic deterministic runtime is disabled.")
        }
        if let generateHandler {
            return try await generateHandler(request)
        }
        return "Diagnostic deterministic runtime response."
    }

    func handleMemoryPressure() async {}
}

struct LlamaRuntimeAdapter: LocalTextGenerationRuntime {
    private enum LiveGenerationDefaults {
        static let temperature = 0.7
        static let topP = 0.9
        static let repetitionPenalty = 1.1
        static let modelName = "agent-kernel-llama"
        static let streamErrorPrefix = "Generation error:"
    }

    let kind: AssistantRuntimeKind = .llama
    let unavailableReason: String?
    private let generateHandler: (@Sendable (TextGenerationRequest) async throws -> String)?
    private let embedHandler: (@Sendable (EmbeddingRequest) async throws -> [Float])?
    private let liveService: (any LlamaRuntimeStreamingService)?
    private let liveSlot: LumenModelSlot

    var isAvailable: Bool {
        generateHandler != nil || liveService != nil
    }

    var supportsEmbeddings: Bool {
        embedHandler != nil || liveService != nil
    }

    var hasKnownSelectableEmbeddingRuntime: Bool {
        embedHandler != nil
    }

    init(
        isAvailable: Bool = false,
        unavailableReason: String? = "llama text runtime is not directly wired to AssistantKernel",
        generateHandler: (@Sendable (TextGenerationRequest) async throws -> String)? = nil,
        embedHandler: (@Sendable (EmbeddingRequest) async throws -> [Float])? = nil
    ) {
        self.generateHandler = generateHandler
        self.embedHandler = embedHandler
        self.liveService = nil
        self.liveSlot = .mouth
        if generateHandler != nil {
            self.unavailableReason = nil
        } else if isAvailable {
            self.unavailableReason = unavailableReason ?? "llama text runtime unavailable: generation adapter missing"
        } else {
            self.unavailableReason = unavailableReason
        }
    }

    private init(liveService: any LlamaRuntimeStreamingService, liveSlot: LumenModelSlot) {
        self.generateHandler = nil
        self.embedHandler = nil
        self.liveService = liveService
        self.liveSlot = liveSlot
        self.unavailableReason = nil
    }

    static func live(service: any LlamaRuntimeStreamingService = AppLlamaService.shared, slot: LumenModelSlot = .mouth) -> LlamaRuntimeAdapter {
        LlamaRuntimeAdapter(liveService: service, liveSlot: slot)
    }

    func streamStructured(_ request: GenerateRequest, slot: LumenModelSlot) async throws -> AsyncStream<GenerationToken> {
        guard let liveService else {
            throw LocalRuntimeError.unavailable(unavailableReason ?? "llama structured streaming runtime unavailable")
        }
        guard await liveService.isChatLoaded else {
            throw LocalRuntimeError.unavailable("llama structured streaming runtime has no loaded chat model")
        }
        return await liveService.stream(request, slot: slot)
    }

    func takeCompletedStructuredTracePayload(requestID: UUID) async -> CompletedGenerationTracePayload? {
        guard let liveService else { return nil }
        return await liveService.takeCompletedTracePayload(requestID: requestID)
    }

    func generate(request: TextGenerationRequest) async throws -> String {
        if let generateHandler {
            return try await generateHandler(request)
        }
        guard let liveService else {
            throw LocalRuntimeError.unavailable(unavailableReason ?? "llama runtime unavailable")
        }
        guard await liveService.isChatLoaded else {
            throw LocalRuntimeError.unavailable("llama runtime has no loaded chat model")
        }

        let generationRequest = GenerateRequest(
            systemPrompt: request.systemPrompt,
            history: request.history,
            userMessage: request.prompt,
            temperature: request.temperature,
            topP: request.topP,
            repetitionPenalty: request.repetitionPenalty,
            maxTokens: request.maxTokens,
            modelName: LiveGenerationDefaults.modelName,
            relevantMemories: request.relevantMemories,
            attachments: request.attachments
        )

        var output = ""
        streamLoop: for await token in await liveService.stream(generationRequest, slot: liveSlot) {
            switch token {
            case .text(let delta):
                output += delta
            case .done:
                break streamLoop
            }
        }

        let final = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !final.isEmpty else {
            throw LocalRuntimeError.unavailable("llama runtime produced no visible output")
        }
        guard !final.hasPrefix(LiveGenerationDefaults.streamErrorPrefix) else {
            throw LocalRuntimeError.unavailable("llama runtime stream failed during generation")
        }
        return final
    }

    func isEmbeddingSelectable() async -> Bool {
        if embedHandler != nil { return true }
        guard let liveService else { return false }
        return await liveService.isEmbedLoaded
    }

    func embed(request: EmbeddingRequest) async throws -> [Float] {
        if let embedHandler {
            return try await embedHandler(request)
        }
        guard let liveService else {
            throw LocalRuntimeError.unavailable("llama embedding runtime unavailable")
        }
        guard await liveService.isEmbedLoaded else {
            throw LocalRuntimeError.unavailable("llama embedding runtime has no loaded embedding model")
        }
        let vector = try await liveService.embed(request.text)
        guard !vector.isEmpty else {
            throw LocalRuntimeError.unavailable("llama embedding runtime produced an empty vector")
        }
        return vector.map(Float.init)
    }

    func capabilityStatus(embeddingSelectable: Bool? = nil) -> String {
        let generation = isAvailable ? "generation available" : "generation unavailable"
        let embeddings: String
        if embeddingSelectable ?? hasKnownSelectableEmbeddingRuntime {
            embeddings = "embeddings available"
        } else if supportsEmbeddings {
            embeddings = "embeddings not loaded"
        } else {
            embeddings = "embeddings unavailable"
        }
        return "\(generation); \(embeddings)"
    }

    func handleMemoryPressure() async {
        await MainActor.run { FleetRuntimeCleanup.unloadOptionalChatSlots() }
    }
}

struct FoundationModelsRuntimeAdapter: LocalTextGenerationRuntime {
    let kind: AssistantRuntimeKind = .foundationModels
    let isAvailable: Bool
    let unavailableReason: String?
    let availabilityStatus: String
    let supportsGeneration: Bool = false

    init(unavailableReason: String? = nil) {
        self.isAvailable = false
        self.availabilityStatus = "experimental runtime excluded from Release routing"
        self.unavailableReason = unavailableReason ?? "FoundationModels generation is experimental and is excluded from Release routing."
    }

    func generate(request: TextGenerationRequest) async throws -> String {
        throw LocalRuntimeError.unavailable(unavailableReason ?? "FoundationModels runtime is disabled.")
    }

    func handleMemoryPressure() async {}
}

struct CoreMLRuntimeAdapter: LocalEmbeddingRuntime {
    let kind: AssistantRuntimeKind = .coreML
    let modelURL: URL?
    let supportsEmbeddings: Bool = false

    var isAvailable: Bool {
        guard supportsEmbeddings else { return false }
        #if canImport(CoreML)
        guard let modelURL else { return false }
        return FileManager.default.fileExists(atPath: modelURL.path)
        #else
        return false
        #endif
    }

    var unavailableReason: String? {
        guard supportsEmbeddings else { return "CoreML embedding runtime is experimental and is excluded from Release routing." }
        #if canImport(CoreML)
        guard let modelURL else { return "No Core ML embedding model configured" }
        return FileManager.default.fileExists(atPath: modelURL.path) ? nil : "Configured Core ML model file is missing"
        #else
        return "CoreML framework unavailable"
        #endif
    }

    var availabilityStatus: String {
        #if canImport(CoreML)
        guard supportsEmbeddings else { return "experimental runtime excluded from Release routing" }
        guard let modelURL else { return "model missing: not configured" }
        return FileManager.default.fileExists(atPath: modelURL.path) ? "available" : "model missing: configured file missing"
        #else
        return "framework unavailable"
        #endif
    }

    func embed(request: EmbeddingRequest) async throws -> [Float] {
        guard supportsEmbeddings else { throw CoreMLRuntimeError.experimentalRuntimeDisabled }
        #if canImport(CoreML)
        guard let modelURL else { throw CoreMLRuntimeError.modelNotConfigured }
        guard FileManager.default.fileExists(atPath: modelURL.path) else { throw CoreMLRuntimeError.modelNotFound }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        do {
            _ = try MLModel(contentsOf: modelURL, configuration: config)
            throw CoreMLRuntimeError.experimentalRuntimeDisabled
        } catch let error as CoreMLRuntimeError {
            throw error
        } catch {
            throw CoreMLRuntimeError.computeFailure("model_load_failed")
        }
        #else
        throw CoreMLRuntimeError.unsupportedOnPlatform
        #endif
    }

    func handleMemoryPressure() async {}
}
