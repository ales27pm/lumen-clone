import Foundation
#if canImport(CoreML)
import CoreML
#endif

enum LocalRuntimeError: Error, Sendable, Equatable {
    case unavailable(String)
    case generationNotImplemented(AssistantRuntimeKind)
}


protocol LlamaRuntimeStreamingService: Sendable {
    var isChatLoaded: Bool { get async }
    func stream(_ req: GenerateRequest, slot: LumenModelSlot) async -> AsyncStream<GenerationToken>
}

extension AppLlamaService: LlamaRuntimeStreamingService {}

enum CoreMLRuntimeError: Error, Sendable, Equatable {
    case unsupportedOnPlatform
    case modelNotConfigured
    case modelNotFound
    case incompatibleModel(String)
    case shapeMismatch
    case embeddingExtractionNotImplemented
    case computeFailure(String)
}

struct DeterministicFallbackRuntime: LocalTextGenerationRuntime {
    let kind: AssistantRuntimeKind = .deterministicFallback
    let isAvailable: Bool = true
    let unavailableReason: String? = nil
    private let generateHandler: (@Sendable (TextGenerationRequest) async throws -> String)?

    init(generateHandler: (@Sendable (TextGenerationRequest) async throws -> String)? = nil) {
        self.generateHandler = generateHandler
    }

    func generate(request: TextGenerationRequest) async throws -> String {
        if let generateHandler {
            return try await generateHandler(request)
        }
        return "Lumen is running in limited local mode."
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
    private let explicitlyAvailable: Bool
    private let generateHandler: (@Sendable (TextGenerationRequest) async throws -> String)?
    private let liveService: (any LlamaRuntimeStreamingService)?
    private let liveSlot: LumenModelSlot

    var isAvailable: Bool {
        explicitlyAvailable || generateHandler != nil || liveService != nil
    }

    init(
        isAvailable: Bool = false,
        unavailableReason: String? = "llama text runtime is not directly wired to AssistantKernel",
        generateHandler: (@Sendable (TextGenerationRequest) async throws -> String)? = nil
    ) {
        self.explicitlyAvailable = isAvailable
        self.generateHandler = generateHandler
        self.liveService = nil
        self.liveSlot = .mouth
        if generateHandler != nil {
            self.unavailableReason = nil
        } else if isAvailable {
            self.unavailableReason = unavailableReason ?? "llama text runtime was marked available without a generation adapter"
        } else {
            self.unavailableReason = unavailableReason
        }
    }

    private init(liveService: any LlamaRuntimeStreamingService, liveSlot: LumenModelSlot) {
        self.explicitlyAvailable = false
        self.generateHandler = nil
        self.liveService = liveService
        self.liveSlot = liveSlot
        self.unavailableReason = nil
    }

    static func live(service: any LlamaRuntimeStreamingService = AppLlamaService.shared, slot: LumenModelSlot = .mouth) -> LlamaRuntimeAdapter {
        LlamaRuntimeAdapter(liveService: service, liveSlot: slot)
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

    func handleMemoryPressure() async {
        await FleetRuntimeCleanup.unloadOptionalChatSlots()
    }
}

struct FoundationModelsRuntimeAdapter: LocalTextGenerationRuntime {
    let kind: AssistantRuntimeKind = .foundationModels
    let isAvailable: Bool
    let unavailableReason: String?

    init(unavailableReason: String? = nil) {
        if #available(iOS 26.0, *) {
            self.isAvailable = false
            self.unavailableReason = unavailableReason ?? "FoundationModels generation is not wired"
        } else {
            self.isAvailable = false
            self.unavailableReason = "FoundationModels requires iOS 26 or later"
        }
    }

    func generate(request: TextGenerationRequest) async throws -> String {
        throw LocalRuntimeError.generationNotImplemented(.foundationModels)
    }

    func handleMemoryPressure() async {}
}

struct CoreMLRuntimeAdapter: LocalEmbeddingRuntime {
    let kind: AssistantRuntimeKind = .coreML
    let modelURL: URL?

    var isAvailable: Bool {
        #if canImport(CoreML)
        guard let modelURL else { return false }
        return FileManager.default.fileExists(atPath: modelURL.path)
        #else
        return false
        #endif
    }

    var unavailableReason: String? {
        #if canImport(CoreML)
        guard let modelURL else { return "No Core ML embedding model configured" }
        return FileManager.default.fileExists(atPath: modelURL.path) ? nil : "Configured Core ML model file is missing"
        #else
        return "CoreML framework unavailable"
        #endif
    }

    func embed(request: EmbeddingRequest) async throws -> [Float] {
        #if canImport(CoreML)
        guard let modelURL else { throw CoreMLRuntimeError.modelNotConfigured }
        guard FileManager.default.fileExists(atPath: modelURL.path) else { throw CoreMLRuntimeError.modelNotFound }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        do {
            _ = try MLModel(contentsOf: modelURL, configuration: config)
            throw CoreMLRuntimeError.embeddingExtractionNotImplemented
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
