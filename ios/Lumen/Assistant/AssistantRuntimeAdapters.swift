import Foundation
#if canImport(CoreML)
import CoreML
#endif

enum LocalRuntimeError: Error, Sendable, Equatable {
    case unavailable(String)
    case generationNotImplemented(AssistantRuntimeKind)
}

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

    func generate(request: TextGenerationRequest) async throws -> String {
        "Lumen is running in limited local mode."
    }

    func handleMemoryPressure() async {}
}

struct LlamaRuntimeAdapter: LocalTextGenerationRuntime {
    let kind: AssistantRuntimeKind = .llama
    let isAvailable: Bool
    let unavailableReason: String?

    init(isAvailable: Bool = false, unavailableReason: String? = "llama text runtime is not directly wired to AssistantKernel") {
        self.isAvailable = isAvailable
        self.unavailableReason = isAvailable ? nil : unavailableReason
    }

    func generate(request: TextGenerationRequest) async throws -> String {
        throw LocalRuntimeError.unavailable(unavailableReason ?? "llama runtime unavailable")
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
