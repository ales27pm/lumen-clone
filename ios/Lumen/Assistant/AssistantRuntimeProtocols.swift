import Foundation

enum AssistantRuntimeKind: String, Sendable {
    case foundationModels
    case coreML
    case llama
    case deterministicFallback
}

struct TextGenerationRequest: Sendable {
    let prompt: String
    let systemPrompt: String
    let maxTokens: Int
}

struct EmbeddingRequest: Sendable {
    let text: String
}

protocol RuntimeHealthReporting: Sendable {
    var isAvailable: Bool { get }
    var unavailableReason: String? { get }
}

protocol RuntimeMemoryPressureHandling: Sendable {
    func handleMemoryPressure() async
}

protocol LocalTextGenerationRuntime: RuntimeHealthReporting, RuntimeMemoryPressureHandling {
    var kind: AssistantRuntimeKind { get }
    func generate(request: TextGenerationRequest) async throws -> String
}

protocol LocalEmbeddingRuntime: RuntimeHealthReporting, RuntimeMemoryPressureHandling {
    var kind: AssistantRuntimeKind { get }
    func embed(request: EmbeddingRequest) async throws -> [Float]
}
