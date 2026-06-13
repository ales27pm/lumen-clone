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
    let history: [(role: MessageRole, content: String)]
    let temperature: Double
    let topP: Double
    let repetitionPenalty: Double
    let maxTokens: Int
    let relevantMemories: [MemoryContextItem]
    let attachments: [ChatAttachment]

    init(
        prompt: String,
        systemPrompt: String,
        history: [(role: MessageRole, content: String)] = [],
        temperature: Double = 0.7,
        topP: Double = 0.9,
        repetitionPenalty: Double = 1.1,
        maxTokens: Int,
        relevantMemories: [MemoryContextItem] = [],
        attachments: [ChatAttachment] = []
    ) {
        self.prompt = prompt
        self.systemPrompt = systemPrompt
        self.history = history
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = max(1, maxTokens)
        self.relevantMemories = relevantMemories
        self.attachments = attachments
    }
}

struct EmbeddingRequest: Sendable {
    let text: String

    init(text: String) {
        self.text = text
    }

    init(text: String, dimensions: Int?) {
        self.text = text
    }
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
