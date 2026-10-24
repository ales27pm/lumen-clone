import Foundation

enum AssistantRuntimeKind: String, Sendable {
    case foundationModels
    case coreML
    case coreAI
    case llama
    case deterministicFallback
    case unavailable
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
    let hostSelfModel: HostSelfModelContext?

    init(
        prompt: String,
        systemPrompt: String,
        history: [(role: MessageRole, content: String)] = [],
        temperature: Double = 0.7,
        topP: Double = 0.9,
        repetitionPenalty: Double = 1.1,
        maxTokens: Int,
        relevantMemories: [MemoryContextItem] = [],
        attachments: [ChatAttachment] = [],
        hostSelfModel: HostSelfModelContext? = nil
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
        self.hostSelfModel = hostSelfModel
    }
}

nonisolated enum EmbeddingPurpose: Sendable, Equatable {
    /// Compatibility for callers that already supplied the legacy model's task prefix.
    /// New production retrieval calls must use an explicit purpose instead.
    case legacyFormatted
    case retrievalQuery
    case retrievalDocument(title: String?)
    case semanticSimilarity
    case codeQuery
}

nonisolated struct EmbeddingRequest: Sendable, Equatable {
    let text: String
    let purpose: EmbeddingPurpose
    let dimensions: Int?

    init(text: String, purpose: EmbeddingPurpose = .legacyFormatted, dimensions: Int? = nil) {
        self.text = text
        self.purpose = purpose
        self.dimensions = dimensions
    }
}

nonisolated struct EmbeddingRuntimeResult: Sendable, Equatable {
    let vector: [Double]
    let modelIdentifier: String
    let formatVersion: Int

    init(vector: [Double], modelIdentifier: String, formatVersion: Int = SemanticEmbeddingText.formatVersion) {
        self.vector = vector
        self.modelIdentifier = modelIdentifier
        self.formatVersion = formatVersion
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
