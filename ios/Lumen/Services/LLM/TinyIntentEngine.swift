import Foundation

actor TinyIntentEngine: LLMEngine {
    nonisolated let id: String
    nonisolated let displayName: String
    nonisolated let capabilities: LLMEngineCapabilities

    private var loadedModel: LocalLLMModel?
    private var loadedProfile: InferenceProfile?
    private var currentGenerationID: UUID?
    private var cancelledGenerationIDs: Set<UUID> = []

    init(
        id: String = "tiny-intent",
        displayName: String = "Tiny Intent Engine",
        capabilities: LLMEngineCapabilities = .tinyIntent
    ) {
        self.id = id
        self.displayName = displayName
        self.capabilities = capabilities
    }

    func load(model: LocalLLMModel, profile: InferenceProfile) async throws {
        guard model.backend == .tinyIntent else {
            throw LLMEngineError.backendUnavailable(model.backend.rawValue)
        }
        if let currentGenerationID {
            cancelledGenerationIDs.insert(currentGenerationID)
        } else {
            cancelledGenerationIDs.removeAll(keepingCapacity: true)
        }
        loadedModel = model
        loadedProfile = profile
    }

    func unload() async {
        loadedModel = nil
        loadedProfile = nil
        if let currentGenerationID {
            cancelledGenerationIDs.insert(currentGenerationID)
        }
    }

    func isLoaded(modelID: String?) async -> Bool {
        guard let loadedModel else { return false }
        guard let modelID else { return true }
        return loadedModel.id == modelID
    }

    nonisolated func generate(_ request: LLMRequest) -> AsyncThrowingStream<LLMTokenEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                await self.runGeneration(request, continuation: continuation)
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    func cancelCurrentGeneration() async {
        guard let currentGenerationID else { return }
        cancelledGenerationIDs.insert(currentGenerationID)
    }

    private func runGeneration(
        _ request: LLMRequest,
        continuation: AsyncThrowingStream<LLMTokenEvent, Error>.Continuation
    ) async {
        let startedAt = Date()

        guard let loadedModel else {
            continuation.finish(throwing: LLMEngineError.modelNotLoaded)
            return
        }

        guard currentGenerationID == nil else {
            continuation.finish(throwing: LLMEngineError.generationAlreadyRunning)
            return
        }

        currentGenerationID = request.id
        defer {
            currentGenerationID = nil
            cancelledGenerationIDs.remove(request.id)
        }

        guard request.context.compactMap(\.tokenEstimate).reduce(0, +) <= capabilities.maximumContextTokens else {
            let actual = request.context.compactMap(\.tokenEstimate).reduce(0, +)
            continuation.finish(throwing: LLMEngineError.contextTooLarge(max: capabilities.maximumContextTokens, actual: actual))
            return
        }

        guard request.tools.isEmpty || capabilities.supportsToolCalling else {
            continuation.finish(throwing: LLMEngineError.unsupportedFeature("tool calling"))
            return
        }

        guard request.responseFormat == .plainText else {
            continuation.finish(throwing: LLMEngineError.unsupportedFeature("structured response formatting"))
            return
        }

        continuation.yield(.started(requestID: request.id))
        await Task.yield()

        guard !Task.isCancelled, !cancelledGenerationIDs.contains(request.id) else {
            continuation.yield(.cancelled)
            continuation.finish()
            return
        }

        let response = Self.response(for: request)
        continuation.yield(.token(response))
        await Task.yield()

        guard !Task.isCancelled, !cancelledGenerationIDs.contains(request.id) else {
            continuation.yield(.cancelled)
            continuation.finish()
            return
        }

        continuation.yield(.completed(LLMCompletionSummary(
            requestID: request.id,
            modelID: loadedModel.id,
            durationSeconds: max(0, Date().timeIntervalSince(startedAt)),
            finishReason: .stop
        )))
        continuation.finish()
    }

    private static func response(for request: LLMRequest) -> String {
        let lastUserMessage = request.messages.last { $0.role == .user }?.content ?? ""
        let normalized = normalize(lastUserMessage)

        guard !normalized.isEmpty else {
            return "Please provide input so I can classify the request."
        }

        if containsAny(normalized, phrases: ["search", "find", "look up"]) {
            return "This looks like a search or retrieval request."
        }

        if containsAny(normalized, phrases: ["summarize", "summary", "resume"]) {
            return "This looks like a summarization request."
        }

        if containsAny(normalized, phrases: ["open", "launch"]) {
            return "This looks like an app or tool action request."
        }

        return "This request requires the main reasoning engine."
    }

    private static func normalize(_ text: String) -> String {
        text
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func containsAny(_ text: String, phrases: [String]) -> Bool {
        phrases.contains { text.contains($0) }
    }
}
