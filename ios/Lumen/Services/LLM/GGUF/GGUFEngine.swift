import Foundation

actor GGUFEngine: LLMEngine {
    nonisolated let id = "gguf"
    nonisolated let displayName = "GGUF Local Engine"
    nonisolated let capabilities = LLMEngineCapabilities.localGGUF

    private let nativeBridge: any GGUFNativeBridge
    private var loadedModel: LocalLLMModel?
    private var loadedProfile: InferenceProfile?
    private var activeRequestID: UUID?
    private var cancelledRequestIDs: Set<UUID> = []
    private var lifecycleTransitionInProgress = false
    private var generationInProgress = false
    private var generationInProgressWaiters: [CheckedContinuation<Void, Never>] = []

    init(nativeBridge: any GGUFNativeBridge = UnavailableGGUFNativeBridge()) {
        self.nativeBridge = nativeBridge
    }

    func load(model: LocalLLMModel, profile: InferenceProfile) async throws {
        guard model.backend == .gguf else {
            throw LLMEngineError.backendUnavailable(model.backend.rawValue)
        }

        await cancelActiveGenerationAndWait()
        lifecycleTransitionInProgress = true
        defer { lifecycleTransitionInProgress = false }

        guard let modelURL = model.localURL else {
            throw LLMEngineError.modelNotFound
        }

        do {
            try ModelFileValidator.validateReadableFile(modelURL)
            try ModelFileValidator.validateExtension(for: modelURL, backend: model.backend)
        } catch {
            throw GGUFEngineErrorMapper.map(error)
        }

        let config = GGUFBridgeLoadConfig(
            modelPath: modelURL.path,
            contextTokens: profile.contextTokens,
            batchSize: profile.batchSize,
            threadCount: profile.threadCount,
            gpuLayerCount: profile.gpuLayerCount,
            useMetal: profile.useMetal,
            useMemoryMapping: profile.useMemoryMapping
        )

        do {
            _ = try await nativeBridge.load(config: config)
            loadedModel = model
            loadedProfile = profile
            cancelledRequestIDs.removeAll(keepingCapacity: true)
        } catch {
            throw GGUFEngineErrorMapper.map(error)
        }
    }

    func unload() async {
        await cancelActiveGenerationAndWait()
        lifecycleTransitionInProgress = true
        defer { lifecycleTransitionInProgress = false }

        await nativeBridge.unload()
        loadedModel = nil
        loadedProfile = nil
        activeRequestID = nil
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
            continuation.onTermination = { @Sendable termination in
                if case .cancelled = termination {
                    task.cancel()
                    Task {
                        await self.cancelCurrentGeneration()
                    }
                }
            }
        }
    }

    func cancelCurrentGeneration() async {
        guard let activeRequestID else { return }
        cancelledRequestIDs.insert(activeRequestID)
        await nativeBridge.cancel()
    }

    private func runGeneration(
        _ request: LLMRequest,
        continuation: AsyncThrowingStream<LLMTokenEvent, Error>.Continuation
    ) async {
        let startedAt = Date()

        guard lifecycleTransitionInProgress == false else {
            continuation.finish(throwing: LLMEngineError.generationAlreadyRunning)
            return
        }

        guard let loadedModel, let loadedProfile else {
            continuation.finish(throwing: LLMEngineError.modelNotLoaded)
            return
        }

        guard activeRequestID == nil else {
            continuation.finish(throwing: LLMEngineError.generationAlreadyRunning)
            return
        }

        do {
            try validateSupportedFeatures(for: request)
            let prompt = try GGUFPromptBuilder.buildPrompt(from: request)
            let promptTokens = approximateTokenCount(for: prompt)
            let remainingCompletionTokens = remainingCompletionTokens(
                promptTokens: promptTokens,
                profile: loadedProfile
            )

            activeRequestID = request.id
            generationInProgress = true
            defer {
                finishGeneration(requestID: request.id)
            }

            let generationConfig = GGUFBridgeGenerateConfig(
                prompt: prompt,
                sampling: makeSamplingConfig(
                    from: request.sampling,
                    budget: request.budget,
                    maximumTokens: remainingCompletionTokens
                )
            )

            continuation.yield(.started(requestID: request.id))

            var runningCompletionCharacters = 0
            var completionTokens = 0
            var parser = ReasoningAwareStreamParser(
                config: ReasoningAwareStreamParserConfig(
                    captureReasoning: request.metadata.booleanValue(forAnyKey: [
                        "reasoningCaptureEnabled",
                        "developerReasoningCaptureEnabled",
                        "allowThinking"
                    ]),
                    reasoningTraceBudgetCharacters: request.metadata.intValue(forAnyKey: [
                        "reasoningTraceBudgetCharacters",
                        "reasoning_trace_budget_characters"
                    ]) ?? 16_384
                )
            )

            do {
                for try await token in nativeBridge.generate(config: generationConfig) {
                    guard !Task.isCancelled, cancelledRequestIDs.contains(request.id) == false else {
                        await nativeBridge.cancel()
                        continuation.yield(.cancelled)
                        continuation.finish()
                        return
                    }

                    runningCompletionCharacters += token.count
                    completionTokens = approximateTokenCount(forCharacterCount: runningCompletionCharacters)
                    let parsedDelta = parser.ingest(token)
                    if parsedDelta.visibleDelta.isEmpty == false {
                        continuation.yield(.token(parsedDelta.visibleDelta))
                    }
                }
            } catch {
                let mapped = GGUFEngineErrorMapper.map(error)
                if let engineError = mapped as? LLMEngineError, engineError == .generationCancelled {
                    continuation.yield(.cancelled)
                    continuation.finish()
                } else {
                    continuation.finish(throwing: mapped)
                }
                return
            }

            let parserFinalDelta = parser.finish()
            if parserFinalDelta.visibleDelta.isEmpty == false {
                continuation.yield(.token(parserFinalDelta.visibleDelta))
            }

            guard !Task.isCancelled, cancelledRequestIDs.contains(request.id) == false else {
                await nativeBridge.cancel()
                continuation.yield(.cancelled)
                continuation.finish()
                return
            }

            continuation.yield(.completed(LLMCompletionSummary(
                requestID: request.id,
                modelID: loadedModel.id,
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                totalTokens: promptTokens + completionTokens,
                durationSeconds: max(0, Date().timeIntervalSince(startedAt)),
                finishReason: .stop
            )))
            continuation.finish()
        } catch {
            continuation.finish(throwing: GGUFEngineErrorMapper.map(error))
        }
    }

    private func validateSupportedFeatures(for request: LLMRequest) throws {
        if request.metadata.booleanValue(forAnyKey: ["requiresEmbeddings", "requires_embeddings"]) {
            throw LLMEngineError.unsupportedFeature("embeddings")
        }

        if request.metadata.booleanValue(forAnyKey: ["requiresVision", "requires_vision"]) {
            throw LLMEngineError.unsupportedFeature("vision")
        }
    }

    private func makeSamplingConfig(
        from sampling: LLMSamplingConfig,
        budget: InferenceBudget,
        maximumTokens: Int
    ) -> GGUFBridgeSamplingConfig {
        let requestedMaxTokens = min(
            sampling.maxTokens,
            budget.maxCompletionTokens,
            capabilities.maximumOutputTokens
        )

        return GGUFBridgeSamplingConfig(
            temperature: sampling.temperature,
            topP: sampling.topP,
            topK: sampling.topK,
            repeatPenalty: sampling.repeatPenalty,
            seed: sampling.seed,
            maxTokens: min(requestedMaxTokens, max(0, maximumTokens)),
            stopSequences: sampling.stopSequences
        )
    }

    private func remainingCompletionTokens(promptTokens: Int, profile: InferenceProfile) -> Int {
        let contextLimit = min(profile.contextTokens, capabilities.maximumContextTokens)
        return max(0, contextLimit - promptTokens)
    }

    private func approximateTokenCount(for text: String) -> Int {
        approximateTokenCount(forCharacterCount: text.count)
    }

    private func approximateTokenCount(forCharacterCount characterCount: Int) -> Int {
        Int(ceil(Double(characterCount) / 4.0))
    }

    private func cancelActiveGenerationAndWait() async {
        guard generationInProgress else { return }
        if let activeRequestID {
            cancelledRequestIDs.insert(activeRequestID)
        }
        await nativeBridge.cancel()
        await waitForActiveGenerationToFinish()
    }

    private func waitForActiveGenerationToFinish() async {
        guard generationInProgress else { return }
        await withCheckedContinuation { continuation in
            generationInProgressWaiters.append(continuation)
        }
    }

    private func finishGeneration(requestID: UUID) {
        activeRequestID = nil
        generationInProgress = false
        cancelledRequestIDs.remove(requestID)
        let waiters = generationInProgressWaiters
        generationInProgressWaiters.removeAll(keepingCapacity: true)
        for waiter in waiters {
            waiter.resume()
        }
    }
}

private extension Dictionary where Key == String, Value == String {
    nonisolated func booleanValue(forAnyKey keys: [String]) -> Bool {
        for key in keys {
            guard let value = self[key]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() else {
                continue
            }
            if ["1", "true", "yes"].contains(value) {
                return true
            }
        }
        return false
    }

    nonisolated func intValue(forAnyKey keys: [String]) -> Int? {
        for key in keys {
            guard let value = self[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
                  let parsed = Int(value) else {
                continue
            }
            return parsed
        }
        return nil
    }
}
