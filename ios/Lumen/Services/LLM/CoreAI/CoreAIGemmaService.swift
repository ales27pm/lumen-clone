import Foundation

#if canImport(CoreAI) && canImport(CoreAILanguageModels) && canImport(Tokenizers)
import CoreAI
import CoreAILanguageModels
import Tokenizers
#endif

/// One owner for the imported Gemma engine. This is an inference adapter, never an
/// agent loop: structured actions still cross the production kernel/tool boundary.
/// Adapted from Assist's actor coordinator at 6a26dcf27042d73833d8ce5b51178bdeb5f17562.
actor CoreAIGemmaService {
    static let shared = CoreAIGemmaService()

    nonisolated static var isSupported: Bool {
        #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI) && canImport(CoreAILanguageModels) && canImport(Tokenizers)
        if #available(iOS 27.2, *) { return true }
        #endif
        return false
    }

    private var operationInFlight = false
    private var cancellationWaiters = 0
    private var replacementInFlight = false
    private var loadingTask: Task<Void, Error>?
    private var generationTask: Task<String, Error>?
    private var cancellationRegistration: (id: UUID, category: AppCancellationCategory)?

    #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI) && canImport(CoreAILanguageModels) && canImport(Tokenizers)
    private var engine: (any InferenceEngine)?
    private var tokenizer: (any Tokenizers.Tokenizer)?
    private var stopTokens: Set<Int32> = []
    #endif
    private var capacity = 0
    private var vocabularySize = 0

    private init() {}

    private var busy: Bool { operationInFlight || cancellationWaiters > 0 || replacementInFlight }

    func isLoaded() -> Bool {
        #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI) && canImport(CoreAILanguageModels) && canImport(Tokenizers)
        return engine != nil && tokenizer != nil
        #else
        return false
        #endif
    }

    /// The loaded bundle's validated context; no file access or model preparation.
    func contextCapacity() -> Int? { capacity > 0 ? capacity : nil }

    /// Called only by explicit user/kernel work; initialization never opens a model.
    func prepare() async throws {
        try Task.checkCancellation()
        guard CoreAIGemmaModelStore.isEnabled else { throw CoreAIGemmaError.resourceDenied }
        guard Self.isSupported else { throw CoreAIGemmaError.unsupported }
        guard !busy else { throw CoreAIGemmaError.busy }
        if isLoaded() { return }
        operationInFlight = true
        let task = Task {
            defer { finishOperation() }
            try await loadConfiguredModel()
        }
        loadingTask = task
        registerCancellation({ task.cancel() }, category: .modelLoad)
        try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    func preflight(messages: [CoreAIGemmaMessage], maxTokens: Int) async throws -> CoreAIGemmaPromptPreflight {
        try CoreAIGemmaPolicy.validateMessages(messages)
        guard maxTokens > 0 else { throw CoreAIGemmaError.invalidTokenBudget }
        try await prepare()
        try Task.checkCancellation()
        guard !busy else { throw CoreAIGemmaError.busy }
        #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI) && canImport(CoreAILanguageModels) && canImport(Tokenizers)
        let prompt = try encodedPrompt(messages)
        _ = try CoreAIGemmaPolicy.outputBudget(promptTokens: prompt.count, capacity: capacity, requested: maxTokens)
        return CoreAIGemmaPromptPreflight(
            contextSize: capacity, promptTokens: prompt.count,
            promptChars: messages.reduce(0) { $0 + $1.content.count }
        )
        #else
        throw CoreAIGemmaError.unsupported
        #endif
    }

    func generate(
        messages: [CoreAIGemmaMessage], temperature: Double, maxTokens: Int,
        onSnapshot: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> String {
        try CoreAIGemmaPolicy.validateMessages(messages)
        try CoreAIGemmaPolicy.validateTemperature(temperature)
        guard maxTokens > 0 else { throw CoreAIGemmaError.invalidTokenBudget }
        try await prepare()
        try Task.checkCancellation()
        guard !busy else { throw CoreAIGemmaError.busy }
        operationInFlight = true
        let task = Task {
            defer { finishOperation() }
            return try await runGeneration(
                messages: messages, temperature: temperature, maxTokens: maxTokens,
                onSnapshot: onSnapshot
            )
        }
        generationTask = task
        registerCancellation({ task.cancel() }, category: .chatGeneration)
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    /// Acknowledgement means load/generation and its cleanup have fully drained.
    func cancelAndWait() async {
        cancellationWaiters += 1
        defer { cancellationWaiters -= 1 }
        let load = loadingTask
        let generation = generationTask
        load?.cancel()
        generation?.cancel()
        #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI) && canImport(CoreAILanguageModels) && canImport(Tokenizers)
        // The pinned engine explicitly permits concurrent cooperative cancel.
        if generation != nil { try? await engine?.cancel() }
        #endif
        if let load { _ = await load.result }
        if let generation { _ = await generation.result }
    }

    func unload() async {
        // Keep admission closed through draining AND release, including actor reentrancy.
        cancellationWaiters += 1
        defer { cancellationWaiters -= 1 }
        await cancelAndWait()
        clearResidentIdentity()
    }

    /// The model store uses this lease while atomically replacing owned model files.
    func beginModelReplacement() async throws {
        guard !replacementInFlight else { throw CoreAIGemmaError.busy }
        replacementInFlight = true
        await cancelAndWait()
        clearResidentIdentity()
    }

    func endModelReplacement() {
        replacementInFlight = false
    }

    private func registerCancellation(_ cancellation: @escaping @Sendable () -> Void, category: AppCancellationCategory) {
        let id = AppCancellationBus.shared.registerCancellation(cancellation, category: category)
        cancellationRegistration = (id, category)
    }

    private func finishOperation() {
        if let registration = cancellationRegistration {
            AppCancellationBus.shared.unregister(registration.id, category: registration.category)
        }
        cancellationRegistration = nil
        loadingTask = nil
        generationTask = nil
        operationInFlight = false
    }

    private func clearResidentIdentity() {
        #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI) && canImport(CoreAILanguageModels) && canImport(Tokenizers)
        engine = nil
        tokenizer = nil
        stopTokens = []
        #endif
        capacity = 0
        vocabularySize = 0
    }

    private func loadConfiguredModel() async throws {
        #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI) && canImport(CoreAILanguageModels) && canImport(Tokenizers)
        do {
            guard await ResourceBudgetGate.allowsForegroundModelLoad(reason: "coreai-gemma") else {
                throw CoreAIGemmaError.resourceDenied
            }
            try Task.checkCancellation()
            // The legacy actor refuses this handoff until all of its streams drain.
            // Embeddings retain their independent runtime and are not unloaded here.
            try await AppLlamaService.shared.suspendForCoreAI()
            guard await ResourceBudgetGate.allowsForegroundModelLoad(reason: "coreai-gemma-handoff") else {
                throw CoreAIGemmaError.resourceDenied
            }
            try Task.checkCancellation()
            // This verifies the imported inventory in a bounded read stream, off MainActor.
            let root = try CoreAIGemmaModelStore.configuredURL()
            _ = try CoreAIGemmaPolicy.requireFile("metadata.json", root: root)
            let bundle = try LanguageModelBundle(at: root)
            try CoreAIGemmaPolicy.validateModel(
                tokenizer: bundle.tokenizer, context: bundle.maxContextLength,
                embeddedTokenizer: bundle.hasEmbeddedTokenizer
            )
            guard bundle.modelBundle.kind == .llm, bundle.visionConfig == nil,
                  Set(bundle.modelBundle.assets.keys) == ["main"],
                  Set(bundle.modelBundle.auxiliaryAssets.keys) == ["per_layer_embeddings"] else {
                throw CoreAIGemmaError.invalidModel
            }
            for relative in bundle.modelBundle.assets.values {
                _ = try CoreAIGemmaPolicy.confinedURL(relative, root: root)
            }
            for relative in bundle.modelBundle.auxiliaryAssets.values {
                _ = try CoreAIGemmaPolicy.requireFile(relative, root: root)
            }
            let tokenizerURL = try CoreAIGemmaPolicy.confinedURL("tokenizer", root: root)
            _ = try CoreAIGemmaPolicy.requireFile("tokenizer/tokenizer.json", root: root)
            _ = try CoreAIGemmaPolicy.requireFile("tokenizer/tokenizer_config.json", root: root)
            _ = try CoreAIGemmaPolicy.requireFile("tokenizer/chat_template.jinja", root: root)
            let modelURL = try bundle.modelBundle.requireModelURL(for: "main")
            let asset = try AIModelAsset(contentsOf: modelURL)
            guard let summary = try asset.summary(includingStatistics: false) else {
                throw CoreAIGemmaError.invalidModel
            }
            try CoreAIGemmaPolicy.validateGraphs(summary.functions.map(\.name), context: bundle.maxContextLength)
            try Task.checkCancellation()
            // Never use bundle.loadTokenizer() or AutoTokenizer.from(pretrained:),
            // both of which may resolve missing data through the network.
            let loadedTokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
            let stops = try CoreAIGemmaPolicy.stopTokenIDs(
                metadata: bundle.rawMetadata, vocabSize: bundle.vocabSize,
                tokenizerEOS: loadedTokenizer.eosTokenId
            )
            try Task.checkCancellation()
            guard await ResourceBudgetGate.allowsForegroundModelLoad(reason: "coreai-gemma-specialization") else {
                throw CoreAIGemmaError.resourceDenied
            }
            try Task.checkCancellation()
            let runner = CoreAIRunner(bundle: bundle, variant: "static-shape")
            let loadedEngine = try await runner.makeInferenceEngine()
            guard loadedEngine.config.maxContextLength == bundle.maxContextLength else {
                throw CoreAIGemmaError.invalidModel
            }
            try Task.checkCancellation()
            guard await ResourceBudgetGate.allowsForegroundModelLoad(reason: "coreai-gemma-warmup") else {
                throw CoreAIGemmaError.resourceDenied
            }
            try Task.checkCancellation()
            try await loadedEngine.warmup(queryLength: 1, sampling: nil)
            try Task.checkCancellation()
            tokenizer = loadedTokenizer
            engine = loadedEngine
            capacity = bundle.maxContextLength
            vocabularySize = bundle.vocabSize
            stopTokens = stops
        } catch {
            clearResidentIdentity()
            if error is CancellationError || Task.isCancelled { throw CancellationError() }
            if let failure = error as? CoreAIGemmaModelStore.Failure {
                switch failure {
                case .notConfigured: throw CoreAIGemmaError.modelNotConfigured
                case .integrity: throw CoreAIGemmaError.integrityFailure
                case .invalidPackage, .localDirectory: throw CoreAIGemmaError.invalidModel
                case .storage: throw CoreAIGemmaError.loadFailed
                }
            }
            throw (error as? CoreAIGemmaError) ?? .loadFailed
        }
        #else
        throw CoreAIGemmaError.unsupported
        #endif
    }

    private func runGeneration(
        messages: [CoreAIGemmaMessage], temperature: Double, maxTokens: Int,
        onSnapshot: @escaping @Sendable (String) -> Void
    ) async throws -> String {
        #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI) && canImport(CoreAILanguageModels) && canImport(Tokenizers)
        guard CoreAIGemmaModelStore.isEnabled else { throw CoreAIGemmaError.resourceDenied }
        guard let engine, let tokenizer else { throw CoreAIGemmaError.loadFailed }
        guard await ResourceBudgetGate.allowsHeavyModelWork(reason: "coreai-gemma-generation") else {
            throw CoreAIGemmaError.resourceDenied
        }
        try Task.checkCancellation()
        let prompt = try encodedPrompt(messages)
        let budget = try CoreAIGemmaPolicy.outputBudget(promptTokens: prompt.count, capacity: capacity, requested: maxTokens)
        let diskWriteLease = DiskWriteBudget.shared.beginGeneration()
        defer { diskWriteLease.end() }
        do {
            try await engine.reset(to: 0)
            let text = try await Self.consumeGeneration(
                engine: engine, tokenizer: tokenizer, prompt: prompt, stopTokens: stopTokens,
                vocabularySize: vocabularySize, temperature: temperature, budget: budget, onSnapshot: onSnapshot
            )
            // The whole stream has stopped before resetting KV. No context survives a turn.
            try await engine.reset(to: 0)
            try Task.checkCancellation()
            return text
        } catch {
            // cancel() invalidates the generation token; consumeGeneration has already drained.
            try? await engine.cancel()
            if error is CancellationError || Task.isCancelled {
                do { try await engine.reset(to: 0) }
                catch { clearResidentIdentity() }
                throw CancellationError()
            }
            clearResidentIdentity()
            throw (error as? CoreAIGemmaError) ?? .inferenceFailed
        }
        #else
        throw CoreAIGemmaError.unsupported
        #endif
    }

    #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI) && canImport(CoreAILanguageModels) && canImport(Tokenizers)
    private func encodedPrompt(_ messages: [CoreAIGemmaMessage]) throws -> [Int] {
        guard let tokenizer else { throw CoreAIGemmaError.loadFailed }
        let prompt: [Int]
        do {
            let chat: [Tokenizers.Message] = messages.map { ["role": $0.role, "content": $0.content] }
            prompt = try tokenizer.applyChatTemplate(
                messages: chat, tools: nil,
                additionalContext: ["enable_thinking": false, "preserve_thinking": false]
            )
        } catch { throw CoreAIGemmaError.invalidMessages }
        guard prompt.allSatisfy({ $0 >= 0 && $0 < vocabularySize && $0 <= Int(Int32.max) }) else {
            throw CoreAIGemmaError.invalidMessages
        }
        return prompt
    }

    /// Apple's sequence is not Sendable: construct and consume it on this one
    /// generic-executor operation. Only the engine's thread-safe cancel may overlap.
    @concurrent
    private static func consumeGeneration(
        engine: any InferenceEngine, tokenizer: any Tokenizers.Tokenizer,
        prompt: [Int], stopTokens: Set<Int32>, vocabularySize: Int, temperature: Double, budget: Int,
        onSnapshot: @escaping @Sendable (String) -> Void
    ) async throws -> String {
        var generated = CoreAIGemmaOutputBuffer()
        let clock = ContinuousClock()
        var lastSnapshot = clock.now
        do {
            try Task.checkCancellation()
            let stream = try await engine.generate(
                with: prompt.map(Int32.init),
                samplingConfiguration: SamplingConfiguration(temperature: temperature),
                inferenceOptions: InferenceOptions(maxTokens: budget, includeLogits: false)
            )
            for try await output in stream {
                try Task.checkCancellation()
                if stopTokens.contains(output.tokenId) {
                    stream.setStopReason(.eos)
                    break
                }
                guard output.tokenId >= 0, Int(output.tokenId) < vocabularySize else {
                    throw CoreAIGemmaError.inferenceFailed
                }
                generated.append(Int(output.tokenId))
                if generated.count == 1 || lastSnapshot.duration(to: clock.now) >= .milliseconds(100) {
                    let raw = generated.progressSnapshot { tokenizer.decode(tokens: $0) }
                    if let visible = CoreAIGemmaPolicy.visibleSnapshot(raw) { onSnapshot(visible) }
                    lastSnapshot = clock.now
                }
            }
            let cancelled = stream.stopReason == .cancelled || Task.isCancelled
            let failed = stream.stopReason == .error
            // Breaking at EOS otherwise leaves an active generation lease in the SDK.
            try await engine.cancel()
            if cancelled { throw CancellationError() }
            if failed { throw CoreAIGemmaError.inferenceFailed }
            try Task.checkCancellation()
        } catch {
            try? await engine.cancel()
            if error is CancellationError || Task.isCancelled { throw CancellationError() }
            throw (error as? CoreAIGemmaError) ?? .inferenceFailed
        }
        let text = try CoreAIGemmaPolicy.visibleOutput(generated.snapshot { tokenizer.decode(tokens: $0) })
        try Task.checkCancellation()
        onSnapshot(text)
        return text
    }
    #endif
}
