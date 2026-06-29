import Foundation
import Metal
import OSLog
import SwiftLlama
import llama

nonisolated struct GenerateRequest: Sendable {
    let id: UUID
    let sessionID: String?
    let systemPrompt: String
    let history: [(role: MessageRole, content: String)]
    let userMessage: String
    let temperature: Double
    let topP: Double
    let repetitionPenalty: Double
    let maxTokens: Int
    let modelName: String
    let relevantMemories: [MemoryContextItem]
    let attachments: [ChatAttachment]
    let responseFormat: LLMResponseFormat
    let seed: UInt32?
    let developerTraceModeEnabled: Bool
    let reasoningCaptureEnabled: Bool
    let reasoningTraceBudgetCharacters: Int
    let allowsMemoryPressureContinuation: Bool

    init(
        id: UUID = UUID(),
        sessionID: String? = nil,
        systemPrompt: String,
        history: [(role: MessageRole, content: String)],
        userMessage: String,
        temperature: Double,
        topP: Double,
        repetitionPenalty: Double,
        maxTokens: Int,
        modelName: String,
        relevantMemories: [MemoryContextItem],
        attachments: [ChatAttachment] = [],
        responseFormat: LLMResponseFormat = .plainText,
        seed: UInt32? = nil,
        developerTraceModeEnabled: Bool = false,
        reasoningCaptureEnabled: Bool = false,
        reasoningTraceBudgetCharacters: Int = 16_384,
        allowsMemoryPressureContinuation: Bool = false
    ) {
        self.id = id
        self.sessionID = sessionID
        self.systemPrompt = systemPrompt
        self.history = history
        self.userMessage = userMessage
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = maxTokens
        self.modelName = modelName
        self.relevantMemories = relevantMemories
        self.attachments = attachments
        self.responseFormat = responseFormat
        self.seed = seed
        self.developerTraceModeEnabled = developerTraceModeEnabled
        self.reasoningCaptureEnabled = developerTraceModeEnabled && reasoningCaptureEnabled
        self.reasoningTraceBudgetCharacters = max(0, reasoningTraceBudgetCharacters)
        self.allowsMemoryPressureContinuation = allowsMemoryPressureContinuation
    }

    init(
        id: UUID = UUID(),
        sessionID: String? = nil,
        systemPrompt: String,
        history: [(role: MessageRole, content: String)],
        userMessage: String,
        temperature: Double,
        topP: Double,
        repetitionPenalty: Double,
        maxTokens: Int,
        modelName: String,
        legacyRelevantMemories: [String],
        attachments: [ChatAttachment] = [],
        responseFormat: LLMResponseFormat = .plainText,
        seed: UInt32? = nil,
        developerTraceModeEnabled: Bool = false,
        reasoningCaptureEnabled: Bool = false,
        reasoningTraceBudgetCharacters: Int = 16_384,
        allowsMemoryPressureContinuation: Bool = false
    ) {
        self.init(
            id: id,
            sessionID: sessionID,
            systemPrompt: systemPrompt,
            history: history,
            userMessage: userMessage,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            maxTokens: maxTokens,
            modelName: modelName,
            relevantMemories: MemoryContextAdapter.fromLegacyStrings(legacyRelevantMemories),
            attachments: attachments,
            responseFormat: responseFormat,
            seed: seed,
            developerTraceModeEnabled: developerTraceModeEnabled,
            reasoningCaptureEnabled: reasoningCaptureEnabled,
            reasoningTraceBudgetCharacters: reasoningTraceBudgetCharacters,
            allowsMemoryPressureContinuation: allowsMemoryPressureContinuation
        )
    }

    func cappedForDeveloperReasoning() -> GenerateRequest {
        guard reasoningCaptureEnabled else { return self }
        return GenerateRequest(
            id: id,
            sessionID: sessionID,
            systemPrompt: systemPrompt,
            history: history,
            userMessage: userMessage,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            maxTokens: min(maxTokens, 768),
            modelName: modelName,
            relevantMemories: relevantMemories,
            attachments: attachments,
            responseFormat: responseFormat,
            seed: seed,
            developerTraceModeEnabled: developerTraceModeEnabled,
            reasoningCaptureEnabled: reasoningCaptureEnabled,
            reasoningTraceBudgetCharacters: reasoningTraceBudgetCharacters,
            allowsMemoryPressureContinuation: allowsMemoryPressureContinuation
        )
    }

    var preservesRawStructuredAgentOutput: Bool {
        modelName == "agent-json" || responseFormat.requiresRawStructuredOutput
    }
}

private extension LLMResponseFormat {
    var requiresRawStructuredOutput: Bool {
        switch self {
        case .plainText:
            return false
        case .json, .toolCallJSON, .constrainedJSON:
            return true
        }
    }
}

nonisolated enum GenerationToken: Sendable {
    case text(String)
    case done
}

final class LlamaGenerationCancellationToken: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelledReason: String?

    func cancel(reason: String) {
        lock.lock()
        if cancelledReason == nil { cancelledReason = reason }
        lock.unlock()
    }

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelledReason != nil
    }

    var reason: String? {
        lock.lock()
        defer { lock.unlock() }
        return cancelledReason
    }

    func checkCancellation() throws {
        if isCancelled { throw CancellationError() }
        try Task.checkCancellation()
    }
}

private final class LlamaRuntimeStopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var requested = false

    func requestStop() {
        lock.lock()
        requested = true
        lock.unlock()
    }

    func reset() {
        lock.lock()
        requested = false
        lock.unlock()
    }

    var isStopRequested: Bool {
        lock.lock()
        defer { lock.unlock() }
        return requested
    }
}

nonisolated enum LlamaError: Error, Sendable {
    case noModelLoaded
    case slotModelNotLoaded(String)
    case modelFileNotFound(String)
    case failedToInitializeContext(String)
    case embeddingModelNotLoaded
    case embeddingFailed(String)
}

extension LlamaError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .noModelLoaded:
            return "No chat model is currently loaded."
        case .slotModelNotLoaded(let slot):
            return "No chat model is currently loaded for slot \(slot)."
        case .modelFileNotFound(let path):
            return "Model file not found at \(path)."
        case .failedToInitializeContext(let details):
            return "Failed to initialize context: \(details)"
        case .embeddingModelNotLoaded:
            return "No embedding model is currently loaded."
        case .embeddingFailed(let details):
            return "Failed to compute embedding: \(details)"
        }
    }
}

private struct ChatRuntime {
    var service: SwiftLlama.LlamaService
    var modelPath: String
    var contextSize: Int
    var batchSize: UInt32
}

private enum LlamaRuntimeScheduling {
    static let inferenceTaskPriority: TaskPriority = .utility
    static let maxForegroundDecodeThreads: Int32 = 2
    static let minInteractiveBatchSize: UInt32 = 32
    static let maxInteractiveBatchSize: UInt32 = 128
    static let decodeYieldTokenInterval = 4

    static func decodeThreadCount(detectedCores: Int) -> Int32 {
        let efficientForegroundCount = max(1, min(Int(maxForegroundDecodeThreads), max(1, detectedCores / 2)))
        return Int32(efficientForegroundCount)
    }

    static func batchSize(_ requested: UInt32) -> UInt32 {
        min(max(requested, minInteractiveBatchSize), maxInteractiveBatchSize)
    }
}

private final class LlamaRuntimeLogCapture: @unchecked Sendable {
    static let shared = LlamaRuntimeLogCapture()
    private static let callback: ggml_log_callback = { _, text, _ in
        guard let text else { return }
        let line = String(cString: text).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty else { return }
        LlamaRuntimeLogCapture.shared.record(line)
    }

    private let lock = NSLock()
    private let logger = Logger(subsystem: "com.lumen.runtime", category: "llama.cpp")
    nonisolated(unsafe) private var installed = false
    nonisolated(unsafe) private var lines: [String] = []

    nonisolated func installIfNeeded() {
        lock.lock()
        let shouldInstall = !installed
        installed = true
        lock.unlock()
        guard shouldInstall else { return }
        llama_log_set(Self.callback, nil)
    }

    nonisolated func markLoadBoundary() {
        lock.lock()
        lines.removeAll(keepingCapacity: true)
        lock.unlock()
    }

    nonisolated func record(_ line: String) {
        let lower = line.lowercased()
        let isBackendSignal = lower.contains("using device")
            || lower.contains("picking default device")
            || lower.contains("gpu name")
            || lower.contains("recommendedmaxworkingsetsize")
            || lower.contains("assigned to device")
            || lower.contains("offload")
            || lower.contains("kv")
            || lower.contains("prompt eval")
            || lower.contains("eval time")
            || lower.contains("tokens per second")
        let isKernelCatalogNoise = lower.contains("loaded kernel") || lower.contains("skipping kernel")
        guard isBackendSignal, !isKernelCatalogNoise else { return }

        lock.lock()
        lines.append(line)
        if lines.count > 80 {
            lines.removeFirst(lines.count - 80)
        }
        lock.unlock()

        logger.info("event=llama.cpp.log \(line, privacy: .public)")
    }

    nonisolated func snapshot() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return lines
    }
}

private struct LlamaOffloadSnapshot {
    let actualBackend: String?
    let offloadedLayers: Int?
    let totalLayers: Int?
    let kqvOffloaded: Bool?
    let notes: [String]

    nonisolated static func fromRuntimeLogs(totalModelLayers: Int?, requestedKQVOffload: Bool) -> LlamaOffloadSnapshot {
        let lines = LlamaRuntimeLogCapture.shared.snapshot()
        let lowerLines = lines.map { $0.lowercased() }
        let backend = lowerLines.contains { $0.contains("metal") || $0.contains("gpu") } ? "metal" : nil
        var offloaded: Int?
        var total: Int? = totalModelLayers

        for line in lowerLines {
            if let parsed = parseOffloadedLayerCount(from: line) {
                offloaded = parsed.offloaded
                total = parsed.total ?? total
            }
        }

        let kqv = lowerLines.contains { line in
            (line.contains("kv") || line.contains("kqv")) && line.contains("offload")
        } ? true : (requestedKQVOffload && backend != nil ? true : nil)

        let selectedNotes = lines
            .filter { line in
                let lower = line.lowercased()
                return lower.contains("metal") || lower.contains("offload") || lower.contains("kv")
            }
            .suffix(8)
        return LlamaOffloadSnapshot(actualBackend: backend, offloadedLayers: offloaded, totalLayers: total, kqvOffloaded: kqv, notes: Array(selectedNotes))
    }

    nonisolated private static func parseOffloadedLayerCount(from line: String) -> (offloaded: Int, total: Int?)? {
        let patterns = [
            #"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers"#,
            #"offloading\s+(\d+)\s+repeating layers"#,
            #"offloading\s+(\d+)\s+layers"#
        ]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let ns = line as NSString
            let range = NSRange(location: 0, length: ns.length)
            guard let match = regex.firstMatch(in: line, range: range), match.numberOfRanges >= 2 else { continue }
            let offloaded = Int(ns.substring(with: match.range(at: 1)))
            let total = match.numberOfRanges >= 3 && match.range(at: 2).location != NSNotFound ? Int(ns.substring(with: match.range(at: 2))) : nil
            if let offloaded { return (offloaded, total) }
        }
        return nil
    }
}

private actor AdapterChatRuntime {
    private let model: LlamaModel
    private let context: LlamaContext
    private let stopFlag = LlamaRuntimeStopFlag()
    let modelPath: String
    private let contextSize: Int
    private let batchSize: UInt32
    private var batch: LlamaBatch
    private var processedTokens: [llama_token] = []
    private var currentTokenPosition: Int32 = 0
    private var loadedAdapters: [LumenModelSlot: LlamaLoraAdapter] = [:]
    private let accelerationDiagnostics: RuntimeAccelerationDiagnostics

    init(path: String, contextSize: Int, batchSize: UInt32) throws {
        LlamaRuntimeLogCapture.shared.installIfNeeded()
        LlamaRuntimeLogCapture.shared.markLoadBoundary()
        let detectedCores = ProcessInfo.processInfo.processorCount
        let runtimeThreadCount = LlamaRuntimeScheduling.decodeThreadCount(detectedCores: detectedCores)
        let effectiveBatchSize = LlamaRuntimeScheduling.batchSize(batchSize)
        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = 999
        guard let model = LlamaModel(path: path, parameters: modelParams) else {
            throw LlamaError.failedToInitializeContext("Unable to load shared chat base GGUF")
        }
        var contextParams = llama_context_default_params()
        contextParams.n_ctx = UInt32(max(1, contextSize))
        contextParams.n_batch = effectiveBatchSize
        contextParams.n_ubatch = effectiveBatchSize
        contextParams.n_threads = runtimeThreadCount
        contextParams.n_threads_batch = runtimeThreadCount
        contextParams.offload_kqv = true
        guard let context = LlamaContext(model: model, parameters: contextParams) else {
            throw LlamaError.failedToInitializeContext("Unable to create shared chat context")
        }
        self.model = model
        self.context = context
        self.modelPath = path
        self.contextSize = contextSize
        self.batchSize = effectiveBatchSize
        self.batch = LlamaBatch(initialSize: Int32(effectiveBatchSize))
        let layerCount = Int(model.nLayer())
        let offload = LlamaOffloadSnapshot.fromRuntimeLogs(totalModelLayers: layerCount, requestedKQVOffload: true)
        self.accelerationDiagnostics = RuntimeAccelerationDiagnostics.forCurrentRuntime(
            requestedBackend: "metal",
            requestedGpuLayers: 999,
            requestedKQVOffload: true,
            actualBackend: offload.actualBackend,
            actualOffloadedLayers: offload.offloadedLayers,
            actualTotalLayers: offload.totalLayers,
            metalDeviceUsed: offload.actualBackend == "metal" ? MTLCreateSystemDefaultDevice()?.name : nil,
            actualKQVOffload: offload.kqvOffloaded,
            notes: offload.notes
        )
    }

    func configuredContextSize() -> Int { contextSize }
    func runtimeAccelerationDiagnostics() -> RuntimeAccelerationDiagnostics { accelerationDiagnostics }

    nonisolated func stopCompletion() {
        stopFlag.requestStop()
    }

    func loadRoleAdapter(slot: LumenModelSlot, path: String) throws {
        loadedAdapters[slot] = try LlamaLoraAdapter(model: model, path: path)
    }

    func activateRoleAdapter(slot: LumenModelSlot, scale: Float) throws {
        clearAdapters()
        guard let adapter = loadedAdapters[slot] else { return }
        try context.apply(loraAdapter: adapter, scale: scale)
    }

    func clearAdapters() {
        context.removeAllLoraAdapters()
    }

    func resetKVCache() {
        stopFlag.reset()
        context.clearKVCache()
        processedTokens.removeAll()
        currentTokenPosition = 0
        batch = LlamaBatch(initialSize: Int32(batchSize))
    }

    func streamCompletion(
        of messages: [LlamaChatMessage],
        samplingConfig: LlamaSamplingConfig,
        maxTokens: Int?,
        cancellationToken: LlamaGenerationCancellationToken? = nil
    ) -> AsyncThrowingStream<String, Error> {
        stopFlag.reset()
        return AsyncThrowingStream<String, Error> { continuation in
            let task = Task.detached(priority: LlamaRuntimeScheduling.inferenceTaskPriority) { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                await self.generateCompletion(
                    messages: messages,
                    samplingConfig: samplingConfig,
                    maxTokens: maxTokens,
                    cancellationToken: cancellationToken,
                    continuation: continuation
                )
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func generateCompletion(
        messages: [LlamaChatMessage],
        samplingConfig: LlamaSamplingConfig,
        maxTokens: Int?,
        cancellationToken: LlamaGenerationCancellationToken?,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async {
        do {
            try checkGenerationCancellation(cancellationToken)
            try Task.checkCancellation()
            try initializeCompletion(messages: messages, cancellationToken: cancellationToken)
            let sampler = LlamaSampler(config: samplingConfig, model: model)
            let limit = min(maxTokens ?? Int.max, max(0, contextSize - Int(currentTokenPosition) - 1))
            var emitted = 0
            // Keep the native decode critical section non-suspending. Any `await`
            // here would make this actor reentrant while it owns the shared llama
            // context, batch, processed tokens, and current token position.
            while emitted < limit, !Task.isCancelled, !stopFlag.isStopRequested {
                try checkGenerationCancellation(cancellationToken)
                let tokenStarted = ProcessInfo.processInfo.systemUptime
                let token = sampler.sample(context: context)
                if model.isEogToken(token) { break }
                batch.reset()
                batch.addToken(token, at: currentTokenPosition, logits: true)
                processedTokens.append(token)
                currentTokenPosition += 1
                try context.decode(batch: batch)
                continuation.yield(model.piece(from: token))
                emitted += 1
                try recordCPUWorkAndCheckBudget(since: tokenStarted, cancellationToken: cancellationToken)
            }
            continuation.finish()
        } catch is CancellationError {
            continuation.finish()
        } catch {
            continuation.finish(throwing: error)
        }
    }

    private func initializeCompletion(messages: [LlamaChatMessage], cancellationToken: LlamaGenerationCancellationToken? = nil) throws {
        try checkGenerationCancellation(cancellationToken)
        let prompt = model.applyChatTemplate(to: messages, addAssistant: nil)
        try checkGenerationCancellation(cancellationToken)
        let tokens = model.tokenize(text: prompt, addBos: model.shouldAddBos(), special: true)
        guard tokens.count < contextSize - 4 else {
            throw LlamaError.failedToInitializeContext("Prompt exceeds shared chat context window")
        }
        guard !tokens.isEmpty else {
            resetKVCache()
            return
        }

        context.clearKVCache()
        processedTokens.removeAll()
        currentTokenPosition = 0
        batch.reset()

        let lastIndex = tokens.count - 1
        // Keep prompt evaluation non-suspending for the same reason as the decode
        // loop above: this method mutates the shared llama context state.
        for (index, token) in tokens.enumerated() {
            try checkGenerationCancellation(cancellationToken)
            let isLast = index == lastIndex
            batch.addToken(token, at: Int32(index), logits: isLast)
            processedTokens.append(token)
            if batch.size == Int32(batchSize) || isLast {
                let batchStarted = ProcessInfo.processInfo.systemUptime
                try context.decode(batch: batch)
                batch.reset()
                try recordCPUWorkAndCheckBudget(since: batchStarted, cancellationToken: cancellationToken)
            }
        }
        currentTokenPosition = Int32(processedTokens.count)
    }

    private func checkGenerationCancellation(_ cancellationToken: LlamaGenerationCancellationToken?) throws {
        if stopFlag.isStopRequested {
            cancellationToken?.cancel(reason: "runtime-stop-requested")
            throw CancellationError()
        }
        try cancellationToken?.checkCancellation()
        try Task.checkCancellation()
    }

    private func recordCPUWorkAndCheckBudget(since start: TimeInterval, cancellationToken: LlamaGenerationCancellationToken?) throws {
        CPUWatchdogGuard.shared.recordWork(category: .chatGeneration, duration: ProcessInfo.processInfo.systemUptime - start)
        if CPUWatchdogGuard.shared.shouldDegrade(category: .chatGeneration) {
            cancellationToken?.cancel(reason: "cpu-watchdog-degraded")
            throw CancellationError()
        }
    }
}

private struct LoadedRoleAdapter {
    let slot: LumenModelSlot
    let path: String
    let scale: Float
    let loadedAt: Date
}

private final class LlamaGenerationTaskBox: @unchecked Sendable {
    private let lock = NSLock()
    private var task: Task<Void, Never>?

    func set(_ task: Task<Void, Never>) {
        lock.lock()
        self.task = task
        lock.unlock()
    }

    func cancel() {
        lock.lock()
        let task = self.task
        lock.unlock()
        task?.cancel()
    }
}

private struct ActiveLlamaGeneration {
    let requestID: UUID
    let slot: LumenModelSlot
    let token: LlamaGenerationCancellationToken
    let taskBox: LlamaGenerationTaskBox
    let diskWriteLease: DiskWriteGenerationLease
    let startedAt: Date
}

nonisolated struct PromptBuildResult: Sendable {
    let messages: [LlamaChatMessage]
    let assembly: PromptAssembly
    let initialPromptChars: Int
    let finalPromptChars: Int
    let estimatedPromptTokens: Int
    let latencySelection: PromptLatencySelection
}

nonisolated struct LlamaAdapterTraceMetadata: Codable, Sendable, Hashable {
    let modelFamily: String?
    let baseModelPath: String?
    let adapterID: String?
    let adapterSlot: String?
    let adapterPath: String?
    let adapterApplied: Bool
    let adapterScale: Float?
    let adapterFailureReason: String?
}

private enum LlamaErrorCode: String, Sendable {
    case network = "network"
    case decode = "decode"
    case modelLoad = "model-load"
    case timeout = "timeout"
    case runtime = "runtime"
}

final actor AppLlamaService {
    static let shared = AppLlamaService()

    private let logger = Logger(subsystem: "com.lumen.runtime", category: "llama.service")
    private var lastAccelerationDiagnostics = RuntimeAccelerationDiagnostics.forCurrentRuntime(requestedBackend: "unknown", requestedGpuLayers: nil, requestedKQVOffload: nil)

    private var chatRuntimes: [LumenModelSlot: ChatRuntime] = [:]
    private var primaryChatSlot: LumenModelSlot = .cortex
    private var sharedChatRuntime: AdapterChatRuntime?
    private var sharedChatBasePath: String?
    private var roleAdapters: [LumenModelSlot: LoadedRoleAdapter] = [:]
    private var activeAdapterSlot: LumenModelSlot?
    private var lastAdapterFailureReason: String?
    private var completedTracePayloads: [UUID: CompletedGenerationTracePayload] = [:]
    private var activeGenerations: [UUID: ActiveLlamaGeneration] = [:]
    private var lastCancellationReasonByRequest: [UUID: String] = [:]

    private var embeddingModelPath: String?
    private var embeddingModel: LlamaModel?
    private var embeddingContext: LlamaContext?
    private var embeddingContextSize: UInt32 = 2048
    private var embeddingBatchSize: UInt32 = 256
    private var embeddingThreads: Int32 = 1

    private init() {}

    var isChatLoaded: Bool { sharedChatRuntime != nil || chatRuntimes[primaryChatSlot] != nil || !chatRuntimes.isEmpty }
    var isEmbedLoaded: Bool { embeddingContext != nil }
    var hasSemanticEmbeddingRuntime: Bool { embeddingContext != nil }
    var loadedChatPath: String? { sharedChatBasePath ?? chatRuntimes[primaryChatSlot]?.modelPath ?? chatRuntimes.values.first?.modelPath }
    var loadedEmbedPath: String? { embeddingModelPath }

    var loadedChatPathsBySlot: [LumenModelSlot: String] {
        Dictionary(uniqueKeysWithValues: chatRuntimes.map { ($0.key, $0.value.modelPath) })
    }

    var activeAdapterSlotValue: LumenModelSlot? { activeAdapterSlot }

    func takeCompletedTracePayload(requestID: UUID) -> CompletedGenerationTracePayload? {
        completedTracePayloads.removeValue(forKey: requestID)
    }

    func contextSizeForDiagnostics(slot: LumenModelSlot) async -> Int {
        await contextSizeForGeneration(slot: slot)
    }

    func cancelActiveGeneration(reason: String) async {
        let generations = activeGenerations
        guard !generations.isEmpty else {
            logger.info("event=llama.chat.cancel_active_generation_empty reason=\(reason, privacy: .public)")
            return
        }
        for (requestID, generation) in generations {
            generation.token.cancel(reason: reason)
            generation.taskBox.cancel()
            lastCancellationReasonByRequest[requestID] = reason
        }
        let legacySlots = Set(generations.values.map(\.slot))
        logger.info("event=llama.chat.cancel_active_generation count=\(generations.count, privacy: .public) reason=\(reason, privacy: .public)")
        sharedChatRuntime?.stopCompletion()
        for slot in legacySlots {
            await stopCompletion(for: slot)
        }
    }

    func hasActiveGenerationForTesting(requestID: UUID) -> Bool {
        activeGenerations[requestID] != nil
    }

    private func registerActiveGeneration(requestID: UUID, slot: LumenModelSlot, token: LlamaGenerationCancellationToken, taskBox: LlamaGenerationTaskBox, diskWriteLease: DiskWriteGenerationLease) {
        activeGenerations[requestID] = ActiveLlamaGeneration(requestID: requestID, slot: slot, token: token, taskBox: taskBox, diskWriteLease: diskWriteLease, startedAt: Date())
        Task { @MainActor in DeferredMaintenanceQueue.shared.setChatOrVoiceActive(true) }
    }

    private func unregisterActiveGeneration(requestID: UUID) {
        let generation = activeGenerations.removeValue(forKey: requestID)
        generation?.diskWriteLease.end()
        lastCancellationReasonByRequest.removeValue(forKey: requestID)
        if activeGenerations.isEmpty {
            Task { @MainActor in DeferredMaintenanceQueue.shared.setChatOrVoiceActive(false) }
        }
    }

    func isChatLoaded(for slot: LumenModelSlot) -> Bool {
        sharedChatRuntime != nil || chatRuntimes[slot] != nil
    }

    func loadedChatPath(for slot: LumenModelSlot) -> String? {
        sharedChatBasePath ?? chatRuntimes[slot]?.modelPath
    }

    func slotLoaded(withPath path: String) -> LumenModelSlot? {
        chatRuntimes.first(where: { $0.value.modelPath == path })?.key
    }

    func aliasChatRuntime(from sourceSlot: LumenModelSlot, to targetSlot: LumenModelSlot) {
        guard let runtime = chatRuntimes[sourceSlot] else { return }
        chatRuntimes[targetSlot] = runtime
    }

    func loadSharedChatModel(path: String, contextSize: Int, batchSize: UInt32 = 256) async throws {
        if sharedChatBasePath == path, sharedChatRuntime != nil { return }
        guard FileManager.default.fileExists(atPath: path) else { throw LlamaError.modelFileNotFound(path) }
        logger.info(
            "event=llama.chat.runtime_init_start path=\(path, privacy: .public) context_size=\(contextSize, privacy: .public) batch_size=\(batchSize, privacy: .public) gpu_target_layers=999"
        )
        do {
            sharedChatRuntime = try AdapterChatRuntime(path: path, contextSize: contextSize, batchSize: batchSize)
            if let diagnostics = await sharedChatRuntime?.runtimeAccelerationDiagnostics() {
                lastAccelerationDiagnostics = diagnostics
                logger.info(
                    "event=llama.chat.acceleration_verified backend=\(diagnostics.actualBackend ?? "unknown", privacy: .public) metal_device=\(diagnostics.metalDeviceUsed ?? diagnostics.metalDeviceName ?? "unknown", privacy: .public) offloaded_layers=\(diagnostics.actualOffloadedLayers.map(String.init) ?? "unknown", privacy: .public) total_layers=\(diagnostics.actualTotalLayers.map(String.init) ?? "unknown", privacy: .public) kqv_offload=\(diagnostics.actualKQVOffload.map { String($0) } ?? "unknown", privacy: .public) verification=\(diagnostics.verificationLevel, privacy: .public)"
                )
            }
            logger.info(
                "event=llama.chat.runtime_init_success path=\(path, privacy: .public) context_size=\(contextSize, privacy: .public) batch_size=\(batchSize, privacy: .public)"
            )
        } catch {
            logger.error(
                "event=llama.chat.runtime_init_failure path=\(path, privacy: .public) context_size=\(contextSize, privacy: .public) batch_size=\(batchSize, privacy: .public) message=\(error.localizedDescription, privacy: .public) fallback=cpu_or_nonoffload"
            )
            throw error
        }
        sharedChatBasePath = path
        activeAdapterSlot = nil
        roleAdapters.removeAll()
        chatRuntimes.removeAll()
    }

    func loadRoleAdapter(slot: LumenModelSlot, path: String, scale: Float = 1.0) async throws {
        guard let runtime = sharedChatRuntime else { throw LlamaError.noModelLoaded }
        guard FileManager.default.fileExists(atPath: path) else { throw LlamaError.modelFileNotFound(path) }
        if roleAdapters[slot]?.path == path { return }
        try await runtime.loadRoleAdapter(slot: slot, path: path)
        roleAdapters[slot] = LoadedRoleAdapter(slot: slot, path: path, scale: scale, loadedAt: Date())
    }

    func loadRoleAdapterIfNeeded(slot: LumenModelSlot, path: String, scale: Float = 1.0) async throws -> Bool {
        if roleAdapters[slot]?.path == path { return false }
        try await loadRoleAdapter(slot: slot, path: path, scale: scale)
        return true
    }

    func activateRoleAdapter(slot: LumenModelSlot) async throws {
        guard let runtime = sharedChatRuntime else { throw LlamaError.noModelLoaded }
        if activeAdapterSlot == slot { return }
        guard let loaded = roleAdapters[slot] else {
            await runtime.clearAdapters()
            activeAdapterSlot = nil
            return
        }
        do {
            try await runtime.activateRoleAdapter(slot: loaded.slot, scale: loaded.scale)
            activeAdapterSlot = slot
            lastAdapterFailureReason = nil
        } catch {
            await runtime.clearAdapters()
            activeAdapterSlot = nil
            lastAdapterFailureReason = error.localizedDescription
            throw error
        }
    }

    func activateRoleAdapterIfNeeded(slot: LumenModelSlot) async throws -> Bool {
        if activeAdapterSlot == slot { return false }
        try await activateRoleAdapter(slot: slot)
        return true
    }

    func clearActiveRoleAdapter() async {
        if let sharedChatRuntime {
            await sharedChatRuntime.clearAdapters()
        }
        activeAdapterSlot = nil
    }

    func unloadRoleAdapter(slot: LumenModelSlot) async {
        if activeAdapterSlot == slot { await clearActiveRoleAdapter() }
        roleAdapters.removeValue(forKey: slot)
    }

    func unloadAllRoleAdapters() async {
        await clearActiveRoleAdapter()
        roleAdapters.removeAll()
    }

    func loadModel(named name: String, contextSize: UInt32 = 2048, batchSize: UInt32 = 256) throws {
        guard let url = Bundle.main.url(forResource: name, withExtension: "gguf") else {
            throw LlamaError.modelFileNotFound("Bundle resource: \(name).gguf")
        }
        try loadChatModelSync(path: url.path, slot: primaryChatSlot, contextSize: Int(contextSize), batchSize: batchSize)
    }

    func loadChatModel(path: String, contextSize: Int = 2048) async throws {
        try loadChatModelSync(path: path, slot: primaryChatSlot, contextSize: contextSize, batchSize: 256)
    }

    func loadChatModel(path: String, for slot: LumenModelSlot, contextSize: Int = 2048) async throws {
        try loadChatModelSync(path: path, slot: slot, contextSize: contextSize, batchSize: 256)
        primaryChatSlot = slot
    }

    func loadFleetChatModels(assignments: [LumenModelSlot: LumenModelAssignment], contextSize: Int = 2048) async -> [LumenModelSlot: String] {
        var failures: [LumenModelSlot: String] = [:]
        for slot in [LumenModelSlot.cortex, .executor, .mouth, .mimicry, .rem] {
            guard let assignment = assignments[slot] else { continue }
            do {
                if assignment.usesRoleAdapter || assignment.modelFamily == .qwen3 {
                    try await loadSharedChatModel(path: assignment.localPath, contextSize: contextSize)
                    if let adapterPath = assignment.adapterPath {
                        try await loadRoleAdapter(slot: slot, path: adapterPath, scale: assignment.adapterScale)
                    }
                } else {
                    await unloadAllChat()
                    try await loadChatModel(path: assignment.localPath, for: slot, contextSize: contextSize)
                }
            } catch {
                failures[slot] = error.localizedDescription
            }
        }
        return failures
    }

    func loadEmbeddingModel(path: String) async throws {
        guard FileManager.default.fileExists(atPath: path) else {
            throw LlamaError.modelFileNotFound(path)
        }

        LlamaRuntimeLogCapture.shared.installIfNeeded()
        LlamaRuntimeLogCapture.shared.markLoadBoundary()

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = 0

        guard let model = LlamaModel(path: path, parameters: modelParams) else {
            throw LlamaError.failedToInitializeContext("Unable to load embedding GGUF")
        }

        guard let context = makeEmbeddingContext(for: model) else {
            throw LlamaError.failedToInitializeContext("Unable to create embedding context")
        }

        context.setEmbeddingsOutput(true)
        context.setCausalAttention(false)

        embeddingModel = model
        embeddingContext = context
        embeddingModelPath = path
    }

    func unloadChat() async {
        chatRuntimes.removeValue(forKey: primaryChatSlot)
        if let first = chatRuntimes.keys.first {
            primaryChatSlot = first
        }
    }

    func unloadChat(for slot: LumenModelSlot) async {
        chatRuntimes.removeValue(forKey: slot)
        if primaryChatSlot == slot, let first = chatRuntimes.keys.first {
            primaryChatSlot = first
        }
    }

    func unloadAllChat() async {
        chatRuntimes.removeAll()
        sharedChatRuntime = nil
        sharedChatBasePath = nil
        roleAdapters.removeAll()
        activeAdapterSlot = nil
        primaryChatSlot = .cortex
    }

    func unloadEmbed() async {
        embeddingModelPath = nil
        embeddingModel = nil
        embeddingContext = nil
    }

    func reloadChat(contextSize: Int = 2048) async throws {
        guard let runtime = chatRuntimes[primaryChatSlot] ?? chatRuntimes.values.first else { throw LlamaError.noModelLoaded }
        try loadChatModelSync(path: runtime.modelPath, slot: primaryChatSlot, contextSize: contextSize, batchSize: runtime.batchSize)
    }

    func reloadChat(for slot: LumenModelSlot, contextSize: Int = 2048) async throws {
        guard let runtime = chatRuntimes[slot] else { throw LlamaError.slotModelNotLoaded(slot.rawValue) }
        try loadChatModelSync(path: runtime.modelPath, slot: slot, contextSize: contextSize, batchSize: runtime.batchSize)
    }

    func reloadEmbed() async throws {
        guard let embeddingModelPath else { throw LlamaError.embeddingModelNotLoaded }
        try await loadEmbeddingModel(path: embeddingModelPath)
    }

    func streamResponse(
        messages: [LlamaChatMessage],
        temperature: Float = 0.8,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.1,
        maxTokens: Int? = nil,
        seed: UInt32? = nil,
        cancellationToken: LlamaGenerationCancellationToken? = nil
    ) async throws -> AsyncThrowingStream<String, Error> {
        if let runtime = sharedChatRuntime {
            return try await streamResponse(
                adapterRuntime: runtime,
                messages: messages,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                maxTokens: maxTokens,
                seed: seed,
                cancellationToken: cancellationToken
            )
        }
        guard let runtime = chatRuntimes[primaryChatSlot] ?? chatRuntimes.values.first else {
            throw LlamaError.noModelLoaded
        }
        return try await streamResponse(
            runtime: runtime,
            stopSlot: primaryChatSlot,
            messages: messages,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            maxTokens: maxTokens,
            seed: seed,
            cancellationToken: cancellationToken
        )
    }

    func streamResponse(
        slot: LumenModelSlot,
        messages: [LlamaChatMessage],
        temperature: Float = 0.8,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.1,
        maxTokens: Int? = nil,
        seed: UInt32? = nil,
        cancellationToken: LlamaGenerationCancellationToken? = nil
    ) async throws -> AsyncThrowingStream<String, Error> {
        if let runtime = sharedChatRuntime {
            return try await streamResponse(
                adapterRuntime: runtime,
                messages: messages,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                maxTokens: maxTokens,
                seed: seed,
                cancellationToken: cancellationToken
            )
        }
        guard let runtime = chatRuntimes[slot] else {
            throw LlamaError.slotModelNotLoaded(slot.rawValue)
        }
        return try await streamResponse(
            runtime: runtime,
            stopSlot: slot,
            messages: messages,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            maxTokens: maxTokens,
            seed: seed,
            cancellationToken: cancellationToken
        )
    }


    private func streamResponse(
        adapterRuntime runtime: AdapterChatRuntime,
        messages: [LlamaChatMessage],
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int?,
        seed: UInt32?,
        cancellationToken: LlamaGenerationCancellationToken? = nil
    ) async throws -> AsyncThrowingStream<String, Error> {
        let resolvedSeed = seed ?? makeRandomSeed()
        let sampling = LlamaSamplingConfig(
            temperature: temperature,
            seed: resolvedSeed,
            topP: topP,
            repetitionPenaltyConfig: LlamaRepetitionPenaltyConfig(repeatPenalty: repetitionPenalty)
        )
        return await runtime.streamCompletion(of: messages, samplingConfig: sampling, maxTokens: maxTokens, cancellationToken: cancellationToken)
    }

    private func streamResponse(
        runtime: ChatRuntime,
        stopSlot: LumenModelSlot,
        messages: [LlamaChatMessage],
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int?,
        seed: UInt32?,
        cancellationToken: LlamaGenerationCancellationToken? = nil
    ) async throws -> AsyncThrowingStream<String, Error> {
        try cancellationToken?.checkCancellation()

        let resolvedSeed = seed ?? makeRandomSeed()
        let sampling = LlamaSamplingConfig(
            temperature: temperature,
            seed: resolvedSeed,
            topP: topP,
            repetitionPenaltyConfig: LlamaRepetitionPenaltyConfig(repeatPenalty: repetitionPenalty)
        )
        let rawStream = try await runtime.service.streamCompletion(of: messages, samplingConfig: sampling)

        return AsyncThrowingStream { continuation in
            let cap = maxTokens.map { max(0, $0) }
            let task = Task.detached(priority: LlamaRuntimeScheduling.inferenceTaskPriority) { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }

                do {
                    try cancellationToken?.checkCancellation()

                    if cap == 0 {
                        await self.stopCompletion(for: stopSlot)
                        continuation.finish()
                        return
                    }

                    var emitted = 0
                    for try await chunk in rawStream {
                        try cancellationToken?.checkCancellation()
                        try Task.checkCancellation()

                        continuation.yield(chunk)
                        emitted += 1

                        if let cap, emitted >= cap {
                            await self.stopCompletion(for: stopSlot)
                            break
                        }

                        if emitted.isMultiple(of: LlamaRuntimeScheduling.decodeYieldTokenInterval) {
                            await Task.yield()
                        }
                    }

                    continuation.finish()
                } catch is CancellationError {
                    await self.stopCompletion(for: stopSlot)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                cancellationToken?.cancel(reason: "legacy-stream-terminated")
                task.cancel()
            }
        }
    }

    func respond(
        messages: [LlamaChatMessage],
        temperature: Float = 0.8,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.1,
        maxTokens: Int? = nil,
        seed: UInt32? = nil
    ) async throws -> String {
        let stream = try await streamResponse(
            messages: messages,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            maxTokens: maxTokens,
            seed: seed
        )
        var parser = ReasoningAwareStreamParser()
        for try await chunk in stream {
            _ = parser.ingest(chunk)
        }
        _ = parser.finish()
        return FinalOutputSanitizer.sanitizeUserVisibleText(parser.result.visibleAnswer).text
    }

    func respond(
        slot: LumenModelSlot,
        messages: [LlamaChatMessage],
        temperature: Float = 0.8,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.1,
        maxTokens: Int? = nil,
        seed: UInt32? = nil
    ) async throws -> String {
        let stream = try await streamResponse(
            slot: slot,
            messages: messages,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            maxTokens: maxTokens,
            seed: seed
        )
        var parser = ReasoningAwareStreamParser()
        for try await chunk in stream {
            _ = parser.ingest(chunk)
        }
        _ = parser.finish()
        return FinalOutputSanitizer.sanitizeUserVisibleText(parser.result.visibleAnswer).text
    }

    func resetKVCache() async {
        if let sharedChatRuntime {
            await sharedChatRuntime.resetKVCache()
            return
        }
        let runtimes = chatRuntimes
        for (slot, runtime) in runtimes {
            do {
                try loadChatModelSync(path: runtime.modelPath, slot: slot, contextSize: runtime.contextSize, batchSize: runtime.batchSize)
            } catch {
                chatRuntimes.removeValue(forKey: slot)
            }
        }
    }

    func resetKVCache(for slot: LumenModelSlot) async {
        if let sharedChatRuntime {
            await sharedChatRuntime.resetKVCache()
            return
        }
        guard let runtime = chatRuntimes[slot] else { return }
        do {
            try loadChatModelSync(path: runtime.modelPath, slot: slot, contextSize: runtime.contextSize, batchSize: runtime.batchSize)
        } catch {
            chatRuntimes.removeValue(forKey: slot)
        }
    }

    func stream(_ req: GenerateRequest) -> AsyncStream<GenerationToken> {
        stream(req, slot: primaryChatSlot)
    }

    private func allowsGenerationWork(
        request: GenerateRequest,
        slot: LumenModelSlot,
        resourceReason: String
    ) async -> Bool {
        let snapshot = await MainActor.run { ResourceBudgetGate.diagnosticSnapshot() }
        if await MainActor.run(body: { ResourceBudgetGate.allowsHeavyModelWork(snapshot: snapshot, reason: resourceReason) }) {
            return true
        }

        guard request.allowsMemoryPressureContinuation,
              request.preservesRawStructuredAgentOutput,
              await SlotModelRuntimeCoordinator.shared.hasLoadedRuntimeReadyForContinuation(slot: slot) else {
            return false
        }

        return await MainActor.run {
            ResourceBudgetGate.allowsLoadedForegroundContinuationAfterMemoryPressure(snapshot: snapshot, reason: resourceReason)
        }
    }

    func stream(_ req: GenerateRequest, slot: LumenModelSlot) -> AsyncStream<GenerationToken> {
        return AsyncStream<GenerationToken>(bufferingPolicy: .unbounded) { (continuation: AsyncStream<GenerationToken>.Continuation) in
            let cancellationToken = LlamaGenerationCancellationToken()
            let taskBox = LlamaGenerationTaskBox()
            let diskWriteLease = DiskWriteBudget.shared.beginGeneration()
            let generationTask = Task.detached(priority: LlamaRuntimeScheduling.inferenceTaskPriority) { [weak self] in
                guard let self else {
                    diskWriteLease.end()
                    continuation.yield(GenerationToken.done)
                    continuation.finish()
                    return
                }

                await self.registerActiveGeneration(requestID: req.id, slot: slot, token: cancellationToken, taskBox: taskBox, diskWriteLease: diskWriteLease)
                let requestForGeneration = req.cappedForDeveloperReasoning()
                let startedAt = Date()
                var traceRequest = requestForGeneration
                var selectedRuntime: String?
                var selectedAdapter: String?
                var modelIdentifier: String?
                var modelLoaded: Bool?
                var estimatedPromptTokenCountForDiagnostics: Int?
                var promptCharsForDiagnostics: Int?
                var firstTokenMs: Int?
                var streamStarted = false
                var firstChunkReceived = false
                var textChunkCount = 0
                var finalChunkReceived = false
                var cancellationStateBeforeStream: String?
                var streamTerminationReason: String?

                do {
                    try cancellationToken.checkCancellation()
                    guard requestForGeneration.maxTokens > 0 else {
                        let emptyOutputReason = "decodeBudgetZero"
                        await self.storeCompletedTracePayloadIfNeeded(
                            request: requestForGeneration,
                            payload: CompletedGenerationTracePayload(
                                requestID: requestForGeneration.id,
                                rawModelOutput: "",
                                reasoningText: nil,
                                visibleAnswer: "",
                                parserWarnings: [],
                                tokenUsage: TraceTokenUsage(promptTokens: nil, completionTokens: 0, reasoningTokens: nil, visibleTokens: 0, totalTokens: nil),
                                finishReason: "decodeBudgetZero",
                                error: nil,
                                streamStarted: false,
                                selectedRuntime: nil,
                                selectedAdapter: nil,
                                modelIdentifier: requestForGeneration.modelName,
                                modelLoaded: await SlotModelRuntimeCoordinator.shared.hasLoadedRuntimeReadyForContinuation(slot: slot),
                                maxTokensRequested: req.maxTokens,
                                maxTokensEffective: requestForGeneration.maxTokens,
                                stopSequences: [],
                                temperature: requestForGeneration.temperature,
                                topP: requestForGeneration.topP,
                                promptCharCount: nil,
                                estimatedPromptTokenCount: nil,
                                cancellationStateBeforeStream: cancellationToken.isCancelled ? (cancellationToken.reason ?? "cancelled") : "notCancelled",
                                firstChunkReceived: false,
                                textChunkCount: 0,
                                finalChunkReceived: false,
                                streamTerminationReason: emptyOutputReason,
                                elapsedMs: Int(Date().timeIntervalSince(startedAt) * 1000),
                                outputTokenCount: 0,
                                emptyOutputReason: emptyOutputReason
                            )
                        )
                        await self.recordModelTrace(
                            slot: slot,
                            request: requestForGeneration,
                            output: "",
                            parseError: AgentTurnParseError.empty.rawValue,
                            generationElapsedMs: Int(Date().timeIntervalSince(startedAt) * 1000),
                            outputTokenCount: 0,
                            runtimePath: nil,
                            maxTokensRequested: req.maxTokens,
                            maxTokensEffective: requestForGeneration.maxTokens,
                            emptyOutputReason: emptyOutputReason,
                            streamStarted: false,
                            selectedRuntime: nil,
                            selectedAdapter: nil,
                            modelIdentifier: requestForGeneration.modelName,
                            modelLoaded: await SlotModelRuntimeCoordinator.shared.hasLoadedRuntimeReadyForContinuation(slot: slot),
                            stopSequences: [],
                            temperature: requestForGeneration.temperature,
                            topP: requestForGeneration.topP,
                            cancellationStateBeforeStream: cancellationToken.isCancelled ? (cancellationToken.reason ?? "cancelled") : "notCancelled",
                            firstChunkReceived: false,
                            textChunkCount: 0,
                            finalChunkReceived: false,
                            streamTerminationReason: emptyOutputReason
                        )
                        return
                    }

                    let backgroundTask = await MainActor.run {
                        BackgroundRuntimeContinuation.begin(name: "Lumen Chat Generation", allowsContinuedProcessing: true)
                    }
                    defer {
                        Task { @MainActor in
                            backgroundTask?.end()
                        }
                    }
                    let allowsWork = await self.allowsGenerationWork(
                        request: requestForGeneration,
                        slot: slot,
                        resourceReason: ModelLoadIntent.userChat.rawValue
                    )
                    guard allowsWork else {
                        cancellationToken.cancel(reason: "resource-budget-denied-before-prompt-eval")
                        throw CancellationError()
                    }
                    let readyMetrics = try await SlotModelRuntimeCoordinator.shared.ensureReadyWithMetrics(
                        slot: slot,
                        allowsLoadedMemoryPressureContinuation: requestForGeneration.allowsMemoryPressureContinuation
                            && requestForGeneration.preservesRawStructuredAgentOutput
                    )
                    let readyAdapterPath = await self.roleAdapterPath(for: slot)
                    let readySlotLoadedPath = await self.loadedChatPath(for: slot)
                    let readyFallbackLoadedPath = await self.loadedChatPath
                    selectedRuntime = readyMetrics.runtimePath
                    selectedAdapter = readyMetrics.activeAdapterSlot ?? readyAdapterPath
                    modelLoaded = await SlotModelRuntimeCoordinator.shared.hasLoadedRuntimeReadyForContinuation(slot: slot)
                    modelIdentifier = readySlotLoadedPath ?? readyFallbackLoadedPath ?? requestForGeneration.modelName
                    try cancellationToken.checkCancellation()
                    let contextSize = await self.contextSizeForGeneration(slot: slot)
                    let groundedRequest = requestForGeneration.groundingSystemPrompt(for: slot)
                    traceRequest = groundedRequest
                    let messageBuildStarted = Date()
                    var promptBuild = await self.buildMessages(req: groundedRequest, contextSize: contextSize, slot: slot)
                    if promptBuild.latencySelection.latencyClass == .fastInteractive, promptBuild.finalPromptChars > PromptBudgetConstants.fastInteractiveTotalChars {
                        let before = promptBuild.finalPromptChars
                        promptBuild = await self.buildMessages(req: groundedRequest, contextSize: contextSize, slot: slot, forceFastBudget: true)
                        logger.info("event=llama.chat.prompt_fast_reslim before_chars=\(before, privacy: .public) after_chars=\(promptBuild.finalPromptChars, privacy: .public)")
                    }
                    let messageBuildMs = Int(Date().timeIntervalSince(messageBuildStarted) * 1000)
                    let messages = promptBuild.messages
                    let promptChars = promptBuild.finalPromptChars
                    let estimatedPromptTokenCount = promptBuild.estimatedPromptTokens
                    promptCharsForDiagnostics = promptChars
                    estimatedPromptTokenCountForDiagnostics = estimatedPromptTokenCount
                    logger.info("event=llama.chat.prompt_budget slot=\(slot.rawValue, privacy: .public) latency_class=\(promptBuild.latencySelection.latencyClass.rawValue, privacy: .public) reason=\(promptBuild.latencySelection.reason, privacy: .public) initial_chars=\(promptBuild.initialPromptChars, privacy: .public) final_chars=\(promptBuild.finalPromptChars, privacy: .public) budget_chars=\(promptBuild.assembly.budgetChars, privacy: .public) estimated_tokens=\(estimatedPromptTokenCount, privacy: .public)")
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .llamaPromptBudget, values: [
                        "latencyClass": promptBuild.latencySelection.latencyClass.rawValue,
                        "initialChars": String(promptBuild.initialPromptChars),
                        "finalChars": String(promptBuild.finalPromptChars),
                        "estimatedTokens": String(estimatedPromptTokenCount)
                    ]))
                    try cancellationToken.checkCancellation()
                    let stillAllowsWork = await self.allowsGenerationWork(
                        request: groundedRequest,
                        slot: slot,
                        resourceReason: ModelLoadIntent.userChat.rawValue
                    )
                    guard stillAllowsWork else {
                        cancellationToken.cancel(reason: "resource-budget-denied-after-prompt-build")
                        throw CancellationError()
                    }
                    cancellationStateBeforeStream = cancellationToken.isCancelled
                        ? (cancellationToken.reason ?? "cancelled")
                        : (Task.isCancelled ? "taskCancelled" : "notCancelled")
                    let stream = try await self.streamResponse(
                        slot: slot,
                        messages: messages,
                        temperature: Float(groundedRequest.temperature),
                        topP: Float(groundedRequest.topP),
                        repetitionPenalty: Float(groundedRequest.repetitionPenalty),
                        maxTokens: groundedRequest.maxTokens,
                        seed: groundedRequest.seed,
                        cancellationToken: cancellationToken
                    )
                    streamStarted = true
                    streamTerminationReason = "streamStarted"
                    let preservesStructuredAgentOutput = groundedRequest.preservesRawStructuredAgentOutput
                    var parser = ReasoningAwareStreamParser(
                        config: ReasoningAwareStreamParserConfig(
                            captureReasoning: groundedRequest.reasoningCaptureEnabled,
                            reasoningTraceBudgetCharacters: groundedRequest.reasoningTraceBudgetCharacters
                        )
                    )
                    var streamingSanitizer = StreamingFinalOutputSanitizer()
                    var streamedSanitized = ""
                    var outputChunks = 0
                    for try await chunk in stream {
                        try Task.checkCancellation()
                        firstChunkReceived = true
                        if !chunk.isEmpty { textChunkCount += 1 }
                        if firstTokenMs == nil {
                            firstTokenMs = Int(Date().timeIntervalSince(startedAt) * 1000)
                            PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .llamaFirstToken, values: ["latencyMs": String(firstTokenMs ?? 0)]))
                        }
                        outputChunks += 1
                        if preservesStructuredAgentOutput {
                            streamedSanitized += chunk
                            continuation.yield(GenerationToken.text(chunk))
                        } else {
                            let parsedDelta = parser.ingest(chunk)
                            let safeDelta = streamingSanitizer.ingest(parsedDelta.visibleDelta)
                            if !safeDelta.isEmpty {
                                streamedSanitized += safeDelta
                                continuation.yield(GenerationToken.text(safeDelta))
                            }
                        }
                        if outputChunks.isMultiple(of: LlamaRuntimeScheduling.decodeYieldTokenInterval) {
                            await Task.yield()
                        }
                    }
                    finalChunkReceived = true
                    streamTerminationReason = textChunkCount == 0
                        ? (firstChunkReceived ? "eosBeforeText" : "stoppedBeforeFirstToken")
                        : "stop"
                    let parserResult: ReasoningAwareStreamParserResult
                    let sanitized: String
                    if preservesStructuredAgentOutput {
                        sanitized = streamedSanitized
                        parserResult = ReasoningAwareStreamParserResult(
                            rawModelOutput: streamedSanitized,
                            reasoningText: nil,
                            visibleAnswer: streamedSanitized,
                            parserWarnings: [],
                            unterminatedReasoningBlock: false,
                            reasoningWasTruncated: false
                        )
                    } else {
                        let parserFinishDelta = parser.finish()
                        let finishSafeDelta = streamingSanitizer.ingest(parserFinishDelta.visibleDelta)
                        if !finishSafeDelta.isEmpty {
                            streamedSanitized += finishSafeDelta
                            continuation.yield(GenerationToken.text(finishSafeDelta))
                        }
                        let finalization = streamingSanitizer.finish()
                        switch finalization {
                        case let .append(final, remainingDelta):
                            sanitized = final.text
                            if !remainingDelta.isEmpty {
                                streamedSanitized += remainingDelta
                                continuation.yield(GenerationToken.text(remainingDelta))
                            }
                        case let .replace(final):
                            sanitized = final.text
                            logger.error("stream_finalization_mismatch slot=\(slot.rawValue, privacy: .public)")
                        }
                        parserResult = parser.result
                    }
                    let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
                    let decodeMs = firstTokenMs.map { max(0, elapsedMs - $0) }
                    let preFirstTokenMs = firstTokenMs
                    let outputTokenEstimate = max(0, sanitized.split(whereSeparator: \.isWhitespace).count)
                    let emptyOutputReason: String?
                    if sanitized.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        emptyOutputReason = Self.emptyStreamReason(
                            streamStarted: streamStarted,
                            firstChunkReceived: firstChunkReceived,
                            textChunkCount: textChunkCount,
                            finalChunkReceived: finalChunkReceived,
                            cancellationReason: cancellationToken.reason,
                            maxTokensEffective: groundedRequest.maxTokens
                        )
                    } else {
                        emptyOutputReason = nil
                    }
                    let reasoningTokenEstimate = parserResult.reasoningText.map { max(0, $0.split(whereSeparator: \.isWhitespace).count) }
                    let tokenUsage = TraceTokenUsage(
                        promptTokens: estimatedPromptTokenCount,
                        completionTokens: nil,
                        reasoningTokens: reasoningTokenEstimate,
                        visibleTokens: outputTokenEstimate,
                        totalTokens: nil
                    )
                    let tps = decodeMs.flatMap { $0 > 0 ? Double(outputTokenEstimate) / (Double($0) / 1000.0) : nil }
                    let promptEvalMs = firstTokenMs.map { max(1, $0 - readyMetrics.ensureReadyMs - messageBuildMs) }
                    let promptEvalTps = promptEvalMs.map { Double(estimatedPromptTokenCount) / (Double($0) / 1000.0) }
                    let accelerationDiagnostics = readyMetrics.accelerationDiagnostics.withPerformance(
                        promptEvalTokensPerSecond: promptEvalTps,
                        decodeTokensPerSecond: tps
                    )
                    await self.updateLastAccelerationDiagnostics(accelerationDiagnostics)
                    logger.info(
                        "event=llama.chat.generation_perf slot=\(slot.rawValue, privacy: .public) prompt_eval_tps=\(promptEvalTps ?? -1, privacy: .public) decode_tps=\(tps ?? -1, privacy: .public) prompt_tokens_est=\(estimatedPromptTokenCount, privacy: .public) output_words=\(outputTokenEstimate, privacy: .public)"
                    )
                    if let emptyOutputReason {
                        let slotLoadedPath = await self.loadedChatPath(for: slot)
                        let fallbackLoadedPath = await self.loadedChatPath
                        let loadedPath = slotLoadedPath ?? fallbackLoadedPath ?? "none"
                        let adapterPath = await self.roleAdapterPath(for: slot) ?? "none"
                        logger.error(
                            "event=llama.chat.empty_output slot=\(slot.rawValue, privacy: .public) model_name=\(groundedRequest.modelName, privacy: .public) reason=\(emptyOutputReason, privacy: .public) requested_tokens=\(req.maxTokens, privacy: .public) effective_tokens=\(groundedRequest.maxTokens, privacy: .public) runtime_path=\(readyMetrics.runtimePath, privacy: .public) active_adapter_slot=\(readyMetrics.activeAdapterSlot ?? "none", privacy: .public) loaded_path=\(loadedPath, privacy: .public) adapter_path=\(adapterPath, privacy: .public) cancelled=\(Task.isCancelled ? "true" : "false", privacy: .public)"
                        )
                        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .llamaEmptyOutput, values: [
                            "slot": slot.rawValue,
                            "modelName": groundedRequest.modelName,
                            "reason": emptyOutputReason,
                            "maxTokensRequested": String(req.maxTokens),
                            "maxTokensEffective": String(groundedRequest.maxTokens),
                            "runtimePath": readyMetrics.runtimePath,
                            "activeAdapterSlot": readyMetrics.activeAdapterSlot ?? "none",
                            "loadedModelPath": loadedPath,
                            "adapterPath": adapterPath,
                            "cancelled": Task.isCancelled ? "true" : "false",
                            "firstChunkReceived": firstChunkReceived ? "true" : "false",
                            "textChunkCount": String(textChunkCount),
                            "finalChunkReceived": finalChunkReceived ? "true" : "false",
                            "streamTerminationReason": streamTerminationReason ?? "unknown"
                        ]))
                    }
                    await self.storeCompletedTracePayloadIfNeeded(
                        request: groundedRequest,
                        payload: CompletedGenerationTracePayload(
                            requestID: groundedRequest.id,
                            rawModelOutput: parserResult.rawModelOutput,
                            reasoningText: parserResult.reasoningText,
                            visibleAnswer: sanitized,
                            parserWarnings: parserResult.parserWarnings,
                            tokenUsage: tokenUsage,
                            finishReason: "stop",
                            error: nil,
                            streamStarted: streamStarted,
                            selectedRuntime: readyMetrics.runtimePath,
                            selectedAdapter: selectedAdapter,
                            modelIdentifier: modelIdentifier ?? groundedRequest.modelName,
                            modelLoaded: modelLoaded,
                            maxTokensRequested: req.maxTokens,
                            maxTokensEffective: groundedRequest.maxTokens,
                            stopSequences: [],
                            temperature: groundedRequest.temperature,
                            topP: groundedRequest.topP,
                            promptCharCount: promptChars,
                            estimatedPromptTokenCount: estimatedPromptTokenCount,
                            cancellationStateBeforeStream: cancellationStateBeforeStream,
                            firstChunkReceived: firstChunkReceived,
                            textChunkCount: textChunkCount,
                            finalChunkReceived: finalChunkReceived,
                            streamTerminationReason: streamTerminationReason,
                            elapsedMs: elapsedMs,
                            outputTokenCount: emptyOutputReason == nil ? nil : 0,
                            emptyOutputReason: emptyOutputReason
                        )
                    )
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .llamaComplete, values: [
                        "elapsedMs": String(elapsedMs),
                        "firstTokenLatencyMs": String(firstTokenMs ?? -1),
                        "estimatedPromptTokens": String(estimatedPromptTokenCount),
                        "finalPromptChars": String(promptChars)
                    ]))
                    let parsedOutput = AgentTurnParser.parse(sanitized)
                    await self.recordModelTrace(
                        slot: slot,
                        request: groundedRequest,
                        output: sanitized,
                        parseError: parsedOutput.parseError?.rawValue,
                        generationElapsedMs: elapsedMs,
                        outputTokenCount: nil,
                        firstTokenLatencyMs: firstTokenMs,
                        estimatedPromptTokenCount: estimatedPromptTokenCount,
                        preFirstTokenMs: preFirstTokenMs,
                        messageBuildMs: messageBuildMs,
                        decodeMs: decodeMs,
                        tokensPerSecond: tps,
                        ensureReadyMs: readyMetrics.ensureReadyMs,
                        adapterActivationMs: readyMetrics.adapterActivationMs,
                        runtimePath: readyMetrics.runtimePath,
                        activeAdapterSlot: readyMetrics.activeAdapterSlot,
                        maxTokensRequested: req.maxTokens,
                        maxTokensEffective: groundedRequest.maxTokens,
                        promptCharCount: promptChars,
                        accelerationDiagnostic: readyMetrics.accelerationDiagnostic,
                        accelerationDiagnostics: accelerationDiagnostics,
                        emptyOutputReason: emptyOutputReason,
                        streamStarted: streamStarted,
                        selectedRuntime: readyMetrics.runtimePath,
                        selectedAdapter: selectedAdapter,
                        modelIdentifier: modelIdentifier ?? groundedRequest.modelName,
                        modelLoaded: modelLoaded,
                        stopSequences: [],
                        temperature: groundedRequest.temperature,
                        topP: groundedRequest.topP,
                        cancellationStateBeforeStream: cancellationStateBeforeStream,
                        firstChunkReceived: firstChunkReceived,
                        textChunkCount: textChunkCount,
                        finalChunkReceived: finalChunkReceived,
                        streamTerminationReason: streamTerminationReason
                    )
                } catch is CancellationError {
                    let cancelReason = cancellationToken.reason ?? AppCancellationBus.shared.lastCancellationReason ?? "task-cancelled"
                    logger.info("event=llama.chat.generation_cancelled slot=\(slot.rawValue, privacy: .public) reason=\(cancelReason, privacy: .public)")
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .llamaCancel, values: ["reason": cancelReason]))
                    streamTerminationReason = cancelReason
                    let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
                    let emptyOutputReason = firstTokenMs == nil ? cancelReason : "completedWithoutText"
                    let currentModelLoaded: Bool
                    if let modelLoaded {
                        currentModelLoaded = modelLoaded
                    } else {
                        currentModelLoaded = await SlotModelRuntimeCoordinator.shared.hasLoadedRuntimeReadyForContinuation(slot: slot)
                    }
                    await self.storeCompletedTracePayloadIfNeeded(
                        request: traceRequest,
                        payload: CompletedGenerationTracePayload(
                            requestID: traceRequest.id,
                            rawModelOutput: "",
                            reasoningText: nil,
                            visibleAnswer: "",
                            parserWarnings: [],
                            tokenUsage: TraceTokenUsage(promptTokens: estimatedPromptTokenCountForDiagnostics, completionTokens: 0, reasoningTokens: nil, visibleTokens: 0, totalTokens: nil),
                            finishReason: "cancelled",
                            error: cancelReason,
                            streamStarted: streamStarted,
                            selectedRuntime: selectedRuntime,
                            selectedAdapter: selectedAdapter,
                            modelIdentifier: modelIdentifier ?? traceRequest.modelName,
                            modelLoaded: currentModelLoaded,
                            maxTokensRequested: req.maxTokens,
                            maxTokensEffective: traceRequest.maxTokens,
                            stopSequences: [],
                            temperature: traceRequest.temperature,
                            topP: traceRequest.topP,
                            promptCharCount: promptCharsForDiagnostics,
                            estimatedPromptTokenCount: estimatedPromptTokenCountForDiagnostics,
                            cancellationStateBeforeStream: cancellationStateBeforeStream ?? cancelReason,
                            firstChunkReceived: firstChunkReceived,
                            textChunkCount: textChunkCount,
                            finalChunkReceived: finalChunkReceived,
                            streamTerminationReason: cancelReason,
                            elapsedMs: elapsedMs,
                            outputTokenCount: 0,
                            emptyOutputReason: emptyOutputReason
                        )
                    )
                    await self.recordModelTrace(
                        slot: slot,
                        request: traceRequest,
                        output: "",
                        parseError: AgentTurnParseError.empty.rawValue,
                        generationElapsedMs: elapsedMs,
                        outputTokenCount: 0,
                        firstTokenLatencyMs: firstTokenMs,
                        estimatedPromptTokenCount: estimatedPromptTokenCountForDiagnostics,
                        runtimePath: selectedRuntime,
                        activeAdapterSlot: selectedAdapter,
                        maxTokensRequested: req.maxTokens,
                        maxTokensEffective: traceRequest.maxTokens,
                        promptCharCount: promptCharsForDiagnostics,
                        emptyOutputReason: emptyOutputReason,
                        streamStarted: streamStarted,
                        selectedRuntime: selectedRuntime,
                        selectedAdapter: selectedAdapter,
                        modelIdentifier: modelIdentifier ?? traceRequest.modelName,
                        modelLoaded: currentModelLoaded,
                        stopSequences: [],
                        temperature: traceRequest.temperature,
                        topP: traceRequest.topP,
                        cancellationStateBeforeStream: cancellationStateBeforeStream ?? cancelReason,
                        firstChunkReceived: firstChunkReceived,
                        textChunkCount: textChunkCount,
                        finalChunkReceived: finalChunkReceived,
                        streamTerminationReason: cancelReason
                    )
                } catch {
                    let promptOverflow = req.preservesRawStructuredAgentOutput && Self.isPromptContextWindowExceeded(error)
                    let errorText = promptOverflow
                        ? Self.promptContextWindowExceededMessage
                        : "Generation error: \(error.localizedDescription)"
                    let parseError = promptOverflow
                        ? AgentTurnParseError.contextWindowExceeded.rawValue
                        : "generation_error"
                    let terminationReason = promptOverflow
                        ? "contextWindowExceeded"
                        : Self.streamErrorTerminationReason(error)
                    let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
                    let currentModelLoaded: Bool
                    if let modelLoaded {
                        currentModelLoaded = modelLoaded
                    } else {
                        currentModelLoaded = await SlotModelRuntimeCoordinator.shared.hasLoadedRuntimeReadyForContinuation(slot: slot)
                    }
                    await self.storeCompletedTracePayloadIfNeeded(
                        request: traceRequest,
                        payload: CompletedGenerationTracePayload(
                            requestID: traceRequest.id,
                            rawModelOutput: "",
                            reasoningText: nil,
                            visibleAnswer: errorText,
                            parserWarnings: [],
                            tokenUsage: nil,
                            finishReason: "error",
                            error: error.localizedDescription,
                            streamStarted: streamStarted,
                            selectedRuntime: selectedRuntime,
                            selectedAdapter: selectedAdapter,
                            modelIdentifier: modelIdentifier ?? traceRequest.modelName,
                            modelLoaded: currentModelLoaded,
                            maxTokensRequested: req.maxTokens,
                            maxTokensEffective: traceRequest.maxTokens,
                            stopSequences: [],
                            temperature: traceRequest.temperature,
                            topP: traceRequest.topP,
                            promptCharCount: promptCharsForDiagnostics,
                            estimatedPromptTokenCount: estimatedPromptTokenCountForDiagnostics,
                            cancellationStateBeforeStream: cancellationStateBeforeStream,
                            firstChunkReceived: firstChunkReceived,
                            textChunkCount: textChunkCount,
                            finalChunkReceived: finalChunkReceived,
                            streamTerminationReason: terminationReason,
                            elapsedMs: elapsedMs,
                            outputTokenCount: 0,
                            emptyOutputReason: promptOverflow ? nil : terminationReason
                        )
                    )
                    PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .llamaFailure, values: ["errorCode": self.classifyError(error).rawValue]))
                    await self.recordModelTrace(
                        slot: slot,
                        request: traceRequest,
                        output: errorText,
                        parseError: parseError,
                        generationElapsedMs: elapsedMs,
                        outputTokenCount: 0,
                        estimatedPromptTokenCount: estimatedPromptTokenCountForDiagnostics,
                        runtimePath: promptOverflow ? "model_initialization_failed_prompt_too_large" : selectedRuntime,
                        activeAdapterSlot: selectedAdapter,
                        maxTokensRequested: req.maxTokens,
                        maxTokensEffective: traceRequest.maxTokens,
                        promptCharCount: promptCharsForDiagnostics,
                        emptyOutputReason: promptOverflow ? nil : terminationReason,
                        streamStarted: streamStarted,
                        selectedRuntime: selectedRuntime,
                        selectedAdapter: selectedAdapter,
                        modelIdentifier: modelIdentifier ?? traceRequest.modelName,
                        modelLoaded: currentModelLoaded,
                        stopSequences: [],
                        temperature: traceRequest.temperature,
                        topP: traceRequest.topP,
                        cancellationStateBeforeStream: cancellationStateBeforeStream,
                        firstChunkReceived: firstChunkReceived,
                        textChunkCount: textChunkCount,
                        finalChunkReceived: finalChunkReceived,
                        streamTerminationReason: terminationReason
                    )
                    continuation.yield(GenerationToken.text(errorText))
                }

                await self.unregisterActiveGeneration(requestID: req.id)
                continuation.yield(GenerationToken.done)
                continuation.finish()
            }

            taskBox.set(generationTask)
            continuation.onTermination = { @Sendable _ in
                diskWriteLease.end()
                cancellationToken.cancel(reason: "stream-terminated")
                generationTask.cancel()
                Task.detached(priority: .utility) {
                    await AppLlamaService.shared.unregisterActiveGeneration(requestID: req.id)
                }
            }
        }
    }

    func embed(_ text: String) async throws -> [Double] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        guard let embeddingModel else { throw LlamaError.embeddingModelNotLoaded }
        guard let embeddingContext else { throw LlamaError.embeddingModelNotLoaded }

        embeddingContext.clearKVCache()
        embeddingContext.setEmbeddingsOutput(true)
        embeddingContext.setCausalAttention(false)

        let tokens = embeddingModel.tokenize(text: trimmed, addBos: embeddingModel.shouldAddBos(), special: false)
        guard !tokens.isEmpty else { return [] }

        if tokens.count >= Int(embeddingContext.contextSize()) {
            throw LlamaError.embeddingFailed("Input exceeds embedding context window")
        }

        let batch = LlamaBatch(initialSize: 1)
        do {
            for (index, token) in tokens.enumerated() {
                batch.reset()
                batch.addToken(token, at: Int32(index), logits: index == (tokens.count - 1))
                try embeddingContext.decode(batch: batch)
            }
        } catch {
            throw LlamaError.embeddingFailed(error.localizedDescription)
        }

        let raw = embeddingContext.pooledEmbeddings(for: 0) ?? embeddingContext.embeddings(at: -1) ?? []
        guard !raw.isEmpty else {
            throw LlamaError.embeddingFailed("Model returned an empty embedding vector")
        }

        return normalize(raw.map(Double.init))
    }

    func embed(text: String, dimensions: Int = 256) async -> [Double] {
        let requestID = UUID().uuidString
        do {
            return try await embed(text)
        } catch {
            let errorCode = classifyError(error)
            Logger(subsystem: "com.lumen.runtime", category: "llama.service").error(
                "event=llama.embedding.failure severity=error error_code=\(errorCode.rawValue, privacy: .public) request_id=\(requestID, privacy: .public) dimensions=\(dimensions, privacy: .public) message=\(error.localizedDescription, privacy: .public)"
            )
            return []
        }
    }

    private func loadChatModelSync(path: String, slot: LumenModelSlot, contextSize: Int, batchSize: UInt32) throws {
        guard slot != .embedding else {
            throw LlamaError.failedToInitializeContext("Embedding slot cannot be loaded as chat")
        }
        guard FileManager.default.fileExists(atPath: path) else {
            throw LlamaError.modelFileNotFound(path)
        }
        guard contextSize > 0 else {
            throw LlamaError.failedToInitializeContext("Context size must be greater than 0")
        }

        let preferredConfig = LlamaConfig(
            batchSize: batchSize,
            maxTokenCount: UInt32(max(1, contextSize)),
            useGPU: true
        )
        LlamaRuntimeLogCapture.shared.installIfNeeded()
        LlamaRuntimeLogCapture.shared.markLoadBoundary()
        let service = SwiftLlama.LlamaService(modelUrl: URL(fileURLWithPath: path), config: preferredConfig)
        let offload = LlamaOffloadSnapshot.fromRuntimeLogs(totalModelLayers: nil, requestedKQVOffload: true)
        self.lastAccelerationDiagnostics = RuntimeAccelerationDiagnostics.forCurrentRuntime(
            requestedBackend: "metal",
            requestedGpuLayers: 999,
            requestedKQVOffload: true,
            actualBackend: offload.actualBackend,
            actualOffloadedLayers: offload.offloadedLayers,
            actualTotalLayers: offload.totalLayers,
            metalDeviceUsed: offload.actualBackend == "metal" ? MTLCreateSystemDefaultDevice()?.name : nil,
            actualKQVOffload: offload.kqvOffloaded,
            notes: offload.notes
        )
        let diagnostics = self.lastAccelerationDiagnostics
        logger.info(
            "event=llama.chat.acceleration_verified backend=\(diagnostics.actualBackend ?? "unknown", privacy: .public) metal_device=\(diagnostics.metalDeviceUsed ?? diagnostics.metalDeviceName ?? "unknown", privacy: .public) offloaded_layers=\(diagnostics.actualOffloadedLayers.map(String.init) ?? "unknown", privacy: .public) total_layers=\(diagnostics.actualTotalLayers.map(String.init) ?? "unknown", privacy: .public) kqv_offload=\(diagnostics.actualKQVOffload.map { String($0) } ?? "unknown", privacy: .public) verification=\(diagnostics.verificationLevel, privacy: .public)"
        )
        chatRuntimes[slot] = ChatRuntime(
            service: service,
            modelPath: path,
            contextSize: contextSize,
            batchSize: batchSize
        )
        primaryChatSlot = slot
    }

    func currentAccelerationDiagnostics() -> RuntimeAccelerationDiagnostics {
        lastAccelerationDiagnostics
    }

    private func updateLastAccelerationDiagnostics(_ diagnostics: RuntimeAccelerationDiagnostics) {
        lastAccelerationDiagnostics = diagnostics
    }

    private func makeEmbeddingContext(for model: LlamaModel) -> LlamaContext? {
        var contextParams = llama_context_default_params()
        contextParams.n_ctx = embeddingContextSize
        contextParams.n_batch = embeddingBatchSize
        contextParams.n_ubatch = embeddingBatchSize
        contextParams.n_threads = embeddingThreads
        contextParams.n_threads_batch = embeddingThreads
        contextParams.offload_kqv = false
        return LlamaContext(model: model, parameters: contextParams)
    }


    private func contextSizeForGeneration(slot: LumenModelSlot) async -> Int {
        if let sharedChatRuntime {
            return await sharedChatRuntime.configuredContextSize()
        }
        return chatRuntimes[slot]?.contextSize ?? 2048
    }

    private func stopCompletion(for slot: LumenModelSlot) async {
        sharedChatRuntime?.stopCompletion()
        await chatRuntimes[slot]?.service.stopCompletion()
    }

    private func recordModelTrace(
        slot: LumenModelSlot,
        request: GenerateRequest,
        output: String,
        parseError: String?,
        generationElapsedMs: Int? = nil,
        outputTokenCount: Int? = nil,
        firstTokenLatencyMs: Int? = nil,
        estimatedPromptTokenCount: Int? = nil,
        preFirstTokenMs: Int? = nil,
        messageBuildMs: Int? = nil,
        decodeMs: Int? = nil,
        tokensPerSecond: Double? = nil,
        ensureReadyMs: Int? = nil,
        adapterActivationMs: Int? = nil,
        runtimePath: String? = nil,
        activeAdapterSlot: String? = nil,
        maxTokensRequested: Int? = nil,
        maxTokensEffective: Int? = nil,
        promptCharCount: Int? = nil,
        accelerationDiagnostic: String? = nil,
        accelerationDiagnostics: RuntimeAccelerationDiagnostics? = nil,
        emptyOutputReason: String? = nil,
        streamStarted: Bool? = nil,
        selectedRuntime: String? = nil,
        selectedAdapter: String? = nil,
        modelIdentifier: String? = nil,
        modelLoaded: Bool? = nil,
        stopSequences: [String] = [],
        temperature: Double? = nil,
        topP: Double? = nil,
        cancellationStateBeforeStream: String? = nil,
        firstChunkReceived: Bool? = nil,
        textChunkCount: Int? = nil,
        finalChunkReceived: Bool? = nil,
        streamTerminationReason: String? = nil
    ) async {
        let adapterMetadata = currentAdapterTraceMetadata(slot: slot)
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: .modelTurn,
                slot: slot.rawValue,
                stage: request.modelName,
                intent: nil,
                promptPrefix: ModelOutputSanitizer.boundedPrefix(request.userMessage, limit: 1200),
                rawOutputPrefix: ModelOutputSanitizer.boundedPrefix(output, limit: 1600),
                selectedToolID: AgentTurnParser.parse(output).action.map { ToolRouteGuard.canonicalToolID($0.tool) },
                toolArguments: AgentTurnParser.parse(output).action?.args.stringCoerced ?? [:],
                allowedToolIDs: allowedToolIDs(for: request.userMessage, slot: slot),
                requiresApproval: nil,
                approvalMode: nil,
                parseError: parseError,
                emittedFinalInActionTurn: output.lowercased().contains("\"final\""),
                modelFamily: adapterMetadata.modelFamily,
                baseModelPath: adapterMetadata.baseModelPath,
                adapterID: adapterMetadata.adapterID,
                adapterSlot: adapterMetadata.adapterSlot,
                adapterPath: adapterMetadata.adapterPath,
                adapterApplied: adapterMetadata.adapterApplied,
                adapterScale: adapterMetadata.adapterScale,
                adapterFailureReason: adapterMetadata.adapterFailureReason,
                generationElapsedMs: generationElapsedMs,
                firstTokenLatencyMs: firstTokenLatencyMs,
                outputTokenCount: outputTokenCount,
                estimatedPromptTokenCount: estimatedPromptTokenCount,
                preFirstTokenMs: preFirstTokenMs,
                messageBuildMs: messageBuildMs,
                decodeMs: decodeMs,
                tokensPerSecond: tokensPerSecond,
                ensureReadyMs: ensureReadyMs,
                adapterActivationMs: adapterActivationMs,
                runtimePath: runtimePath,
                activeAdapterSlot: activeAdapterSlot,
                maxTokensRequested: maxTokensRequested,
                maxTokensEffective: maxTokensEffective,
                promptCharCount: promptCharCount,
                accelerationDiagnostic: accelerationDiagnostic,
                accelerationDiagnostics: accelerationDiagnostics,
                emptyOutputReason: emptyOutputReason,
                streamStarted: streamStarted,
                selectedRuntime: selectedRuntime,
                selectedAdapter: selectedAdapter,
                modelIdentifier: modelIdentifier,
                modelLoaded: modelLoaded,
                stopSequences: stopSequences,
                temperature: temperature,
                topP: topP,
                cancellationStateBeforeStream: cancellationStateBeforeStream,
                firstChunkReceived: firstChunkReceived,
                textChunkCount: textChunkCount,
                finalChunkReceived: finalChunkReceived,
                streamTerminationReason: streamTerminationReason,
                selfModel: AgentBehaviorTrace.SelfModelDecisionSummary.fromPrompt(
                    request.userMessage,
                    selectedToolID: AgentTurnParser.parse(output).action.map { ToolRouteGuard.canonicalToolID($0.tool) },
                    requiresApproval: nil,
                    approvalMode: nil
                )
            )
        )
    }

    private func storeCompletedTracePayloadIfNeeded(request: GenerateRequest, payload: CompletedGenerationTracePayload) {
        guard request.sessionID != nil || request.developerTraceModeEnabled || request.preservesRawStructuredAgentOutput else { return }
        completedTracePayloads[payload.requestID] = payload
        if completedTracePayloads.count > 32 {
            let overflow = completedTracePayloads.count - 32
            for key in completedTracePayloads.keys.prefix(overflow) {
                completedTracePayloads.removeValue(forKey: key)
            }
        }
    }



    private func allowedToolIDs(for prompt: String, slot: LumenModelSlot) -> [String] {
        var ids: Set<String> = []
        let lines = prompt.split(whereSeparator: \.isNewline).map(String.init)
        var insideAvailableTools = false
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed == "Available tools:" || trimmed == "[AVAILABLE LOCAL TOOLS]" {
                insideAvailableTools = true
                continue
            }
            if insideAvailableTools, trimmed.hasPrefix("["), trimmed.hasSuffix("]") {
                insideAvailableTools = false
                continue
            }
            if insideAvailableTools, trimmed.hasSuffix(":") && !trimmed.hasPrefix("-") {
                insideAvailableTools = false
            }
            guard insideAvailableTools, trimmed.hasPrefix("- ") else { continue }
            let candidate = String(trimmed.dropFirst(2)).split(separator: ":", maxSplits: 1).first.map(String.init) ?? ""
            if !candidate.isEmpty { ids.insert(ToolRouteGuard.canonicalToolID(candidate)) }
        }
        if ids.isEmpty, slot == .cortex || slot == .executor {
            ids = IntentRouter.classify(traceUserRequest(from: prompt)).allowedToolIDs
        }
        return Array(ids).sorted()
    }

    private func traceUserRequest(from prompt: String) -> String {
        guard let marker = prompt.range(of: "User request:") else {
            return PromptGroundingIdempotencyGuard.stripExistingGrounding(from: prompt).text
        }
        var tail = String(prompt[marker.upperBound...])
        if tail.hasPrefix("\n") {
            tail.removeFirst()
        }
        if let grounding = tail.range(of: PromptGroundingIdempotencyGuard.marker) {
            tail = String(tail[..<grounding.lowerBound])
        }
        if let nextInstruction = tail.range(of: "\n\nEmit ") {
            tail = String(tail[..<nextInstruction.lowerBound])
        }
        return tail.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func currentAdapterTraceMetadata(slot: LumenModelSlot) -> LlamaAdapterTraceMetadata {
        let loaded = roleAdapters[slot]
        let roleContract = LumenTrainedModelRuntimeRegistry.contract(for: .qwen3).adapterRole(for: slot)
        return LlamaAdapterTraceMetadata(
            modelFamily: sharedChatRuntime == nil ? nil : LumenModelFamily.qwen3.rawValue,
            baseModelPath: sharedChatBasePath,
            adapterID: loaded.map { _ in roleContract?.adapterID ?? "\(slot.rawValue):adapter" },
            adapterSlot: loaded?.slot.rawValue,
            adapterPath: loaded?.path,
            adapterApplied: activeAdapterSlot == slot && loaded != nil,
            adapterScale: loaded?.scale,
            adapterFailureReason: lastAdapterFailureReason
        )
    }

    private func roleAdapterPath(for slot: LumenModelSlot) -> String? {
        roleAdapters[slot]?.path
    }

    private func normalize(_ vector: [Double]) -> [Double] {
        let norm = sqrt(vector.reduce(0.0) { $0 + ($1 * $1) })
        guard norm > 0 else { return vector }
        return vector.map { $0 / norm }
    }

    private func makeRandomSeed() -> UInt32 {
        UInt32.random(in: UInt32.min...UInt32.max)
    }

    private struct HiddenBlockStreamSanitizer {
        private var carry = ""
        private var insideHidden = false

        mutating func sanitize(_ chunk: String) -> String {
            guard !chunk.isEmpty else { return "" }
            carry += chunk
            let lower = carry.lowercased()
            var out = ""
            var cursor = lower.startIndex
            var flushed = lower.startIndex

            while cursor < lower.endIndex {
                if !insideHidden,
                   let open = lower[cursor...].range(of: "<think>") ?? lower[cursor...].range(of: "<thinking>") {
                    if flushed < open.lowerBound {
                        out += String(carry[flushed..<open.lowerBound])
                    }
                    insideHidden = true
                    cursor = open.upperBound
                    flushed = cursor
                    continue
                }
                if insideHidden,
                   let close = lower[cursor...].range(of: "</think>") ?? lower[cursor...].range(of: "</thinking>") {
                    insideHidden = false
                    cursor = close.upperBound
                    flushed = cursor
                    continue
                }
                break
            }

            if !insideHidden {
                if flushed < lower.endIndex {
                    out += String(carry[flushed..<lower.endIndex])
                }
                carry = ""
            } else {
                carry = String(carry[flushed...])
            }
            return out
        }
    }

    private func buildMessages(req: GenerateRequest, contextSize: Int? = nil, slot: LumenModelSlot? = nil, forceFastBudget: Bool = false) -> PromptBuildResult {
        let latencySelection = forceFastBudget
            ? PromptLatencySelection(latencyClass: .fastInteractive, reason: "forced-fast-slimming")
            : PromptLatencyClassifier.classify(
                userMessage: req.userMessage,
                attachments: req.attachments,
                developerTraceModeEnabled: req.developerTraceModeEnabled,
                reasoningCaptureEnabled: req.reasoningCaptureEnabled,
                modelName: req.modelName
            )
        let formattedSystemPrompt = Self.systemPrompt(
            req.systemPrompt,
            responseFormat: req.responseFormat
        )
        let budget: PromptBudget
        if req.preservesRawStructuredAgentOutput {
            budget = PromptBudget.agentJSON(
                contextSize: contextSize ?? 2048,
                maxTokens: req.maxTokens
            )
        } else {
            switch latencySelection.latencyClass {
            case .fastInteractive:
                budget = PromptBudget.fastInteractive()
            case .normalInteractive, .documentGrounded, .developerTrace:
                budget = PromptBudget.make(
                    contextSize: contextSize ?? 2048,
                    maxTokens: req.maxTokens,
                    systemPromptChars: formattedSystemPrompt.count,
                    userMessageChars: req.userMessage.count,
                    hasAttachments: !req.attachments.isEmpty,
                    hasMemories: !req.relevantMemories.isEmpty
                )
            }
        }

        let assembly = PromptAssembler.assemble(
            systemPrompt: formattedSystemPrompt,
            history: req.history,
            userMessage: req.userMessage,
            memories: req.relevantMemories,
            attachments: req.attachments,
            budget: budget,
            attachmentNormalization: req.modelName == "agent-json" ? .agentRouting : .preserveRaw,
            latencyClass: latencySelection.latencyClass
        )
        let useQwenDirective = currentChatModelLooksLikeQwen3(slot: slot)
        let requireFinalAnswerOnly = !req.responseFormat.requiresRawStructuredOutput
            && !req.modelName.lowercased().contains("json")
        let allowReasoningCapture = req.reasoningCaptureEnabled && requireFinalAnswerOnly
        let systemPrompt = ModelThinkingControl.systemPrompt(
            assembly.systemPrompt,
            reasoningCaptureEnabled: allowReasoningCapture,
            requireFinalAnswerOnly: requireFinalAnswerOnly
        )
        let userMessage = ModelThinkingControl.userMessage(
            assembly.userMessage,
            reasoningCaptureEnabled: allowReasoningCapture,
            useQwenThinkingDirective: useQwenDirective
        )

        var messages: [LlamaChatMessage] = [
            LlamaChatMessage(role: .system, content: systemPrompt)
        ]

        for h in assembly.history {
            switch h.role {
            case .system:
                continue
            case .user:
                messages.append(LlamaChatMessage(role: .user, content: h.content))
            case .assistant:
                messages.append(LlamaChatMessage(role: .assistant, content: h.content))
            case .tool:
                messages.append(LlamaChatMessage(role: .user, content: h.content))
            }
        }

        messages.append(LlamaChatMessage(role: .user, content: userMessage))
        let finalPromptChars = messages.reduce(0) { $0 + $1.content.count }
        return PromptBuildResult(
            messages: messages,
            assembly: assembly,
            initialPromptChars: req.systemPrompt.count + req.userMessage.count + req.history.reduce(0) { $0 + $1.content.count } + req.relevantMemories.reduce(0) { $0 + $1.content.count },
            finalPromptChars: finalPromptChars,
            estimatedPromptTokens: max(1, finalPromptChars / 4),
            latencySelection: latencySelection
        )
    }

    private nonisolated static func systemPrompt(_ base: String, responseFormat: LLMResponseFormat) -> String {
        let instruction: String?
        switch responseFormat {
        case .plainText:
            instruction = nil
        case .json:
            instruction = "Response format contract: output exactly one valid JSON object. Do not include prose, markdown, code fences, or hidden reasoning."
        case .toolCallJSON:
            instruction = #"Response format contract: output exactly one valid JSON object shaped as {"action":{"tool":"<tool id>","args":{...}}}. Do not include prose, markdown, code fences, or hidden reasoning."#
        case .constrainedJSON(let schema):
            instruction = """
            Response format contract: output exactly one valid JSON object matching this schema. Do not include prose, markdown, code fences, or hidden reasoning.
            Enforcement diagnostic: \(responseFormat.enforcementDiagnostic ?? "none"). TODO: wire llama.cpp grammar enforcement when the native bridge exposes grammar.
            JSON schema:
            \(schema)
            """
        }
        guard let instruction else { return base }
        let trimmed = base.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.localizedCaseInsensitiveContains("Response format contract:") else { return base }
        guard !trimmed.isEmpty else { return instruction }
        return "\(trimmed)\n\n\(instruction)"
    }

    func buildMessagesForDiagnostics(req: GenerateRequest, contextSize: Int? = nil, slot: LumenModelSlot? = nil, forceFastBudget: Bool = false) -> PromptBuildResult {
        buildMessages(req: req, contextSize: contextSize, slot: slot, forceFastBudget: forceFastBudget)
    }

    #if DEBUG
    func buildMessagesForTesting(req: GenerateRequest, contextSize: Int? = nil, slot: LumenModelSlot? = nil, forceFastBudget: Bool = false) -> PromptBuildResult {
        buildMessages(req: req, contextSize: contextSize, slot: slot, forceFastBudget: forceFastBudget)
    }
    #endif

    private func currentChatModelLooksLikeQwen3(slot: LumenModelSlot?) -> Bool {
        if sharedChatRuntime != nil { return true }
        let path = slot.flatMap { chatRuntimes[$0]?.modelPath } ?? chatRuntimes[primaryChatSlot]?.modelPath ?? chatRuntimes.values.first?.modelPath
        let lower = (path ?? "").lowercased()
        return lower.contains("qwen3") || lower.contains("qwen-3")
    }

    private nonisolated func classifyError(_ error: Error) -> LlamaErrorCode {
        if let llamaError = error as? LlamaError {
            switch llamaError {
            case .modelFileNotFound, .failedToInitializeContext, .noModelLoaded, .slotModelNotLoaded, .embeddingModelNotLoaded:
                return .modelLoad
            case .embeddingFailed:
                return .decode
            }
        }

        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain {
            switch nsError.code {
            case NSURLErrorTimedOut:
                return .timeout
            case NSURLErrorCannotFindHost, NSURLErrorCannotConnectToHost, NSURLErrorNetworkConnectionLost, NSURLErrorNotConnectedToInternet:
                return .network
            default:
                return .runtime
            }
        }
        return .runtime
    }

    private nonisolated static func emptyStreamReason(
        streamStarted: Bool,
        firstChunkReceived: Bool,
        textChunkCount: Int,
        finalChunkReceived: Bool,
        cancellationReason: String?,
        maxTokensEffective: Int
    ) -> String {
        if maxTokensEffective <= 0 {
            return "decodeBudgetZero"
        }
        if let cancellationReason, !cancellationReason.isEmpty {
            return firstChunkReceived ? "completedWithoutText" : cancellationReason
        }
        if !streamStarted {
            return "runtimeUnavailable"
        }
        if finalChunkReceived, textChunkCount == 0 {
            return firstChunkReceived ? "eosBeforeText" : "stoppedBeforeFirstToken"
        }
        if !firstChunkReceived {
            return "stoppedBeforeFirstToken"
        }
        return "unknownEmptyStream"
    }

    private nonisolated static func streamErrorTerminationReason(_ error: Error) -> String {
        if case LocalRuntimeError.unavailable(let message) = error {
            if message.localizedCaseInsensitiveContains("resource budget") {
                return "resource-budget-denied-ensure-ready"
            }
            if message.localizedCaseInsensitiveContains("adapter") {
                return "adapterUnavailable"
            }
        }
        if let llamaError = error as? LlamaError {
            switch llamaError {
            case .noModelLoaded, .modelFileNotFound, .embeddingModelNotLoaded:
                return "modelNotLoaded"
            case .slotModelNotLoaded:
                return "slotUnavailable"
            case .failedToInitializeContext:
                return "runtimeUnavailable"
            case .embeddingFailed:
                return "runtimeUnavailable"
            }
        }
        return "runtimeUnavailable"
    }

    nonisolated static let promptContextWindowExceededMessage = "Prompt exceeded context window before generation"

    nonisolated static func isPromptContextWindowExceeded(_ error: Error) -> Bool {
        if case LlamaError.failedToInitializeContext(let details) = error {
            return details.localizedCaseInsensitiveContains("prompt exceeds shared chat context window")
                || details.localizedCaseInsensitiveContains("prompt exceeded context window")
        }
        let text = error.localizedDescription
        return text.localizedCaseInsensitiveContains("prompt exceeds shared chat context window")
            || text.localizedCaseInsensitiveContains("prompt exceeded context window")
    }
}
