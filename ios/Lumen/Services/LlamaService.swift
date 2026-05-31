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
    let seed: UInt32?
    let developerTraceModeEnabled: Bool
    let reasoningCaptureEnabled: Bool
    let reasoningTraceBudgetCharacters: Int

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
        seed: UInt32? = nil,
        developerTraceModeEnabled: Bool = false,
        reasoningCaptureEnabled: Bool = false,
        reasoningTraceBudgetCharacters: Int = 16_384
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
        self.seed = seed
        self.developerTraceModeEnabled = developerTraceModeEnabled
        self.reasoningCaptureEnabled = developerTraceModeEnabled && reasoningCaptureEnabled
        self.reasoningTraceBudgetCharacters = max(0, reasoningTraceBudgetCharacters)
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
        seed: UInt32? = nil,
        developerTraceModeEnabled: Bool = false,
        reasoningCaptureEnabled: Bool = false,
        reasoningTraceBudgetCharacters: Int = 16_384
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
            seed: seed,
            developerTraceModeEnabled: developerTraceModeEnabled,
            reasoningCaptureEnabled: reasoningCaptureEnabled,
            reasoningTraceBudgetCharacters: reasoningTraceBudgetCharacters
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
            seed: seed,
            developerTraceModeEnabled: developerTraceModeEnabled,
            reasoningCaptureEnabled: reasoningCaptureEnabled,
            reasoningTraceBudgetCharacters: reasoningTraceBudgetCharacters
        )
    }
}

nonisolated enum GenerationToken: Sendable {
    case text(String)
    case done
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

private final class LlamaRuntimeLogCapture: @unchecked Sendable {
    nonisolated(unsafe) static let shared = LlamaRuntimeLogCapture()
    nonisolated(unsafe) private static let callback: ggml_log_callback = { _, text, _ in
        guard let text else { return }
        let line = String(cString: text).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty else { return }
        LlamaRuntimeLogCapture.shared.record(line)
    }

    nonisolated(unsafe) private let lock = NSLock()
    nonisolated(unsafe) private let logger = Logger(subsystem: "com.lumen.runtime", category: "llama.cpp")
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
        let runtimeThreadCount = Int32(max(2, detectedCores - 2))
        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = 999
        guard let model = LlamaModel(path: path, parameters: modelParams) else {
            throw LlamaError.failedToInitializeContext("Unable to load shared chat base GGUF")
        }
        var contextParams = llama_context_default_params()
        contextParams.n_ctx = UInt32(max(1, contextSize))
        contextParams.n_batch = batchSize
        contextParams.n_ubatch = batchSize
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
        self.batchSize = batchSize
        self.batch = LlamaBatch(initialSize: Int32(batchSize))
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
        context.clearKVCache()
        processedTokens.removeAll()
        currentTokenPosition = 0
        batch = LlamaBatch(initialSize: Int32(batchSize))
    }

    func streamCompletion(
        of messages: [LlamaChatMessage],
        samplingConfig: LlamaSamplingConfig,
        maxTokens: Int?
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                await self.generateCompletion(
                    messages: messages,
                    samplingConfig: samplingConfig,
                    maxTokens: maxTokens,
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
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) {
        do {
            try initializeCompletion(messages: messages)
            let sampler = LlamaSampler(config: samplingConfig, model: model)
            let limit = min(maxTokens ?? Int.max, max(0, contextSize - Int(currentTokenPosition) - 1))
            var emitted = 0
            while emitted < limit, !Task.isCancelled {
                let token = sampler.sample(context: context)
                if model.isEogToken(token) { break }
                batch.reset()
                batch.addToken(token, at: currentTokenPosition, logits: true)
                processedTokens.append(token)
                currentTokenPosition += 1
                try context.decode(batch: batch)
                continuation.yield(model.piece(from: token))
                emitted += 1
            }
            continuation.finish()
        } catch {
            continuation.finish(throwing: error)
        }
    }

    private func initializeCompletion(messages: [LlamaChatMessage]) throws {
        let prompt = model.applyChatTemplate(to: messages, addAssistant: nil)
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
        for (index, token) in tokens.enumerated() {
            let isLast = index == lastIndex
            batch.addToken(token, at: Int32(index), logits: isLast)
            processedTokens.append(token)
            if batch.size == Int32(batchSize) || isLast {
                try context.decode(batch: batch)
                batch.reset()
            }
        }
        currentTokenPosition = Int32(processedTokens.count)
    }
}

private struct LoadedRoleAdapter {
    let slot: LumenModelSlot
    let path: String
    let scale: Float
    let loadedAt: Date
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

private enum LlamaErrorCode: String {
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
        seed: UInt32? = nil
    ) async throws -> AsyncThrowingStream<String, Error> {
        if let runtime = sharedChatRuntime {
            return try await streamResponse(
                adapterRuntime: runtime,
                messages: messages,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                maxTokens: maxTokens,
                seed: seed
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
            seed: seed
        )
    }

    func streamResponse(
        slot: LumenModelSlot,
        messages: [LlamaChatMessage],
        temperature: Float = 0.8,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.1,
        maxTokens: Int? = nil,
        seed: UInt32? = nil
    ) async throws -> AsyncThrowingStream<String, Error> {
        if let runtime = sharedChatRuntime {
            return try await streamResponse(
                adapterRuntime: runtime,
                messages: messages,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                maxTokens: maxTokens,
                seed: seed
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
            seed: seed
        )
    }


    private func streamResponse(
        adapterRuntime runtime: AdapterChatRuntime,
        messages: [LlamaChatMessage],
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int?,
        seed: UInt32?
    ) async throws -> AsyncThrowingStream<String, Error> {
        let resolvedSeed = seed ?? makeRandomSeed()
        let sampling = LlamaSamplingConfig(
            temperature: temperature,
            seed: resolvedSeed,
            topP: topP,
            repetitionPenaltyConfig: LlamaRepetitionPenaltyConfig(repeatPenalty: repetitionPenalty)
        )
        return await runtime.streamCompletion(of: messages, samplingConfig: sampling, maxTokens: maxTokens)
    }

    private func streamResponse(
        runtime: ChatRuntime,
        stopSlot: LumenModelSlot,
        messages: [LlamaChatMessage],
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float,
        maxTokens: Int?,
        seed: UInt32?
    ) async throws -> AsyncThrowingStream<String, Error> {
        let resolvedSeed = seed ?? makeRandomSeed()
        let sampling = LlamaSamplingConfig(
            temperature: temperature,
            seed: resolvedSeed,
            topP: topP,
            repetitionPenaltyConfig: LlamaRepetitionPenaltyConfig(repeatPenalty: repetitionPenalty)
        )
        let rawStream = try await runtime.service.streamCompletion(of: messages, samplingConfig: sampling)
        guard let maxTokens else { return rawStream }

        return AsyncThrowingStream { continuation in
            let cap = max(0, maxTokens)
            let task = Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                if cap == 0 {
                    await self.stopCompletion(for: stopSlot)
                    continuation.finish()
                    return
                }

                var emitted = 0
                do {
                    for try await chunk in rawStream {
                        continuation.yield(chunk)
                        emitted += 1
                        if emitted >= cap {
                            await self.stopCompletion(for: stopSlot)
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
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

    func stream(_ req: GenerateRequest, slot: LumenModelSlot) -> AsyncStream<GenerationToken> {
        return AsyncStream<GenerationToken>(bufferingPolicy: .unbounded) { (continuation: AsyncStream<GenerationToken>.Continuation) in
            let generationTask = Task { [weak self] in
                guard let self else {
                    continuation.yield(GenerationToken.done)
                    continuation.finish()
                    return
                }

                do {
                    let requestForGeneration = req.cappedForDeveloperReasoning()
                    guard requestForGeneration.maxTokens > 0 else {
                        continuation.yield(GenerationToken.done)
                        continuation.finish()
                        return
                    }

                    let startedAt = Date()
                    let readyMetrics = try await SlotModelRuntimeCoordinator.shared.ensureReadyWithMetrics(slot: slot)
                    let contextSize = await self.contextSizeForGeneration(slot: slot)
                    let groundedRequest = requestForGeneration.groundingSystemPrompt(for: slot)
                    let messageBuildStarted = Date()
                    let messages = await self.buildMessages(req: groundedRequest, contextSize: contextSize, slot: slot)
                    let messageBuildMs = Int(Date().timeIntervalSince(messageBuildStarted) * 1000)
                    let promptChars = messages.reduce(0) { $0 + $1.content.count }
                    let estimatedPromptTokenCount = max(1, promptChars / 4)
                    let stream = try await self.streamResponse(
                        slot: slot,
                        messages: messages,
                        temperature: Float(groundedRequest.temperature),
                        topP: Float(groundedRequest.topP),
                        repetitionPenalty: Float(groundedRequest.repetitionPenalty),
                        maxTokens: groundedRequest.maxTokens,
                        seed: groundedRequest.seed
                    )
                    var parser = ReasoningAwareStreamParser(
                        config: ReasoningAwareStreamParserConfig(
                            captureReasoning: groundedRequest.reasoningCaptureEnabled,
                            reasoningTraceBudgetCharacters: groundedRequest.reasoningTraceBudgetCharacters
                        )
                    )
                    var streamingSanitizer = StreamingFinalOutputSanitizer()
                    var streamedSanitized = ""
                    var firstTokenMs: Int?
                    var outputChunks = 0
                    for try await chunk in stream {
                        if firstTokenMs == nil {
                            firstTokenMs = Int(Date().timeIntervalSince(startedAt) * 1000)
                        }
                        outputChunks += 1
                        let parsedDelta = parser.ingest(chunk)
                        let safeDelta = streamingSanitizer.ingest(parsedDelta.visibleDelta)
                        if !safeDelta.isEmpty {
                            streamedSanitized += safeDelta
                            continuation.yield(GenerationToken.text(safeDelta))
                        }
                    }
                    let parserFinishDelta = parser.finish()
                    let finishSafeDelta = streamingSanitizer.ingest(parserFinishDelta.visibleDelta)
                    if !finishSafeDelta.isEmpty {
                        streamedSanitized += finishSafeDelta
                        continuation.yield(GenerationToken.text(finishSafeDelta))
                    }
                    let finalization = streamingSanitizer.finish()
                    let sanitized: String
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
                    let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
                    let decodeMs = firstTokenMs.map { max(0, elapsedMs - $0) }
                    let preFirstTokenMs = firstTokenMs
                    let outputTokenEstimate = max(0, streamedSanitized.split(whereSeparator: \.isWhitespace).count)
                    let parserResult = parser.result
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
                            error: nil
                        )
                    )
                    await self.recordModelTrace(
                        slot: slot,
                        request: groundedRequest,
                        output: sanitized,
                        parseError: AgentTurnParser.parse(sanitized).parseError?.rawValue,
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
                        accelerationDiagnostics: accelerationDiagnostics
                    )
                } catch {
                    let errorText = "Generation error: \(error.localizedDescription)"
                    await self.storeCompletedTracePayloadIfNeeded(
                        request: req,
                        payload: CompletedGenerationTracePayload(
                            requestID: req.id,
                            rawModelOutput: "",
                            reasoningText: nil,
                            visibleAnswer: errorText,
                            parserWarnings: [],
                            tokenUsage: nil,
                            finishReason: "error",
                            error: error.localizedDescription
                        )
                    )
                    await self.recordModelTrace(slot: slot, request: req, output: errorText, parseError: "generation_error")
                    continuation.yield(GenerationToken.text(errorText))
                }

                continuation.yield(GenerationToken.done)
                continuation.finish()
            }

            continuation.onTermination = { @Sendable _ in
                generationTask.cancel()
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
        let service: SwiftLlama.LlamaService
        do {
            LlamaRuntimeLogCapture.shared.installIfNeeded()
            LlamaRuntimeLogCapture.shared.markLoadBoundary()
            service = SwiftLlama.LlamaService(modelUrl: URL(fileURLWithPath: path), config: preferredConfig)
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
        } catch {
            logger.error(
                "event=llama.chat.runtime_init_failure path=\(path, privacy: .public) context_size=\(contextSize, privacy: .public) batch_size=\(batchSize, privacy: .public) message=\(error.localizedDescription, privacy: .public) fallback=cpu_or_nonoffload"
            )
            let fallbackConfig = LlamaConfig(
                batchSize: batchSize,
                maxTokenCount: UInt32(max(1, contextSize)),
                useGPU: false
            )
            service = SwiftLlama.LlamaService(modelUrl: URL(fileURLWithPath: path), config: fallbackConfig)
            lastAccelerationDiagnostics = RuntimeAccelerationDiagnostics.forCurrentRuntime(requestedBackend: "cpu", requestedGpuLayers: 0, requestedKQVOffload: false, actualBackend: "cpu")
            logger.info(
                "event=llama.chat.runtime_init_cpu_fallback_success path=\(path, privacy: .public) context_size=\(contextSize, privacy: .public) batch_size=\(batchSize, privacy: .public)"
            )
        }
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
        accelerationDiagnostics: RuntimeAccelerationDiagnostics? = nil
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
                accelerationDiagnostics: accelerationDiagnostics
            )
        )
    }

    private func storeCompletedTracePayloadIfNeeded(request: GenerateRequest, payload: CompletedGenerationTracePayload) {
        guard request.sessionID != nil || request.developerTraceModeEnabled else { return }
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
            if trimmed == "Available tools:" {
                insideAvailableTools = true
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
            ids = IntentRouter.classify(prompt).allowedToolIDs
        }
        return Array(ids).sorted()
    }

    private func currentAdapterTraceMetadata(slot: LumenModelSlot) -> LlamaAdapterTraceMetadata {
        let loaded = roleAdapters[slot]
        return LlamaAdapterTraceMetadata(
            modelFamily: sharedChatRuntime == nil ? nil : LumenModelFamily.qwen3.rawValue,
            baseModelPath: sharedChatBasePath,
            adapterID: loaded.map { "\($0.slot.rawValue):\($0.path)" },
            adapterSlot: loaded?.slot.rawValue,
            adapterPath: loaded?.path,
            adapterApplied: activeAdapterSlot == slot && loaded != nil,
            adapterScale: loaded?.scale,
            adapterFailureReason: lastAdapterFailureReason
        )
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

    private func buildMessages(req: GenerateRequest, contextSize: Int? = nil, slot: LumenModelSlot? = nil) -> [LlamaChatMessage] {
        let budget = PromptBudget.make(
            contextSize: contextSize ?? 2048,
            maxTokens: req.maxTokens,
            systemPromptChars: req.systemPrompt.count,
            userMessageChars: req.userMessage.count,
            hasAttachments: !req.attachments.isEmpty,
            hasMemories: !req.relevantMemories.isEmpty
        )

        let assembly = PromptAssembler.assemble(
            systemPrompt: req.systemPrompt,
            history: req.history,
            userMessage: req.userMessage,
            memories: req.relevantMemories,
            attachments: req.attachments,
            budget: budget,
            attachmentNormalization: req.modelName == "agent-json" ? .agentRouting : .preserveRaw
        )
        let useQwenDirective = currentChatModelLooksLikeQwen3(slot: slot)
        let systemPrompt = ModelThinkingControl.systemPrompt(
            assembly.systemPrompt,
            reasoningCaptureEnabled: req.reasoningCaptureEnabled,
            requireFinalAnswerOnly: !req.modelName.lowercased().contains("json")
        )
        let userMessage = ModelThinkingControl.userMessage(
            assembly.userMessage,
            reasoningCaptureEnabled: req.reasoningCaptureEnabled,
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
        return messages
    }

    private func currentChatModelLooksLikeQwen3(slot: LumenModelSlot?) -> Bool {
        if sharedChatRuntime != nil { return true }
        let path = slot.flatMap { chatRuntimes[$0]?.modelPath } ?? chatRuntimes[primaryChatSlot]?.modelPath ?? chatRuntimes.values.first?.modelPath
        let lower = (path ?? "").lowercased()
        return lower.contains("qwen3") || lower.contains("qwen-3")
    }

    private func classifyError(_ error: Error) -> LlamaErrorCode {
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
}
