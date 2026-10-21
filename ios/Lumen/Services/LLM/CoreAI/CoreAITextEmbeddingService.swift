import Foundation
#if canImport(CoreAI)
import CoreAI
#endif

/// Independent sentence-retrieval encoder. It never constructs a chat engine or
/// downloads assets, and a vector request cannot implicitly prepare the model.
actor CoreAITextEmbeddingService {
    static let shared = CoreAITextEmbeddingService()

    nonisolated static var isSupported: Bool {
        #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI)
        if #available(iOS 27.2, *) { return true }
        #endif
        return false
    }

    private var operationInFlight = false
    private var cancellationWaiters = 0
    private var replacementInFlight = false
    private var loadingTask: Task<Void, Error>?
    private var embeddingTask: Task<EmbeddingRuntimeResult, Error>?
    private var cancellationRegistration: (id: UUID, category: AppCancellationCategory)?
    private var loadedIdentifier: String?
    #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI)
    private var model: AIModel?
    private var functions: [Int: InferenceFunction] = [:]
    private var tokenizer: GemmaTokenizer?
    #endif

    private init() {}

    private var busy: Bool { operationInFlight || cancellationWaiters > 0 || replacementInFlight }

    func isLoaded() -> Bool { loadedIdentifier != nil }

    /// Cached metadata only. Dimension identifies an embedding space, not a task;
    /// retrieval query and document prefixes intentionally share that space.
    func loadedMetadata(dimensions: Int = CoreAITextEmbeddingPolicy.defaultDimensions) -> RAGEmbeddingIndexMetadata? {
        guard loadedIdentifier != nil, CoreAITextEmbeddingModelStore.isSelected,
              CoreAITextEmbeddingPolicy.dimensions.contains(dimensions) else { return nil }
        return RAGEmbeddingIndexMetadata(
            formatVersion: CoreAITextEmbeddingPolicy.formatVersion,
            modelIdentifier: CoreAITextEmbeddingPolicy.modelIdentifier(dimensions: dimensions), dimension: dimensions
        )
    }

    /// The foreground ModelLoader readiness seam calls this after explicit user
    /// work. Scene/bootstrap/background discovery must never call it.
    func prepare() async throws {
        try Task.checkCancellation()
        guard CoreAITextEmbeddingModelStore.isSelected else { throw CoreAITextEmbeddingError.notSelected }
        guard Self.isSupported else { throw CoreAITextEmbeddingError.unsupported }
        guard !busy else { throw CoreAITextEmbeddingError.busy }
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
        } onCancel: { task.cancel() }
    }

    func embed(_ request: EmbeddingRequest) async throws -> EmbeddingRuntimeResult {
        // Validate bounded raw requests before entering the framework, including on
        // an unsupported host. No hidden legacy-prefix compatibility is accepted.
        _ = try CoreAITextEmbeddingPolicy.formattedText(request)
        try Task.checkCancellation()
        guard CoreAITextEmbeddingModelStore.isSelected else { throw CoreAITextEmbeddingError.notSelected }
        guard Self.isSupported else { throw CoreAITextEmbeddingError.unsupported }
        guard !busy else { throw CoreAITextEmbeddingError.busy }
        guard isLoaded() else { throw CoreAITextEmbeddingError.notPrepared }
        operationInFlight = true
        let task = Task {
            defer { finishOperation() }
            return try await runEmbedding(request)
        }
        embeddingTask = task
        registerCancellation({ task.cancel() }, category: .memoryMaintenance)
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: { task.cancel() }
    }

    /// CoreAI's generic function API exposes cooperative task cancellation. Hold
    /// admission closed until the awaited framework call actually returns; never
    /// release model buffers or replace files while inference is still in flight.
    func cancelAndWait() async {
        cancellationWaiters += 1
        defer { cancellationWaiters -= 1 }
        let load = loadingTask
        let embedding = embeddingTask
        load?.cancel()
        embedding?.cancel()
        if let load { _ = await load.result }
        if let embedding { _ = await embedding.result }
    }

    func unload() async {
        cancellationWaiters += 1
        defer { cancellationWaiters -= 1 }
        await cancelAndWait()
        clearModel()
    }

    @discardableResult
    func beginModelReplacement(if condition: @Sendable () -> Bool = { true }) async throws -> Bool {
        // A stale rollback must not drain or unload a newer selected runtime.
        guard condition() else { return false }
        guard !replacementInFlight else { throw CoreAITextEmbeddingError.busy }
        replacementInFlight = true
        await cancelAndWait()
        clearModel()
        return true
    }

    func endModelReplacement() { replacementInFlight = false }

    private func registerCancellation(_ cancellation: @escaping @Sendable () -> Void, category: AppCancellationCategory) {
        cancellationRegistration = (AppCancellationBus.shared.registerCancellation(cancellation, category: category), category)
    }

    private func finishOperation() {
        if let registration = cancellationRegistration {
            AppCancellationBus.shared.unregister(registration.id, category: registration.category)
        }
        cancellationRegistration = nil
        loadingTask = nil
        embeddingTask = nil
        operationInFlight = false
    }

    private func clearModel() {
        #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI)
        functions = [:]
        model = nil
        tokenizer = nil
        #endif
        loadedIdentifier = nil
    }

    private func loadConfiguredModel() async throws {
        #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI)
        do {
            guard await ResourceBudgetGate.allowsForegroundModelLoad(reason: "coreai-text-embeddings") else {
                throw CoreAITextEmbeddingError.resourceDenied
            }
            try Task.checkCancellation()
            // Store hashing is streamed off MainActor; only an owned, immutable,
            // fully pinned package can reach the tokenizer or specialization.
            let package = try CoreAITextEmbeddingModelStore.configuredPackage()
            guard package.identifier == CoreAITextEmbeddingPolicy.baseIdentifier,
                  package.artifactDigest == CoreAITextEmbeddingPolicy.graphDigest,
                  package.configuration.buckets == CoreAITextEmbeddingPolicy.buckets else {
                throw CoreAITextEmbeddingError.invalidModel
            }
            let loadedTokenizer = try GemmaTokenizer(contentsOf: package.tokenizerURL)
            guard loadedTokenizer.bosID == 2, loadedTokenizer.eosID == 1,
                  loadedTokenizer.padID == 0, loadedTokenizer.vocabSize == 262_144 else {
                throw CoreAITextEmbeddingError.invalidModel
            }
            try Task.checkCancellation()
            let asset = try AIModelAsset(contentsOf: package.modelURL)
            guard let summary = try asset.summary(includingStatistics: false),
                  Set(summary.functions.map(\.name)) == Set(CoreAITextEmbeddingPolicy.functions) else {
                throw CoreAITextEmbeddingError.invalidModel
            }
            guard CoreAITextEmbeddingModelStore.isSelected,
                  await ResourceBudgetGate.allowsForegroundModelLoad(reason: "coreai-text-embeddings-specialization") else {
                throw CoreAITextEmbeddingError.resourceDenied
            }
            try Task.checkCancellation()
            let loadedModel = try await AIModel(
                contentsOf: package.modelURL,
                options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
            )
            var loadedFunctions: [Int: InferenceFunction] = [:]
            for bucket in CoreAITextEmbeddingPolicy.buckets {
                try Task.checkCancellation()
                guard CoreAITextEmbeddingModelStore.isSelected,
                      await ResourceBudgetGate.allowsForegroundModelLoad(reason: "coreai-text-embeddings-functions") else {
                    throw CoreAITextEmbeddingError.resourceDenied
                }
                try Task.checkCancellation()
                guard let function = try loadedModel.loadFunction(named: "mm_\(bucket)") else {
                    throw CoreAITextEmbeddingError.invalidModel
                }
                try validate(function.descriptor, bucket: bucket)
                loadedFunctions[bucket] = function
            }
            try Task.checkCancellation()
            guard CoreAITextEmbeddingModelStore.isSelected else { throw CoreAITextEmbeddingError.notSelected }
            model = loadedModel
            functions = loadedFunctions
            tokenizer = loadedTokenizer
            loadedIdentifier = package.identifier
        } catch is CancellationError { throw CancellationError() }
        catch let error as CoreAITextEmbeddingError { throw error }
        catch let error as CoreAILocalPackageIO.Failure {
            if Task.isCancelled { throw CancellationError() }
            switch error {
            case .notConfigured: throw CoreAITextEmbeddingError.modelNotConfigured
            case .integrity: throw CoreAITextEmbeddingError.integrityFailure
            case .invalidPackage, .localDirectory: throw CoreAITextEmbeddingError.invalidModel
            case .storage: throw CoreAITextEmbeddingError.loadFailed
            }
        } catch {
            if Task.isCancelled { throw CancellationError() }
            throw CoreAITextEmbeddingError.loadFailed
        }
        #else
        throw CoreAITextEmbeddingError.unsupported
        #endif
    }

    private func runEmbedding(_ request: EmbeddingRequest) async throws -> EmbeddingRuntimeResult {
        #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI)
        do {
            guard CoreAITextEmbeddingModelStore.isSelected else { throw CoreAITextEmbeddingError.notSelected }
            guard let tokenizer, model != nil, loadedIdentifier != nil else { throw CoreAITextEmbeddingError.notPrepared }
            guard await ResourceBudgetGate.allowsWork(policy: .embedding, reason: "coreai-text-embedding") else {
                throw CoreAITextEmbeddingError.resourceDenied
            }
            try Task.checkCancellation()
            let dimension = try CoreAITextEmbeddingPolicy.requestedDimensions(request)
            let text = try CoreAITextEmbeddingPolicy.formattedText(request)
            let ids = try CoreAITextEmbeddingPolicy.tokenIDs(tokenizer.encode(text))
            let bucket = try CoreAITextEmbeddingPolicy.bucket(forTokenCount: ids.count)
            guard let function = functions[bucket] else { throw CoreAITextEmbeddingError.invalidModel }
            // There is no conversation state, KV cache, soft token input, pooling
            // shim, image/audio asset or generative model involved in this call.
            var inputIDs = NDArray(shape: [1, bucket], scalarType: .int32)
            var attentionMask = NDArray(shape: [1, bucket], scalarType: .float16)
            var multimodalEmbeddings = NDArray(shape: [1, bucket, 512], scalarType: .float16)
            var multimodalMask = NDArray(shape: [1, bucket], scalarType: .float16)
            inputIDs.mutableView(as: Int32.self).withUnsafeMutablePointer { p, _, _ in
                for index in 0..<bucket { p[index] = index < ids.count ? ids[index] : 0 }
            }
            attentionMask.mutableView(as: Float16.self).withUnsafeMutablePointer { p, _, _ in
                for index in 0..<bucket { p[index] = index < ids.count ? 1 : 0 }
            }
            multimodalEmbeddings.mutableView(as: Float16.self).withUnsafeMutablePointer { p, _, _ in
                for index in 0..<(bucket * 512) { p[index] = 0 }
            }
            multimodalMask.mutableView(as: Float16.self).withUnsafeMutablePointer { p, _, _ in
                for index in 0..<bucket { p[index] = 0 }
            }
            try Task.checkCancellation()
            guard CoreAITextEmbeddingModelStore.isSelected,
                  await ResourceBudgetGate.allowsWork(policy: .embedding, reason: "coreai-text-embedding-run") else {
                throw CoreAITextEmbeddingError.resourceDenied
            }
            try Task.checkCancellation()
            var outputs = try await function.run(inputs: [
                "input_ids": inputIDs, "attention_mask": attentionMask,
                "mm_embeds": multimodalEmbeddings, "mm_mask": multimodalMask
            ])
            try Task.checkCancellation()
            guard CoreAITextEmbeddingModelStore.isSelected else { throw CoreAITextEmbeddingError.notSelected }
            guard let output = outputs.remove("embedding")?.ndArray,
                  output.shape == [1, 768], output.scalarType == .float16 else {
                throw CoreAITextEmbeddingError.invalidOutput
            }
            let view = output.view(as: Float16.self)
            guard view.isContiguous else { throw CoreAITextEmbeddingError.invalidOutput }
            var values = [Double](repeating: 0, count: 768)
            view.withUnsafePointer { p, _, _ in
                for index in 0..<768 { values[index] = Double(p[index]) }
            }
            let vector = try CoreAITextEmbeddingPolicy.normalizedVector(values, dimensions: dimension)
            try Task.checkCancellation()
            return EmbeddingRuntimeResult(
                vector: vector, modelIdentifier: CoreAITextEmbeddingPolicy.modelIdentifier(dimensions: dimension),
                formatVersion: CoreAITextEmbeddingPolicy.formatVersion
            )
        } catch is CancellationError { throw CancellationError() }
        catch let error as CoreAITextEmbeddingError { throw error }
        catch {
            if Task.isCancelled { throw CancellationError() }
            throw CoreAITextEmbeddingError.inferenceFailed
        }
        #else
        throw CoreAITextEmbeddingError.unsupported
        #endif
    }

    #if os(iOS) && !targetEnvironment(simulator) && canImport(CoreAI)
    private func validate(_ descriptor: InferenceFunctionDescriptor, bucket: Int) throws {
        guard Set(descriptor.inputNames) == ["input_ids", "attention_mask", "mm_embeds", "mm_mask"],
              descriptor.outputNames == ["embedding"], descriptor.stateNames.isEmpty else {
            throw CoreAITextEmbeddingError.invalidModel
        }
        let inputs: [(String, [Int], NDArray.ScalarType)] = [
            ("input_ids", [1, bucket], .int32), ("attention_mask", [1, bucket], .float16),
            ("mm_embeds", [1, bucket, 512], .float16), ("mm_mask", [1, bucket], .float16)
        ]
        for (name, shape, scalarType) in inputs {
            guard case .ndArray(let array)? = descriptor.inputDescriptor(of: name),
                  array.shape == shape, array.scalarType == scalarType else { throw CoreAITextEmbeddingError.invalidModel }
        }
        guard case .ndArray(let array)? = descriptor.outputDescriptor(of: "embedding"),
              array.shape == [1, 768], array.scalarType == .float16 else { throw CoreAITextEmbeddingError.invalidModel }
    }
    #endif
}
