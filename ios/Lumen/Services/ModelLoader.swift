import Foundation
import SwiftData
import OSLog


nonisolated struct ModelLoadSnapshot: Sendable {
    let activeChatModelID: String?
    let activeEmbeddingModelID: String?
    let contextSize: Int
    let selectedModelFamily: LumenModelFamily
    let storedModels: [StoredModelLoadItem]

    @MainActor
    init(appState: AppState, stored: [StoredModel]) {
        self.activeChatModelID = appState.activeChatModelID
        self.activeEmbeddingModelID = appState.activeEmbeddingModelID
        self.contextSize = appState.contextSize
        self.selectedModelFamily = LumenModelFamily.persistedSelected
        self.storedModels = stored.map(StoredModelLoadItem.init(stored:))
    }

    init(
        activeChatModelID: String?,
        activeEmbeddingModelID: String?,
        contextSize: Int,
        selectedModelFamily: LumenModelFamily,
        storedModels: [StoredModelLoadItem]
    ) {
        self.activeChatModelID = activeChatModelID
        self.activeEmbeddingModelID = activeEmbeddingModelID
        self.contextSize = contextSize
        self.selectedModelFamily = selectedModelFamily
        self.storedModels = storedModels
    }

    func replacingActiveChatModelID(_ activeChatModelID: String?) -> Self {
        Self(
            activeChatModelID: activeChatModelID,
            activeEmbeddingModelID: activeEmbeddingModelID,
            contextSize: contextSize,
            selectedModelFamily: selectedModelFamily,
            storedModels: storedModels
        )
    }

    func replacingActiveEmbeddingModelID(_ activeEmbeddingModelID: String?) -> Self {
        Self(
            activeChatModelID: activeChatModelID,
            activeEmbeddingModelID: activeEmbeddingModelID,
            contextSize: contextSize,
            selectedModelFamily: selectedModelFamily,
            storedModels: storedModels
        )
    }
}

nonisolated struct ModelLoadSnapshotResult: Sendable {
    let snapshot: ModelLoadSnapshot?
    let diagnostic: String?

    var isReady: Bool { snapshot != nil && diagnostic == nil }
}

nonisolated struct ModelLaunchLoadResult: Sendable, Equatable {
    let chatLoaded: Bool
    let embeddingLoaded: Bool
    let resourceRetryAfterSeconds: TimeInterval?

    var allSelectedModelsLoaded: Bool { chatLoaded && embeddingLoaded }
}

nonisolated struct StoredModelLoadItem: Identifiable, Sendable, Hashable {
    let id: UUID
    let name: String
    let repoId: String
    let fileName: String
    let sizeBytes: Int64
    let quantization: String
    let parameters: String
    let role: String
    let downloadedAt: Date
    let localPath: String
    let resolvedPath: String
    let expectedSHA256: String?

    var modelRole: ModelRole { ModelRole(rawValue: role) ?? .chat }

    @MainActor
    init(stored: StoredModel) {
        self.id = stored.id
        self.name = stored.name
        self.repoId = stored.repoId
        self.fileName = stored.fileName
        let catalog = ModelCatalog.catalogModel(repoId: stored.repoId, fileName: stored.fileName)
        let qwen3Contract = LumenTrainedModelRuntimeRegistry.contract(for: .qwen3)
        let isQwen3SharedBaseIdentity = stored.repoId == qwen3Contract.sharedBaseRepoID
            && stored.fileName == qwen3Contract.sharedBaseFileName
        if isQwen3SharedBaseIdentity {
            self.sizeBytes = stored.sizeBytes
            if let catalog,
               catalog.repoId == qwen3Contract.sharedBaseRepoID,
               catalog.fileName == qwen3Contract.sharedBaseFileName,
               catalog.sizeBytes == qwen3Contract.sharedBaseSizeBytes,
               qwen3Contract.matchesSharedBase(
                   repoID: stored.repoId,
                   fileName: stored.fileName,
                   sizeBytes: stored.sizeBytes,
                   expectedSHA256: catalog.expectedSHA256
               ) {
                self.expectedSHA256 = qwen3Contract.sharedBaseExpectedSHA256
            } else {
                self.expectedSHA256 = nil
            }
        } else {
            self.sizeBytes = catalog?.sizeBytes ?? stored.sizeBytes
            self.expectedSHA256 = catalog?.expectedSHA256
        }
        self.quantization = stored.quantization
        self.parameters = stored.parameters
        self.role = stored.role
        self.downloadedAt = stored.downloadedAt
        self.localPath = stored.localPath
        self.resolvedPath = ModelStorage.resolvedModelURL(from: stored.localPath, fileName: stored.fileName).path
    }
}

enum ModelLoadIntent: String, Sendable, Equatable {
    case userChat
    case userVoice
    case appStartup
    case diagnostics
    case background
}

@MainActor
enum ModelLoader {
    private struct ChatLoadResult: Sendable {
        let loaded: Bool
        let selectedChatModelID: String?
    }

    private struct EmbeddingLoadResult: Sendable {
        let loaded: Bool
        let selectedEmbeddingModelID: String?
    }

    private struct PendingChatModelLoad {
        let id = UUID()
        let epoch: UInt64
        let requestKey: RoleLoadRequestKey
        let task: Task<ChatLoadResult, Never>
    }

    private struct PendingModelLoad {
        let id = UUID()
        let epoch: UInt64
        let requestKey: RoleLoadRequestKey
        let task: Task<EmbeddingLoadResult, Never>
    }

    private struct RoleLoadRequestKey: Equatable, Sendable {
        let activeModelID: String?
        let modelFamily: LumenModelFamily

        init(activeModelID: String?, modelFamily: LumenModelFamily) {
            self.activeModelID = activeModelID
            self.modelFamily = modelFamily
        }

        init(snapshot: ModelLoadSnapshot) {
            activeModelID = snapshot.activeChatModelID
            modelFamily = snapshot.selectedModelFamily
        }
    }

    private static var chatLoadTask: PendingChatModelLoad?
    private static var embedLoadTask: PendingModelLoad?
    private static var chatLoadEpoch: UInt64 = 0
    private static var embedLoadEpoch: UInt64 = 0
    private static var chatEpochRequestKey: RoleLoadRequestKey?
    private static var embedEpochRequestKey: RoleLoadRequestKey?

    static func canStartModelLoad(intent: ModelLoadIntent) -> Bool {
        switch intent {
        case .userChat, .userVoice, .appStartup:
            return ResourceBudgetGate.allowsForegroundModelLoad(reason: intent.rawValue)
        case .diagnostics, .background:
            return false
        }
    }

    static func cancelActiveLoads() {
        advanceChatLoadEpoch()
        advanceEmbedLoadEpoch()
        if let pending = chatLoadTask {
            pending.task.cancel()
            Task { @MainActor in
                _ = await finishChatLoad(pending)
            }
        }
        if let pending = embedLoadTask {
            pending.task.cancel()
            Task { @MainActor in
                _ = await finishEmbedLoad(pending)
            }
        }
    }

    #if DEBUG
    static func installChatLoadTaskForTesting(_ task: Task<Bool, Never>) {
        chatLoadTask?.task.cancel()
        chatLoadTask = PendingChatModelLoad(
            epoch: advanceChatLoadEpoch(),
            requestKey: RoleLoadRequestKey(activeModelID: nil, modelFamily: LumenModelFamily.persistedSelected),
            task: Task {
                await withTaskCancellationHandler {
                    ChatLoadResult(loaded: await task.value, selectedChatModelID: nil)
                } onCancel: {
                    task.cancel()
                }
            }
        )
    }

    static func installEmbeddingLoadTaskForTesting(_ task: Task<Bool, Never>) {
        embedLoadTask?.task.cancel()
        embedLoadTask = PendingModelLoad(
            epoch: advanceEmbedLoadEpoch(),
            requestKey: RoleLoadRequestKey(activeModelID: nil, modelFamily: LumenModelFamily.persistedSelected),
            task: Task {
                await withTaskCancellationHandler {
                    EmbeddingLoadResult(loaded: await task.value, selectedEmbeddingModelID: nil)
                } onCancel: {
                    task.cancel()
                }
            }
        )
    }

    static var hasActiveChatLoadTaskForTesting: Bool {
        chatLoadTask != nil
    }

    static var hasActiveEmbeddingLoadTaskForTesting: Bool {
        embedLoadTask != nil
    }

    static func awaitInstalledChatLoadMutationEligibilityForTesting(
        onCaptured: @MainActor () async -> Void
    ) async -> Bool {
        guard let pending = chatLoadTask else { return false }
        await onCaptured()
        let result = await finishChatLoad(pending)
        return result.loaded && chatLoadEpoch == pending.epoch
    }

    static func awaitInstalledEmbeddingLoadMutationEligibilityForTesting(
        onCaptured: @MainActor () async -> Void
    ) async -> Bool {
        guard let pending = embedLoadTask else { return false }
        await onCaptured()
        let result = await finishEmbedLoad(pending)
        return result.loaded && embedLoadEpoch == pending.epoch
    }

    static func resetLoadTasksForTesting() {
        advanceChatLoadEpoch()
        advanceEmbedLoadEpoch()
        chatLoadTask?.task.cancel()
        embedLoadTask?.task.cancel()
        chatLoadTask = nil
        embedLoadTask = nil
        chatEpochRequestKey = nil
        embedEpochRequestKey = nil
    }
    #endif

    static func modelLoadSnapshot(appState: AppState, context: ModelContext) -> ModelLoadSnapshotResult {
        modelLoadSnapshot(appState: appState) {
            try context.fetch(FetchDescriptor<StoredModel>())
        }
    }

    static func modelLoadSnapshotForTests(appState: AppState, fetch: () throws -> [StoredModel]) -> ModelLoadSnapshotResult {
        modelLoadSnapshot(appState: appState, fetch: fetch)
    }

    static func modelCatalogFetchFailureMessage(error: Error) -> String {
        ModelLaunchBootstrap.modelCatalogFetchFailureMessage(error: error)
    }

    private static func modelLoadSnapshot(appState: AppState, fetch: () throws -> [StoredModel]) -> ModelLoadSnapshotResult {
        do {
            return ModelLoadSnapshotResult(
                snapshot: ModelLoadSnapshot(appState: appState, stored: try fetch()),
                diagnostic: nil
            )
        } catch {
            return ModelLoadSnapshotResult(
                snapshot: nil,
                diagnostic: modelCatalogFetchFailureMessage(error: error)
            )
        }
    }

    static func syncChat(appState: AppState, stored: [StoredModel]) async {
        await ensureFleetChatLoaded(appState: appState, stored: stored, intent: .appStartup)
    }

    static func syncEmbed(appState: AppState, stored: [StoredModel]) async {
        await ensureEmbedLoaded(appState: appState, stored: stored, intent: .appStartup)
    }

    /// Restores the selected chat and embedding runtimes after a normal foreground
    /// launch. Large role-baked GGUFs remain assignment-first: one primary chat
    /// runtime is warmed while individual role slots continue to load on demand.
    @discardableResult
    static func loadAtLaunch(appState: AppState, stored: [StoredModel]) async -> ModelLaunchLoadResult {
        var latestResult = ModelLaunchLoadResult(
            chatLoaded: false,
            embeddingLoaded: false,
            resourceRetryAfterSeconds: nil
        )

        // A selection can change while a multi-gigabyte runtime is loading. Retry
        // against the newest selection instead of publishing a stale launch result.
        for _ in 0..<3 {
            let requestedChatID = appState.activeChatModelID
            let requestedEmbeddingID = appState.activeEmbeddingModelID
            let requestedFamily = LumenModelFamily.persistedSelected
            let chatLoaded = await ensureFleetChatLoaded(appState: appState, stored: stored, intent: .appStartup)
            guard !Task.isCancelled else { return latestResult }
            let embeddingLoaded = await ensureEmbedLoaded(appState: appState, stored: stored, intent: .appStartup)
            let allSelectedModelsLoaded = chatLoaded && embeddingLoaded
            latestResult = ModelLaunchLoadResult(
                chatLoaded: chatLoaded,
                embeddingLoaded: embeddingLoaded,
                resourceRetryAfterSeconds: allSelectedModelsLoaded
                    ? nil
                    : ResourceBudgetGate.foregroundModelLoadRetryDelay()
            )
            guard !Task.isCancelled else { return latestResult }
            if appState.activeChatModelID == requestedChatID,
               appState.activeEmbeddingModelID == requestedEmbeddingID,
               LumenModelFamily.persistedSelected == requestedFamily {
                return latestResult
            }
        }

        return ModelLaunchLoadResult(
            chatLoaded: latestResult.chatLoaded,
            embeddingLoaded: latestResult.embeddingLoaded,
            resourceRetryAfterSeconds: nil
        )
    }

    /// Backward-compatible entry point. In v1 this registers all available chat
    /// assignments for on-demand slot loading instead of loading all contexts.
    @discardableResult
    static func ensureChatLoaded(appState: AppState, stored: [StoredModel], intent: ModelLoadIntent = .userChat) async -> Bool {
        await ensureFleetChatLoaded(snapshot: ModelLoadSnapshot(appState: appState, stored: stored), appState: appState, intent: intent)
    }

    @discardableResult
    static func ensureChatLoaded(snapshot: ModelLoadSnapshot, intent: ModelLoadIntent = .userChat) async -> Bool {
        await ensureFleetChatLoaded(snapshot: snapshot, appState: nil, intent: intent)
    }

    @discardableResult
    static func ensureFleetChatLoaded(appState: AppState, stored: [StoredModel], intent: ModelLoadIntent = .userChat) async -> Bool {
        await ensureFleetChatLoaded(snapshot: ModelLoadSnapshot(appState: appState, stored: stored), appState: appState, intent: intent)
    }

    @discardableResult
    private static func ensureFleetChatLoaded(snapshot: ModelLoadSnapshot, appState: AppState?, intent: ModelLoadIntent) async -> Bool {
        let requestKey = RoleLoadRequestKey(snapshot: snapshot)
        let requestEpoch = registerChatRequest(requestKey)
        await Task.yield()
        guard chatLoadEpoch == requestEpoch,
              chatRequestIsCurrent(snapshot: snapshot, appState: appState) else { return false }
        if let pending = chatLoadTask,
           pending.requestKey != requestKey || pending.epoch != requestEpoch {
            pending.task.cancel()
            _ = await finishChatLoad(pending)
            guard chatLoadEpoch == requestEpoch,
                  chatRequestIsCurrent(snapshot: snapshot, appState: appState) else { return false }
            return await ensureFleetChatLoaded(snapshot: snapshot, appState: appState, intent: intent)
        }
        guard chatSelectionIsCompatible(snapshot: snapshot) else {
            if let pending = chatLoadTask {
                advanceChatLoadEpoch()
                pending.task.cancel()
                _ = await finishChatLoad(pending)
            }
            return false
        }
        guard await configureFleetRuntimeIfSelectionIsCurrent(
            snapshot: snapshot,
            appState: appState,
            epoch: requestEpoch
        ) else {
            return false
        }
        if await hasLoadedChatRuntime(snapshot: snapshot) {
            guard chatLoadEpoch == requestEpoch,
                  chatRequestIsCurrent(snapshot: snapshot, appState: appState) else { return false }
            return true
        }
        guard canStartModelLoad(intent: intent) else { return false }
        if let pending = chatLoadTask {
            let result = await finishChatLoad(pending)
            return await completeChatLoad(result, epoch: pending.epoch, snapshot: snapshot, appState: appState)
        }
        let pending = PendingChatModelLoad(
            epoch: requestEpoch,
            requestKey: requestKey,
            task: Task.detached(priority: .userInitiated) {
                await performEnsureFleetChatLoaded(snapshot: snapshot, intent: intent)
            }
        )
        chatLoadTask = pending
        let result = await finishChatLoad(pending)
        return await completeChatLoad(result, epoch: pending.epoch, snapshot: snapshot, appState: appState)
    }

    private static func completeChatLoad(
        _ result: ChatLoadResult,
        epoch: UInt64,
        snapshot: ModelLoadSnapshot,
        appState: AppState?
    ) async -> Bool {
        guard result.loaded, chatLoadEpoch == epoch else { return false }

        var completedSnapshot = snapshot
        if let appState {
            let currentID = appState.activeChatModelID
            guard loadRequestRemainsOwned(
                currentID: currentID,
                requestedID: snapshot.activeChatModelID,
                loadedID: result.selectedChatModelID,
                currentFamily: LumenModelFamily.persistedSelected,
                requestedFamily: snapshot.selectedModelFamily
            ) else {
                await restoreCurrentFleetConfiguration(from: snapshot, appState: appState, epoch: epoch)
                return false
            }
            completedSnapshot = snapshot.replacingActiveChatModelID(currentID)
        }

        guard chatLoadEpoch == epoch,
              await configureFleetRuntimeIfSelectionIsCurrent(
                  snapshot: completedSnapshot,
                  appState: appState,
                  epoch: epoch
              ) else {
            return false
        }
        let loaded = await hasLoadedChatRuntime(snapshot: completedSnapshot)
        guard chatLoadEpoch == epoch,
              chatRequestIsCurrent(snapshot: completedSnapshot, appState: appState) else {
            await restoreCurrentFleetConfiguration(from: completedSnapshot, appState: appState, epoch: epoch)
            return false
        }
        return loaded
    }

    private static func configureFleetRuntimeIfSelectionIsCurrent(
        snapshot: ModelLoadSnapshot,
        appState: AppState?,
        epoch: UInt64
    ) async -> Bool {
        guard epoch == chatLoadEpoch else { return false }
        guard chatRequestIsCurrent(snapshot: snapshot, appState: appState) else { return false }
        _ = await configureFleetRuntime(snapshot: snapshot)
        guard epoch == chatLoadEpoch,
              chatRequestIsCurrent(snapshot: snapshot, appState: appState) else { return false }
        return true
    }

    private static func restoreCurrentFleetConfiguration(
        from snapshot: ModelLoadSnapshot,
        appState: AppState?,
        epoch: UInt64
    ) async {
        guard epoch == chatLoadEpoch else { return }
        guard let appState else { return }
        let currentSnapshot = ModelLoadSnapshot(
            activeChatModelID: appState.activeChatModelID,
            activeEmbeddingModelID: appState.activeEmbeddingModelID,
            contextSize: appState.contextSize,
            selectedModelFamily: LumenModelFamily.persistedSelected,
            storedModels: snapshot.storedModels
        )
        guard epoch == chatLoadEpoch else { return }
        _ = await configureFleetRuntime(snapshot: currentSnapshot)
    }

    private static func chatRequestIsCurrent(
        snapshot: ModelLoadSnapshot,
        appState: AppState?
    ) -> Bool {
        guard LumenModelFamily.persistedSelected == snapshot.selectedModelFamily else { return false }
        return appState?.activeChatModelID == snapshot.activeChatModelID || appState == nil
    }

    static func chatSelectionIsCompatible(snapshot: ModelLoadSnapshot) -> Bool {
        guard let activeChatModelID = snapshot.activeChatModelID,
              let selected = snapshot.storedModels.first(where: {
                  $0.id.uuidString == activeChatModelID && $0.modelRole == .chat
              })
        else { return false }
        return LumenModelSelectionPolicy.isChatModelCompatible(
            repoID: selected.repoId,
            fileName: selected.fileName,
            sizeBytes: selected.sizeBytes,
            expectedSHA256: selected.expectedSHA256,
            family: snapshot.selectedModelFamily
        )
    }

    static func selectionRemainsOwned(
        currentID: String?,
        requestedID: String?,
        loadedID: String?
    ) -> Bool {
        currentID == requestedID
            && (loadedID == nil || loadedID == requestedID)
    }

    static func loadRequestRemainsOwned(
        currentID: String?,
        requestedID: String?,
        loadedID: String?,
        currentFamily: LumenModelFamily,
        requestedFamily: LumenModelFamily
    ) -> Bool {
        currentFamily == requestedFamily
            && selectionRemainsOwned(currentID: currentID, requestedID: requestedID, loadedID: loadedID)
    }

    @discardableResult
    nonisolated private static func configureFleetRuntime(snapshot loadSnapshot: ModelLoadSnapshot) async -> LumenModelFleetSnapshot {
        let snapshot = LumenModelFleetResolver.resolveV1(snapshot: loadSnapshot)
        await SlotModelRuntimeCoordinator.shared.configure(
            assignments: snapshot.assignments,
            contextSize: loadSnapshot.contextSize,
            preferExclusiveChatRuntime: true
        )
        return snapshot
    }

    @discardableResult
    nonisolated private static func performEnsureFleetChatLoaded(snapshot loadSnapshot: ModelLoadSnapshot, intent: ModelLoadIntent) async -> ChatLoadResult {
        guard !Task.isCancelled, await MainActor.run(body: { canStartModelLoad(intent: intent) }) else { return ChatLoadResult(loaded: false, selectedChatModelID: nil) }
        await Task.yield()
        let snapshot = await configureFleetRuntime(snapshot: loadSnapshot)

        let runnableSlots = [LumenModelSlot.cortex, .executor, .mouth, .mimicry, .rem]
            .filter { snapshot.assignment(for: $0) != nil }
        guard !runnableSlots.isEmpty else {
            return await ensurePrimaryChatLoaded(snapshot: loadSnapshot)
        }

        // Keep one chat runtime warm for non-agent/plain chat. Slot-agent turns load
        // each role-baked GGUF lazily, one slot at a time, through the coordinator.
        await Task.yield()
        let primaryReady = await SlotModelRuntimeCoordinator.shared.ensurePrimaryReady(preferredSlots: [.mouth, .cortex])
        guard !Task.isCancelled else { return ChatLoadResult(loaded: false, selectedChatModelID: nil) }
        return ChatLoadResult(loaded: primaryReady, selectedChatModelID: nil)
    }

    private static func finishChatLoad(_ pending: PendingChatModelLoad) async -> ChatLoadResult {
        let result = await pending.task.value
        if chatLoadTask?.id == pending.id {
            chatLoadTask = nil
        }
        return result
    }

    @discardableResult
    private static func advanceChatLoadEpoch() -> UInt64 {
        chatLoadEpoch &+= 1
        return chatLoadEpoch
    }

    private static func registerChatRequest(_ requestKey: RoleLoadRequestKey) -> UInt64 {
        if chatEpochRequestKey != requestKey {
            chatEpochRequestKey = requestKey
            advanceChatLoadEpoch()
        }
        return chatLoadEpoch
    }

    nonisolated private static func hasLoadedChatRuntime(snapshot: ModelLoadSnapshot) async -> Bool {
        guard let preferredID = snapshot.activeChatModelID,
              let preferred = snapshot.storedModels.first(where: { $0.id.uuidString == preferredID && $0.modelRole == .chat })
        else { return false }
        guard await AppLlamaService.shared.loadedChatPath == preferred.resolvedPath else { return false }
        guard snapshot.selectedModelFamily == .qwen3 else { return true }
        guard LumenModelSelectionPolicy.isChatModelCompatible(
            repoID: preferred.repoId,
            fileName: preferred.fileName,
            sizeBytes: preferred.sizeBytes,
            expectedSHA256: preferred.expectedSHA256,
            family: snapshot.selectedModelFamily
        ) else { return false }
        if case .success = await ModelFileIntegrity.validateInstalledFileWithDiagnosticsAsync(preferred) {
            return true
        }
        return false
    }

    @discardableResult
    nonisolated private static func ensurePrimaryChatLoaded(snapshot: ModelLoadSnapshot) async -> ChatLoadResult {
        guard let preferredID = snapshot.activeChatModelID,
              let preferred = snapshot.storedModels.first(where: {
                  $0.id.uuidString == preferredID && $0.modelRole == .chat
              }),
              FileManager.default.fileExists(atPath: preferred.resolvedPath)
        else {
            if await AppLlamaService.shared.isChatLoaded {
                await AppLlamaService.shared.unloadAllChat()
            }
            return ChatLoadResult(loaded: false, selectedChatModelID: nil)
        }
        if await AppLlamaService.shared.isChatLoaded,
           await AppLlamaService.shared.loadedChatPath == preferred.resolvedPath {
            return ChatLoadResult(loaded: true, selectedChatModelID: nil)
        }
        if await AppLlamaService.shared.isChatLoaded {
            await AppLlamaService.shared.unloadAllChat()
        }
        await SlotModelRuntimeCoordinator.shared.configure(
            assignments: [:],
            contextSize: snapshot.contextSize,
            preferExclusiveChatRuntime: true
        )
        await Task.yield()
        let selectedChatModelID = await SlotModelRuntimeCoordinator.shared.ensureChatModelSelection(
            candidates: [preferred],
            preferredID: preferredID
        )
        return ChatLoadResult(loaded: selectedChatModelID != nil, selectedChatModelID: selectedChatModelID)
    }

    @discardableResult
    static func ensureEmbedLoaded(appState: AppState, stored: [StoredModel], intent: ModelLoadIntent = .userChat) async -> Bool {
        let snapshot = ModelLoadSnapshot(appState: appState, stored: stored)
        let requestKey = RoleLoadRequestKey(
            activeModelID: snapshot.activeEmbeddingModelID,
            modelFamily: snapshot.selectedModelFamily
        )
        let requestEpoch = registerEmbedRequest(requestKey)
        if let pending = embedLoadTask,
           pending.requestKey != requestKey || pending.epoch != requestEpoch {
            pending.task.cancel()
            _ = await finishEmbedLoad(pending)
            guard embedLoadEpoch == requestEpoch,
                  embeddingRequestIsCurrent(snapshot: snapshot, appState: appState) else { return false }
            return await ensureEmbedLoaded(appState: appState, stored: stored, intent: intent)
        }
        if await hasLoadedEmbeddingRuntime(snapshot: snapshot) {
            return embedLoadEpoch == requestEpoch
                && embeddingRequestIsCurrent(snapshot: snapshot, appState: appState)
        }
        guard canStartModelLoad(intent: intent) else { return false }
        if let pending = embedLoadTask {
            let result = await finishEmbedLoad(pending)
            return await completeEmbeddingLoad(result, epoch: pending.epoch, snapshot: snapshot, appState: appState)
        }
        let pending = PendingModelLoad(
            epoch: requestEpoch,
            requestKey: requestKey,
            task: Task.detached(priority: .userInitiated) {
                await performEnsureEmbedLoaded(snapshot: snapshot, intent: intent)
            }
        )
        embedLoadTask = pending
        let result = await finishEmbedLoad(pending)
        return await completeEmbeddingLoad(result, epoch: pending.epoch, snapshot: snapshot, appState: appState)
    }

    private static func finishEmbedLoad(_ pending: PendingModelLoad) async -> EmbeddingLoadResult {
        let result = await pending.task.value
        if embedLoadTask?.id == pending.id {
            embedLoadTask = nil
        }
        return result
    }

    @discardableResult
    private static func advanceEmbedLoadEpoch() -> UInt64 {
        embedLoadEpoch &+= 1
        return embedLoadEpoch
    }

    private static func registerEmbedRequest(_ requestKey: RoleLoadRequestKey) -> UInt64 {
        if embedEpochRequestKey != requestKey {
            embedEpochRequestKey = requestKey
            advanceEmbedLoadEpoch()
        }
        return embedLoadEpoch
    }

    private static func completeEmbeddingLoad(
        _ result: EmbeddingLoadResult,
        epoch: UInt64,
        snapshot: ModelLoadSnapshot,
        appState: AppState
    ) async -> Bool {
        guard result.loaded, embedLoadEpoch == epoch else { return false }
        let currentID = appState.activeEmbeddingModelID
        guard loadRequestRemainsOwned(
            currentID: currentID,
            requestedID: snapshot.activeEmbeddingModelID,
            loadedID: result.selectedEmbeddingModelID,
            currentFamily: LumenModelFamily.persistedSelected,
            requestedFamily: snapshot.selectedModelFamily
        ) else {
            return false
        }
        let completedSnapshot = snapshot.replacingActiveEmbeddingModelID(currentID)
        let loaded = await hasLoadedEmbeddingRuntime(snapshot: completedSnapshot)
        guard embedLoadEpoch == epoch,
              embeddingRequestIsCurrent(snapshot: completedSnapshot, appState: appState) else {
            return false
        }
        return loaded
    }

    private static func embeddingRequestIsCurrent(
        snapshot: ModelLoadSnapshot,
        appState: AppState
    ) -> Bool {
        appState.activeEmbeddingModelID == snapshot.activeEmbeddingModelID
            && LumenModelFamily.persistedSelected == snapshot.selectedModelFamily
    }

    nonisolated private static func hasLoadedEmbeddingRuntime(snapshot: ModelLoadSnapshot) async -> Bool {
        guard let preferredID = snapshot.activeEmbeddingModelID,
              let preferred = snapshot.storedModels.first(where: { $0.id.uuidString == preferredID && $0.modelRole == .embedding })
        else { return false }
        return await AppLlamaService.shared.loadedEmbedPath == preferred.resolvedPath
    }

    nonisolated private static func performEnsureEmbedLoaded(
        snapshot: ModelLoadSnapshot,
        intent: ModelLoadIntent
    ) async -> EmbeddingLoadResult {
        guard !Task.isCancelled,
              await MainActor.run(body: { canStartModelLoad(intent: intent) }) else {
            return EmbeddingLoadResult(loaded: false, selectedEmbeddingModelID: nil)
        }
        guard let preferredID = snapshot.activeEmbeddingModelID,
              let preferred = snapshot.storedModels.first(where: {
                  $0.id.uuidString == preferredID && $0.modelRole == .embedding
              }),
              FileManager.default.fileExists(atPath: preferred.resolvedPath)
        else {
            if await AppLlamaService.shared.isEmbedLoaded {
                await AppLlamaService.shared.unloadEmbed()
            }
            return EmbeddingLoadResult(loaded: false, selectedEmbeddingModelID: nil)
        }
        if await AppLlamaService.shared.isEmbedLoaded,
           await AppLlamaService.shared.loadedEmbedPath == preferred.resolvedPath {
            return EmbeddingLoadResult(loaded: true, selectedEmbeddingModelID: nil)
        }
        if await AppLlamaService.shared.isEmbedLoaded {
            await AppLlamaService.shared.unloadEmbed()
        }

        await SlotModelRuntimeCoordinator.shared.configure(
            assignments: await SlotModelRuntimeCoordinator.shared.configuredAssignments,
            contextSize: snapshot.contextSize,
            preferExclusiveChatRuntime: true
        )
        guard !Task.isCancelled else {
            return EmbeddingLoadResult(loaded: false, selectedEmbeddingModelID: nil)
        }
        let selectedEmbeddingModelID = await SlotModelRuntimeCoordinator.shared.ensureEmbeddingModelSelection(
            candidates: [preferred],
            preferredID: preferredID
        )
        guard !Task.isCancelled else {
            return EmbeddingLoadResult(loaded: false, selectedEmbeddingModelID: nil)
        }
        return EmbeddingLoadResult(
            loaded: selectedEmbeddingModelID != nil,
            selectedEmbeddingModelID: selectedEmbeddingModelID
        )
    }
}
