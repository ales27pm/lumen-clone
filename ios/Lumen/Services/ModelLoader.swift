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
}

nonisolated struct StoredModelLoadItem: Identifiable, Sendable, Hashable {
    let id: UUID
    let name: String
    let repoId: String
    let fileName: String
    let quantization: String
    let parameters: String
    let role: String
    let downloadedAt: Date
    let localPath: String
    let resolvedPath: String

    var modelRole: ModelRole { ModelRole(rawValue: role) ?? .chat }

    @MainActor
    init(stored: StoredModel) {
        self.id = stored.id
        self.name = stored.name
        self.repoId = stored.repoId
        self.fileName = stored.fileName
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

    private struct PendingChatModelLoad {
        let id = UUID()
        let task: Task<ChatLoadResult, Never>
    }

    private struct PendingModelLoad {
        let id = UUID()
        let task: Task<Bool, Never>
    }

    private static var chatLoadTask: PendingChatModelLoad?
    private static var embedLoadTask: PendingModelLoad?

    static func canStartModelLoad(intent: ModelLoadIntent) -> Bool {
        switch intent {
        case .userChat, .userVoice:
            return ResourceBudgetGate.allowsForegroundModelLoad(reason: intent.rawValue)
        case .diagnostics, .background:
            return ResourceBudgetGate.allowsHeavyModelWork(reason: intent.rawValue)
        case .appStartup:
            return false
        }
    }

    static func cancelActiveLoads() {
        chatLoadTask?.task.cancel()
        embedLoadTask?.task.cancel()
    }

    #if DEBUG
    static func installChatLoadTaskForTesting(_ task: Task<Bool, Never>) {
        chatLoadTask = PendingChatModelLoad(task: Task {
            ChatLoadResult(loaded: await task.value, selectedChatModelID: nil)
        })
    }

    static var hasActiveChatLoadTaskForTesting: Bool {
        chatLoadTask != nil
    }

    static func resetLoadTasksForTesting() {
        chatLoadTask?.task.cancel()
        embedLoadTask?.task.cancel()
        chatLoadTask = nil
        embedLoadTask = nil
    }
    #endif
    static func syncChat(appState: AppState, stored: [StoredModel]) async {
        await ensureFleetChatLoaded(appState: appState, stored: stored, intent: .appStartup)
    }

    static func syncEmbed(appState: AppState, stored: [StoredModel]) async {
        await ensureEmbedLoaded(appState: appState, stored: stored, intent: .appStartup)
    }

    /// Launch registration is assignment-first. Large role-baked GGUFs are not all
    /// preloaded; the runtime lazily loads the specific slot that is about to run.
    static func loadAtLaunch(appState: AppState, stored: [StoredModel]) async {
        _ = await ensureFleetChatLoaded(appState: appState, stored: stored, intent: .appStartup)
        _ = await ensureEmbedLoaded(appState: appState, stored: stored, intent: .appStartup)
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
        await Task.yield()
        _ = await configureFleetRuntime(snapshot: snapshot)
        if await hasLoadedChatRuntime(snapshot: snapshot) { return true }
        guard canStartModelLoad(intent: intent) else { return false }
        if let chatLoadTask {
            let result = await finishChatLoad(chatLoadTask)
            _ = await configureFleetRuntime(snapshot: snapshot)
            if let selectedChatModelID = result.selectedChatModelID {
                appState?.activeChatModelID = selectedChatModelID
            }
            return result.loaded
        }
        let pending = PendingChatModelLoad(task: Task.detached(priority: .userInitiated) {
            await performEnsureFleetChatLoaded(snapshot: snapshot, intent: intent)
        })
        chatLoadTask = pending
        let result = await finishChatLoad(pending)
        _ = await configureFleetRuntime(snapshot: snapshot)
        if let selectedChatModelID = result.selectedChatModelID {
            appState?.activeChatModelID = selectedChatModelID
        }
        return result.loaded
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
        return ChatLoadResult(loaded: primaryReady || !runnableSlots.isEmpty, selectedChatModelID: nil)
    }

    private static func finishChatLoad(_ pending: PendingChatModelLoad) async -> ChatLoadResult {
        let result = await pending.task.value
        if chatLoadTask?.id == pending.id {
            chatLoadTask = nil
        }
        return result
    }

    nonisolated private static func hasLoadedChatRuntime(snapshot: ModelLoadSnapshot) async -> Bool {
        let preferredID = snapshot.activeChatModelID
        if let preferredID,
           let preferred = snapshot.storedModels.first(where: { $0.id.uuidString == preferredID && $0.modelRole == .chat }) {
            return await AppLlamaService.shared.loadedChatPath == preferred.resolvedPath
        }
        return await AppLlamaService.shared.isChatLoaded
    }

    @discardableResult
    nonisolated private static func ensurePrimaryChatLoaded(snapshot: ModelLoadSnapshot) async -> ChatLoadResult {
        let preferredID = snapshot.activeChatModelID
        if let preferredID,
           let preferred = snapshot.storedModels.first(where: { $0.id.uuidString == preferredID && $0.modelRole == .chat }) {
            guard FileManager.default.fileExists(atPath: preferred.resolvedPath) else { return ChatLoadResult(loaded: false, selectedChatModelID: nil) }
            if await AppLlamaService.shared.isChatLoaded,
               await AppLlamaService.shared.loadedChatPath == preferred.resolvedPath {
                return ChatLoadResult(loaded: true, selectedChatModelID: nil)
            }
        } else if await AppLlamaService.shared.isChatLoaded {
            return ChatLoadResult(loaded: true, selectedChatModelID: nil)
        }
        let candidates = snapshot.storedModels.filter { $0.modelRole == .chat }
        await SlotModelRuntimeCoordinator.shared.configure(
            assignments: [:],
            contextSize: snapshot.contextSize,
            preferExclusiveChatRuntime: true
        )
        await Task.yield()
        let selectedChatModelID = await SlotModelRuntimeCoordinator.shared.ensureChatModelSelection(
            candidates: candidates,
            preferredID: preferredID
        )
        return ChatLoadResult(loaded: selectedChatModelID != nil, selectedChatModelID: selectedChatModelID)
    }

    @discardableResult
    static func ensureEmbedLoaded(appState: AppState, stored: [StoredModel], intent: ModelLoadIntent = .userChat) async -> Bool {
        if await hasLoadedEmbeddingRuntime(appState: appState, stored: stored) { return true }
        guard canStartModelLoad(intent: intent) else { return false }
        if let embedLoadTask { return await finishEmbedLoad(embedLoadTask) }
        let pending = PendingModelLoad(task: Task { @MainActor in
            await performEnsureEmbedLoaded(appState: appState, stored: stored, intent: intent)
        })
        embedLoadTask = pending
        return await finishEmbedLoad(pending)
    }

    private static func finishEmbedLoad(_ pending: PendingModelLoad) async -> Bool {
        let result = await pending.task.value
        if embedLoadTask?.id == pending.id {
            embedLoadTask = nil
        }
        return result
    }

    private static func hasLoadedEmbeddingRuntime(appState: AppState, stored: [StoredModel]) async -> Bool {
        let snapshot = ModelLoadSnapshot(appState: appState, stored: stored)
        return await hasLoadedEmbeddingRuntime(snapshot: snapshot)
    }

    nonisolated private static func hasLoadedEmbeddingRuntime(snapshot: ModelLoadSnapshot) async -> Bool {
        let preferredID = snapshot.activeEmbeddingModelID
        if let preferredID,
           let preferred = snapshot.storedModels.first(where: { $0.id.uuidString == preferredID && $0.modelRole == .embedding }) {
            return await AppLlamaService.shared.loadedEmbedPath == preferred.resolvedPath
        }
        return await AppLlamaService.shared.hasSemanticEmbeddingRuntime
    }

    private static func performEnsureEmbedLoaded(appState: AppState, stored: [StoredModel], intent: ModelLoadIntent) async -> Bool {
        guard !Task.isCancelled, canStartModelLoad(intent: intent) else { return false }
        let preferredID = appState.activeEmbeddingModelID
        if let preferredID,
           let preferred = stored.first(where: { $0.id.uuidString == preferredID && $0.modelRole == .embedding }) {
            let resolvedPath = ModelStorage.resolvedModelURL(from: preferred.localPath, fileName: preferred.fileName).path
            if await AppLlamaService.shared.isEmbedLoaded,
               await AppLlamaService.shared.loadedEmbedPath == resolvedPath {
                return true
            }
        } else if await AppLlamaService.shared.isEmbedLoaded {
            return true
        }

        let candidates = stored.filter { $0.modelRole == .embedding }
        await SlotModelRuntimeCoordinator.shared.configure(
            assignments: await SlotModelRuntimeCoordinator.shared.configuredAssignments,
            contextSize: appState.contextSize,
            preferExclusiveChatRuntime: true
        )
        guard !Task.isCancelled else { return false }
        return await SlotModelRuntimeCoordinator.shared.ensureEmbeddingModel(
            appState: appState,
            candidates: candidates,
            preferredID: preferredID
        )
    }
}
