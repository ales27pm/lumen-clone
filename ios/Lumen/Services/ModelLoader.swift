import Foundation
import SwiftData
import OSLog

enum ModelLoadIntent: String, Sendable, Equatable {
    case userChat
    case userVoice
    case appStartup
    case diagnostics
    case background
}

@MainActor
enum ModelLoader {
    private static var chatLoadTask: Task<Bool, Never>?
    private static var embedLoadTask: Task<Bool, Never>?

    static func canStartModelLoad(intent: ModelLoadIntent) -> Bool {
        switch intent {
        case .userChat, .userVoice:
            return ResourceBudgetGate.allowsForegroundModelLoad(reason: intent.rawValue)
        case .appStartup, .diagnostics, .background:
            return false
        }
    }

    static func cancelActiveLoads() {
        chatLoadTask?.cancel()
        chatLoadTask = nil
        embedLoadTask?.cancel()
        embedLoadTask = nil
    }
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
        await ensureFleetChatLoaded(appState: appState, stored: stored, intent: intent)
    }

    @discardableResult
    static func ensureFleetChatLoaded(appState: AppState, stored: [StoredModel], intent: ModelLoadIntent = .userChat) async -> Bool {
        if await hasLoadedChatRuntime(appState: appState, stored: stored) { return true }
        guard canStartModelLoad(intent: intent) else { return false }
        if let chatLoadTask { return await chatLoadTask.value }
        let task = Task { @MainActor in
            await performEnsureFleetChatLoaded(appState: appState, stored: stored, intent: intent)
        }
        chatLoadTask = task
        let result = await task.value
        chatLoadTask = nil
        return result
    }

    @discardableResult
    private static func performEnsureFleetChatLoaded(appState: AppState, stored: [StoredModel], intent: ModelLoadIntent) async -> Bool {
        guard !Task.isCancelled, ResourceBudgetGate.allowsForegroundModelLoad(reason: intent.rawValue) else { return false }
        let snapshot = LumenModelFleetResolver.resolveV1(appState: appState, storedModels: stored)
        SlotModelRuntimeCoordinator.shared.configure(
            assignments: snapshot.assignments,
            contextSize: appState.contextSize,
            preferExclusiveChatRuntime: true
        )

        let runnableSlots = [LumenModelSlot.cortex, .executor, .mouth, .mimicry, .rem]
            .filter { snapshot.assignment(for: $0) != nil }
        guard !runnableSlots.isEmpty else {
            return await ensurePrimaryChatLoaded(appState: appState, stored: stored)
        }

        // Keep one chat runtime warm for non-agent/plain chat. Slot-agent turns load
        // each role-baked GGUF lazily, one slot at a time, through the coordinator.
        let primaryReady = await SlotModelRuntimeCoordinator.shared.ensurePrimaryReady(preferredSlots: [.mouth, .cortex])
        guard !Task.isCancelled else { return false }
        return primaryReady || !runnableSlots.isEmpty
    }

    private static func hasLoadedChatRuntime(appState: AppState, stored: [StoredModel]) async -> Bool {
        let preferredID = appState.activeChatModelID
        if let preferredID,
           let preferred = stored.first(where: { $0.id.uuidString == preferredID && $0.modelRole == .chat }) {
            let resolvedPath = ModelStorage.resolvedModelURL(from: preferred.localPath, fileName: preferred.fileName).path
            return await AppLlamaService.shared.loadedChatPath == resolvedPath
        }
        return await AppLlamaService.shared.isChatLoaded
    }

    @discardableResult
    private static func ensurePrimaryChatLoaded(appState: AppState, stored: [StoredModel]) async -> Bool {
        let preferredID = appState.activeChatModelID
        if let preferredID,
           let preferred = stored.first(where: { $0.id.uuidString == preferredID && $0.modelRole == .chat }) {
            let resolvedPath = ModelStorage.resolvedModelURL(from: preferred.localPath, fileName: preferred.fileName).path
            guard FileManager.default.fileExists(atPath: resolvedPath) else { return false }
            if await AppLlamaService.shared.isChatLoaded,
               await AppLlamaService.shared.loadedChatPath == resolvedPath {
                return true
            }
        } else if await AppLlamaService.shared.isChatLoaded {
            return true
        }
        let candidates = stored.filter { $0.modelRole == .chat }
        SlotModelRuntimeCoordinator.shared.configure(
            assignments: [:],
            contextSize: appState.contextSize,
            preferExclusiveChatRuntime: true
        )
        return await SlotModelRuntimeCoordinator.shared.ensureChatModel(
            appState: appState,
            candidates: candidates,
            preferredID: preferredID
        )
    }

    @discardableResult
    static func ensureEmbedLoaded(appState: AppState, stored: [StoredModel], intent: ModelLoadIntent = .userChat) async -> Bool {
        if await hasLoadedEmbeddingRuntime(appState: appState, stored: stored) { return true }
        guard canStartModelLoad(intent: intent) else { return false }
        if let embedLoadTask { return await embedLoadTask.value }
        let task = Task { @MainActor in
            await performEnsureEmbedLoaded(appState: appState, stored: stored, intent: intent)
        }
        embedLoadTask = task
        let result = await task.value
        embedLoadTask = nil
        return result
    }

    private static func hasLoadedEmbeddingRuntime(appState: AppState, stored: [StoredModel]) async -> Bool {
        let preferredID = appState.activeEmbeddingModelID
        if let preferredID,
           let preferred = stored.first(where: { $0.id.uuidString == preferredID && $0.modelRole == .embedding }) {
            let resolvedPath = ModelStorage.resolvedModelURL(from: preferred.localPath, fileName: preferred.fileName).path
            return await AppLlamaService.shared.loadedEmbedPath == resolvedPath
        }
        return await AppLlamaService.shared.hasSemanticEmbeddingRuntime
    }

    private static func performEnsureEmbedLoaded(appState: AppState, stored: [StoredModel], intent: ModelLoadIntent) async -> Bool {
        guard !Task.isCancelled, ResourceBudgetGate.allowsForegroundModelLoad(reason: intent.rawValue) else { return false }
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
        SlotModelRuntimeCoordinator.shared.configure(
            assignments: SlotModelRuntimeCoordinator.shared.configuredAssignments,
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
