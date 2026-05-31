import Foundation
import SwiftData
import OSLog

@MainActor
enum ModelLoader {
    static func syncChat(appState: AppState, stored: [StoredModel]) async {
        await ensureFleetChatLoaded(appState: appState, stored: stored)
    }

    static func syncEmbed(appState: AppState, stored: [StoredModel]) async {
        await ensureEmbedLoaded(appState: appState, stored: stored)
    }

    /// Launch registration is assignment-first. Large role-baked GGUFs are not all
    /// preloaded; the runtime lazily loads the specific slot that is about to run.
    static func loadAtLaunch(appState: AppState, stored: [StoredModel]) async {
        async let chat = ensureFleetChatLoaded(appState: appState, stored: stored)
        async let embed = ensureEmbedLoaded(appState: appState, stored: stored)
        _ = await (chat, embed)
    }

    /// Backward-compatible entry point. In v1 this registers all available chat
    /// assignments for on-demand slot loading instead of loading all contexts.
    @discardableResult
    static func ensureChatLoaded(appState: AppState, stored: [StoredModel]) async -> Bool {
        await ensureFleetChatLoaded(appState: appState, stored: stored)
    }

    @discardableResult
    static func ensureFleetChatLoaded(appState: AppState, stored: [StoredModel]) async -> Bool {
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
        return primaryReady || !runnableSlots.isEmpty
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
    static func ensureEmbedLoaded(appState: AppState, stored: [StoredModel]) async -> Bool {
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
        return await SlotModelRuntimeCoordinator.shared.ensureEmbeddingModel(
            appState: appState,
            candidates: candidates,
            preferredID: preferredID
        )
    }
}
