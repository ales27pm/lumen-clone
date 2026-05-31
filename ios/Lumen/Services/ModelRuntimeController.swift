import Foundation
import SwiftData

@MainActor
final class ModelRuntimeController {
    private var didStart = false
    private var refreshRequestID: UUID?

    func startupIfNeeded(refresh: @escaping () async -> Void) {
        guard !didStart else { return }
        didStart = true
        Task { await refresh() }
    }

    func refreshLoadedPaths() async -> Set<String>? {
        let requestID = UUID()
        refreshRequestID = requestID

        var set: Set<String> = []
        if let sharedPath = await AppLlamaService.shared.loadedChatPath,
           FileManager.default.fileExists(atPath: sharedPath) {
            let fileName = URL(fileURLWithPath: sharedPath).lastPathComponent
            set.insert(ModelStorage.resolvedModelURL(from: sharedPath, fileName: fileName).path)
        }
        let chatPaths = await AppLlamaService.shared.loadedChatPathsBySlot
        for path in chatPaths.values where FileManager.default.fileExists(atPath: path) {
            let fileName = URL(fileURLWithPath: path).lastPathComponent
            set.insert(ModelStorage.resolvedModelURL(from: path, fileName: fileName).path)
        }
        if let p = await AppLlamaService.shared.loadedEmbedPath,
           await AppLlamaService.shared.hasSemanticEmbeddingRuntime,
           FileManager.default.fileExists(atPath: p) {
            let fileName = URL(fileURLWithPath: p).lastPathComponent
            set.insert(ModelStorage.resolvedModelURL(from: p, fileName: fileName).path)
        }

        guard refreshRequestID == requestID else { return nil }
        return set
    }

    func load(_ sm: StoredModel, appState: AppState, storedModels: [StoredModel]) async throws {
        if sm.modelRole == .chat || sm.modelRole == .roleAdapter {
            await ModelLoader.ensureFleetChatLoaded(appState: appState, stored: storedModels)
        } else {
            let resolvedPath = ModelStorage.resolvedModelURL(from: sm.localPath, fileName: sm.fileName).path
            try await AppLlamaService.shared.loadEmbeddingModel(path: resolvedPath)
        }
    }

    func unload(_ sm: StoredModel, adapterSlot: (StoredModel) -> LumenModelSlot?) async {
        if sm.modelRole == .chat {
            let resolvedPath = ModelStorage.resolvedModelURL(from: sm.localPath, fileName: sm.fileName).path
            if await AppLlamaService.shared.loadedChatPath == resolvedPath {
                await AppLlamaService.shared.unloadAllChat()
                return
            }
            let slots = await AppLlamaService.shared.loadedChatPathsBySlot.filter { $0.value == resolvedPath }.map(\.key)
            for slot in slots { await AppLlamaService.shared.unloadChat(for: slot) }
        } else if sm.modelRole == .roleAdapter {
            if let slot = adapterSlot(sm) { await AppLlamaService.shared.unloadRoleAdapter(slot: slot) }
        } else {
            await AppLlamaService.shared.unloadEmbed()
        }
    }

    func reload(_ sm: StoredModel, appState: AppState, storedModels: [StoredModel]) async throws {
        if sm.modelRole == .chat || sm.modelRole == .roleAdapter {
            await ModelLoader.ensureFleetChatLoaded(appState: appState, stored: storedModels)
        } else {
            try await AppLlamaService.shared.reloadEmbed()
        }
    }
}
