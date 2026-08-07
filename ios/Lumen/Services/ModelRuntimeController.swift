import Foundation
import SwiftData

@MainActor
final class ModelRuntimeController {
    enum ChatUnloadPlan: Equatable {
        case allChat
        case slots([LumenModelSlot])
        case none
    }

    enum Failure: LocalizedError {
        case runtimeDidNotLoad(ModelRole)

        var errorDescription: String? {
            switch self {
            case .runtimeDidNotLoad(.chat):
                return "The selected chat model could not be loaded."
            case .runtimeDidNotLoad(.embedding):
                return "The selected embedding model could not be loaded."
            case .runtimeDidNotLoad(.roleAdapter):
                return "The selected role adapter could not be loaded."
            }
        }
    }

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
            guard await ModelLoader.ensureFleetChatLoaded(appState: appState, stored: storedModels, intent: .userChat) else {
                throw Failure.runtimeDidNotLoad(sm.modelRole)
            }
        } else {
            guard await ModelLoader.ensureEmbedLoaded(appState: appState, stored: storedModels, intent: .userChat) else {
                throw Failure.runtimeDidNotLoad(.embedding)
            }
        }
    }

    func unload(_ sm: StoredModel, adapterSlot: (StoredModel) -> LumenModelSlot?) async {
        let resolvedPath = ModelStorage.resolvedModelURL(from: sm.localPath, fileName: sm.fileName).path
        await unloadResolvedModel(
            role: sm.modelRole,
            resolvedPath: resolvedPath,
            adapterSlot: sm.modelRole == .roleAdapter ? adapterSlot(sm) : nil
        )
    }

    func unloadResolvedModel(
        role: ModelRole,
        resolvedPath: String,
        adapterSlot: LumenModelSlot?
    ) async {
        if role == .chat {
            let loadedSharedPath = await AppLlamaService.shared.loadedChatPath
            let loadedSlotPaths = await AppLlamaService.shared.loadedChatPathsBySlot
            switch Self.chatUnloadPlan(
                resolvedPath: resolvedPath,
                loadedSharedPath: loadedSharedPath,
                loadedSlotPaths: loadedSlotPaths
            ) {
            case .allChat:
                await AppLlamaService.shared.unloadAllChat()
            case .slots(let slots):
                for slot in slots { await AppLlamaService.shared.unloadChat(for: slot) }
            case .none:
                break
            }
        } else if role == .roleAdapter {
            if let adapterSlot { await AppLlamaService.shared.unloadRoleAdapter(slot: adapterSlot) }
        } else {
            await AppLlamaService.shared.unloadEmbed()
        }
    }

    static func chatUnloadPlan(
        resolvedPath: String,
        loadedSharedPath: String?,
        loadedSlotPaths: [LumenModelSlot: String]
    ) -> ChatUnloadPlan {
        if loadedSharedPath == resolvedPath {
            return .allChat
        }
        let slots = loadedSlotPaths
            .filter { $0.value == resolvedPath }
            .map(\.key)
            .sorted { $0.rawValue < $1.rawValue }
        return slots.isEmpty ? .none : .slots(slots)
    }

    func reload(_ sm: StoredModel, appState: AppState, storedModels: [StoredModel]) async throws {
        await unload(sm) { model in
            let text = [model.name, model.fileName, model.localPath].joined(separator: " ").lowercased()
            return [LumenModelSlot.cortex, .executor, .mouth, .mimicry, .rem]
                .first(where: { text.contains($0.rawValue) })
        }
        try await load(sm, appState: appState, storedModels: storedModels)
    }
}
