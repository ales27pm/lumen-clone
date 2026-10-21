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
        case selectionSuperseded

        var errorDescription: String? {
            switch self {
            case .runtimeDidNotLoad(.chat):
                return "The selected chat model could not be loaded."
            case .runtimeDidNotLoad(.embedding):
                return "The selected embedding model could not be loaded."
            case .runtimeDidNotLoad(.roleAdapter):
                return "The selected role adapter could not be loaded."
            case .selectionSuperseded:
                return "A newer model selection replaced this request."
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

    /// A legacy selection owns its provider revision and persisted IDs until
    /// another user selection supersedes it. Restore only that owned revision;
    /// a failed old load must never undo a newer selection.
    func preservingEmbeddingSelection(
        for role: ModelRole,
        readSelection: () -> CoreAITextEmbeddingModelStore.SelectionSnapshot = { CoreAITextEmbeddingModelStore.selectionSnapshot },
        chooseLegacy: () async throws -> CoreAITextEmbeddingModelStore.SelectionSnapshot = { try await CoreAITextEmbeddingModelStore.setSelected(false) },
        restoreSelection: (Bool, CoreAITextEmbeddingModelStore.SelectionSnapshot) async throws -> Bool = {
            try await CoreAITextEmbeddingModelStore.restoreSelection($0, ifUnchanged: $1)
        },
        isCurrent: () -> Bool = { true },
        rollback: () throws -> Void = {},
        operation: () async throws -> Void
    ) async throws {
        try Task.checkCancellation()
        let previousSelection = readSelection()
        let attemptedSelection: CoreAITextEmbeddingModelStore.SelectionSnapshot?
        if role == .embedding { attemptedSelection = try await chooseLegacy() }
        else { attemptedSelection = nil }
        do {
            try Task.checkCancellation()
            guard isCurrent(), attemptedSelection.map({ readSelection() == $0 }) ?? true else {
                throw Failure.selectionSuperseded
            }
            try await operation()
            try Task.checkCancellation()
            guard isCurrent(), attemptedSelection.map({ readSelection() == $0 }) ?? true else {
                throw Failure.selectionSuperseded
            }
        } catch {
            if let attemptedSelection {
                guard try await restoreSelection(previousSelection.selected, attemptedSelection) else { throw error }
                // The provider restore awaits its actor gate. IDs may have been
                // changed by a newer request during that suspension.
            }
            guard isCurrent() else { throw error }
            try rollback()
            throw error
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
