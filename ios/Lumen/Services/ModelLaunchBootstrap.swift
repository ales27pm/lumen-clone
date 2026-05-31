import Foundation
import SwiftData
import OSLog

@MainActor
enum ModelLaunchBootstrap {
    private static let logger = Logger(subsystem: "ai.lumen.app", category: "persistence")

    private static func persist(_ context: ModelContext, operation: String, scope: String) throws {
        do { try context.save() } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error=\(String(describing: error), privacy: .public)")
            throw error
        }
    }

    static func auditPersistence(operation: String, scope: String, save: () throws -> Void) -> Bool {
        do {
            try save()
            return true
        } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error=\(String(describing: error), privacy: .public)")
            return false
        }
    }

    private static let storageSafetyBufferBytes: Int64 = 500_000_000

    static func ensureFleetDownloaded(appState: AppState, context: ModelContext) async {
        guard appState.autoDownloadFleetModels else {
            appState.runtime.updateBootStep(id: "models", detail: "Fleet auto-download disabled", state: .warning)
            linkExistingFleetFiles(appState: appState, context: context)
            return
        }
        guard !appState.confirmFleetDownloads else {
            appState.runtime.updateBootStep(id: "models", detail: "Fleet download waiting for manual repair", state: .warning)
            linkExistingFleetFiles(appState: appState, context: context)
            return
        }
        await repairFleet(appState: appState, context: context, source: .launch)
    }

    static func repairFleet(appState: AppState, context: ModelContext, source: RepairSource = .manual) async {
        let family = LumenModelFamily.persistedSelected
        let models = fleetModelsForInstall(family: family)
        guard !models.isEmpty else {
            appState.runtime.updateBootStep(id: "models", detail: "No \(family.shortLabel) catalog entries", state: .warning)
            return
        }

        let missing = missingModels(from: models, context: context)
        let missingBytes = missing.reduce(Int64(0)) { $0 + $1.sizeBytes }
        let requiredBytes = max(0, missingBytes + (missing.isEmpty ? 0 : storageSafetyBufferBytes))
        let availableBytes = availableStorageBytes()

        if requiredBytes > 0, availableBytes < requiredBytes {
            appState.runtime.updateBootStep(
                id: "models",
                detail: "\(family.shortLabel): need \(formatBytesForBoot(requiredBytes)); only \(formatBytesForBoot(availableBytes)) free",
                state: .warning
            )
            linkExistingFleetFiles(appState: appState, context: context)
            return
        }

        appState.runtime.updateBootStep(
            id: "models",
            detail: source == .launch ? "Checking \(models.count) \(family.shortLabel) artifacts" : "Repairing \(models.count) \(family.shortLabel) artifacts",
            state: .running
        )

        var alreadyPresent = 0
        var startedDownloads = 0
        var linkedLocalFiles = 0

        for model in models {
            let result = ensureModelPresent(model, expectedFleetCount: models.count, appState: appState, context: context)
            switch result {
            case .alreadyStored, .alreadyDownloading:
                alreadyPresent += 1
            case .linkedLocalFile:
                linkedLocalFiles += 1
            case .startedDownload:
                startedDownloads += 1
            }
        }

        let fragments = [
            alreadyPresent > 0 ? "\(alreadyPresent) ready" : nil,
            linkedLocalFiles > 0 ? "\(linkedLocalFiles) linked" : nil,
            startedDownloads > 0 ? "\(startedDownloads) downloading" : nil
        ].compactMap { $0 }

        let detail = fragments.isEmpty ? "\(family.shortLabel) model check complete" : "\(family.shortLabel): " + fragments.joined(separator: " · ")
        appState.runtime.updateBootStep(id: "models", detail: detail, state: startedDownloads > 0 ? .running : .complete)
    }

    static func switchFamily(_ family: LumenModelFamily, appState: AppState, context: ModelContext) async {
        LumenModelFamily.persistedSelected = family
        appState.activeChatModelID = nil
        appState.activeEmbeddingModelID = nil
        await repairFleet(appState: appState, context: context, source: .manual)
    }

    enum RepairSource: Sendable {
        case launch
        case manual
    }

    private enum EnsureResult {
        case alreadyStored
        case linkedLocalFile
        case alreadyDownloading
        case startedDownload
    }

    private static func fleetModelsForInstall(family: LumenModelFamily = LumenModelFamily.persistedSelected) -> [CatalogModel] {
        uniqueByArtifact(LumenModelFleetCatalog.bootstrapModels(for: family))
    }

    private static func ensureModelPresent(
        _ model: CatalogModel,
        expectedFleetCount: Int,
        appState: AppState,
        context: ModelContext
    ) -> EnsureResult {
        let existingStored = storedModel(for: model, context: context)
        let localURL = ModelDownloader.shared.localURL(for: model)

        if FileManager.default.fileExists(atPath: localURL.path) {
            if existingStored == nil {
                guard let stored = insertStoredModel(for: model, localURL: localURL, appState: appState, context: context) else {
                    return .alreadyStored
                }
                Task { @MainActor in
                    await loadIfSelected(stored, appState: appState, context: context)
                    updateFleetBootProgress(expectedCount: expectedFleetCount, appState: appState, context: context)
                }
                return .linkedLocalFile
            } else if let existingStored {
                activateIfNeeded(existingStored, appState: appState)
                Task { @MainActor in
                    await loadIfSelected(existingStored, appState: appState, context: context)
                    updateFleetBootProgress(expectedCount: expectedFleetCount, appState: appState, context: context)
                }
            }
            return .alreadyStored
        }

        guard existingStored == nil || !FileManager.default.fileExists(atPath: existingStored?.localPath ?? "") else {
            if let existingStored {
                activateIfNeeded(existingStored, appState: appState)
                Task { @MainActor in
                    await loadIfSelected(existingStored, appState: appState, context: context)
                    updateFleetBootProgress(expectedCount: expectedFleetCount, appState: appState, context: context)
                }
            }
            return .alreadyStored
        }

        guard !ModelDownloader.shared.isDownloading(model) else { return .alreadyDownloading }

        ModelDownloader.shared.start(model) { localURL in
            Task { @MainActor in
                let stored: StoredModel
                if let existing = storedModel(for: model, context: context) {
                    activateIfNeeded(existing, appState: appState)
                    stored = existing
                } else {
                    guard let inserted = insertStoredModel(for: model, localURL: localURL, appState: appState, context: context) else {
                        return
                    }
                    stored = inserted
                }

                await loadIfSelected(stored, appState: appState, context: context)
                updateFleetBootProgress(expectedCount: expectedFleetCount, appState: appState, context: context)
            }
        }
        return .startedDownload
    }

    private static func linkExistingFleetFiles(appState: AppState, context: ModelContext) {
        for model in fleetModelsForInstall() {
            let localURL = ModelDownloader.shared.localURL(for: model)
            guard FileManager.default.fileExists(atPath: localURL.path) else { continue }
            if storedModel(for: model, context: context) == nil {
                _ = insertStoredModel(for: model, localURL: localURL, appState: appState, context: context)
            }
        }
    }

    private static func missingModels(from models: [CatalogModel], context: ModelContext) -> [CatalogModel] {
        models.filter { model in
            let localURL = ModelDownloader.shared.localURL(for: model)
            if FileManager.default.fileExists(atPath: localURL.path) { return false }
            return storedModel(for: model, context: context) == nil
        }
    }

    private static func loadIfSelected(_ stored: StoredModel, appState: AppState, context: ModelContext) async {
        let allStored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
        switch stored.modelRole {
        case .chat:
            guard appState.activeChatModelID == stored.id.uuidString else { return }
            _ = await ModelLoader.ensureChatLoaded(appState: appState, stored: allStored)
        case .embedding:
            guard appState.activeEmbeddingModelID == stored.id.uuidString else { return }
            _ = await ModelLoader.ensureEmbedLoaded(appState: appState, stored: allStored)
        case .roleAdapter:
            _ = await ModelLoader.ensureChatLoaded(appState: appState, stored: allStored)
        }
    }

    private static func updateFleetBootProgress(expectedCount: Int, appState: AppState, context: ModelContext) {
        let readyCount = readyFleetArtifactCount(context: context)
        let state: BootStepState = readyCount >= expectedCount ? .complete : .running
        appState.runtime.updateBootStep(
            id: "models",
            detail: "\(min(readyCount, expectedCount)) / \(expectedCount) \(LumenModelFamily.persistedSelected.shortLabel) artifacts ready",
            state: state
        )
    }

    private static func readyFleetArtifactCount(context: ModelContext) -> Int {
        let stored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
        let installedKeys = Set(stored.map { artifactKey(repoId: $0.repoId, fileName: $0.fileName) })
        return fleetModelsForInstall().reduce(0) { count, model in
            let localReady = FileManager.default.fileExists(atPath: ModelDownloader.shared.localURL(for: model).path)
            let storedReady = installedKeys.contains(artifactKey(repoId: model.repoId, fileName: model.fileName))
            return localReady || storedReady ? count + 1 : count
        }
    }

    private static func storedModel(for catalog: CatalogModel, context: ModelContext) -> StoredModel? {
        let models = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
        return models.first { stored in
            artifactKey(repoId: stored.repoId, fileName: stored.fileName) == artifactKey(repoId: catalog.repoId, fileName: catalog.fileName)
        }
    }

    @discardableResult
    private static func insertStoredModel(for catalog: CatalogModel, localURL: URL, appState: AppState, context: ModelContext) -> StoredModel? {
        let stored = StoredModel(
            name: catalog.name,
            repoId: catalog.repoId,
            fileName: catalog.fileName,
            sizeBytes: catalog.sizeBytes,
            quantization: catalog.quantization,
            parameters: catalog.parameters,
            role: catalog.role,
            localPath: localURL.path
        )
        context.insert(stored)
        do {
            try persist(context, operation: "insertStoredModel", scope: "StoredModel")
        } catch {
            logger.error("persist_blocked op=insertStoredModel scope=StoredModel error=\(String(describing: error), privacy: .public)")
            context.delete(stored)
            return nil
        }
        activateIfNeeded(stored, appState: appState)
        return stored
    }

    private static func activateIfNeeded(_ stored: StoredModel, appState: AppState) {
        switch stored.modelRole {
        case .chat:
            if appState.activeChatModelID == nil {
                appState.activeChatModelID = stored.id.uuidString
            }
        case .embedding:
            if appState.activeEmbeddingModelID == nil {
                appState.activeEmbeddingModelID = stored.id.uuidString
            }
        case .roleAdapter:
            break
        }
    }

    private static func uniqueByArtifact(_ models: [CatalogModel]) -> [CatalogModel] {
        var seen: Set<String> = []
        var unique: [CatalogModel] = []
        unique.reserveCapacity(models.count)

        for model in models {
            let key = artifactKey(repoId: model.repoId, fileName: model.fileName)
            guard seen.insert(key).inserted else { continue }
            unique.append(model)
        }

        return unique
    }

    private static func availableStorageBytes(fileManager: FileManager = .default) -> Int64 {
        let url = ModelStorage.modelsDirectoryURL(fileManager: fileManager)
        if let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
           let important = values.volumeAvailableCapacityForImportantUsage {
            return important
        }
        if let attrs = try? fileManager.attributesOfFileSystem(forPath: url.path),
           let free = attrs[.systemFreeSize] as? NSNumber {
            return free.int64Value
        }
        return 0
    }

    private static func formatBytesForBoot(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    private static func artifactKey(repoId: String, fileName: String) -> String {
        "\(repoId.lowercased())/\(fileName.lowercased())"
    }
}
