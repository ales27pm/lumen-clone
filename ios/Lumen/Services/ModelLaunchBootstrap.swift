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

        // Fetch all stored models once to avoid repeated O(n) fetches in the loop below.
        let allStored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []

        let missing = missingModels(from: models, allStored: allStored)
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
            let result = ensureModelPresent(model, expectedFleetCount: models.count, appState: appState, context: context, allStored: allStored)
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

    static func prepareLiveRuntimeArtifacts(appState: AppState, context: ModelContext, timeoutSeconds: TimeInterval = 300) async -> Bool {
        let family = LumenModelFamily.persistedSelected
        let models = liveRuntimeModelsForInstall(family: family)
        guard !models.isEmpty else {
            appState.runtime.updateBootStep(id: "models", detail: "No \(family.shortLabel) live runtime artifacts", state: .warning)
            return false
        }

        guard appState.autoDownloadFleetModels else {
            appState.runtime.updateBootStep(id: "models", detail: "Fleet auto-download disabled", state: .warning)
            linkExistingFleetFiles(appState: appState, context: context)
            return readyArtifactCount(for: models, context: context) >= models.count
        }
        guard !appState.confirmFleetDownloads else {
            appState.runtime.updateBootStep(id: "models", detail: "Fleet download waiting for manual repair", state: .warning)
            linkExistingFleetFiles(appState: appState, context: context)
            return readyArtifactCount(for: models, context: context) >= models.count
        }

        let allStored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
        let missing = missingModels(from: models, allStored: allStored)
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
            return readyArtifactCount(for: models, context: context) >= models.count
        }

        let deadline = Date().addingTimeInterval(timeoutSeconds)
        var startedDownloads = startMissingLiveRuntimeDownloads(models: models, appState: appState, context: context)
        while !Task.isCancelled {
            let readyCount = readyArtifactCount(for: models, context: context)
            if readyCount >= models.count {
                appState.runtime.updateBootStep(id: "models", detail: "\(family.shortLabel): \(readyCount) / \(models.count) live runtime artifacts ready", state: .complete)
                return true
            }

            appState.runtime.updateBootStep(
                id: "models",
                detail: startedDownloads > 0
                    ? "\(family.shortLabel): \(readyCount) / \(models.count) live runtime artifacts ready · \(startedDownloads) downloading"
                    : "\(family.shortLabel): \(readyCount) / \(models.count) live runtime artifacts ready",
                state: .running
            )

            if Date() >= deadline {
                appState.runtime.updateBootStep(id: "models", detail: "\(family.shortLabel): timed out waiting for live runtime artifacts", state: .warning)
                return false
            }

            try? await Task.sleep(nanoseconds: 1_000_000_000)
            if !hasActiveDownloads(for: models) {
                startedDownloads = startMissingLiveRuntimeDownloads(models: models, appState: appState, context: context)
            }
        }

        return false
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

    static func liveRuntimeModelsForInstall(family: LumenModelFamily = LumenModelFamily.persistedSelected) -> [CatalogModel] {
        let bootstrap = LumenModelFleetCatalog.bootstrapModels(for: family)
        switch family {
        case .qwen25:
            return uniqueByArtifact(bootstrap.filter { $0.role == .chat })
        case .qwen3:
            let contract = LumenTrainedModelRuntimeRegistry.contract(for: family)
            let requiredAdapterFiles = Set(contract.adapterRoles.compactMap { role in
                role.slot == nil ? nil : role.adapterFileName
            })
            return uniqueByArtifact(bootstrap.filter { model in
                if model.role == .chat {
                    return contract.matchesSharedBase(repoID: model.repoId, fileName: model.fileName)
                }
                if model.role == .roleAdapter {
                    return requiredAdapterFiles.contains(model.fileName)
                }
                return false
            })
        }
    }

    static func liveRuntimeArtifactReadiness(context: ModelContext, family: LumenModelFamily = LumenModelFamily.persistedSelected) -> (ready: Int, required: Int) {
        let models = liveRuntimeModelsForInstall(family: family)
        return (readyArtifactCount(for: models, context: context), models.count)
    }

    private static func startMissingLiveRuntimeDownloads(models: [CatalogModel], appState: AppState, context: ModelContext) -> Int {
        let allStored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
        var started = 0
        for model in missingModels(from: models, allStored: allStored) {
            switch ensureModelPresent(model, expectedFleetCount: models.count, appState: appState, context: context, allStored: allStored) {
            case .startedDownload:
                started += 1
            default:
                break
            }
        }
        return started
    }

    private static func ensureModelPresent(
        _ model: CatalogModel,
        expectedFleetCount: Int,
        appState: AppState,
        context: ModelContext,
        allStored: [StoredModel]
    ) -> EnsureResult {
        let existingStored = storedModel(for: model, in: allStored)
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
                let freshStored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
                let stored: StoredModel
                if let existing = storedModel(for: model, in: freshStored) {
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

    private static func missingModels(from models: [CatalogModel], allStored: [StoredModel]) -> [CatalogModel] {
        models.filter { model in
            let localURL = ModelDownloader.shared.localURL(for: model)
            if FileManager.default.fileExists(atPath: localURL.path) { return false }
            return !(storedModel(for: model, in: allStored).map(storedModelFileExists) ?? false)
        }
    }

    private static func storedModelFileExists(_ stored: StoredModel) -> Bool {
        FileManager.default.fileExists(atPath: ModelStorage.resolvedModelURL(from: stored.localPath, fileName: stored.fileName).path)
    }

    private static func loadIfSelected(_ stored: StoredModel, appState: AppState, context: ModelContext) async {
        let allStored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
        switch stored.modelRole {
        case .chat:
            guard appState.activeChatModelID == stored.id.uuidString else { return }
            _ = await ModelLoader.ensureChatLoaded(appState: appState, stored: allStored, intent: .appStartup)
        case .embedding:
            guard appState.activeEmbeddingModelID == stored.id.uuidString else { return }
            _ = await ModelLoader.ensureEmbedLoaded(appState: appState, stored: allStored, intent: .appStartup)
        case .roleAdapter:
            _ = await ModelLoader.ensureChatLoaded(appState: appState, stored: allStored, intent: .appStartup)
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
        readyArtifactCount(for: fleetModelsForInstall(), context: context)
    }

    private static func readyArtifactCount(for models: [CatalogModel], context: ModelContext) -> Int {
        let stored = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
        return models.reduce(0) { count, model in
            let localReady = FileManager.default.fileExists(atPath: ModelDownloader.shared.localURL(for: model).path)
            let storedReady = storedModel(for: model, in: stored).map(storedModelFileExists) ?? false
            return localReady || storedReady ? count + 1 : count
        }
    }

    private static func hasActiveDownloads(for models: [CatalogModel]) -> Bool {
        models.contains { ModelDownloader.shared.isDownloading($0) }
    }

    private static func storedModel(for catalog: CatalogModel, context: ModelContext) -> StoredModel? {
        let models = (try? context.fetch(FetchDescriptor<StoredModel>())) ?? []
        return storedModel(for: catalog, in: models)
    }

    private static func storedModel(for catalog: CatalogModel, in models: [StoredModel]) -> StoredModel? {
        models.first { stored in
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
