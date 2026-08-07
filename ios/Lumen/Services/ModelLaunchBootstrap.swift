import Foundation
import SwiftData
import OSLog

nonisolated struct LiveRuntimeArtifactReadiness: Sendable, Equatable {
    let ready: Int
    let required: Int
    let missingAdapterSlots: [String]
    let missingArtifactFileNames: [String]
    let diagnostic: String?
}

nonisolated struct ModelFamilyProvisioningResult: Sendable, Equatable {
    let ready: Int
    let required: Int
    let errorMessage: String?

    var succeeded: Bool { ready == required && required > 0 && errorMessage == nil }
}

nonisolated enum ModelProvisioningAuthorization: Sendable, Equatable {
    case persistedConsent
    case explicitUserConsent
}

@MainActor
enum ModelLaunchBootstrap {
    private static let logger = Logger(subsystem: "ai.lumen.app", category: "persistence")

    private static func persist(_ context: ModelContext, operation: String, scope: String) throws {
        do { try context.save() } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            throw error
        }
    }

    static func auditPersistence(operation: String, scope: String, save: () throws -> Void) -> Bool {
        do {
            try save()
            return true
        } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return false
        }
    }

    static func modelCatalogFetchFailureMessage(error: Error) -> String {
        "Model catalog fetch failed (\(RuntimeMetricErrorSanitizer.code(for: error)))."
    }

    private static func fetchStoredModels(context: ModelContext, operation: String, appState: AppState? = nil) -> [StoredModel]? {
        do {
            return try context.fetch(FetchDescriptor<StoredModel>())
        } catch {
            logger.error("fetch_failed op=\(operation, privacy: .public) scope=StoredModel error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            appState?.runtime.updateBootStep(id: "models", detail: modelCatalogFetchFailureMessage(error: error), state: .warning)
            return nil
        }
    }

    private static let storageSafetyBufferBytes: Int64 = 500_000_000

    static func ensureFleetDownloaded(appState: AppState, context: ModelContext) async {
        guard appState.autoDownloadFleetModels else {
            appState.runtime.updateBootStep(id: "models", detail: "Fleet auto-download disabled", state: .warning)
            await linkExistingFleetFiles(appState: appState, context: context)
            return
        }
        guard !appState.confirmFleetDownloads else {
            appState.runtime.updateBootStep(id: "models", detail: "Fleet download waiting for manual repair", state: .warning)
            await linkExistingFleetFiles(appState: appState, context: context)
            return
        }
        let family = LumenModelFamily.persistedSelected
        guard ModelProvisioningReceipt.isConsented(family: family) else {
            appState.runtime.updateBootStep(id: "models", detail: "Model download requires explicit consent", state: .warning)
            await linkExistingFleetFiles(appState: appState, context: context)
            return
        }
        _ = await provisionSelectedFamily(family: family, appState: appState, context: context)
    }

    static func repairFleet(appState: AppState, context: ModelContext, source: RepairSource = .manual) async {
        let family = LumenModelFamily.persistedSelected
        if source == .manual {
            guard ModelProvisioningReceipt.markConsented(family: family) else {
                appState.runtime.updateBootStep(id: "models", detail: "Could not record model download consent", state: .failed)
                return
            }
        } else if !ModelProvisioningReceipt.isConsented(family: family) {
            appState.runtime.updateBootStep(id: "models", detail: "Model download requires explicit consent", state: .warning)
            return
        }
        _ = await provisionSelectedFamily(family: family, appState: appState, context: context)
    }

    static func provisionSelectedFamily(
        family: LumenModelFamily = LumenModelFamily.persistedSelected,
        appState: AppState,
        context: ModelContext,
        timeoutSeconds: TimeInterval = 3_600,
        authorization: ModelProvisioningAuthorization = .persistedConsent
    ) async -> ModelFamilyProvisioningResult {
        let models = provisioningModelsForInstall(family: family)
        guard !models.isEmpty else {
            return ModelFamilyProvisioningResult(ready: 0, required: 0, errorMessage: "No verified artifacts are configured for \(family.shortLabel).")
        }
        let isAuthorized = authorization == .explicitUserConsent
            || ModelProvisioningReceipt.isConsented(family: family)
        guard isAuthorized else {
            return ModelFamilyProvisioningResult(
                ready: 0,
                required: models.count,
                errorMessage: "Confirm the verified \(family.shortLabel) download before setup begins."
            )
        }

        if let errorMessage = await startProvisioningDownloads(models: models, appState: appState, context: context) {
            return ModelFamilyProvisioningResult(ready: 0, required: models.count, errorMessage: errorMessage)
        }
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        var idlePasses = 0

        while !Task.isCancelled {
            guard let stored = fetchStoredModels(context: context, operation: "provisionSelectedFamily", appState: appState) else {
                return ModelFamilyProvisioningResult(ready: 0, required: models.count, errorMessage: "Could not read the installed model catalog.")
            }
            let persistedReady = await persistedArtifactCount(for: models, stored: stored)
            if persistedReady == models.count {
                guard let provisionedSelection = provisionedRoleIDs(models: models, stored: stored) else {
                    return ModelFamilyProvisioningResult(ready: persistedReady, required: models.count, errorMessage: "Verified chat and embedding selections could not be restored.")
                }
                let previousSelection = PersistedModelSelectionStore.loadOrMigrate()
                let planID = ModelProvisioningReceipt.catalogIdentity(for: family)
                guard ModelProvisioningSwitchJournalStore.prepare(targetFamily: family) else {
                    return ModelFamilyProvisioningResult(
                        ready: persistedReady,
                        required: models.count,
                        errorMessage: "Verified models were installed, but the atomic selection update could not begin."
                    )
                }
                do {
                    try appState.commitActiveModelSelection(
                        chatModelID: provisionedSelection.chatModelID,
                        embeddingModelID: provisionedSelection.embeddingModelID,
                        family: family,
                        provisioningPlanID: planID
                    )
                } catch {
                    _ = ModelProvisioningSwitchJournalStore.rollback()
                    return ModelFamilyProvisioningResult(
                        ready: persistedReady,
                        required: models.count,
                        errorMessage: "Verified models were installed, but the paired selection could not be saved."
                    )
                }

                ModelLoader.cancelActiveLoads()
                await unloadResidentModelRuntimes()
                let loadResult = await loadProvisionedSelection(appState: appState, stored: stored)
                guard loadResult.allSelectedModelsLoaded else {
                    await restorePreviousSelection(previousSelection, appState: appState, stored: stored)
                    _ = ModelProvisioningSwitchJournalStore.rollback()
                    return ModelFamilyProvisioningResult(
                        ready: persistedReady,
                        required: models.count,
                        errorMessage: "Verified models were installed, but the chat and embedding runtimes did not both initialize. Retry while the app is active and the device has enough free memory."
                    )
                }

                guard ModelProvisioningReceipt.markCurrent(
                    family: family,
                    chatModelID: provisionedSelection.chatModelID,
                    embeddingModelID: provisionedSelection.embeddingModelID
                ) else {
                    await restorePreviousSelection(previousSelection, appState: appState, stored: stored)
                    _ = ModelProvisioningSwitchJournalStore.rollback()
                    return ModelFamilyProvisioningResult(
                        ready: persistedReady,
                        required: models.count,
                        errorMessage: "The verified setup receipt could not be saved."
                    )
                }
                guard ModelProvisioningReceipt.isCurrent(
                    family: family,
                    chatModelID: provisionedSelection.chatModelID,
                    embeddingModelID: provisionedSelection.embeddingModelID
                ), ModelProvisioningSwitchJournalStore.markCommitted(
                    targetFamily: family,
                    chatModelID: provisionedSelection.chatModelID,
                    embeddingModelID: provisionedSelection.embeddingModelID,
                    provisioningPlanID: planID
                ) else {
                    await restorePreviousSelection(previousSelection, appState: appState, stored: stored)
                    _ = ModelProvisioningSwitchJournalStore.rollback()
                    return ModelFamilyProvisioningResult(
                        ready: persistedReady,
                        required: models.count,
                        errorMessage: "The verified setup transaction could not be committed."
                    )
                }
                _ = ModelProvisioningSwitchJournalStore.clearCommitted()
                appState.runtime.modelAutoloadState = .finished(chatLoaded: true, embeddingLoaded: true)
                appState.runtime.updateBootStep(
                    id: "models",
                    detail: "\(family.shortLabel): \(persistedReady) / \(models.count) verified artifacts ready · chat and embedding loaded",
                    state: .complete
                )
                return ModelFamilyProvisioningResult(ready: persistedReady, required: models.count, errorMessage: nil)
            }

            if let failureMessage = models.compactMap({ model -> String? in
                guard let progress = ModelDownloader.shared.progresses[model.id],
                      case .failed(let message) = progress.state else { return nil }
                return "\(model.name): \(message)"
            }).first {
                return ModelFamilyProvisioningResult(ready: persistedReady, required: models.count, errorMessage: failureMessage)
            }

            if hasActiveDownloads(for: models) {
                idlePasses = 0
            } else {
                idlePasses += 1
                if idlePasses == 2 {
                    if let errorMessage = await startProvisioningDownloads(models: models, appState: appState, context: context) {
                        return ModelFamilyProvisioningResult(ready: persistedReady, required: models.count, errorMessage: errorMessage)
                    }
                } else if idlePasses >= 8 {
                    return ModelFamilyProvisioningResult(
                        ready: persistedReady,
                        required: models.count,
                        errorMessage: "Model setup stopped before every verified artifact was installed. Check storage and network access, then retry."
                    )
                }
            }

            if Date() >= deadline {
                return ModelFamilyProvisioningResult(
                    ready: persistedReady,
                    required: models.count,
                    errorMessage: "Model setup timed out before every verified artifact was installed."
                )
            }

            do {
                try await Task.sleep(nanoseconds: 500_000_000)
            } catch {
                return ModelFamilyProvisioningResult(ready: persistedReady, required: models.count, errorMessage: "Model setup was cancelled.")
            }
        }

        return ModelFamilyProvisioningResult(ready: 0, required: models.count, errorMessage: "Model setup was cancelled.")
    }

    private static func startProvisioningDownloads(
        models: [CatalogModel],
        appState: AppState,
        context: ModelContext
    ) async -> String? {
        guard let stored = fetchStoredModels(context: context, operation: "startProvisioningDownloads", appState: appState) else {
            return "Could not read the installed model catalog."
        }
        let missing = await missingModels(from: models, allStored: stored)
        let requiredBytes = missing.reduce(Int64(0)) { $0 + $1.sizeBytes }
            + (missing.isEmpty ? 0 : storageSafetyBufferBytes)
        let availableBytes = availableStorageBytes()
        guard requiredBytes <= availableBytes else {
            return "Model setup needs \(formatBytesForBoot(requiredBytes)); only \(formatBytesForBoot(availableBytes)) is free."
        }

        for model in models {
            let result = await ensureModelPresent(
                model,
                expectedFleetCount: models.count,
                appState: appState,
                context: context,
                allStored: stored
            )
            if result == .failed {
                return "Could not start the verified download for \(model.name)."
            }
        }
        return nil
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
            await linkExistingFleetFiles(appState: appState, context: context)
            guard let ready = await readyArtifactCount(for: models, context: context, operation: "prepareLiveRuntimeArtifacts.autoDownloadDisabled", appState: appState) else {
                return false
            }
            return ready >= models.count
        }
        guard ModelProvisioningReceipt.isConsented(family: family) else {
            appState.runtime.updateBootStep(id: "models", detail: "Model download requires explicit consent", state: .warning)
            await linkExistingFleetFiles(appState: appState, context: context)
            guard let ready = await readyArtifactCount(
                for: models,
                context: context,
                operation: "prepareLiveRuntimeArtifacts.consentRequired",
                appState: appState
            ) else {
                return false
            }
            return ready >= models.count
        }
        guard !appState.confirmFleetDownloads else {
            appState.runtime.updateBootStep(id: "models", detail: "Fleet download waiting for manual repair", state: .warning)
            await linkExistingFleetFiles(appState: appState, context: context)
            guard let ready = await readyArtifactCount(for: models, context: context, operation: "prepareLiveRuntimeArtifacts.confirmFleetDownloads", appState: appState) else {
                return false
            }
            return ready >= models.count
        }

        guard let allStored = fetchStoredModels(context: context, operation: "prepareLiveRuntimeArtifacts", appState: appState) else {
            return false
        }
        let missing = await missingModels(from: models, allStored: allStored)
        let missingBytes = missing.reduce(Int64(0)) { $0 + $1.sizeBytes }
        let requiredBytes = max(0, missingBytes + (missing.isEmpty ? 0 : storageSafetyBufferBytes))
        let availableBytes = availableStorageBytes()
        if requiredBytes > 0, availableBytes < requiredBytes {
            appState.runtime.updateBootStep(
                id: "models",
                detail: "\(family.shortLabel): need \(formatBytesForBoot(requiredBytes)); only \(formatBytesForBoot(availableBytes)) free",
                state: .warning
            )
            await linkExistingFleetFiles(appState: appState, context: context)
            guard let ready = await readyArtifactCount(for: models, context: context, operation: "prepareLiveRuntimeArtifacts.storagePressure", appState: appState) else {
                return false
            }
            return ready >= models.count
        }

        let deadline = Date().addingTimeInterval(timeoutSeconds)
        var startedDownloads = await startMissingLiveRuntimeDownloads(models: models, appState: appState, context: context)
        while !Task.isCancelled {
            guard let readyCount = await readyArtifactCount(for: models, context: context, operation: "prepareLiveRuntimeArtifacts.pollReady", appState: appState) else {
                return false
            }
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
                startedDownloads = await startMissingLiveRuntimeDownloads(models: models, appState: appState, context: context)
            }
        }

        return false
    }

    static func switchFamily(
        _ family: LumenModelFamily,
        appState: AppState,
        context: ModelContext
    ) async -> ModelFamilyProvisioningResult {
        await provisionSelectedFamily(
            family: family,
            appState: appState,
            context: context,
            authorization: .explicitUserConsent
        )
    }

    enum RepairSource: Sendable, Equatable {
        case launch
        case manual
    }

    private enum EnsureResult: Equatable {
        case alreadyStored
        case linkedLocalFile
        case alreadyDownloading
        case startedDownload
        case failed
    }

    private static func fleetModelsForInstall(family: LumenModelFamily = LumenModelFamily.persistedSelected) -> [CatalogModel] {
        uniqueByArtifact(LumenModelFleetCatalog.bootstrapModels(for: family))
    }

    static func provisioningModelsForInstall(
        family: LumenModelFamily = LumenModelFamily.persistedSelected
    ) -> [CatalogModel] {
        let bootstrap = uniqueByArtifact(LumenModelFleetCatalog.bootstrapModels(for: family))
        guard family == .qwen3 else { return bootstrap }
        let runtimeAdapterFiles = Set(
            LumenTrainedModelRuntimeRegistry.contract(for: family).adapterRoles.compactMap { role in
                role.slot == nil ? nil : role.adapterFileName
            }
        )
        return bootstrap.filter { model in
            model.role != .roleAdapter || runtimeAdapterFiles.contains(model.fileName)
        }
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
                    return contract.matchesSharedBase(
                        repoID: model.repoId,
                        fileName: model.fileName,
                        sizeBytes: model.sizeBytes,
                        expectedSHA256: model.expectedSHA256
                    )
                }
                if model.role == .roleAdapter {
                    return requiredAdapterFiles.contains(model.fileName)
                }
                return false
            })
        }
    }

    static func liveRuntimeArtifactReadiness(context: ModelContext, family: LumenModelFamily = LumenModelFamily.persistedSelected) async -> (ready: Int, required: Int) {
        let details = await liveRuntimeArtifactReadinessDetails(context: context, family: family)
        return (details.ready, details.required)
    }

    static func liveRuntimeArtifactReadinessDetails(context: ModelContext, family: LumenModelFamily = LumenModelFamily.persistedSelected) async -> LiveRuntimeArtifactReadiness {
        let models = liveRuntimeModelsForInstall(family: family)
        guard let stored = fetchStoredModels(context: context, operation: "liveRuntimeArtifactReadinessDetails") else {
            return LiveRuntimeArtifactReadiness(
                ready: 0,
                required: models.count,
                missingAdapterSlots: [],
                missingArtifactFileNames: [],
                diagnostic: "model_catalog_fetch_failed"
            )
        }
        let missingFiles = await missingModels(from: models, allStored: stored).map(\.fileName).sorted()
        var missingAdapterSlots: [String] = []
        if family == .qwen3 {
            let contract = LumenTrainedModelRuntimeRegistry.contract(for: family)
            for role in contract.adapterRoles {
                guard let slot = role.slot else { continue }
                let catalog = models.first { model in
                    artifactKey(repoId: model.repoId, fileName: model.fileName)
                        == artifactKey(repoId: role.adapterRepoID, fileName: role.adapterFileName)
                }
                let storedReady: Bool
                if let catalog,
                   let matchingStored = stored.first(where: { stored in
                    artifactKey(repoId: stored.repoId, fileName: stored.fileName) == artifactKey(repoId: role.adapterRepoID, fileName: role.adapterFileName)
                   }) {
                    storedReady = await storedModelFileIsValid(matchingStored, catalog: catalog)
                } else {
                    storedReady = false
                }
                if !storedReady { missingAdapterSlots.append(slot.rawValue) }
            }
            missingAdapterSlots.sort()
        }
        return LiveRuntimeArtifactReadiness(
            ready: await readyArtifactCount(for: models, stored: stored),
            required: models.count,
            missingAdapterSlots: missingAdapterSlots,
            missingArtifactFileNames: missingFiles,
            diagnostic: nil
        )
    }

    private static func startMissingLiveRuntimeDownloads(models: [CatalogModel], appState: AppState, context: ModelContext) async -> Int {
        guard let allStored = fetchStoredModels(context: context, operation: "startMissingLiveRuntimeDownloads", appState: appState) else {
            return 0
        }
        var started = 0
        for model in await missingModels(from: models, allStored: allStored) {
            switch await ensureModelPresent(model, expectedFleetCount: models.count, appState: appState, context: context, allStored: allStored) {
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
    ) async -> EnsureResult {
        let existingStored = storedModel(for: model, in: allStored)
        let localURL = ModelDownloader.shared.localURL(for: model)

        if await catalogFileIsValid(model, at: localURL) {
            if existingStored == nil {
                guard let stored = insertStoredModel(for: model, localURL: localURL, appState: appState, context: context) else {
                    return .failed
                }
                Task { @MainActor in
                    await loadIfSelected(stored, appState: appState, context: context)
                    await updateFleetBootProgress(expectedCount: expectedFleetCount, appState: appState, context: context)
                }
                return .linkedLocalFile
            } else if let existingStored {
                Task { @MainActor in
                    await loadIfSelected(existingStored, appState: appState, context: context)
                    await updateFleetBootProgress(expectedCount: expectedFleetCount, appState: appState, context: context)
                }
            }
            return .alreadyStored
        }

        if let existingStored,
           await storedModelFileIsValid(existingStored, catalog: model) {
            Task { @MainActor in
                await loadIfSelected(existingStored, appState: appState, context: context)
                await updateFleetBootProgress(expectedCount: expectedFleetCount, appState: appState, context: context)
            }
            return .alreadyStored
        }

        guard !ModelDownloader.shared.isDownloading(model) else { return .alreadyDownloading }

        let startResult = await ModelDownloader.shared.start(model) { localURL in
            Task { @MainActor in
                guard let freshStored = fetchStoredModels(context: context, operation: "downloadCompletion", appState: appState) else {
                    return
                }
                let stored: StoredModel
                if let existing = storedModel(for: model, in: freshStored) {
                    guard refreshStoredModel(existing, from: model, localURL: localURL, appState: appState, context: context) else {
                        return
                    }
                    stored = existing
                } else {
                    guard let inserted = insertStoredModel(for: model, localURL: localURL, appState: appState, context: context) else {
                        return
                    }
                    stored = inserted
                }

                await loadIfSelected(stored, appState: appState, context: context)
                await updateFleetBootProgress(expectedCount: expectedFleetCount, appState: appState, context: context)
            }
        }
        if case .failure(let error) = startResult {
            appState.runtime.updateBootStep(
                id: "models",
                detail: "Could not start \(model.name): \(error.localizedDescription)",
                state: .warning
            )
            return .failed
        }
        return .startedDownload
    }

    private static func linkExistingFleetFiles(appState: AppState, context: ModelContext) async {
        guard var allStored = fetchStoredModels(context: context, operation: "linkExistingFleetFiles", appState: appState) else {
            return
        }
        for model in fleetModelsForInstall() {
            let localURL = ModelDownloader.shared.localURL(for: model)
            guard await catalogFileIsValid(model, at: localURL) else { continue }
            if storedModel(for: model, in: allStored) == nil {
                if let inserted = insertStoredModel(for: model, localURL: localURL, appState: appState, context: context) {
                    allStored.append(inserted)
                }
            }
        }
    }

    private static func missingModels(from models: [CatalogModel], allStored: [StoredModel]) async -> [CatalogModel] {
        var missing: [CatalogModel] = []
        for model in models {
            let localURL = ModelDownloader.shared.localURL(for: model)
            if await catalogFileIsValid(model, at: localURL) { continue }
            if let stored = storedModel(for: model, in: allStored),
               await storedModelFileIsValid(stored, catalog: model) {
                continue
            }
            missing.append(model)
        }
        return missing
    }

    private static func catalogFileIsValid(_ catalog: CatalogModel, at url: URL) async -> Bool {
        if case .success = await ModelFileIntegrity.validateDownloadedCatalogFileAsync(catalog, at: url) {
            return true
        }
        return false
    }

    private static func storedModelFileIsValid(_ stored: StoredModel, catalog: CatalogModel) async -> Bool {
        let url = ModelStorage.resolvedModelURL(from: stored.localPath, fileName: stored.fileName)
        return await catalogFileIsValid(catalog, at: url)
    }

    private static func loadIfSelected(_ stored: StoredModel, appState: AppState, context: ModelContext) async {
        guard let allStored = fetchStoredModels(context: context, operation: "loadIfSelected", appState: appState) else {
            return
        }
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

    private static func updateFleetBootProgress(expectedCount: Int, appState: AppState, context: ModelContext) async {
        guard let readyCount = await readyFleetArtifactCount(context: context, appState: appState) else {
            return
        }
        let state: BootStepState = readyCount >= expectedCount ? .complete : .running
        appState.runtime.updateBootStep(
            id: "models",
            detail: "\(min(readyCount, expectedCount)) / \(expectedCount) \(LumenModelFamily.persistedSelected.shortLabel) artifacts ready",
            state: state
        )
    }

    private static func readyFleetArtifactCount(context: ModelContext, appState: AppState) async -> Int? {
        await readyArtifactCount(for: fleetModelsForInstall(), context: context, operation: "readyFleetArtifactCount", appState: appState)
    }

    private static func readyArtifactCount(for models: [CatalogModel], context: ModelContext, operation: String, appState: AppState? = nil) async -> Int? {
        guard let stored = fetchStoredModels(context: context, operation: operation, appState: appState) else {
            return nil
        }
        return await readyArtifactCount(for: models, stored: stored)
    }

    private static func readyArtifactCount(for models: [CatalogModel], stored: [StoredModel]) async -> Int {
        var ready = 0
        for model in models {
            let localReady = await catalogFileIsValid(model, at: ModelDownloader.shared.localURL(for: model))
            let storedReady: Bool
            if let matchingStored = storedModel(for: model, in: stored) {
                storedReady = await storedModelFileIsValid(matchingStored, catalog: model)
            } else {
                storedReady = false
            }
            if localReady || storedReady { ready += 1 }
        }
        return ready
    }

    private static func persistedArtifactCount(for models: [CatalogModel], stored: [StoredModel]) async -> Int {
        var ready = 0
        for model in models {
            guard let matchingStored = storedModel(for: model, in: stored) else { continue }
            if await storedModelFileIsValid(matchingStored, catalog: model) {
                ready += 1
            }
        }
        return ready
    }

    static func isProvisionedSelectionValid(
        appState: AppState,
        context: ModelContext
    ) async -> Bool {
        let family = LumenModelFamily.persistedSelected
        guard ModelProvisioningReceipt.isCurrent(
            family: family,
            chatModelID: appState.activeChatModelID,
            embeddingModelID: appState.activeEmbeddingModelID
        ), let stored = fetchStoredModels(
            context: context,
            operation: "isProvisionedSelectionValid",
            appState: appState
        ) else { return false }

        let models = provisioningModelsForInstall(family: family)
        guard let selection = provisionedRoleIDs(models: models, stored: stored),
              selection.chatModelID == appState.activeChatModelID,
              selection.embeddingModelID == appState.activeEmbeddingModelID
        else { return false }
        return await persistedArtifactCount(for: models, stored: stored) == models.count
    }

    private static func provisionedRoleIDs(
        models: [CatalogModel],
        stored: [StoredModel]
    ) -> (chatModelID: String, embeddingModelID: String)? {
        guard let chatCatalog = models.first(where: { $0.role == .chat }),
              let embeddingCatalog = models.first(where: { $0.role == .embedding }),
              let chat = storedModel(for: chatCatalog, in: stored),
              let embedding = storedModel(for: embeddingCatalog, in: stored)
        else { return nil }
        return (chat.id.uuidString, embedding.id.uuidString)
    }

    private static func loadProvisionedSelection(
        appState: AppState,
        stored: [StoredModel]
    ) async -> ModelLaunchLoadResult {
        var result = ModelLaunchLoadResult(chatLoaded: false, embeddingLoaded: false, resourceRetryAfterSeconds: nil)
        for attempt in 0...ModelAutoloadRetryPolicy.maximumRetryCount {
            guard !Task.isCancelled else { return result }
            result = await ModelLoader.loadAtLaunch(appState: appState, stored: stored)
            if result.allSelectedModelsLoaded { return result }
            guard attempt < ModelAutoloadRetryPolicy.maximumRetryCount,
                  let suggested = ModelAutoloadRetryPolicy.boundedDelaySeconds(result.resourceRetryAfterSeconds)
            else { return result }
            do {
                try await Task.sleep(nanoseconds: UInt64(min(10, suggested) * 1_000_000_000))
            } catch {
                return result
            }
        }
        return result
    }

    private static func restorePreviousSelection(
        _ previous: PersistedModelSelectionV2,
        appState: AppState,
        stored: [StoredModel]
    ) async {
        let previousFamily = LumenModelFamily.fromStoredID(previous.familyID)
        _ = try? appState.commitActiveModelSelection(
            chatModelID: previous.chatModelID,
            embeddingModelID: previous.embeddingModelID,
            family: previousFamily,
            provisioningPlanID: previous.provisioningPlanID
        )
        ModelLoader.cancelActiveLoads()
        await unloadResidentModelRuntimes()
        if previous.chatModelID != nil || previous.embeddingModelID != nil {
            _ = await ModelLoader.loadAtLaunch(appState: appState, stored: stored)
        }
        appState.runtime.requestModelAutoload()
    }

    private static func unloadResidentModelRuntimes() async {
        await AppLlamaService.shared.unloadAllRoleAdapters()
        await AppLlamaService.shared.unloadAllChat()
        await AppLlamaService.shared.unloadEmbed()
    }

    private static func hasActiveDownloads(for models: [CatalogModel]) -> Bool {
        models.contains { ModelDownloader.shared.isDownloading($0) }
    }

    private static func storedModel(for catalog: CatalogModel, context: ModelContext) -> StoredModel? {
        guard let models = fetchStoredModels(context: context, operation: "storedModel") else {
            return nil
        }
        return storedModel(for: catalog, in: models)
    }

    private static func storedModel(for catalog: CatalogModel, in models: [StoredModel]) -> StoredModel? {
        models.first { stored in
            artifactKey(repoId: stored.repoId, fileName: stored.fileName) == artifactKey(repoId: catalog.repoId, fileName: catalog.fileName)
        }
    }

    private static func refreshStoredModel(
        _ stored: StoredModel,
        from catalog: CatalogModel,
        localURL: URL,
        appState: AppState,
        context: ModelContext
    ) -> Bool {
        let previous = (
            name: stored.name,
            repoId: stored.repoId,
            fileName: stored.fileName,
            sizeBytes: stored.sizeBytes,
            quantization: stored.quantization,
            parameters: stored.parameters,
            role: stored.role,
            downloadedAt: stored.downloadedAt,
            localPath: stored.localPath
        )
        stored.name = catalog.name
        stored.repoId = catalog.repoId
        stored.fileName = catalog.fileName
        stored.sizeBytes = catalog.sizeBytes
        stored.quantization = catalog.quantization
        stored.parameters = catalog.parameters
        stored.role = catalog.role.rawValue
        stored.downloadedAt = Date()
        stored.localPath = localURL.path
        do {
            try persist(context, operation: "refreshStoredModel", scope: "StoredModel")
        } catch {
            stored.name = previous.name
            stored.repoId = previous.repoId
            stored.fileName = previous.fileName
            stored.sizeBytes = previous.sizeBytes
            stored.quantization = previous.quantization
            stored.parameters = previous.parameters
            stored.role = previous.role
            stored.downloadedAt = previous.downloadedAt
            stored.localPath = previous.localPath
            return false
        }
        return true
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
            logger.error("persist_blocked op=insertStoredModel scope=StoredModel error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            context.delete(stored)
            return nil
        }
        return stored
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
