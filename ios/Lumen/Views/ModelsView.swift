import SwiftUI
import SwiftData

struct ModelsView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \StoredModel.downloadedAt, order: .reverse) private var storedModels: [StoredModel]
    @State private var showAddModel = false
    @State private var downloader = ModelDownloader.shared
    @State private var loadedPaths: Set<String> = []
    @State private var selectedModelFamily = LumenModelFamily.persistedSelected
    @State private var isRepairingSelectedFamily = false
    @State private var modelOperationError: String?
    @State private var runtimeController = ModelRuntimeController()

    var body: some View {
        NavigationStack {
            ZStack {
                AppBackground()
                ScrollView {
                    VStack(spacing: 24) {
                        activeRow
                        modelFamilyCard
                        FleetStatusCard(
                            snapshot: runtimeAwareFleetSnapshot,
                            progresses: downloader.progresses,
                            loadedPaths: loadedPaths,
                            onRepair: repairFleet
                        )

                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text("Featured — \(selectedModelFamily.shortLabel)")
                                    .font(.subheadline.weight(.semibold))
                                    .foregroundStyle(Theme.textPrimary)
                                Spacer()
                                Text("\(featuredModels.count) artifacts")
                                    .font(.caption.monospaced())
                                    .foregroundStyle(Theme.textSecondary)
                            }
                            VStack(spacing: 10) {
                                ForEach(featuredModels) { model in
                                    ModelCard(
                                        catalog: model,
                                        stored: installedStoredModel(for: model),
                                        progress: downloader.progresses[model.id],
                                        onDownload: { download(model) },
                                        onPause: { downloader.pause(model) },
                                        onResume: { download(model) },
                                        onCancel: { downloader.cancel(model) },
                                        onDelete: { deleteStored(for: model) },
                                        onActivate: { activate(model) }
                                    )
                                }
                            }
                        }

                        VStack(alignment: .leading, spacing: 10) {
                            Text("Downloaded")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(Theme.textPrimary)
                            let installedModels = storedModels.filter { modelFileExists($0) }
                            let staleModels = storedModels.filter { !modelFileExists($0) }
                            if installedModels.isEmpty && staleModels.isEmpty {
                                Text("No models yet.")
                                    .font(.footnote)
                                    .foregroundStyle(Theme.textSecondary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(14)
                                    .background(Theme.surface)
                                    .clipShape(.rect(cornerRadius: 10))
                                    .overlay {
                                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                                            .strokeBorder(Theme.border, lineWidth: 1)
                                    }
                            } else {
                                VStack(spacing: 8) {
                                    ForEach(installedModels) { sm in
                                        DownloadedRow(model: sm,
                                                      isActiveChat: sm.id.uuidString == appState.activeChatModelID,
                                                      isActiveEmbed: sm.id.uuidString == appState.activeEmbeddingModelID,
                                                      isLoaded: loadedPaths.contains(ModelStorage.resolvedModelURL(from: sm.localPath, fileName: sm.fileName).path),
                                                      isMissingFile: false,
                                                      isAdapter: sm.modelRole == .roleAdapter,
                                                      onActivate: { activate(stored: sm) },
                                                      onLoad: { load(sm) },
                                                      onUnload: { unload(sm) },
                                                      onReload: { reload(sm) },
                                                      onDelete: { deleteStoredModel(sm) })
                                    }
                                    ForEach(staleModels) { sm in
                                        DownloadedRow(model: sm,
                                                      isActiveChat: false,
                                                      isActiveEmbed: false,
                                                      isLoaded: false,
                                                      isMissingFile: true,
                                                      isAdapter: sm.modelRole == .roleAdapter,
                                                      onActivate: {},
                                                      onLoad: {},
                                                      onUnload: {},
                                                      onReload: {},
                                                      onDelete: { deleteStoredModel(sm) })
                                    }
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
                    .padding(.bottom, 40)
                }
            }
            .navigationTitle("Models")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showAddModel = true } label: { Image(systemName: "plus") }
                }
            }
            .sheet(isPresented: $showAddModel) {
                AddModelSheet().presentationDetents([.medium, .large])
            }
            .task {
                runtimeController.startupIfNeeded {
                    selectedModelFamily = LumenModelFamily.persistedSelected
                    await refreshLoaded()
                }
            }
            .task(id: appState.activeChatModelID) { await refreshLoaded() }
            .task(id: appState.activeEmbeddingModelID) { await refreshLoaded() }
            .task(id: appState.runtime.modelAutoloadState) { await refreshLoaded() }
            .alert("Model operation failed", isPresented: Binding(
                get: { modelOperationError != nil },
                set: { if !$0 { modelOperationError = nil } }
            )) {
                Button("OK", role: .cancel) { modelOperationError = nil }
            } message: {
                Text(modelOperationError ?? "The model operation could not be completed.")
            }
        }
    }

    private var activeRow: some View {
        HStack(spacing: 10) {
            ActivePill(
                title: "Chat",
                name: activeChatModel?.name ?? "None",
                modelID: appState.activeChatModelID,
                icon: "bubble.left.and.bubble.right",
                isLoaded: activeChatModel.map(isRuntimeLoaded) ?? false,
                statusAccessibilityIdentifier: "models.chatRuntimeStatus"
            )
            ActivePill(
                title: "Embed",
                name: activeEmbeddingModel?.name ?? "None",
                modelID: appState.activeEmbeddingModelID,
                icon: "point.3.connected.trianglepath.dotted",
                isLoaded: activeEmbeddingModel.map(isRuntimeLoaded) ?? false,
                statusAccessibilityIdentifier: "models.embeddingRuntimeStatus"
            )
        }
    }

    private var modelFamilyCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Image(systemName: "switch.2").foregroundStyle(Theme.accent)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Model family").font(.subheadline.weight(.semibold)).foregroundStyle(Theme.textPrimary)
                    Text("Choose a family, then confirm Download / repair. Your working selection stays active until setup succeeds.").font(.caption).foregroundStyle(Theme.textSecondary)
                }
                Spacer()
            }

            Picker("Model family", selection: $selectedModelFamily) {
                ForEach(LumenModelFamily.allCases) { family in
                    Text(family.displayName).tag(family)
                }
            }
            .pickerStyle(.segmented)
            .accessibilityIdentifier("models.familyPicker")

            Text(selectedModelFamily.description).font(.caption).foregroundStyle(Theme.textSecondary)

            Button { repairSelectedFamily() } label: {
                HStack {
                    Label(isRepairingSelectedFamily ? "Repairing…" : "Download / repair \(selectedModelFamily.shortLabel)", systemImage: "arrow.down.circle")
                    Spacer()
                    if isRepairingSelectedFamily { ProgressView() }
                }
            }
            .disabled(isRepairingSelectedFamily)
            .buttonStyle(.borderedProminent)
            .tint(Theme.accent)
            .accessibilityIdentifier("models.repairSelectedFamily")
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 14))
        .overlay { RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(Theme.border, lineWidth: 1) }
    }

    private var featuredModels: [CatalogModel] { LumenModelFleetCatalog.bootstrapModels(for: selectedModelFamily) }
    private var installedModels: [StoredModel] { storedModels.filter { modelFileExists($0) } }
    private var activeChatModel: StoredModel? { installedModels.first { $0.id.uuidString == appState.activeChatModelID } }
    private var activeEmbeddingModel: StoredModel? { installedModels.first { $0.id.uuidString == appState.activeEmbeddingModelID } }
    private var runtimeAwareFleetSnapshot: LumenModelFleetSnapshot { fleetSnapshot.withRuntimeResidentPaths(loadedPaths) }
    private var fleetSnapshot: LumenModelFleetSnapshot { LumenModelFleetResolver.resolveV1(appState: appState, storedModels: storedModels) }

    private func isRuntimeLoaded(_ model: StoredModel) -> Bool {
        loadedPaths.contains(ModelStorage.resolvedModelURL(from: model.localPath, fileName: model.fileName).path)
    }

    private func installedStoredModel(for catalog: CatalogModel) -> StoredModel? {
        storedModel(for: catalog).flatMap { modelFileExists($0) ? $0 : nil }
    }

    private func storedModel(for catalog: CatalogModel) -> StoredModel? {
        storedModels.first { stored in
            stored.repoId.caseInsensitiveCompare(catalog.repoId) == .orderedSame && stored.fileName.caseInsensitiveCompare(catalog.fileName) == .orderedSame
        }
    }

    private func modelFileExists(_ model: StoredModel) -> Bool {
        FileManager.default.fileExists(atPath: ModelStorage.resolvedModelURL(from: model.localPath, fileName: model.fileName).path)
    }

    private func repairFleet() { repairSelectedFamily() }

    private func repairSelectedFamily() {
        guard !isRepairingSelectedFamily else { return }
        isRepairingSelectedFamily = true
        Task { @MainActor in
            let result = await ModelLaunchBootstrap.switchFamily(selectedModelFamily, appState: appState, context: modelContext)
            await refreshLoaded()
            isRepairingSelectedFamily = false
            if result.succeeded {
                selectedModelFamily = LumenModelFamily.persistedSelected
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            } else {
                modelOperationError = result.errorMessage
                    ?? "Only \(result.ready) of \(result.required) verified artifacts became ready."
                UINotificationFeedbackGenerator().notificationOccurred(.error)
            }
        }
    }

    private func download(_ model: CatalogModel) {
        Task { @MainActor in
            _ = await downloader.start(model) { localURL in
                Task { @MainActor in
                    do {
                        let stored = try ModelCatalogPersistenceCoordinator.upsertVerifiedCatalogModel(
                            model,
                            localURL: localURL,
                            context: modelContext
                        )
                        if model.role == .chat && appState.activeChatModelID == nil {
                            try select(stored)
                        } else if model.role == .embedding && appState.activeEmbeddingModelID == nil {
                            try select(stored)
                        }
                        if model.role != .roleAdapter {
                            try await runtimeController.load(stored, appState: appState, storedModels: mergedStoredModels(including: stored))
                        }
                        await refreshLoaded()
                        UINotificationFeedbackGenerator().notificationOccurred(.success)
                    } catch {
                        modelOperationError = error.localizedDescription
                        UINotificationFeedbackGenerator().notificationOccurred(.error)
                    }
                }
            }
        }
    }

    private func activate(_ catalog: CatalogModel) {
        guard catalog.role != .roleAdapter else {
            UINotificationFeedbackGenerator().notificationOccurred(.warning)
            return
        }
        guard let stored = installedStoredModel(for: catalog) else { return }
        activate(stored: stored)
    }

    private func activate(stored: StoredModel) {
        guard modelFileExists(stored) else { return }
        guard stored.modelRole != .roleAdapter else {
            UINotificationFeedbackGenerator().notificationOccurred(.warning)
            return
        }
        do {
            try select(stored)
            UIImpactFeedbackGenerator(style: .rigid).impactOccurred()
        } catch {
            modelOperationError = error.localizedDescription
            UINotificationFeedbackGenerator().notificationOccurred(.error)
        }
    }

    private func deleteStored(for catalog: CatalogModel) {
        if let stored = storedModel(for: catalog) {
            deleteStoredModel(stored)
        } else {
            downloader.deleteLocal(catalog)
        }
    }

    private func refreshLoaded() async {
        guard let set = await runtimeController.refreshLoadedPaths() else { return }
        loadedPaths = set
    }

    private func load(_ sm: StoredModel) {
        guard modelFileExists(sm) else { return }
        Task {
            let previousSelection = PersistedModelSelectionStore.loadOrMigrate()
            do {
                if sm.modelRole != .roleAdapter {
                    try select(sm)
                }
                try await runtimeController.load(sm, appState: appState, storedModels: storedModels)
                await refreshLoaded()
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            } catch {
                try? appState.commitActiveModelSelection(
                    chatModelID: previousSelection.chatModelID,
                    embeddingModelID: previousSelection.embeddingModelID,
                    family: LumenModelFamily.fromStoredID(previousSelection.familyID),
                    provisioningPlanID: previousSelection.provisioningPlanID
                )
                modelOperationError = error.localizedDescription
                UINotificationFeedbackGenerator().notificationOccurred(.error)
            }
        }
    }

    private func unload(_ sm: StoredModel) {
        Task {
            await runtimeController.unload(sm) { adapterSlot(for: $0) }
            await refreshLoaded()
            UIImpactFeedbackGenerator(style: .soft).impactOccurred()
        }
    }

    private func reload(_ sm: StoredModel) {
        guard modelFileExists(sm) else { return }
        Task {
            do {
                try await runtimeController.reload(sm, appState: appState, storedModels: storedModels)
                await refreshLoaded()
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            } catch {
                UINotificationFeedbackGenerator().notificationOccurred(.error)
            }
        }
    }

    private func deleteStoredModel(_ sm: StoredModel) {
        let fm = FileManager.default
        let modelID = sm.id.uuidString
        let role = sm.modelRole
        let localPath = sm.localPath
        let resolvedPath = ModelStorage.resolvedModelURL(
            from: localPath,
            fileName: sm.fileName,
            fileManager: fm
        ).path
        let roleAdapterSlot = role == .roleAdapter ? adapterSlot(for: sm) : nil
        Task { @MainActor in
            let previousSelection = PersistedModelSelectionStore.loadOrMigrate()
            let deletingSelectedChat = modelID == appState.activeChatModelID
            let deletingSelectedEmbedding = modelID == appState.activeEmbeddingModelID
            do {
                if deletingSelectedChat || deletingSelectedEmbedding {
                    try appState.commitActiveModelSelection(
                        chatModelID: deletingSelectedChat ? nil : appState.activeChatModelID,
                        embeddingModelID: deletingSelectedEmbedding ? nil : appState.activeEmbeddingModelID,
                        family: LumenModelFamily.persistedSelected,
                        provisioningPlanID: nil
                    )
                }
                do {
                    try ModelCatalogPersistenceCoordinator.delete(sm, context: modelContext)
                } catch {
                    _ = try? appState.commitActiveModelSelection(
                        chatModelID: previousSelection.chatModelID,
                        embeddingModelID: previousSelection.embeddingModelID,
                        family: LumenModelFamily.fromStoredID(previousSelection.familyID),
                        provisioningPlanID: previousSelection.provisioningPlanID
                    )
                    throw error
                }
            } catch {
                modelOperationError = error.localizedDescription
                UINotificationFeedbackGenerator().notificationOccurred(.error)
                return
            }

            await runtimeController.unloadResolvedModel(
                role: role,
                resolvedPath: resolvedPath,
                adapterSlot: roleAdapterSlot
            )
            try? fm.removeItem(atPath: localPath)
            if resolvedPath != localPath { try? fm.removeItem(atPath: resolvedPath) }
            ModelProvisioningReceipt.invalidate()
            appState.runtime.requestModelAutoload()
            await refreshLoaded()
        }
    }

    private func select(_ stored: StoredModel) throws {
        let family = LumenModelFamily.persistedSelected
        if stored.modelRole == .chat {
            try LumenModelSelectionPolicy.validatePersistedChatModel(
                repoID: stored.repoId,
                fileName: stored.fileName,
                sizeBytes: stored.sizeBytes,
                family: family
            )
        }
        let chatID = stored.modelRole == .chat ? stored.id.uuidString : appState.activeChatModelID
        let embeddingID = stored.modelRole == .embedding ? stored.id.uuidString : appState.activeEmbeddingModelID
        try appState.commitActiveModelSelection(
            chatModelID: chatID,
            embeddingModelID: embeddingID,
            family: family,
            provisioningPlanID: nil
        )
        ModelProvisioningReceipt.invalidate()
        appState.runtime.requestModelAutoload()
    }

    private func mergedStoredModels(including stored: StoredModel) -> [StoredModel] {
        storedModels.contains(where: { $0.id == stored.id }) ? storedModels : storedModels + [stored]
    }

    private func adapterSlot(for model: StoredModel) -> LumenModelSlot? {
        let text = [model.name, model.fileName, model.localPath].joined(separator: " ").lowercased()
        for slot in [LumenModelSlot.cortex, .executor, .mouth, .mimicry, .rem] where text.contains(slot.rawValue) {
            return slot
        }
        return nil
    }
}

private extension LumenModelFleetSnapshot {
    func withRuntimeResidentPaths(_ loadedPaths: Set<String>) -> LumenModelFleetSnapshot {
        let runtimeSlots = Set(assignments.compactMap { slot, assignment in loadedPaths.contains(assignment.localPath) ? slot : nil })
        return LumenModelFleetSnapshot(mode: mode, assignments: assignments, missingSlots: missingSlots, missingAdapterSlots: missingAdapterSlots, targetResidentSlots: targetResidentSlots, runtimeResidentSlots: runtimeSlots)
    }
}

struct ActivePill: View {
    let title: String
    let name: String
    let modelID: String?
    let icon: String
    let isLoaded: Bool
    let statusAccessibilityIdentifier: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: icon).font(.caption).foregroundStyle(Theme.textSecondary)
                Text(title).font(.caption).foregroundStyle(Theme.textSecondary)
                Spacer()
                Text(isLoaded ? "Loaded" : "Active")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(isLoaded ? Theme.accent : Theme.textSecondary)
                    .accessibilityIdentifier(statusAccessibilityIdentifier)
                    .accessibilityValue(modelID ?? "")
            }
            Text(name).font(.subheadline.weight(.medium)).foregroundStyle(Theme.textPrimary).lineLimit(1)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 10))
        .overlay { RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Theme.border, lineWidth: 1) }
    }
}

struct ModelCard: View {
    let catalog: CatalogModel
    let stored: StoredModel?
    let progress: DownloadProgress?
    var onDownload: () -> Void
    var onPause: () -> Void = {}
    var onResume: () -> Void = {}
    var onCancel: () -> Void
    var onDelete: () -> Void
    var onActivate: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Image(systemName: catalog.role == .embedding ? "point.3.connected.trianglepath.dotted" : "cpu").font(.body).foregroundStyle(Theme.textSecondary).frame(width: 28, height: 28)
                VStack(alignment: .leading, spacing: 1) {
                    Text(catalog.name).font(.subheadline.weight(.semibold)).foregroundStyle(Theme.textPrimary)
                    Text("\(catalog.parameters) · \(catalog.quantization) · \(formatBytes(catalog.sizeBytes))").font(.caption.monospaced()).foregroundStyle(Theme.textSecondary)
                }
                Spacer()
                actionButton
            }
            Text(catalog.description).font(.footnote).foregroundStyle(Theme.textSecondary)
            if !catalog.tags.isEmpty {
                HStack(spacing: 6) {
                    ForEach(catalog.tags, id: \.self) { tag in
                        Text(tag).font(.caption2).padding(.horizontal, 6).padding(.vertical, 2).foregroundStyle(Theme.textSecondary).background(Theme.surfaceHigh).clipShape(.rect(cornerRadius: 4))
                    }
                }
            }
            if let progress {
                switch progress.state {
                case .downloading:
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: progress.fractionCompleted).tint(Theme.accent)
                        Text("\(formatBytes(progress.bytesReceived)) / \(formatBytes(progress.totalBytes))").font(.caption2.monospaced()).foregroundStyle(Theme.textSecondary)
                    }
                case .paused:
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: progress.fractionCompleted).tint(Theme.textTertiary)
                        Text("Paused — \(formatBytes(progress.bytesReceived)) / \(formatBytes(progress.totalBytes))").font(.caption2.monospaced()).foregroundStyle(Theme.textSecondary)
                    }
                case .failed(let msg):
                    Text("Failed: \(msg)").font(.caption2).foregroundStyle(.red)
                case .queued, .completed:
                    EmptyView()
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 10))
        .overlay { RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Theme.border, lineWidth: 1) }
    }

    @ViewBuilder
    private var actionButton: some View {
        if let progress, case .downloading = progress.state {
            HStack(spacing: 6) {
                Button { onPause() } label: { Image(systemName: "pause.fill").font(.caption) }.buttonStyle(.bordered)
                Button("Cancel") { onCancel() }.font(.caption.weight(.medium)).buttonStyle(.bordered).tint(.red)
            }
        } else if let progress, case .paused = progress.state {
            HStack(spacing: 6) {
                Button { onResume() } label: { Image(systemName: "play.fill").font(.caption) }.buttonStyle(.borderedProminent).tint(Theme.accent)
                Button("Cancel") { onCancel() }.font(.caption.weight(.medium)).buttonStyle(.bordered).tint(.red)
            }
        } else if stored != nil, catalog.role == .roleAdapter {
            Menu {
                Button("Delete", systemImage: "trash", role: .destructive) { onDelete() }
            } label: {
                Text("Adapter").font(.caption.weight(.medium)).foregroundStyle(Theme.textSecondary).padding(.horizontal, 8).padding(.vertical, 4).overlay { RoundedRectangle(cornerRadius: 6).strokeBorder(Theme.border, lineWidth: 1) }
            }
        } else if stored != nil {
            Menu {
                Button("Set as Active", systemImage: "checkmark") { onActivate() }
                Button("Delete", systemImage: "trash", role: .destructive) { onDelete() }
            } label: {
                Text("Installed").font(.caption.weight(.medium)).foregroundStyle(Theme.textSecondary).padding(.horizontal, 8).padding(.vertical, 4).overlay { RoundedRectangle(cornerRadius: 6).strokeBorder(Theme.border, lineWidth: 1) }
            }
        } else {
            Button("Download") { onDownload() }.font(.caption.weight(.medium)).buttonStyle(.borderedProminent).tint(Theme.accent)
        }
    }
}

struct DownloadedRow: View {
    let model: StoredModel
    let isActiveChat: Bool
    let isActiveEmbed: Bool
    let isLoaded: Bool
    let isMissingFile: Bool
    let isAdapter: Bool
    var onActivate: () -> Void
    var onLoad: () -> Void
    var onUnload: () -> Void
    var onReload: () -> Void
    var onDelete: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: model.modelRole == .embedding ? "point.3.connected.trianglepath.dotted" : "cpu").foregroundStyle(isMissingFile ? .orange : Theme.textSecondary)
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    Text(model.name).font(.subheadline.weight(.medium)).foregroundStyle(Theme.textPrimary)
                    if isLoaded { Circle().fill(Theme.accent).frame(width: 6, height: 6) }
                }
                Text(isMissingFile ? "Missing local file · stale record" : "\(model.parameters) · \(model.quantization) · \(formatBytes(model.sizeBytes))")
                    .font(.caption2.monospaced())
                    .foregroundStyle(isMissingFile ? .orange : Theme.textSecondary)
            }
            Spacer()
            if isMissingFile {
                Text("Missing").font(.caption.weight(.medium)).foregroundStyle(.orange)
            } else if isAdapter {
                Text("Adapter").font(.caption.weight(.medium)).foregroundStyle(Theme.textSecondary)
            } else if isActiveChat || isActiveEmbed {
                Text(isLoaded ? "Loaded" : "Active").font(.caption.weight(.medium)).foregroundStyle(Theme.accent)
            } else {
                Button("Use") { onActivate() }.font(.caption.weight(.medium)).buttonStyle(.bordered)
            }
            Menu {
                if !isMissingFile, !isAdapter {
                    if isLoaded {
                        Button("Reload", systemImage: "arrow.clockwise") { onReload() }
                        Button("Unload", systemImage: "eject") { onUnload() }
                    } else {
                        Button("Load", systemImage: "arrow.down.circle") { onLoad() }
                    }
                    Divider()
                }
                Button(isMissingFile ? "Remove stale record" : "Delete", systemImage: "trash", role: .destructive) { onDelete() }
            } label: { Image(systemName: "ellipsis.circle").foregroundStyle(Theme.textTertiary) }
            .buttonStyle(.plain)
        }
        .padding(12)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 10))
        .overlay { RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(Theme.border, lineWidth: 1) }
    }
}

struct ModelPickerSheet: View {
    @Environment(AppState.self) private var appState
    @Environment(\.dismiss) private var dismiss
    @Query private var stored: [StoredModel]
    @State private var selectionError: String?

    var body: some View {
        NavigationStack {
            List {
                Section("Chat model") {
                    ForEach(stored.filter { model in
                        model.modelRole == .chat
                            && FileManager.default.fileExists(atPath: ModelStorage.resolvedModelURL(from: model.localPath, fileName: model.fileName).path)
                            && LumenModelSelectionPolicy.isPersistedChatModelCompatible(
                                repoID: model.repoId,
                                fileName: model.fileName,
                                sizeBytes: model.sizeBytes,
                                family: LumenModelFamily.persistedSelected
                            )
                    }) { m in
                        pickerRow(m, isActive: appState.activeChatModelID == m.id.uuidString) { select(m) }
                    }
                }
                Section("Embedding model") {
                    ForEach(stored.filter { $0.modelRole == .embedding && FileManager.default.fileExists(atPath: ModelStorage.resolvedModelURL(from: $0.localPath, fileName: $0.fileName).path) }) { m in
                        pickerRow(m, isActive: appState.activeEmbeddingModelID == m.id.uuidString) { select(m) }
                    }
                }
            }
            .navigationTitle("Active Models")
            .navigationBarTitleDisplayMode(.inline)
            .alert("Selection not saved", isPresented: Binding(
                get: { selectionError != nil },
                set: { if !$0 { selectionError = nil } }
            )) {
                Button("OK", role: .cancel) { selectionError = nil }
            } message: {
                Text(selectionError ?? "The model selection could not be saved.")
            }
        }
    }

    private func select(_ model: StoredModel) {
        do {
            let family = LumenModelFamily.persistedSelected
            if model.modelRole == .chat {
                try LumenModelSelectionPolicy.validatePersistedChatModel(
                    repoID: model.repoId,
                    fileName: model.fileName,
                    sizeBytes: model.sizeBytes,
                    family: family
                )
            }
            try appState.commitActiveModelSelection(
                chatModelID: model.modelRole == .chat ? model.id.uuidString : appState.activeChatModelID,
                embeddingModelID: model.modelRole == .embedding ? model.id.uuidString : appState.activeEmbeddingModelID,
                family: family,
                provisioningPlanID: nil
            )
            ModelProvisioningReceipt.invalidate()
            appState.runtime.requestModelAutoload()
            dismiss()
        } catch {
            selectionError = error.localizedDescription
        }
    }

    private func pickerRow(_ m: StoredModel, isActive: Bool, onTap: @escaping () -> Void) -> some View {
        Button(action: onTap) {
            HStack {
                VStack(alignment: .leading) {
                    Text(m.name)
                    Text("\(m.parameters) · \(m.quantization)").font(.caption.monospaced()).foregroundStyle(Theme.textSecondary)
                }
                Spacer()
                if isActive { Image(systemName: "checkmark").foregroundStyle(Theme.accent) }
            }
        }
    }
}

struct AddModelSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var modelContext
    @Environment(AppState.self) private var appState
    @State private var candidates: [LocalModelFile] = []
    @State private var importError: String?

    var body: some View {
        NavigationStack {
            List {
                if candidates.isEmpty {
                    Text("No .gguf files found in app bundle or Documents directory.").font(.footnote).foregroundStyle(Theme.textSecondary)
                } else {
                    ForEach(candidates) { model in
                        Button { addLocal(model) } label: {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(model.displayName).font(.body.weight(.medium))
                                Text("\(model.fileName) • \(model.source)").font(.caption).foregroundStyle(Theme.textSecondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Select GGUF")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .topBarTrailing) { Button("Refresh") { candidates = LocalModelDiscovery.discoverGGUF() } }
            }
            .task { candidates = LocalModelDiscovery.discoverGGUF() }
            .alert("Model import", isPresented: Binding(
                get: { importError != nil },
                set: { if !$0 { importError = nil } }
            )) {
                Button("OK", role: .cancel) { importError = nil }
            } message: {
                Text(importError ?? "The local model import could not be completed.")
            }
        }
    }

    private func addLocal(_ file: LocalModelFile) {
        let fileName = file.fileName
        let role: ModelRole = fileName.lowercased().contains("embed") ? .embedding : .chat
        let attrs = (try? FileManager.default.attributesOfItem(atPath: file.url.path)) ?? [:]
        let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        do {
            let stored = try ModelCatalogPersistenceCoordinator.insertLocalModel(
                name: file.displayName,
                repoID: "local/\(file.source.lowercased())",
                fileName: fileName,
                sizeBytes: size,
                role: role,
                localURL: file.url,
                context: modelContext
            )
            let family = LumenModelFamily.persistedSelected
            let chatCompatible = role != .chat || LumenModelSelectionPolicy.isPersistedChatModelCompatible(
                repoID: stored.repoId,
                fileName: stored.fileName,
                sizeBytes: stored.sizeBytes,
                family: family
            )
            let shouldActivateChat = role == .chat
                && appState.activeChatModelID == nil
                && chatCompatible
            let shouldActivateEmbedding = role == .embedding && appState.activeEmbeddingModelID == nil
            if shouldActivateChat || shouldActivateEmbedding {
                try appState.commitActiveModelSelection(
                    chatModelID: role == .chat ? stored.id.uuidString : appState.activeChatModelID,
                    embeddingModelID: role == .embedding ? stored.id.uuidString : appState.activeEmbeddingModelID,
                    family: family,
                    provisioningPlanID: nil
                )
                ModelProvisioningReceipt.invalidate()
                appState.runtime.requestModelAutoload()
            }
            if role == .chat, !chatCompatible {
                importError = "\(file.displayName) was imported, but it cannot be activated while \(family.shortLabel) adapter mode is selected. Switch model families to use this local chat model."
                return
            }
            dismiss()
        } catch {
            importError = error.localizedDescription
        }
    }
}
