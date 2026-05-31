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
    @State private var runtimeController = ModelRuntimeController()

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.background.ignoresSafeArea()
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
            .onChange(of: selectedModelFamily) { _, family in
                LumenModelFamily.persistedSelected = family
            }
        }
    }

    private var activeRow: some View {
        HStack(spacing: 10) {
            ActivePill(title: "Chat", name: installedModels.first { $0.id.uuidString == appState.activeChatModelID }?.name ?? "None", icon: "bubble.left.and.bubble.right")
            ActivePill(title: "Embed", name: installedModels.first { $0.id.uuidString == appState.activeEmbeddingModelID }?.name ?? "None", icon: "point.3.connected.trianglepath.dotted")
        }
    }

    private var modelFamilyCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Image(systemName: "switch.2").foregroundStyle(Theme.accent)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Model family").font(.subheadline.weight(.semibold)).foregroundStyle(Theme.textPrimary)
                    Text("First launch and repair download only this family.").font(.caption).foregroundStyle(Theme.textSecondary)
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
    private var runtimeAwareFleetSnapshot: LumenModelFleetSnapshot { fleetSnapshot.withRuntimeResidentPaths(loadedPaths) }
    private var fleetSnapshot: LumenModelFleetSnapshot { LumenModelFleetResolver.resolveV1(appState: appState, storedModels: storedModels) }

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
            await ModelLaunchBootstrap.switchFamily(selectedModelFamily, appState: appState, context: modelContext)
            await ModelLoader.ensureFleetChatLoaded(appState: appState, stored: storedModels)
            await refreshLoaded()
            isRepairingSelectedFamily = false
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        }
    }

    private func download(_ model: CatalogModel) {
        downloader.start(model) { localURL in
            Task { @MainActor in
                if let existing = installedStoredModel(for: model) {
                    activate(stored: existing)
                    return
                }
                if let stale = storedModel(for: model), !modelFileExists(stale) {
                    modelContext.delete(stale)
                    try? modelContext.save()
                }
                let stored = StoredModel(name: model.name, repoId: model.repoId, fileName: model.fileName, sizeBytes: model.sizeBytes, quantization: model.quantization, parameters: model.parameters, role: model.role, localPath: localURL.path)
                modelContext.insert(stored)
                try? modelContext.save()
                if model.role == .chat && appState.activeChatModelID == nil { appState.activeChatModelID = stored.id.uuidString }
                if model.role == .embedding && appState.activeEmbeddingModelID == nil { appState.activeEmbeddingModelID = stored.id.uuidString }
                if model.role == .chat || model.role == .roleAdapter {
                    await ModelLoader.ensureFleetChatLoaded(appState: appState, stored: storedModels + [stored])
                } else {
                    _ = await ModelLoader.ensureEmbedLoaded(appState: appState, stored: storedModels + [stored])
                }
                await refreshLoaded()
                UINotificationFeedbackGenerator().notificationOccurred(.success)
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
        if stored.modelRole == .chat { appState.activeChatModelID = stored.id.uuidString }
        if stored.modelRole == .embedding { appState.activeEmbeddingModelID = stored.id.uuidString }
        UIImpactFeedbackGenerator(style: .rigid).impactOccurred()
    }

    private func deleteStored(for catalog: CatalogModel) {
        if let stored = storedModel(for: catalog) { deleteStoredModel(stored) }
        downloader.deleteLocal(catalog)
    }

    private func refreshLoaded() async {
        guard let set = await runtimeController.refreshLoadedPaths() else { return }
        loadedPaths = set
    }

    private func load(_ sm: StoredModel) {
        guard modelFileExists(sm) else { return }
        Task {
            do {
                try await runtimeController.load(sm, appState: appState, storedModels: storedModels)
                await refreshLoaded()
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            } catch {
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
        let resolvedPath = ModelStorage.resolvedModelURL(from: sm.localPath, fileName: sm.fileName, fileManager: fm).path
        Task { @MainActor in
            if sm.modelRole == .chat {
                let slots = await AppLlamaService.shared.loadedChatPathsBySlot.filter { $0.value == resolvedPath }.map(\.key)
                for slot in slots { await AppLlamaService.shared.unloadChat(for: slot) }
            } else if sm.modelRole == .roleAdapter {
                if let slot = adapterSlot(for: sm) { await AppLlamaService.shared.unloadRoleAdapter(slot: slot) }
            } else {
                await AppLlamaService.shared.unloadEmbed()
            }
        }
        try? fm.removeItem(atPath: sm.localPath)
        if resolvedPath != sm.localPath { try? fm.removeItem(atPath: resolvedPath) }
        if sm.id.uuidString == appState.activeChatModelID { appState.activeChatModelID = nil }
        if sm.id.uuidString == appState.activeEmbeddingModelID { appState.activeEmbeddingModelID = nil }
        modelContext.delete(sm)
        try? modelContext.save()
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
        return LumenModelFleetSnapshot(mode: mode, assignments: assignments, missingSlots: missingSlots, targetResidentSlots: targetResidentSlots, runtimeResidentSlots: runtimeSlots)
    }
}

struct ActivePill: View {
    let title: String
    let name: String
    let icon: String
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: icon).font(.caption).foregroundStyle(Theme.textSecondary)
                Text(title).font(.caption).foregroundStyle(Theme.textSecondary)
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
                Text("Active").font(.caption.weight(.medium)).foregroundStyle(Theme.accent)
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

    var body: some View {
        NavigationStack {
            List {
                Section("Chat model") {
                    ForEach(stored.filter { $0.modelRole == .chat && FileManager.default.fileExists(atPath: ModelStorage.resolvedModelURL(from: $0.localPath, fileName: $0.fileName).path) }) { m in
                        pickerRow(m, isActive: appState.activeChatModelID == m.id.uuidString) { appState.activeChatModelID = m.id.uuidString; dismiss() }
                    }
                }
                Section("Embedding model") {
                    ForEach(stored.filter { $0.modelRole == .embedding && FileManager.default.fileExists(atPath: ModelStorage.resolvedModelURL(from: $0.localPath, fileName: $0.fileName).path) }) { m in
                        pickerRow(m, isActive: appState.activeEmbeddingModelID == m.id.uuidString) { appState.activeEmbeddingModelID = m.id.uuidString; dismiss() }
                    }
                }
            }
            .navigationTitle("Active Models")
            .navigationBarTitleDisplayMode(.inline)
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
        }
    }

    private func addLocal(_ file: LocalModelFile) {
        let fileName = file.fileName
        let role: ModelRole = fileName.lowercased().contains("embed") ? .embedding : .chat
        let attrs = (try? FileManager.default.attributesOfItem(atPath: file.url.path)) ?? [:]
        let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        let stored = StoredModel(name: file.displayName, repoId: "local/\(file.source.lowercased())", fileName: fileName, sizeBytes: size, quantization: "local", parameters: "local", role: role, localPath: file.url.path)
        modelContext.insert(stored)
        try? modelContext.save()
        if role == .chat && appState.activeChatModelID == nil { appState.activeChatModelID = stored.id.uuidString }
        if role == .embedding && appState.activeEmbeddingModelID == nil { appState.activeEmbeddingModelID = stored.id.uuidString }
        dismiss()
    }
}
