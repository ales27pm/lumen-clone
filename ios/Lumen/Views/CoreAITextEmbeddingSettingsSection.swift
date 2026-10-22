import SwiftUI
import SwiftData
import UniformTypeIdentifiers

/// The text embedding selection is independent from the generative assistant.
/// Appearance reads metadata only; installation, selection and rebuilding require
/// separate explicit actions and never download an artifact.
@MainActor
struct CoreAITextEmbeddingSettingsSection: View {
    var onConfigurationChanged: () -> Void = {}
    @Environment(\.scenePhase) private var scenePhase
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @AppStorage("coreai.textEmbedding.selected") private var isSelected = false
    @State private var isConfigured = CoreAITextEmbeddingModelStore.isConfigured
    @State private var summary: CoreAITextEmbeddingModelStore.ConfigurationSummary?
    @State private var operationOwner = ModelSettingsOperationOwner()
    @State private var showImporter = false
    @State private var errorMessage: String?
    @State private var maintenanceMessage: String?
    @State private var rebuild: EmbeddingSourceRebuild?
    @State private var confirmFreshIndex = false
    @State private var metadataRefreshID = UUID()
    @State private var configurationWasNotified = true

    private var isBusy: Bool { operationOwner.isBusy }

    var body: some View {
        Section {
            Toggle("Use Core AI for text embeddings", isOn: Binding(get: { isSelected }, set: { selected in
                perform { _ = try await CoreAITextEmbeddingModelStore.setSelected(selected) }
            }))
            .disabled(isBusy || (!isSelected && !isConfigured))
            .accessibilityIdentifier("settings.coreaiEmbedding.selected")
            LabeledContent("Memory and document search", value: isSelected ? "EmbeddingGemma 2 · Core AI" : "Selected GGUF embedding model")
            if let summary {
                Text("Text only · 256 dimensions · up to \(summary.maximumTokens) tokens")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            Label(isConfigured ? "Pinned text encoder imported" : "Import the text encoder to select it",
                  systemImage: isConfigured ? "checkmark.circle" : "folder.badge.plus")
                .font(.subheadline)
                .foregroundStyle(Theme.textSecondary)
            Button {
                showImporter = true
            } label: {
                HStack {
                    Label(isBusy ? "Working…" : "Import EmbeddingGemma text folder", systemImage: "folder")
                    Spacer()
                    if isBusy { ProgressView() }
                }
            }
            .disabled(isBusy)
            .accessibilityIdentifier("settings.coreaiEmbedding.import")
            Group {
                Text("Start a fresh semantic index from saved memory and document text. This clears vectors and app embedding indexes first, then rebuilds in bounded pages. Source contents and model files are preserved.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                Button("Start a fresh embedding index", role: .destructive) { confirmFreshIndex = true }
                    .disabled(isBusy || (isSelected && !isConfigured))
                    .accessibilityIdentifier("settings.coreaiEmbedding.freshIndex")
                if let rebuild, rebuild.phase != .complete {
                    Button(rebuild.phase == .reset ? "Continue clearing vectors (up to 32 records)" : "Rebuild next source batch (up to 16 records)") {
                        advanceRebuild(startFresh: false)
                    }
                    .disabled(isBusy)
                    .accessibilityIdentifier("settings.coreaiEmbedding.rebuildSources")
                }
                if EmbeddingIndexLifecycle.resetPending {
                    Text("Vector reset is incomplete. Semantic search and embedding writes are paused. Continue clearing vectors, or start a fresh index to resume after reopening settings.")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                }
            }
            if let maintenanceMessage {
                Text(maintenanceMessage).font(.caption).foregroundStyle(Theme.textSecondary)
                    .accessibilityIdentifier("settings.coreaiEmbedding.maintenanceStatus")
            }
            if isBusy {
                Button("Cancel current embedding operation", role: .cancel) { operationOwner.cancel() }
                    .accessibilityIdentifier("settings.coreaiEmbedding.cancel")
            }
            if isConfigured {
                Button("Remove local text embedding model", role: .destructive) {
                    perform { try await CoreAITextEmbeddingModelStore.remove() }
                }
                .disabled(isBusy)
                .accessibilityIdentifier("settings.coreaiEmbedding.remove")
            }
        } header: {
            Text("Local text embeddings")
        } footer: {
            Text("Requires iOS 27.1 on a supported device. Import the pinned EverPaper EmbeddingGemma 2 text package with model/config.json, local tokenizer, LICENSE, NOTICE and attribution. Vision and audio assets are excluded. Lumen copies and verifies every file locally. Importing does not select or load the model. Leave space for a second copy when replacing it. Selecting Core AI never sends text to a server or silently falls back to GGUF. iPhone performance and Neural Engine placement require device qualification.")
        }
        .task(id: metadataRefreshID) { await refreshMetadata() }
        .onAppear { updateActivity() }
        .onDisappear { operationOwner.deactivate() }
        .onChange(of: scenePhase) { _, _ in updateActivity() }
        .onChange(of: operationOwner.isBusy) { _, busy in
            if !busy, operationOwner.isActive { refreshConfiguration() }
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.folder], allowsMultipleSelection: false) { result in
            guard operationOwner.isActive else { return }
            switch result {
            case .success(let urls):
                guard let directory = urls.first else { return }
                perform { try await CoreAITextEmbeddingModelStore.importDirectory(directory) }
            case .failure:
                errorMessage = "The folder could not be opened. Choose a fully downloaded local folder."
            }
        }
        .confirmationDialog("Start a fresh embedding index?", isPresented: $confirmFreshIndex, titleVisibility: .visible) {
            Button("Prepare model and clear vectors", role: .destructive) { advanceRebuild(startFresh: true) }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Saved memories, document text, conversations and model files are kept. Existing vectors are discarded. Rebuild from source text with the next batch actions.")
        }
        .alert("Text embedding model", isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button("OK", role: .cancel) { errorMessage = nil }
        } message: { Text(errorMessage ?? "") }
    }

    private func updateActivity() {
        if scenePhase == .active { operationOwner.activate() }
        else { operationOwner.deactivate() }
        metadataRefreshID = UUID()
        if operationOwner.isActive, !isBusy { refreshConfiguration() }
    }

    private func refreshConfiguration() {
        isSelected = CoreAITextEmbeddingModelStore.isSelected
        metadataRefreshID = UUID()
        if !configurationWasNotified {
            configurationWasNotified = true
            onConfigurationChanged()
        }
    }

    @MainActor private func refreshMetadata() async {
        guard operationOwner.isActive, !isBusy else { return }
        let configured = CoreAITextEmbeddingModelStore.isConfigured
        let refreshedSummary = try? await CoreAITextEmbeddingModelStore.configurationSummary()
        guard !Task.isCancelled, operationOwner.isActive, !isBusy else { return }
        isConfigured = configured
        summary = refreshedSummary
    }

    private func perform(_ operation: @escaping @Sendable () async throws -> Void) {
        guard operationOwner.start(operation: { try await operation() }, completion: { result in
            switch result {
            case .success: break
            case .failure(is CancellationError): maintenanceMessage = "Embedding operation cancelled."
            case .failure(let error as CoreAITextEmbeddingModelStore.Failure): errorMessage = error.localizedDescription
            case .failure: errorMessage = "The embedding operation is unavailable or busy. Stop active work and try again."
            }
            refreshConfiguration()
        }) != nil else { return }
        configurationWasNotified = false
        errorMessage = nil
        rebuild = nil
    }

    private func advanceRebuild(startFresh: Bool) {
        guard operationOwner.start(operation: { () async throws -> String in
            let metadata = try await preparedMetadata()
            try Task.checkCancellation()
            if startFresh {
                guard !modelContext.hasChanges else {
                    return "Save pending edits before starting a fresh index. No vector reset was started."
                }
                rebuild = EmbeddingSourceRebuild(container: modelContext.container, prepared: metadata)
            }
            guard let rebuild else { return "Start a fresh index before rebuilding source batches." }
            await rebuild.advance(matching: metadata)
            try Task.checkCancellation()
            return rebuild.message
        }, completion: { result in
            switch result {
            case .success(let message): maintenanceMessage = message
            case .failure(is CancellationError): maintenanceMessage = "Paused. Source contents are preserved. Continue the current phase when ready."
            case .failure:
                errorMessage = "The selected embedding model could not be prepared. Check its local files and available device resources."
            }
            refreshConfiguration()
        }) != nil else { return }
        configurationWasNotified = false
        errorMessage = nil
        maintenanceMessage = "Preparing and verifying the selected text model…"
    }

    @MainActor private func preparedMetadata() async throws -> RAGEmbeddingIndexMetadata {
        if CoreAITextEmbeddingModelStore.isSelected {
            try await CoreAITextEmbeddingService.shared.prepare()
            try Task.checkCancellation()
            guard let loaded = await CoreAITextEmbeddingService.shared.loadedMetadata() else {
                throw CoreAITextEmbeddingError.notPrepared
            }
            return loaded
        }
        let stored = try modelContext.fetch(FetchDescriptor<StoredModel>())
        guard await ModelLoader.ensureEmbedLoaded(appState: appState, stored: stored, intent: .userChat) else {
            throw LocalRuntimeError.unavailable("Selected local embedding model is not prepared")
        }
        try Task.checkCancellation()
        let probe = try await AssistantKernel.runEmbeddingWithIdentity(
            text: "Source index readiness", purpose: .retrievalQuery
        )
        return RAGEmbeddingIndexMetadata(
            formatVersion: probe.formatVersion, modelIdentifier: probe.modelIdentifier, dimension: probe.vector.count
        )
    }
}
