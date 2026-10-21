import SwiftUI
import SwiftData
import UniformTypeIdentifiers

/// The text embedding selection is independent from the generative assistant.
/// Appearance reads metadata only; installation, selection and rebuilding require
/// separate explicit actions and never download an artifact.
struct CoreAITextEmbeddingSettingsSection: View {
    var onConfigurationChanged: () -> Void = {}
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @AppStorage("coreai.textEmbedding.selected") private var isSelected = false
    @State private var isConfigured = CoreAITextEmbeddingModelStore.isConfigured
    @State private var summary: CoreAITextEmbeddingModelStore.ConfigurationSummary?
    @State private var isBusy = false
    @State private var showImporter = false
    @State private var errorMessage: String?
    @State private var maintenanceMessage: String?
    @State private var rebuildCursor: MemoryStore.ReembeddingCursor?
    @State private var work: Task<Void, Never>?

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
                Text("Existing memories remain searchable by text. Semantic search uses only vectors from the selected embedding space. Rebuild older memories in bounded batches; documents are rebuilt when their sources are indexed again.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                Button(rebuildCursor == nil ? "Rebuild up to 16 memory embeddings" : "Continue rebuilding up to 16 memory embeddings") { rebuildMemories() }
                    .disabled(isBusy || (isSelected && !isConfigured))
                    .accessibilityIdentifier("settings.coreaiEmbedding.rebuildMemories")
            }
            if let maintenanceMessage {
                Text(maintenanceMessage).font(.caption).foregroundStyle(Theme.textSecondary)
                    .accessibilityIdentifier("settings.coreaiEmbedding.maintenanceStatus")
            }
            if isBusy {
                Button("Cancel current embedding operation", role: .cancel) { work?.cancel() }
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
            Text("Requires iOS 27.2 on a supported device. Import the pinned EverPaper EmbeddingGemma 2 text package with model/config.json, local tokenizer, LICENSE, NOTICE and attribution. Vision and audio assets are excluded. Lumen copies and verifies every file locally. Importing does not select or load the model. Leave space for a second copy when replacing it. Selecting Core AI never sends text to a server or silently falls back to GGUF. iPhone performance and Neural Engine placement require device qualification.")
        }
        .task { await refreshMetadata() }
        .onDisappear { work?.cancel() }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.folder], allowsMultipleSelection: false) { result in
            switch result {
            case .success(let urls):
                guard let directory = urls.first else { return }
                perform { try await CoreAITextEmbeddingModelStore.importDirectory(directory) }
            case .failure:
                errorMessage = "The folder could not be opened. Choose a fully downloaded local folder."
            }
        }
        .alert("Text embedding model", isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button("OK", role: .cancel) { errorMessage = nil }
        } message: { Text(errorMessage ?? "") }
    }

    @MainActor private func refreshMetadata() async {
        isConfigured = CoreAITextEmbeddingModelStore.isConfigured
        summary = try? await CoreAITextEmbeddingModelStore.configurationSummary()
    }

    private func perform(_ operation: @escaping @Sendable () async throws -> Void) {
        guard !isBusy else { return }
        isBusy = true
        rebuildCursor = nil
        work = Task { @MainActor in
            defer { isBusy = false; work = nil; onConfigurationChanged() }
            do { try await operation() }
            catch is CancellationError { maintenanceMessage = "Embedding operation cancelled." }
            catch let error as CoreAITextEmbeddingModelStore.Failure { errorMessage = error.localizedDescription }
            catch { errorMessage = "The embedding operation is unavailable or busy. Stop active work and try again." }
            isSelected = CoreAITextEmbeddingModelStore.isSelected
            await refreshMetadata()
        }
    }

    private func rebuildMemories() {
        guard !isBusy else { return }
        isBusy = true
        maintenanceMessage = "Preparing the selected text model for this memory batch…"
        work = Task { @MainActor in
            defer { isBusy = false; work = nil; onConfigurationChanged() }
            do {
                let metadata: RAGEmbeddingIndexMetadata
                if CoreAITextEmbeddingModelStore.isSelected {
                    try await CoreAITextEmbeddingService.shared.prepare()
                    try Task.checkCancellation()
                    guard let loaded = await CoreAITextEmbeddingService.shared.loadedMetadata() else {
                        throw CoreAITextEmbeddingError.notPrepared
                    }
                    metadata = loaded
                } else {
                    let stored = try modelContext.fetch(FetchDescriptor<StoredModel>())
                    guard await ModelLoader.ensureEmbedLoaded(appState: appState, stored: stored, intent: .userChat) else {
                        throw LocalRuntimeError.unavailable("Selected local embedding model is not prepared")
                    }
                    try Task.checkCancellation()
                    // This probe is part of the explicit foreground rebuild, never
                    // a passive availability check or an implicit model selection.
                    let probe = try await AssistantKernel.runEmbeddingWithIdentity(
                        text: "Memory index readiness", purpose: .retrievalQuery
                    )
                    metadata = RAGEmbeddingIndexMetadata(
                        formatVersion: probe.formatVersion, modelIdentifier: probe.modelIdentifier, dimension: probe.vector.count
                    )
                }
                if let cursor = rebuildCursor, cursor.metadata != metadata {
                    rebuildCursor = nil
                }
                let result = await MemoryStore.reembedStaleEmbeddings(
                    context: modelContext, matching: metadata, limit: 16, after: rebuildCursor
                )
                rebuildCursor = result.nextCursor
                let progress = "Rebuilt \(result.rebuilt) of \(result.attempted) memories; skipped \(result.skipped); \(result.remaining) remain."
                maintenanceMessage = progress
                let skippedInputsOnly = result.skipped > 0 && result.diagnostic == "memory_inputs_skipped:\(result.skipped)"
                if result.diagnostic != nil && !skippedInputsOnly {
                    maintenanceMessage = progress + " The batch stopped. Saved memories are preserved."
                } else if result.nextCursor == nil && result.remaining > 0 {
                    maintenanceMessage = progress + " Scan complete. Skipped memories are preserved without replacement vectors."
                }
            } catch is CancellationError {
                maintenanceMessage = "Memory rebuilding cancelled. Saved memories are preserved."
            } catch {
                errorMessage = "Memory embeddings could not be rebuilt. Check the selected local model and available device resources."
            }
        }
    }
}
