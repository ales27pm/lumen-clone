import SwiftUI
import UniformTypeIdentifiers

struct CoreAIGemmaSettingsSection: View {
    var onConfigurationChanged: () -> Void = {}
    @AppStorage("coreai.gemma.enabled") private var isEnabled = true
    @State private var isConfigured = CoreAIGemmaModelStore.isConfigured
    @State private var isImporting = false
    @State private var showImporter = false
    @State private var errorMessage: String?

    var body: some View {
        Section {
            Toggle("Use Gemma by default", isOn: Binding(get: { isEnabled }, set: { enabled in
                perform { try await CoreAIGemmaModelStore.setEnabled(enabled) }
            }))
            .disabled(isImporting)
            .accessibilityIdentifier("settings.coreai.enabled")
            LabeledContent("Assistant", value: isEnabled ? "Gemma 4 E2B · Core AI" : "llama.cpp")
            if isEnabled {
                Text("One Gemma model handles conversations, voice responses and reasoning for tools and memory. Embeddings use a separate local model.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            Label(isConfigured ? "Local Gemma model imported" : "Import Gemma to use this assistant",
                  systemImage: isConfigured ? "checkmark.circle" : "folder.badge.plus")
                .font(.subheadline)
                .foregroundStyle(Theme.textSecondary)
            Button {
                showImporter = true
            } label: {
                HStack {
                    Label(isImporting ? "Importing and checking files…" : "Import Gemma model folder", systemImage: "folder")
                    Spacer()
                    if isImporting { ProgressView() }
                }
            }
            .disabled(isImporting)
            .accessibilityIdentifier("settings.coreai.import")
            if isConfigured {
                Button("Remove local Gemma model", role: .destructive) {
                    perform { try await CoreAIGemmaModelStore.remove() }
                }
                .disabled(isImporting)
                .accessibilityIdentifier("settings.coreai.remove")
            }
        } header: {
            Text("Local Gemma")
        } footer: {
            Text("Requires iOS 27.1 and a complete Gemma 4 E2B Core AI text export with an 8K or 32K context, PLE weights and local tokenizer. Select a fully downloaded local folder in Files. Lumen copies and verifies it on this device; importing does not download a model. Allow space for a second copy during replacement. Imported hashes detect later changes; they do not certify the model’s publisher.")
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.folder], allowsMultipleSelection: false) { result in
            switch result {
            case .success(let urls):
                guard let directory = urls.first else { return }
                perform { try await CoreAIGemmaModelStore.importDirectory(directory) }
            case .failure:
                errorMessage = "The selected folder could not be opened. Choose a fully downloaded local folder."
            }
        }
        .alert("Gemma model", isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button("OK", role: .cancel) { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "")
        }
    }

    private func perform(_ operation: @escaping @Sendable () async throws -> Void) {
        guard !isImporting else { return }
        isImporting = true
        Task { @MainActor in
            defer {
                isEnabled = CoreAIGemmaModelStore.isEnabled
                isConfigured = CoreAIGemmaModelStore.isConfigured
                isImporting = false
                onConfigurationChanged()
            }
            do { try await operation() }
            catch is CancellationError { }
            catch let error as CoreAIGemmaModelStore.Failure { errorMessage = error.localizedDescription }
            catch { errorMessage = "The model is busy or unavailable. Stop the current conversation and try again." }
        }
    }
}
