import SwiftUI
import UniformTypeIdentifiers

@MainActor
struct CoreAIGemmaSettingsSection: View {
    var onConfigurationChanged: () -> Void = {}
    @Environment(\.scenePhase) private var scenePhase
    @AppStorage("coreai.gemma.enabled") private var isEnabled = true
    @State private var isConfigured = CoreAIGemmaModelStore.isConfigured
    @State private var operationOwner = ModelSettingsOperationOwner()
    @State private var showImporter = false
    @State private var errorMessage: String?
    @State private var configurationWasNotified = true

    var body: some View {
        Section {
            Toggle("Use Gemma by default", isOn: Binding(get: { isEnabled }, set: { enabled in
                perform { try await CoreAIGemmaModelStore.setEnabled(enabled) }
            }))
            .disabled(operationOwner.isBusy)
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
                    Label(operationOwner.isBusy ? "Importing and checking files…" : "Import Gemma model folder", systemImage: "folder")
                    Spacer()
                    if operationOwner.isBusy { ProgressView() }
                }
            }
            .disabled(operationOwner.isBusy)
            .accessibilityIdentifier("settings.coreai.import")
            if operationOwner.isBusy {
                Button("Cancel current Gemma operation", role: .cancel) { operationOwner.cancel() }
                    .accessibilityIdentifier("settings.coreai.cancel")
            }
            if isConfigured {
                Button("Remove local Gemma model", role: .destructive) {
                    perform { try await CoreAIGemmaModelStore.remove() }
                }
                .disabled(operationOwner.isBusy)
                .accessibilityIdentifier("settings.coreai.remove")
            }
        } header: {
            Text("Local Gemma")
        } footer: {
            Text("Requires iOS 27.1 and a complete Gemma 4 E2B Core AI text export with an 8K or 32K context, PLE weights and local tokenizer. Select a fully downloaded local folder in Files. Lumen copies and verifies it on this device; importing does not download a model. Allow space for a second copy during replacement. Imported hashes detect later changes; they do not certify the model’s publisher.")
        }
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

    private func updateActivity() {
        guard scenePhase == .active else { operationOwner.deactivate(); return }
        operationOwner.activate()
        if !operationOwner.isBusy { refreshConfiguration() }
    }

    private func refreshConfiguration() {
        isEnabled = CoreAIGemmaModelStore.isEnabled
        isConfigured = CoreAIGemmaModelStore.isConfigured
        if !configurationWasNotified {
            configurationWasNotified = true
            onConfigurationChanged()
        }
    }

    private func perform(_ operation: @escaping @Sendable () async throws -> Void) {
        guard operationOwner.start(operation: { try await operation() }, completion: { result in
            refreshConfiguration()
            switch result {
            case .success: break
            case .failure(is CancellationError): break
            case .failure(let error as CoreAIGemmaModelStore.Failure): errorMessage = error.localizedDescription
            case .failure: errorMessage = "The model is busy or unavailable. Stop the current conversation and try again."
            }
        }) != nil else { return }
        configurationWasNotified = false
        errorMessage = nil
    }
}
