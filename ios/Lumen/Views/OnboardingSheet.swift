import SwiftUI
import SwiftData

struct OnboardingSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var modelContext
    @Environment(AppState.self) private var appState
    @State private var downloader = ModelDownloader.shared
    @State private var setupTask: Task<Void, Never>?
    @State private var setupError: String?

    var body: some View {
        NavigationStack {
            ZStack {
                AppBackground()
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Welcome to Lumen")
                                .font(.title2.weight(.semibold))
                                .foregroundStyle(Theme.textPrimary)
                            Text("A local-first AI that runs on your iPhone.")
                                .font(.subheadline)
                                .foregroundStyle(Theme.textSecondary)
                        }

                        VStack(spacing: 0) {
                            FeatureRow(icon: "lock.shield", title: "Private by default", subtitle: "Chats, memory, and inference stay on-device")
                            Divider().background(Theme.border).padding(.leading, 44)
                            FeatureRow(icon: "wrench.and.screwdriver", title: "Permission-gated tools", subtitle: "Network and connected services run only when you enable and request them")
                            Divider().background(Theme.border).padding(.leading, 44)
                            FeatureRow(icon: "brain", title: "Vector memory", subtitle: "A local embedding model powers recall across chats")
                            Divider().background(Theme.border).padding(.leading, 44)
                            FeatureRow(icon: "checkmark.shield", title: "Verified model downloads", subtitle: "Pinned artifacts are checked before installation")
                        }
                        .background(Theme.surface)
                        .clipShape(.rect(cornerRadius: 10))
                        .overlay {
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .strokeBorder(Theme.border, lineWidth: 1)
                        }

                        VStack(spacing: 10) {
                            if setupTask != nil {
                                VStack(alignment: .leading, spacing: 7) {
                                    ProgressView(value: setupProgress)
                                        .tint(Theme.accent)
                                    Text(setupProgressText)
                                        .font(.caption)
                                        .foregroundStyle(Theme.textSecondary)
                                }

                                Button("Cancel setup", role: .cancel) {
                                    cancelSetup()
                                }
                                .font(.subheadline)
                            } else {
                                Button {
                                    startDefault()
                                } label: {
                                    Text("Download verified model fleet (\(formattedFleetSize))")
                                        .font(.subheadline.weight(.medium))
                                        .foregroundStyle(.white)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 12)
                                        .background(Theme.accent)
                                        .clipShape(.rect(cornerRadius: 10))
                                }
                                .buttonStyle(.plain)

                                Text("This downloads the selected chat model, embedding model, and required role adapters from Hugging Face after you confirm.")
                                    .font(.caption2)
                                    .foregroundStyle(Theme.textTertiary)
                                    .multilineTextAlignment(.center)
                            }

                            if let setupError {
                                Text(setupError)
                                    .font(.caption)
                                    .foregroundStyle(.red)
                                    .multilineTextAlignment(.center)
                            }

                            Button("Skip for now") {
                                _ = ModelProvisioningReceipt.markDeferred()
                                dismiss()
                            }
                                .font(.subheadline)
                                .foregroundStyle(Theme.textSecondary)
                                .disabled(setupTask != nil)
                        }
                    }
                    .padding(20)
                }
            }
            .navigationTitle("Setup")
            .navigationBarTitleDisplayMode(.inline)
        }
        .interactiveDismissDisabled(setupTask != nil)
    }

    private func startDefault() {
        guard setupTask == nil else { return }
        setupError = nil
        guard ModelProvisioningReceipt.markConsented() else {
            setupError = "Lumen could not save your download confirmation. Check available storage and retry."
            return
        }
        setupTask = Task { @MainActor in
            let result = await ModelLaunchBootstrap.provisionSelectedFamily(
                appState: appState,
                context: modelContext
            )
            guard !Task.isCancelled else { return }
            setupTask = nil
            if result.succeeded {
                dismiss()
            } else {
                setupError = result.errorMessage
                    ?? "Model setup completed only \(result.ready) of \(result.required) verified artifacts. Retry to continue."
            }
        }
    }

    private func cancelSetup() {
        setupTask?.cancel()
        for model in selectedFamilyModels {
            downloader.cancel(model)
        }
        setupTask = nil
        setupError = "Model setup was cancelled. You can retry when ready."
    }

    private var selectedFamilyModels: [CatalogModel] {
        ModelLaunchBootstrap.provisioningModelsForInstall()
    }

    private var formattedFleetSize: String {
        ByteCountFormatter.string(
            fromByteCount: selectedFamilyModels.reduce(Int64(0)) { $0 + $1.sizeBytes },
            countStyle: .file
        )
    }

    private var setupProgress: Double {
        let total = selectedFamilyModels.reduce(Int64(0)) { $0 + $1.sizeBytes }
        guard total > 0 else { return 0 }
        let completed = selectedFamilyModels.reduce(Int64(0)) { partial, model in
            guard let progress = downloader.progresses[model.id] else { return partial }
            if case .completed = progress.state { return partial + model.sizeBytes }
            return partial + min(max(0, progress.bytesReceived), model.sizeBytes)
        }
        return min(1, Double(completed) / Double(total))
    }

    private var setupProgressText: String {
        let total = selectedFamilyModels.reduce(Int64(0)) { $0 + $1.sizeBytes }
        let received = Int64(Double(total) * setupProgress)
        return "Downloading and verifying \(ByteCountFormatter.string(fromByteCount: received, countStyle: .file)) of \(ByteCountFormatter.string(fromByteCount: total, countStyle: .file))"
    }
}

struct FeatureRow: View {
    let icon: String
    let title: String
    let subtitle: String
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.body)
                .foregroundStyle(Theme.textSecondary)
                .frame(width: 22, height: 22)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.subheadline.weight(.medium)).foregroundStyle(Theme.textPrimary)
                Text(subtitle).font(.caption).foregroundStyle(Theme.textSecondary)
            }
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }
}

struct FeatureBullet: View {
    let icon: String
    let tint: Color
    let title: String
    let subtitle: String
    var body: some View {
        FeatureRow(icon: icon, title: title, subtitle: subtitle)
    }
}
