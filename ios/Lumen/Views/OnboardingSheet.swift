import SwiftUI
import SwiftData

struct OnboardingSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var modelContext
    @Environment(AppState.self) private var appState
    @State private var downloader = ModelDownloader.shared

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.background.ignoresSafeArea()
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Welcome to Lumen")
                                .font(.title2.weight(.semibold))
                                .foregroundStyle(Theme.textPrimary)
                            Text("A private AI that runs on your iPhone.")
                                .font(.subheadline)
                                .foregroundStyle(Theme.textSecondary)
                        }

                        VStack(spacing: 0) {
                            FeatureRow(icon: "lock.shield", title: "Fully offline", subtitle: "Nothing ever leaves your device")
                            Divider().background(Theme.border).padding(.leading, 44)
                            FeatureRow(icon: "wrench.and.screwdriver", title: "Agentic tools", subtitle: "Calendar, Reminders, Health, Maps & more")
                            Divider().background(Theme.border).padding(.leading, 44)
                            FeatureRow(icon: "brain", title: "Vector memory", subtitle: "Remembers what matters across chats")
                            Divider().background(Theme.border).padding(.leading, 44)
                            FeatureRow(icon: "cpu", title: "Any GGUF model", subtitle: "Download straight from Hugging Face")
                        }
                        .background(Theme.surface)
                        .clipShape(.rect(cornerRadius: 10))
                        .overlay {
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .strokeBorder(Theme.border, lineWidth: 1)
                        }

                        VStack(spacing: 10) {
                            Button {
                                startDefault()
                            } label: {
                                Text("Download default model (1.1 GB)")
                                    .font(.subheadline.weight(.medium))
                                    .foregroundStyle(.white)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 12)
                                    .background(Theme.accent)
                                    .clipShape(.rect(cornerRadius: 10))
                            }
                            .buttonStyle(.plain)

                            Button("Skip for now") { dismiss() }
                                .font(.subheadline)
                                .foregroundStyle(Theme.textSecondary)
                        }
                    }
                    .padding(20)
                }
            }
            .navigationTitle("Setup")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private func startDefault() {
        let model = ModelCatalog.defaultOnboardingModel
        ModelDownloader.shared.start(model) { localURL in
            Task { @MainActor in
                let stored = StoredModel(
                    name: model.name, repoId: model.repoId, fileName: model.fileName,
                    sizeBytes: model.sizeBytes, quantization: model.quantization,
                    parameters: model.parameters, role: model.role, localPath: localURL.path
                )
                modelContext.insert(stored)
                try? modelContext.save()
                appState.activeChatModelID = stored.id.uuidString
            }
        }
        dismiss()
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
