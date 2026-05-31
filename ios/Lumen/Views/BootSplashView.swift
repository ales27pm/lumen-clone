import SwiftUI

struct BootSplashView: View {
    @Environment(AppState.self) private var appState
    @State private var downloader = ModelDownloader.shared

    private var fleetModels: [CatalogModel] { ModelCatalog.featured.filter { $0.tags.contains("fleet-v1") || $0.tags.contains("memory") } }

    private var activeDownloads: [(CatalogModel, DownloadProgress)] {
        fleetModels.compactMap { model in
            guard let progress = downloader.progresses[model.id] else { return nil }
            return (model, progress)
        }
    }

    private var hasActiveDownloads: Bool {
        activeDownloads.contains { _, progress in
            switch progress.state {
            case .queued, .downloading, .paused:
                return true
            case .completed, .failed:
                return false
            }
        }
    }

    var body: some View {
        ZStack {
            LumenBrandBackground(intensity: 1.0)
            LumenLightBeam()
                .opacity(0.80)

            VStack(spacing: 22) {
                Spacer(minLength: 24)

                VStack(spacing: 8) {
                    LumenAssistantMark(showsWordmark: true, size: 104)
                    Text(appState.runtime.bootHeadline)
                        .font(.subheadline)
                        .foregroundStyle(Theme.textSecondary)
                        .multilineTextAlignment(.center)
                        .padding(.top, 2)
                }

                VStack(alignment: .leading, spacing: 12) {
                    Text("Boot sequence")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)

                    LazyVStack(spacing: 8) {
                        ForEach(appState.runtime.bootSteps.prefix(8)) { step in
                            BootStepRow(step: step)
                        }
                    }
                }
                .surface(cornerRadius: 14, padding: 14)

                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Model downloads")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(Theme.textPrimary)
                        Spacer()
                        if hasActiveDownloads {
                            ProgressView()
                                .controlSize(.small)
                        }
                    }

                    if activeDownloads.isEmpty {
                        Text("No active model downloads. Installed models will load automatically.")
                            .font(.footnote)
                            .foregroundStyle(Theme.textSecondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        LazyVStack(spacing: 10) {
                            ForEach(activeDownloads.prefix(8), id: \.0.id) { model, progress in
                                BootModelDownloadRow(model: model, progress: progress)
                            }
                        }
                    }
                }
                .surface(cornerRadius: 14, padding: 14)

                if appState.runtime.bootCoreComplete {
                    Button {
                        appState.runtime.dismissBootSplash()
                    } label: {
                        Text(hasActiveDownloads ? "Continue while downloads finish" : "Continue")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(LumenBrand.midnight)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Theme.accent)
                            .clipShape(.rect(cornerRadius: 12))
                            .shadow(color: LumenBrand.lumen.opacity(0.40), radius: 18, y: 8)
                    }
                    .buttonStyle(.plain)
                }

                Spacer(minLength: 20)
            }
            .padding(.horizontal, 20)
            .frame(maxWidth: 560)
        }
        .transition(.opacity)
        .onChange(of: hasActiveDownloads) { _, downloading in
            if appState.runtime.bootCoreComplete && !downloading {
                appState.runtime.updateBootStep(id: "models", detail: "Fleet models ready", state: .complete)
            }
        }
    }
}

private struct BootStepRow: View {
    let step: BootStep

    var body: some View {
        HStack(spacing: 10) {
            stateIcon
                .frame(width: 22, height: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(step.title)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(Theme.textPrimary)
                Text(step.detail)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            Spacer()
        }
    }

    @ViewBuilder
    private var stateIcon: some View {
        switch step.state {
        case .pending:
            Image(systemName: "circle")
                .foregroundStyle(Theme.textTertiary)
        case .running:
            ProgressView()
                .controlSize(.small)
        case .complete:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(Theme.accent)
        case .warning:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.yellow)
        case .failed:
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(.red)
        }
    }
}

private struct BootModelDownloadRow: View {
    let model: CatalogModel
    let progress: DownloadProgress

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: model.role == .embedding ? "point.3.connected.trianglepath.dotted" : "cpu")
                    .foregroundStyle(Theme.textSecondary)
                    .frame(width: 18)
                VStack(alignment: .leading, spacing: 1) {
                    Text(model.name)
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(Theme.textPrimary)
                        .lineLimit(1)
                    Text(statusText)
                        .font(.caption2.monospaced())
                        .foregroundStyle(Theme.textSecondary)
                }
                Spacer()
                Text(percentText)
                    .font(.caption2.monospaced().weight(.medium))
                    .foregroundStyle(Theme.textSecondary)
            }

            ProgressView(value: min(max(progress.fractionCompleted, 0), 1))
                .tint(Theme.accent)
        }
    }

    private var percentText: String {
        switch progress.state {
        case .completed:
            return "100%"
        case .failed:
            return "failed"
        default:
            return "\(Int((progress.fractionCompleted * 100).rounded()))%"
        }
    }

    private var statusText: String {
        switch progress.state {
        case .queued:
            return "Queued · \(formatBytes(progress.totalBytes))"
        case .downloading:
            return "\(formatBytes(progress.bytesReceived)) / \(formatBytes(progress.totalBytes))"
        case .paused:
            return "Paused · \(formatBytes(progress.bytesReceived)) / \(formatBytes(progress.totalBytes))"
        case .completed:
            return "Completed · \(formatBytes(progress.totalBytes))"
        case .failed(let message):
            return "Failed · \(message)"
        }
    }
}
