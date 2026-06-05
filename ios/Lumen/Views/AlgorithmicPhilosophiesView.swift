import SwiftUI
import WebKit

struct AlgorithmicPhilosophiesView: View {
    @State private var selectedArtifactID = AlgorithmicPhilosophyCatalog.all.first?.id
    @State private var selectedPanel: ArtifactPanel = .viewer

    private var selectedArtifact: AlgorithmicPhilosophyArtifact? {
        AlgorithmicPhilosophyCatalog.all.first { $0.id == selectedArtifactID } ?? AlgorithmicPhilosophyCatalog.all.first
    }

    var body: some View {
        VStack(spacing: 0) {
            artifactPicker
                .padding(.horizontal, 20)
                .padding(.top, 18)
                .padding(.bottom, 12)

            Divider()
                .overlay(Theme.border.opacity(0.55))

            if let selectedArtifact {
                ArtifactDetailView(artifact: selectedArtifact, selectedPanel: $selectedPanel)
            } else {
                ContentUnavailableView(
                    "No philosophies bundled",
                    systemImage: "sparkles.rectangle.stack",
                    description: Text("Bundled algorithmic philosophy artifacts will appear here when they are included in the app resources.")
                )
            }
        }
        .background(AppBackground())
        .navigationTitle("Philosophies")
    }

    private var artifactPicker: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Algorithmic Philosophies")
                        .font(.title2.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                    Text("Living generative movements bundled as philosophy, viewer, and source code.")
                        .font(.subheadline)
                        .foregroundStyle(Theme.textSecondary)
                }
                Spacer()
                Text("\(AlgorithmicPhilosophyCatalog.all.count) movement")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.accent)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Theme.accent.opacity(0.12), in: Capsule())
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(AlgorithmicPhilosophyCatalog.all) { artifact in
                        Button {
                            selectedArtifactID = artifact.id
                            selectedPanel = .viewer
                        } label: {
                            ArtifactCard(
                                artifact: artifact,
                                isSelected: artifact.id == selectedArtifactID
                            )
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("philosophy.card.\(artifact.id)")
                    }
                }
                .padding(.vertical, 2)
            }
        }
    }
}

private enum ArtifactPanel: String, CaseIterable, Identifiable {
    case viewer = "Viewer"
    case philosophy = "Philosophy"
    case algorithm = "Algorithm"

    var id: String { rawValue }
}

private struct ArtifactCard: View {
    let artifact: AlgorithmicPhilosophyArtifact
    let isSelected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "sparkles")
                    .foregroundStyle(Theme.accent)
                Spacer()
                Text("Seed \(artifact.defaultSeed)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(Theme.textTertiary)
            }

            Text(artifact.title)
                .font(.headline)
                .foregroundStyle(Theme.textPrimary)

            Text(artifact.subtitle)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
                .lineLimit(2)
        }
        .frame(width: 250, alignment: .leading)
        .padding(14)
        .background(isSelected ? Theme.surfaceHigh : Theme.surface, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(isSelected ? Theme.accent.opacity(0.75) : Theme.border, lineWidth: isSelected ? 1.5 : 1)
        )
    }
}

private struct ArtifactDetailView: View {
    let artifact: AlgorithmicPhilosophyArtifact
    @Binding var selectedPanel: ArtifactPanel

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, 20)
                .padding(.vertical, 14)

            Picker("Artifact panel", selection: $selectedPanel) {
                ForEach(ArtifactPanel.allCases) { panel in
                    Text(panel.rawValue).tag(panel)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 20)
            .padding(.bottom, 12)

            Group {
                switch selectedPanel {
                case .viewer:
                    ArtifactViewerPanel(artifact: artifact)
                case .philosophy:
                    ArtifactTextPanel(
                        title: "Manifesto",
                        systemImage: "doc.text",
                        url: artifact.philosophyURL,
                        rendersMarkdown: true
                    )
                case .algorithm:
                    ArtifactTextPanel(
                        title: "Generative Algorithm",
                        systemImage: "chevron.left.forwardslash.chevron.right",
                        url: artifact.algorithmURL,
                        rendersMarkdown: false
                    )
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(artifact.title)
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                    Text(artifact.movementSummary)
                        .font(.subheadline)
                        .foregroundStyle(Theme.textSecondary)
                }
                Spacer()
                Label("Self-contained", systemImage: "shippingbox")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.accent)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Theme.accent.opacity(0.12), in: Capsule())
            }

            Text(artifact.conceptualSeed)
                .font(.caption)
                .foregroundStyle(Theme.textTertiary)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Theme.surface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))

            ParameterChips(parameters: artifact.parameterNames)
        }
    }
}

private struct ParameterChips: View {
    let parameters: [String]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(parameters, id: \.self) { parameter in
                    Text(parameter)
                        .font(.caption2.weight(.medium))
                        .foregroundStyle(Theme.textSecondary)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(Theme.surface.opacity(0.9), in: Capsule())
                        .overlay(Capsule().stroke(Theme.border.opacity(0.7), lineWidth: 1))
                }
            }
        }
    }
}

private struct ArtifactViewerPanel: View {
    let artifact: AlgorithmicPhilosophyArtifact

    var body: some View {
        if let viewerURL = artifact.viewerURL {
            LocalHTMLArtifactView(url: viewerURL)
                .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .stroke(Theme.border.opacity(0.65), lineWidth: 1)
                )
                .padding(.horizontal, 20)
                .padding(.bottom, 20)
                .accessibilityIdentifier("philosophy.viewer.\(artifact.id)")
        } else {
            ContentUnavailableView(
                "Viewer missing",
                systemImage: "exclamationmark.triangle",
                description: Text("The bundled HTML viewer could not be found for \(artifact.title).")
            )
        }
    }
}

private struct ArtifactTextPanel: View {
    let title: String
    let systemImage: String
    let url: URL?
    let rendersMarkdown: Bool

    private var bodyText: String {
        guard let url else { return "Missing bundled resource." }
        return (try? String(contentsOf: url, encoding: .utf8)) ?? "Could not read \(url.lastPathComponent)."
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Label(title, systemImage: systemImage)
                    .font(.headline)
                    .foregroundStyle(Theme.textPrimary)

                if rendersMarkdown, let attributed = try? AttributedString(markdown: bodyText) {
                    Text(attributed)
                        .font(.body)
                        .foregroundStyle(Theme.textSecondary)
                        .textSelection(.enabled)
                } else {
                    Text(bodyText)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(Theme.textSecondary)
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(18)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Theme.border.opacity(0.65), lineWidth: 1)
            )
            .padding(20)
        }
    }
}

private struct LocalHTMLArtifactView: UIViewRepresentable {
    let url: URL

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.allowsBackForwardNavigationGestures = false
        webView.scrollView.keyboardDismissMode = .onDrag
        webView.scrollView.backgroundColor = .clear
        webView.isOpaque = false
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        guard webView.url != url else { return }
        webView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent().deletingLastPathComponent())
    }
}
