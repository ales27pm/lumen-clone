import SwiftUI

struct DeveloperConsoleView: View {
    @Environment(AppState.self) private var appState
    @State private var model = DeveloperConsoleModel()
    @State private var reportTitle = "Developer Report"
    @State private var reportText = ""
    @State private var showReport = false

    var body: some View {
        List {
            Section("Status") {
                HStack {
                    Text(model.statusMessage)
                    Spacer()
                    if model.isCollectingDiagnostics {
                        ProgressView()
                    }
                }
                Button {
                    Task { await model.collectDiagnostics() }
                } label: {
                    Label("Collect diagnostics", systemImage: "waveform.path.ecg")
                }
                Button {
                    presentReport(title: "Storage Checks", text: model.runStorageChecks())
                } label: {
                    Label("Run storage checks", systemImage: "externaldrive.badge.checkmark")
                }
            }

            Section("Surfaces") {
                NavigationLink {
                    LumenControlTowerView()
                } label: {
                    Label("Control Tower", systemImage: "scope")
                }
                NavigationLink {
                    DiagnosticsView()
                } label: {
                    Label("Diagnostics", systemImage: "stethoscope")
                }
                NavigationLink {
                    if let grounding = model.diagnosticsSnapshot?.grounding {
                        GroundingDiagnosticsView(grounding: grounding)
                    } else {
                        ContentUnavailableView("Grounding unavailable", systemImage: "point.3.connected.trianglepath.dotted")
                    }
                } label: {
                    Label("Grounding", systemImage: "point.3.connected.trianglepath.dotted")
                }
                NavigationLink {
                    if let background = model.diagnosticsSnapshot?.background {
                        BackgroundDiagnosticsView(background: background)
                    } else {
                        ContentUnavailableView("Background unavailable", systemImage: "clock.badge.checkmark")
                    }
                } label: {
                    Label("Background", systemImage: "clock.badge.checkmark")
                }
                NavigationLink {
                    if let runtime = model.diagnosticsSnapshot?.runtime {
                        RuntimeDashboardView(runtime: runtime)
                    } else {
                        ContentUnavailableView("Runtime unavailable", systemImage: "gauge.with.dots.needle.50percent")
                    }
                } label: {
                    Label("Runtime Dashboard", systemImage: "gauge.with.dots.needle.50percent")
                }
            }

            Section("Reports") {
                Button {
                    presentReport(title: "Runtime Debug", text: model.debugText(appState: appState))
                } label: {
                    Label("Runtime debug text", systemImage: "doc.text")
                }
                Button {
                    presentReport(title: "Diagnostics Text", text: model.diagnosticText())
                } label: {
                    Label("Diagnostics text", systemImage: "doc.plaintext")
                }
                Button {
                    presentReport(title: "Recent Logs", text: model.logsText())
                } label: {
                    Label("Recent logs", systemImage: "list.bullet.rectangle")
                }
            }

            Section("Evidence Layers") {
                ForEach(model.evidenceLayers) { layerStatus in
                    VStack(alignment: .leading, spacing: 4) {
                        Label(layerStatus.layer.title, systemImage: layerStatus.layer.systemImage)
                            .font(.subheadline.weight(.semibold))
                        Text(layerStatus.detail)
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                        Text(layerStatus.status)
                            .font(.caption2.monospaced())
                            .foregroundStyle(layerStatus.isBlocking ? .red : Theme.textSecondary)
                    }
                    .padding(.vertical, 2)
                }
            }

            Section("Findings") {
                ForEach(model.findings) { finding in
                    Label {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(finding.title)
                                .font(.subheadline.weight(.semibold))
                            Text(finding.detail)
                                .font(.caption)
                                .foregroundStyle(Theme.textSecondary)
                        }
                    } icon: {
                        Image(systemName: finding.severity.systemImage)
                            .foregroundStyle(finding.severity.color)
                    }
                }
            }
        }
        .navigationTitle("Developer Console")
        .navigationBarTitleDisplayMode(.inline)
        .navigationDestination(isPresented: $showReport) {
            DeveloperConsoleReportTextView(title: reportTitle, bodyText: reportText)
        }
        .task {
            model.loadCachedDiagnostics()
        }
    }

    private func presentReport(title: String, text: String) {
        reportTitle = title
        reportText = text
        showReport = true
    }
}

private struct DeveloperConsoleReportTextView: View {
    let title: String
    let bodyText: String

    var body: some View {
        ScrollView {
            Text(bodyText)
                .font(.footnote.monospaced())
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
        }
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
    }
}
