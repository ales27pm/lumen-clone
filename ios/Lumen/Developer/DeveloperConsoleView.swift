import SwiftUI
import SwiftData

struct DeveloperConsoleView: View {
    @Environment(AppState.self) private var appState
    @State private var model = DeveloperConsoleModel()
    @State private var showStorageAlert = false
    @State private var storageAlertMessage = ""

    var body: some View {
        Form {
            Section {
                Picker("Console", selection: $model.selectedTab) {
                    ForEach(DeveloperConsoleTab.allCases) { tab in
                        Text(tab.title).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
            }

            switch model.selectedTab {
            case .overview:
                overviewSections
            case .evidence:
                evidenceSections
            case .workflows:
                workflowSections
            case .reports:
                reportSections
            case .privacy:
                privacySections
            }
        }
        .navigationTitle("Developer Console")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(Theme.background, for: .navigationBar)
        .toolbarBackground(.visible, for: .navigationBar)
        .safeAreaPadding(.top, 8)
        .task {
            model.loadCachedDiagnostics()
        }
        .alert("Storage checks", isPresented: $showStorageAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(storageAlertMessage)
        }
    }

    @ViewBuilder
    private var overviewSections: some View {
        let runtimeContract = LumenTrainedModelRuntimeRegistry.selected

        Section("Status") {
            LabeledContent("Framework", value: "Observe → Diagnose → Plan → Change → Validate → Learn")
            LabeledContent("Live authority", value: "Live E2E")
            LabeledContent("Runtime mode", value: appState.isGenerating ? "Generating" : "Idle")
            LabeledContent("Model family", value: runtimeContract.family.displayName)
            LabeledContent("Trained base", value: runtimeContract.sharedBaseModelID)
            LabeledContent("Adapter runtime", value: runtimeContract.mode)
            LabeledContent("Adapter roles", value: runtimeContract.adapterRoleIDs.joined(separator: " / "))
            Text(model.statusMessage)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
        }

        Section("Findings") {
            ForEach(model.findings) { finding in
                DeveloperFindingRow(finding: finding)
            }
        }

        Section("Core Modules") {
            NavigationLink {
                LumenControlTowerView()
            } label: {
                Label("Control Tower", systemImage: "gauge.with.dots.needle.67percent")
            }

            NavigationLink {
                DiagnosticsView()
            } label: {
                Label("Diagnostics", systemImage: "waveform.path.ecg")
            }

            NavigationLink {
                AgentGroundingAuditView(registryProvider: LiveRuntimeToolRegistryProvider())
            } label: {
                Label("Agent Grounding", systemImage: "checkmark.seal.text.page")
            }

            NavigationLink {
                E2ETestRunnerView()
            } label: {
                Label("Live E2E", systemImage: "testtube.2")
            }

            if PersistentRuntimeDiagnosticsAvailability.isDeveloperVisible {
                NavigationLink {
                    PersistentRuntimeDiagnosticsView()
                } label: {
                    Label("Persistent Runtime Diagnostics", systemImage: "waveform.path.ecg.rectangle")
                }
            }
        }
    }

    @ViewBuilder
    private var evidenceSections: some View {
        Section {
            ForEach(model.evidenceLayers) { status in
                DeveloperEvidenceLayerRow(status: status)
            }
        } header: {
            Text("Evidence Layers")
        } footer: {
            Text("Only Live E2E owns scenario pass/fail. All other layers are diagnostics, runtime traces, static checks, or repair evidence for the offline loop.")
        }

        Section("Export Packet") {
            NavigationLink {
                LumenControlTowerView()
            } label: {
                Label("Inspect Control Tower workflow snapshot", systemImage: "gauge.with.dots.needle.67percent")
            }

            NavigationLink {
                AgentGroundingAuditView(registryProvider: LiveRuntimeToolRegistryProvider())
            } label: {
                Label("Prepare runtime audit package", systemImage: "square.and.arrow.up")
            }

            NavigationLink {
                E2ETestRunnerView()
            } label: {
                Label("Prepare live E2E JSON", systemImage: "arrow.up.doc")
            }

            NavigationLink {
                PersistentRuntimeDiagnosticsView()
            } label: {
                Label("Export chat runtime traces", systemImage: "doc.zipper")
            }
        }
    }

    @ViewBuilder
    private var workflowSections: some View {
        Section("Fixed Actions") {
            NavigationLink {
                LumenControlTowerView()
            } label: {
                Label("Open Control Tower", systemImage: "gauge.with.dots.needle.67percent")
            }

            Button {
                Task { await model.collectDiagnostics() }
            } label: {
                HStack {
                    Label(DeveloperWorkflowAction.collectDiagnostics.title, systemImage: DeveloperWorkflowAction.collectDiagnostics.systemImage)
                    Spacer()
                    if model.isCollectingDiagnostics { ProgressView() }
                }
            }
            .disabled(model.isCollectingDiagnostics)

            Button {
                storageAlertMessage = model.runStorageChecks()
                showStorageAlert = true
            } label: {
                Label("Run storage checks", systemImage: "checkmark.circle")
            }

            NavigationLink {
                AgentGroundingAuditView(registryProvider: LiveRuntimeToolRegistryProvider())
            } label: {
                Label("Run audit and trace workflow", systemImage: "bolt.heart")
            }

            NavigationLink {
                E2ETestRunnerView()
            } label: {
                Label("Run live E2E workflow", systemImage: "testtube.2")
            }

            if PersistentRuntimeDiagnosticsAvailability.isDeveloperVisible {
                NavigationLink {
                    PersistentRuntimeDiagnosticsView()
                } label: {
                    Label("Run persistent diagnostics workflow", systemImage: "repeat.circle")
                }
            }
        }

        Section("Offline Loop Handoff") {
            Text("1. Export Agent Grounding runtime audit package.\n2. Export Live E2E report JSON.\n3. Share both files to the macOS improve-loop.\n4. Ingest with `python -m lumen_manifest_crawler framework ingest improve-loop --runtime-audit <json>`.")
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
        }
    }

    @ViewBuilder
    private var reportSections: some View {
        Section("Reports") {
            NavigationLink {
                DeveloperConsoleTextView(title: "Logs", bodyText: model.logsText())
            } label: {
                Label("Logs", systemImage: "doc.text.magnifyingglass")
            }

            NavigationLink {
                DeveloperConsoleTextView(title: "Debug", bodyText: model.debugText(appState: appState))
            } label: {
                Label("Debug", systemImage: "ladybug")
            }

            NavigationLink {
                DeveloperConsoleTextView(title: "Diagnostic", bodyText: model.diagnosticText())
            } label: {
                Label("Diagnostic", systemImage: "stethoscope")
            }
        }

        Section("Latest E2E") {
            Text(E2ETestLogStore.latestText())
                .font(.caption.monospaced())
                .textSelection(.enabled)
        }
    }

    @ViewBuilder
    private var privacySections: some View {
        Section("Export Boundaries") {
            ForEach(DeveloperEvidenceLayer.allCases) { layer in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Label(layer.title, systemImage: layer.systemImage)
                        Spacer()
                        Text(layer.ownsLiveE2EScenarios ? "live" : "diagnostic")
                            .font(.caption.monospaced())
                            .foregroundStyle(layer.ownsLiveE2EScenarios ? .green : Theme.textSecondary)
                    }
                    Text(layer.privacySummary)
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                }
                .padding(.vertical, 2)
            }
        }

        Section("Policy") {
            Text("Developer exports must not include unrestricted raw transcripts, contact bodies, calendar bodies, file contents, photo data, or full tool payload bodies. Bounded diagnostic snippets and explicit live E2E reports are treated as review-before-sharing artifacts.")
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
        }
    }
}

private struct DeveloperEvidenceLayerRow: View {
    let status: DeveloperEvidenceLayerStatus

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Label(status.layer.title, systemImage: status.layer.systemImage)
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text(status.layer.ownsLiveE2EScenarios ? "live owner" : "diagnostic")
                    .font(.caption.monospaced())
                    .foregroundStyle(status.layer.ownsLiveE2EScenarios ? .green : Theme.textSecondary)
            }
            LabeledContent("sourceLayer", value: status.layer.sourceLayer)
                .font(.caption)
            Text(status.detail)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
        }
        .padding(.vertical, 2)
    }
}

private struct DeveloperFindingRow: View {
    let finding: DeveloperFinding

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(finding.title, systemImage: finding.severity.systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(finding.severity.color)
            Text(finding.detail)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
        }
        .padding(.vertical, 2)
    }
}

struct DeveloperConsoleTextView: View {
    let title: String
    let bodyText: String

    var body: some View {
        ScrollView {
            Text(bodyText)
                .font(.footnote.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
        }
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
    }
}
