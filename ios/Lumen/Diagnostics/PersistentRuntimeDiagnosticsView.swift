import SwiftUI
import UIKit

struct PersistentRuntimeDiagnosticsView: View {
    @State private var campaign = PersistentDiagnosticCampaign()
    @State private var status = PersistentDiagnosticRunnerStatus()
    @State private var isBusy = false
    @State private var exportURL: URL?
    @State private var shareItem: RuntimeDiagnosticsShareItem?
    @State private var message = "Persistent diagnostics are idle."

    var body: some View {
        Form {
            campaignSection
            scenariosSection
            controlsSection
            statusSection
            validationSection
        }
        .navigationTitle("Runtime Diagnostics")
        .task { await load() }
        .sheet(item: $shareItem) { item in
            RuntimeDiagnosticsActivityView(activityItems: [item.url])
        }
    }

    private var campaignSection: some View {
        Section("Campaign") {
            Toggle("Enable Persistent Runtime Diagnostics", isOn: binding(\.enabled))
            Toggle("Run continuously", isOn: binding(\.runContinuously))
            maxRunsStepper
            delayStepper
        }
    }

    private var maxRunsStepper: some View {
        Stepper(value: maxRunsBinding, in: 1...100) {
            HStack {
                Text("Max runs / scenario")
                Spacer()
                Text("\(campaign.maxRunsPerScenario)")
            }
        }
    }

    private var delayStepper: some View {
        Stepper(value: delayBinding, in: 0.5...120, step: 0.5) {
            HStack {
                Text("Delay between runs")
                Spacer()
                Text(String(format: "%.1fs", campaign.delayBetweenRunsSeconds))
            }
        }
    }

    private var scenariosSection: some View {
        Section("Scenarios") {
            ForEach(PersistentDiagnosticScenarioKind.allCases) { scenario in
                Toggle(isOn: scenarioBinding(scenario)) {
                    scenarioLabel(scenario)
                }
            }
        }
    }

    private func scenarioLabel(_ scenario: PersistentDiagnosticScenarioKind) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(scenario.displayName)
            Text(scenario.automationPolicy.rawValue)
                .font(.caption2)
                .foregroundStyle(scenario.automationPolicy == .automatic ? Color.secondary : Color.orange)
        }
    }

    private var controlsSection: some View {
        Section("Controls") {
            Button("Run Once") { runOnce() }.disabled(isBusy)
            Button("Start Continuous Campaign") { startContinuous() }.disabled(isBusy)
            Button("Stop") { stop() }.disabled(isBusy)
            Button("Start Lifecycle Cancellation Probe") { startLifecycleProbe() }.disabled(isBusy)
            Button("Run Live Agent Stream") { runLiveAgentStream() }.disabled(isBusy)
            Button("Export Logs") { exportLogs() }.disabled(isBusy)
            Button("Clear Logs", role: .destructive) { clearLogs() }.disabled(isBusy)
            exportControls
        }
    }

    @ViewBuilder
    private var exportControls: some View {
        if let exportURL,
           PersistentRuntimeDiagnosticsExporter.isPrivacySafeShareURL(exportURL) {
            Button {
                shareItem = RuntimeDiagnosticsShareItem(url: exportURL)
            } label: {
                Label("Share Exported Logs", systemImage: "square.and.arrow.up")
            }

            ShareLink(item: exportURL) {
                Label("Save / Share Logs", systemImage: "square.and.arrow.up")
            }

            Text(exportURL.lastPathComponent)
                .font(.caption.monospaced())
                .textSelection(.enabled)

            Text(exportURL.path)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    private var statusSection: some View {
        Section("Latest Status") {
            LabeledContent("Runner", value: runnerStatusText)
            LabeledContent("Last scenario", value: status.latestScenario?.displayName ?? "—")
            LabeledContent("Pass / fail / skipped", value: passFailSkippedText)
            LabeledContent("First token latency", value: firstTokenLatencyText)
            LabeledContent("Final prompt chars", value: finalPromptCharsText)
            LabeledContent("Cancellation", value: status.lastCancellationReason ?? "—")
            LabeledContent("Crash resume", value: status.lastCrashResumeStatus ?? "—")
            LabeledContent("Remediation", value: status.lastRemediationSummary ?? "—")
            Text(message).font(.caption).foregroundStyle(.secondary)
        }
    }

    private var validationSection: some View {
        Section("TestFlight validation") {
            Text("Run Agent fast prompt, Agent cancellation, and Lifecycle cancellation on a physical device. For lifecycle validation, tap the probe button, then lock or background the app within 3 seconds. On next launch, export logs and verify interrupted_or_terminated or clean_cancel_before_termination.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var runnerStatusText: String {
        guard status.isRunning else { return "Idle" }
        return status.isPaused ? "Paused" : "Running"
    }

    private var passFailSkippedText: String {
        "\(status.passedCount) / \(status.failedCount) / \(status.skippedCount)"
    }

    private var firstTokenLatencyText: String {
        status.lastFirstTokenLatencyMs.map { "\($0) ms" } ?? "—"
    }

    private var finalPromptCharsText: String {
        status.lastPromptFinalChars.map(String.init) ?? "—"
    }

    private var maxRunsBinding: Binding<Int> {
        Binding(
            get: { campaign.maxRunsPerScenario },
            set: { newValue in
                campaign.maxRunsPerScenario = max(1, newValue)
                saveCampaign()
            }
        )
    }

    private var delayBinding: Binding<Double> {
        Binding(
            get: { campaign.delayBetweenRunsSeconds },
            set: { newValue in
                campaign.delayBetweenRunsSeconds = max(0.5, newValue)
                saveCampaign()
            }
        )
    }

    private func binding(_ keyPath: WritableKeyPath<PersistentDiagnosticCampaign, Bool>) -> Binding<Bool> {
        Binding(
            get: { campaign[keyPath: keyPath] },
            set: { newValue in
                campaign[keyPath: keyPath] = newValue
                saveCampaign()
            }
        )
    }

    private func scenarioBinding(_ scenario: PersistentDiagnosticScenarioKind) -> Binding<Bool> {
        Binding(
            get: { campaign.scenarios.contains(scenario) },
            set: { enabled in
                if enabled, !campaign.scenarios.contains(scenario) {
                    campaign.scenarios.append(scenario)
                }
                if !enabled {
                    campaign.scenarios.removeAll { $0 == scenario }
                }
                saveCampaign()
            }
        )
    }

    private func load() async {
        campaign = await PersistentRuntimeDiagnosticsRunner.shared.loadCampaign()
        status = await PersistentRuntimeDiagnosticsRunner.shared.loadStatus()
    }

    private func saveCampaign() {
        let updated = campaign
        Task { await PersistentRuntimeDiagnosticsRunner.shared.saveCampaign(updated); await load() }
    }

    private func runOnce() {
        isBusy = true
        Task {
            let record = await PersistentRuntimeDiagnosticsRunner.shared.runOnce(campaign)
            message = record.map { "Last run: \($0.scenario.displayName) \($0.status.rawValue)" } ?? "No scenario selected."
            await load()
            isBusy = false
        }
    }

    private func startContinuous() {
        isBusy = true
        Task {
            await PersistentRuntimeDiagnosticsRunner.shared.startContinuous(campaign)
            message = "Continuous campaign started."
            await load()
            isBusy = false
        }
    }

    private func stop() {
        isBusy = true
        Task {
            await PersistentRuntimeDiagnosticsRunner.shared.stop()
            message = "Campaign stopped."
            await load()
            isBusy = false
        }
    }

    private func startLifecycleProbe() {
        isBusy = true
        Task {
            _ = await PersistentRuntimeDiagnosticsRunner.shared.startLifecycleCancellationProbe()
            message = "Lifecycle probe armed. Background or lock the device within 3 seconds."
            await load()
            isBusy = false
        }
    }

    private func runLiveAgentStream() {
        isBusy = true
        Task {
            let record = await PersistentRuntimeDiagnosticsRunner.shared.runLiveAgentStream(explicitUserRequested: true)
            message = record.map { "Live stream: \($0.status.rawValue)" } ?? "Live stream requires explicit user request."
            await load()
            isBusy = false
        }
    }

    private func exportLogs() {
        isBusy = true
        Task {
            do {
                let url = try await PersistentRuntimeDiagnosticsExporter.shared.export()
                guard PersistentRuntimeDiagnosticsExporter.isPrivacySafeShareURL(url) else {
                    throw PersistentRuntimeDiagnosticsExportError.unsafeShareURL
                }
                exportURL = url
                shareItem = RuntimeDiagnosticsShareItem(url: url)
                message = "Export ready to share."
            } catch {
                message = "Export failed: \(error.localizedDescription)"
            }
            await load()
            isBusy = false
        }
    }

    private func clearLogs() {
        isBusy = true
        Task {
            await PersistentRuntimeDiagnosticsRunner.shared.clearLogs()
            message = "Logs cleared."
            await load()
            isBusy = false
        }
    }
}

private struct RuntimeDiagnosticsShareItem: Identifiable {
    let id = UUID()
    let url: URL
}

private struct RuntimeDiagnosticsActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
