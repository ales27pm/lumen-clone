import SwiftUI
import UIKit

struct DeveloperConsoleView: View {
    @Environment(AppState.self) private var appState
    @State private var model = DeveloperConsoleModel()
    @State private var reportTitle = "Developer Report"
    @State private var reportText = ""
    @State private var reportExportURL: URL?
    @State private var showReport = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                DeveloperWorkflowHero(
                    status: model.statusMessage,
                    progress: model.workflowProgress,
                    isRunning: model.isWorkflowRunning,
                    monitoringEnabled: model.monitoringEnabled,
                    activeAction: model.activeWorkflowAction,
                    summary: model.lastWorkflowSummary,
                    onStart: {
                        Task { await model.startDeveloperWorkflow() }
                    },
                    onRefresh: {
                        Task { await model.collectDiagnostics() }
                    }
                )

                Picker("Developer view", selection: $model.selectedTab) {
                    ForEach(DeveloperConsoleTab.allCases) { tab in
                        Text(tab.title).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .accessibilityIdentifier("developerConsole.segmentedTabs")

                tabContent
            }
            .padding(.horizontal, 20)
            .padding(.top, 14)
            .padding(.bottom, 28)
        }
        .scrollIndicators(.visible)
        .scrollContentBackground(.hidden)
        .background(AppBackground())
        .navigationTitle("Developer Console")
        .navigationBarTitleDisplayMode(.inline)
        .navigationDestination(isPresented: $showReport) {
            DeveloperConsoleReportTextView(
                title: reportTitle,
                bodyText: reportText,
                exportURL: reportExportURL
            )
        }
        .task {
            model.loadCachedDiagnostics()
            model.loadLatestE2EReport()
            model.startLiveMeters()
        }
        .onDisappear {
            model.stopLiveMeters()
        }
    }

    @ViewBuilder
    private var tabContent: some View {
        switch model.selectedTab {
        case .run:
            DeveloperRunDashboard(
                model: model,
                presentReport: { title, text in
                    presentReport(title: title, text: text)
                },
                presentExportReport: { title, text, exportURL in
                    presentReport(title: title, text: text, exportURL: exportURL)
                },
                appState: appState
            )
        case .telemetry:
            DeveloperTelemetryDashboard(model: model)
        case .reports:
            DeveloperReportsSection(
                model: model,
                presentReport: { title, text in
                    presentReport(title: title, text: text)
                },
                appState: appState
            )
        }
    }

    private func presentReport(title: String, text: String, exportURL: URL? = nil) {
        reportTitle = title
        reportText = text
        reportExportURL = exportURL
        showReport = true
    }
}

private struct DeveloperLiveTelemetrySection: View {
    let meters: [DeveloperTelemetryMeter]

    private let columns = [
        GridItem(.flexible(), spacing: 10),
        GridItem(.flexible(), spacing: 10)
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            DeveloperSectionHeader(
                title: "Live Telemetry",
                subtitle: "RAM is resident memory; CPU is event-density proxy; GPU and ANE show real availability plus utilization when the runtime exposes counters."
            )
            LazyVGrid(columns: columns, spacing: 10) {
                ForEach(meters) { meter in
                    DeveloperSharpTelemetryMeter(meter: meter)
                }
            }
        }
    }
}

private struct DeveloperWorkflowHero: View {
    let status: String
    let progress: Double
    let isRunning: Bool
    let monitoringEnabled: Bool
    let activeAction: DeveloperWorkflowAction?
    let summary: String
    let onStart: () -> Void
    let onRefresh: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top, spacing: 12) {
                ZStack {
                    Circle()
                        .fill(Theme.accent.opacity(0.18))
                    Image(systemName: "point.3.filled.connected.trianglepath.dotted")
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(Theme.accent)
                }
                .frame(width: 44, height: 44)

                VStack(alignment: .leading, spacing: 5) {
                    Text("Developer Operations")
                        .font(.title3.weight(.semibold))
                    Text(status)
                        .font(.subheadline)
                        .foregroundStyle(Theme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer()

                StatusPill(
                    text: monitoringEnabled ? "LIVE" : "IDLE",
                    color: monitoringEnabled ? .green : Theme.textSecondary
                )
            }

            HStack(spacing: 14) {
                DeveloperRingMeter(progress: progress, tint: monitoringEnabled ? .green : Theme.accent)
                    .frame(width: 74, height: 74)

                VStack(alignment: .leading, spacing: 6) {
                    Text(activeAction?.title ?? "Workflow monitor idle")
                        .font(.headline)
                        .lineLimit(2)
                    Text(summary)
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }

            HStack(spacing: 10) {
                Button(action: onStart) {
                    Label(isRunning ? "Starting" : "Start workflow", systemImage: isRunning ? "hourglass" : "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.accent)
                .disabled(isRunning)
                .accessibilityIdentifier("developerConsole.startWorkflow")

                Button(action: onRefresh) {
                    Label("Refresh", systemImage: "arrow.clockwise")
                        .labelStyle(.iconOnly)
                        .frame(width: 44, height: 36)
                }
                .buttonStyle(.bordered)
                .tint(Theme.textSecondary)
                .disabled(isRunning)
                .accessibilityIdentifier("developerConsole.refreshDiagnostics")
            }
        }
        .padding(18)
        .dashboardCard()
    }
}

private struct DeveloperRunDashboard: View {
    @Bindable var model: DeveloperConsoleModel
    let presentReport: (String, String) -> Void
    let presentExportReport: (String, String, URL?) -> Void
    let appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            DeveloperE2ECommandCenter(
                model: model,
                presentReport: presentReport,
                presentExportReport: presentExportReport
            )

            DeveloperSectionHeader(title: "Runtime Capture", subtitle: "Start monitoring, refresh diagnostics, and create evidence inputs for the offline loop.")
            VStack(spacing: 10) {
                DeveloperActionRow(
                    title: model.isWorkflowRunning ? "Workflow running" : "Start workflow",
                    detail: "Arms monitoring, refreshes diagnostics, runs storage checks, then waits for live runtime evidence.",
                    systemImage: model.isWorkflowRunning ? "hourglass" : "play.fill",
                    tint: Theme.accent,
                    isRunning: model.isWorkflowRunning
                ) {
                    Task { await model.startDeveloperWorkflow() }
                }
                DeveloperActionRow(
                    title: "Refresh diagnostics",
                    detail: "Update runtime, permissions, grounding, background, and privacy state.",
                    systemImage: "waveform.path.ecg",
                    tint: Theme.accent,
                    isRunning: model.isCollectingDiagnostics
                ) {
                    Task { await model.collectDiagnostics() }
                }
                DeveloperActionRow(
                    title: "Run storage checks",
                    detail: "Verify model, import, and E2E report directories are readable and writable.",
                    systemImage: "externaldrive.badge.checkmark",
                    tint: .green
                ) {
                    presentReport("Storage Checks", model.runStorageChecks())
                }
                DeveloperActionRow(
                    title: "Open runtime debug text",
                    detail: "Inspect model family, adapter runtime, and generation settings.",
                    systemImage: "doc.text.magnifyingglass",
                    tint: .blue
                ) {
                    presentReport("Runtime Debug", model.debugText(appState: appState))
                }
            }

            DeveloperSectionHeader(title: "Cycle State", subtitle: "Only live evidence and generated loop handoff state are shown here.")
            LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)], spacing: 10) {
                ForEach(model.dashboardMeters) { meter in
                    DeveloperMeterCard(meter: meter)
                }
            }

            if let active = model.activeWorkflowAction {
                DeveloperWorkflowActionCard(
                    action: active,
                    isActive: true,
                    isComplete: isComplete(active)
                )
            }
        }
    }

    private func isComplete(_ action: DeveloperWorkflowAction) -> Bool {
        switch action {
        case .collectDiagnostics:
            return model.diagnosticsSnapshot != nil
        case .runPersistentDiagnostics:
            return model.storageCheckRatio != nil
        case .runLiveTraceSmoke, .exportRecentTraces:
            return !model.workflowSnapshot.events.isEmpty
        case .runE2EStandard:
            return model.hasE2EReport(containing: E2ETestScenario.standard)
        case .runE2ETraining:
            return model.hasE2EReport(containing: E2ETestScenario.trainingValidation)
        case .exportLiveE2E:
            return model.hasExportedLiveE2E
        case .runAgentGrounding, .exportRuntimeAudit:
            return model.monitoringEnabled
        }
    }
}

private struct DeveloperTelemetryDashboard: View {
    @Bindable var model: DeveloperConsoleModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            DeveloperLiveTelemetrySection(meters: model.telemetryMeters)

            DeveloperSectionHeader(title: "Live Surfaces", subtitle: "Open the runtime tools that produce the dashboard signals.")
            DeveloperSurfaceGrid()
        }
    }
}

private struct DeveloperE2ECommandCenter: View {
    @Bindable var model: DeveloperConsoleModel
    let presentReport: (String, String) -> Void
    let presentExportReport: (String, String, URL?) -> Void

    private let columns = [
        GridItem(.flexible(), spacing: 10),
        GridItem(.flexible(), spacing: 10)
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            DeveloperSectionHeader(
                title: "Complete E2E Tests",
                subtitle: "Run standard and training validation from the canonical runner, then export the live E2E JSON evidence layer."
            )

            DeveloperE2ESummaryPanel(model: model)

            LazyVGrid(columns: columns, spacing: 10) {
                DeveloperE2ESuiteCard(
                    title: "Standard",
                    detail: "Regression, tool guard, and live chat coverage",
                    scenarios: E2ETestScenario.standard,
                    isCovered: model.hasE2EReport(containing: E2ETestScenario.standard),
                    tint: .cyan
                )
                DeveloperE2ESuiteCard(
                    title: "Training",
                    detail: "Failure prompts and correction signals",
                    scenarios: E2ETestScenario.trainingValidation,
                    isCovered: model.hasE2EReport(containing: E2ETestScenario.trainingValidation),
                    tint: Theme.accent
                )
            }

            VStack(spacing: 10) {
                NavigationLink {
                    E2ETestRunnerView(initialRunMode: .standard)
                } label: {
                    DeveloperNavigationRow(
                        title: "Run complete standard E2E",
                        detail: "\(E2ETestScenario.standard.count) scenarios; includes \(liveCount(E2ETestScenario.standard)) live model-backed checks.",
                        systemImage: "testtube.2",
                        tint: .cyan
                    )
                }

                NavigationLink {
                    E2ETestRunnerView(initialRunMode: .trainingValidation)
                } label: {
                    DeveloperNavigationRow(
                        title: "Run training E2E validation",
                        detail: "\(E2ETestScenario.trainingValidation.count) scenarios for the next improvement loop.",
                        systemImage: "graduationcap",
                        tint: Theme.accent
                    )
                }

                DeveloperActionRow(
                    title: "Refresh latest E2E report",
                    detail: "Reload latest pass/fail, live/static split, performance, and failure signals.",
                    systemImage: "arrow.clockwise",
                    tint: .blue
                ) {
                    model.loadLatestE2EReport()
                    presentReport("Latest E2E Report", model.e2eReportText)
                }

                DeveloperActionRow(
                    title: "Export live E2E JSON",
                    detail: "Writes the `e2eTestReport` evidence layer for improve-loop ingestion.",
                    systemImage: "arrow.up.doc",
                    tint: .green
                ) {
                    let text = model.exportLatestE2EReport()
                    presentExportReport("Live E2E Export", text, model.e2eLastExportURL)
                }
            }
            .buttonStyle(.plain)

            if let error = model.e2eExportError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            } else if let url = model.e2eLastExportURL {
                Text("Last export: \(url.lastPathComponent)")
                    .font(.caption.monospaced())
                    .foregroundStyle(Theme.textSecondary)
                    .lineLimit(2)
                DeveloperExportHandoffCard(
                    title: "Live E2E JSON",
                    subtitle: "e2eTestReport evidence layer",
                    url: url
                )
            }

            if !model.e2eLatestResults.isEmpty {
                DeveloperE2ELatestResults(results: Array(model.e2eLatestResults.suffix(5)))
            }
        }
    }

    private func liveCount(_ scenarios: [E2ETestScenario]) -> Int {
        scenarios.filter(\.requiresAgentRun).count
    }
}

private struct DeveloperExportHandoffCard: View {
    let title: String
    let subtitle: String
    let url: URL
    @State private var copiedLabel: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "shippingbox.and.arrow.backward")
                    .font(.headline)
                    .foregroundStyle(.green)
                    .frame(width: 34, height: 34)
                    .background(.green.opacity(0.13), in: RoundedRectangle(cornerRadius: 9, style: .continuous))

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                    Text(url.lastPathComponent)
                        .font(.caption2.monospaced())
                        .foregroundStyle(Theme.textSecondary)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }

            HStack(spacing: 8) {
                ShareLink(item: url) {
                    Label("Share JSON", systemImage: "square.and.arrow.up")
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)

                Button {
                    UIPasteboard.general.string = url.path
                    copiedLabel = "path copied"
                } label: {
                    Label("Copy path", systemImage: "doc.on.doc")
                }
                .buttonStyle(.bordered)

                Button {
                    UIPasteboard.general.string = url.lastPathComponent
                    copiedLabel = "name copied"
                } label: {
                    Label("Copy name", systemImage: "textformat")
                }
                .buttonStyle(.bordered)
            }
            .labelStyle(.iconOnly)

            if let copiedLabel {
                Text(copiedLabel)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.green)
            }
        }
        .padding(13)
        .dashboardCard(cornerRadius: 16)
    }
}

private struct DeveloperE2ESummaryPanel: View {
    @Bindable var model: DeveloperConsoleModel

    var body: some View {
        HStack(alignment: .center, spacing: 14) {
            DeveloperRingMeter(progress: max(model.e2ePassRate, model.e2eLatestResults.isEmpty ? 0.08 : 0.02), tint: statusTint)
                .frame(width: 70, height: 70)

            VStack(alignment: .leading, spacing: 7) {
                HStack(spacing: 8) {
                    Text(statusTitle)
                        .font(.headline.weight(.semibold))
                    StatusPill(text: statusPill, color: statusTint)
                }

                Text(statusDetail)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 8) {
                    MiniMetric(title: "pass", value: "\(model.e2eLatestReport?.passed ?? 0)")
                    MiniMetric(title: "fail", value: "\(model.e2eLatestReport?.failed ?? 0)")
                    MiniMetric(title: "live", value: "\(model.e2eLiveResultCount)")
                    MiniMetric(title: "static", value: "\(model.e2eStaticResultCount)")
                }
            }

            Spacer(minLength: 0)
        }
        .padding(14)
        .dashboardCard(cornerRadius: 18)
    }

    private var statusTitle: String {
        guard let report = model.e2eLatestReport else { return "No E2E report loaded" }
        return report.failed == 0 ? "Latest E2E passing" : "Latest E2E failing"
    }

    private var statusDetail: String {
        guard let report = model.e2eLatestReport else {
            return "Open the complete runner, execute standard coverage, then export live E2E JSON."
        }
        return "\(report.results.count) result(s), \(model.e2eLiveResultCount) live model-backed, \(model.e2eStaticResultCount) static guard."
    }

    private var statusPill: String {
        guard let report = model.e2eLatestReport else { return "missing" }
        return report.failed == 0 ? "passing" : "failing"
    }

    private var statusTint: Color {
        guard let report = model.e2eLatestReport else { return Theme.textSecondary }
        return report.failed == 0 ? .green : .red
    }
}

private struct DeveloperE2ESuiteCard: View {
    let title: String
    let detail: String
    let scenarios: [E2ETestScenario]
    let isCovered: Bool
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: isCovered ? "checkmark.seal.fill" : "circle.dashed")
                    .font(.headline)
                    .foregroundStyle(isCovered ? .green : tint)
                Spacer()
                Text(isCovered ? "covered" : "run")
                    .font(.caption2.monospaced().weight(.bold))
                    .foregroundStyle(isCovered ? .green : tint)
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
                    .lineLimit(2)
            }

            HStack(spacing: 8) {
                MiniMetric(title: "total", value: "\(scenarios.count)")
                MiniMetric(title: "live", value: "\(scenarios.filter(\.requiresAgentRun).count)")
                MiniMetric(title: "static", value: "\(scenarios.filter { !$0.requiresAgentRun }.count)")
            }
        }
        .padding(13)
        .frame(minHeight: 142, alignment: .topLeading)
        .dashboardCard(cornerRadius: 16)
    }
}

private struct DeveloperE2ELatestResults: View {
    let results: [E2ETestResult]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Latest scenario signals")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Theme.textSecondary)
            ForEach(results) { result in
                HStack(alignment: .top, spacing: 9) {
                    Image(systemName: result.statusIcon)
                        .foregroundStyle(result.statusColor)
                        .frame(width: 18)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(result.title)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(Theme.textPrimary)
                            .lineLimit(2)
                        Text(result.failures.first ?? "\(result.actualIntent) · \(result.requiresAgentRun ? "live" : "static")")
                            .font(.caption2)
                            .foregroundStyle(result.passed ? Theme.textSecondary : result.statusColor)
                            .lineLimit(2)
                    }
                    Spacer(minLength: 0)
                }
            }
        }
        .padding(14)
        .dashboardCard(cornerRadius: 16)
    }
}

private struct DeveloperEvidenceSection: View {
    let layers: [DeveloperEvidenceLayerStatus]
    let findings: [DeveloperFinding]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            DeveloperSectionHeader(title: "Evidence Layers", subtitle: "Diagnostics stay separate from live scenario ownership.")
            VStack(spacing: 10) {
                ForEach(layers) { layerStatus in
                    DeveloperEvidenceLayerCard(layerStatus: layerStatus)
                }
            }

            DeveloperSectionHeader(title: "Findings", subtitle: "Current gaps and authority rules.")
            VStack(spacing: 10) {
                ForEach(findings) { finding in
                    DeveloperFindingCard(finding: finding)
                }
            }
        }
    }
}

private struct DeveloperReportsSection: View {
    @Bindable var model: DeveloperConsoleModel
    let presentReport: (String, String) -> Void
    let appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            DeveloperSectionHeader(title: "Reports", subtitle: "Readable text exports for debugging and handoff.")
            VStack(spacing: 10) {
                DeveloperReportButton(title: "Runtime debug text", subtitle: "Generation, adapter, model, and runtime flags.", systemImage: "doc.text") {
                    presentReport("Runtime Debug", model.debugText(appState: appState))
                }
                DeveloperReportButton(title: "Diagnostics text", subtitle: "Permissions, parser noise, and latest E2E report.", systemImage: "doc.plaintext") {
                    presentReport("Diagnostics Text", model.diagnosticText())
                }
                DeveloperReportButton(title: "Recent logs", subtitle: "Imported files, model files, and parser summaries.", systemImage: "list.bullet.rectangle") {
                    presentReport("Recent Logs", model.logsText())
                }
                DeveloperReportButton(title: "Storage checks", subtitle: "Directory readability and writeability.", systemImage: "externaldrive.badge.checkmark") {
                    presentReport("Storage Checks", model.runStorageChecks())
                }
            }

            if !model.findings.isEmpty {
                DeveloperSectionHeader(title: "Findings", subtitle: "Dynamic blockers and runtime evidence gaps.")
                VStack(spacing: 10) {
                    ForEach(model.findings) { finding in
                        DeveloperFindingCard(finding: finding)
                    }
                }
            }
        }
    }
}

private struct DeveloperPrivacySection: View {
    let snapshot: DiagnosticsSnapshot?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            DeveloperSectionHeader(title: "Privacy Boundaries", subtitle: "Keep runtime exports useful without weakening local-first policy.")
            VStack(spacing: 10) {
                DeveloperPrivacyRow(title: "Local-only mode", value: snapshot?.privacy.localOnlyMode == true ? "Enabled" : "Review", systemImage: "lock.shield")
                DeveloperPrivacyRow(title: "Network access", value: snapshot?.privacy.networkAccessState ?? "cached", systemImage: "network")
                DeveloperPrivacyRow(title: "Tool categories", value: snapshot?.privacy.recentToolCategories.joined(separator: ", ") ?? "none", systemImage: "wrench.and.screwdriver")
                ForEach(snapshot?.privacy.appIntentLimitations ?? ["Sensitive actions require open-app approval"], id: \.self) { limitation in
                    DeveloperPrivacyRow(title: "Limitation", value: limitation, systemImage: "hand.raised")
                }
            }
        }
    }
}

private struct DeveloperSurfaceGrid: View {
    private let columns = [
        GridItem(.flexible(), spacing: 10),
        GridItem(.flexible(), spacing: 10)
    ]

    var body: some View {
        LazyVGrid(columns: columns, spacing: 10) {
            NavigationLink {
                LumenControlTowerView()
            } label: {
                DeveloperSurfaceTile(title: "Control Tower", subtitle: "Workflow timeline", systemImage: "scope", tint: Theme.accent)
            }
            NavigationLink {
                DiagnosticsView()
            } label: {
                DeveloperSurfaceTile(title: "Diagnostics", subtitle: "Device state", systemImage: "stethoscope", tint: .cyan)
            }
            NavigationLink {
                E2ETestRunnerView(initialRunMode: .standard)
            } label: {
                DeveloperSurfaceTile(title: "E2E Tests", subtitle: "Live suites", systemImage: "testtube.2", tint: .green)
            }
            NavigationLink {
                DeveloperGroundingDestination()
            } label: {
                DeveloperSurfaceTile(title: "Grounding", subtitle: "Evidence audit", systemImage: "point.3.connected.trianglepath.dotted", tint: .orange)
            }
            NavigationLink {
                DeveloperRuntimeDestination()
            } label: {
                DeveloperSurfaceTile(title: "Runtime", subtitle: "Meters", systemImage: "gauge.with.dots.needle.50percent", tint: .purple)
            }
        }
        .buttonStyle(.plain)
    }
}

private struct DeveloperGroundingDestination: View {
    @State private var model = DeveloperConsoleModel()

    var body: some View {
        Group {
            if let grounding = model.diagnosticsSnapshot?.grounding {
                GroundingDiagnosticsView(grounding: grounding)
            } else {
                ContentUnavailableView("Grounding unavailable", systemImage: "point.3.connected.trianglepath.dotted")
            }
        }
        .task { model.loadCachedDiagnostics() }
    }
}

private struct DeveloperRuntimeDestination: View {
    @State private var model = DeveloperConsoleModel()

    var body: some View {
        Group {
            if let runtime = model.diagnosticsSnapshot?.runtime {
                RuntimeDashboardView(runtime: runtime)
            } else {
                ContentUnavailableView("Runtime unavailable", systemImage: "gauge.with.dots.needle.50percent")
            }
        }
        .task { model.loadCachedDiagnostics() }
    }
}

private struct DeveloperMeterCard: View {
    let meter: DeveloperDashboardMeter

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: meter.systemImage)
                    .font(.headline)
                    .foregroundStyle(tint)
                    .frame(width: 28, height: 28)
                    .background(tint.opacity(0.14), in: Circle())
                Spacer()
                Text(meter.value)
                    .font(.caption.monospacedDigit().weight(.semibold))
                    .foregroundStyle(tint)
                    .lineLimit(1)
                    .minimumScaleFactor(0.74)
            }
            Gauge(value: min(max(meter.progress, 0), 1)) {
                Text(meter.title)
            }
            .gaugeStyle(.accessoryLinearCapacity)
            .tint(tint)
            VStack(alignment: .leading, spacing: 3) {
                Text(meter.title)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                Text(meter.detail)
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(13)
        .frame(minHeight: 136, alignment: .topLeading)
        .dashboardCard(cornerRadius: 18)
    }

    private var tint: Color {
        switch meter.level {
        case .nominal: return .green
        case .warning: return Theme.accent
        case .critical: return .red
        case .inactive: return Theme.textSecondary
        }
    }
}

private struct DeveloperSharpTelemetryMeter: View {
    let meter: DeveloperTelemetryMeter

    var body: some View {
        VStack(alignment: .leading, spacing: 11) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: meter.systemImage)
                    .font(.headline)
                    .foregroundStyle(meter.tint)
                    .frame(width: 30, height: 30)
                    .background(meter.tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                Spacer(minLength: 4)
                VStack(alignment: .trailing, spacing: 0) {
                    HStack(alignment: .firstTextBaseline, spacing: 2) {
                        Text(meter.value)
                            .font(.system(size: 19, weight: .bold, design: .rounded).monospacedDigit())
                            .foregroundStyle(Theme.textPrimary)
                            .lineLimit(1)
                            .minimumScaleFactor(0.62)
                        if !meter.unit.isEmpty {
                            Text(meter.unit)
                                .font(.caption2.monospaced().weight(.semibold))
                                .foregroundStyle(Theme.textSecondary)
                        }
                    }
                    Text(meter.styleLabel)
                        .font(.caption2.monospaced())
                        .foregroundStyle(meter.tint)
                        .lineLimit(1)
                }
            }

            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .fill(Theme.border.opacity(0.38))
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .fill(meter.tint)
                        .frame(width: max(6, proxy.size.width * min(max(meter.progress, 0), 1)))
                    HStack(spacing: 3) {
                        ForEach(0..<12, id: \.self) { _ in
                            Rectangle()
                                .fill(Color.black.opacity(0.18))
                                .frame(width: 1)
                            Spacer(minLength: 0)
                        }
                    }
                    .padding(.horizontal, 4)
                }
            }
            .frame(height: 9)

            VStack(alignment: .leading, spacing: 3) {
                Text(meter.title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                    .lineLimit(1)
                Text(meter.detail)
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(13)
        .frame(minHeight: 132, alignment: .topLeading)
        .dashboardCard(cornerRadius: 16)
    }
}

private extension DeveloperTelemetryMeter {
    var styleLabel: String {
        switch style {
        case .percent: return "percent"
        case .value: return "live"
        case .availability: return "state"
        }
    }
}

private struct DeveloperRingMeter: View {
    let progress: Double
    let tint: Color

    var body: some View {
        ZStack {
            Circle()
                .stroke(Theme.border.opacity(0.55), lineWidth: 9)
            Circle()
                .trim(from: 0, to: min(max(progress, 0.04), 1))
                .stroke(tint, style: StrokeStyle(lineWidth: 9, lineCap: .round))
                .rotationEffect(.degrees(-90))
            Text("\(Int(min(max(progress, 0), 1) * 100))")
                .font(.headline.monospacedDigit().weight(.bold))
                .foregroundStyle(Theme.textPrimary)
                .accessibilityHidden(true)
        }
        .accessibilityLabel("Workflow progress \(Int(progress * 100)) percent")
    }
}

private struct DeveloperWorkflowActionCard: View {
    let action: DeveloperWorkflowAction
    let isActive: Bool
    let isComplete: Bool

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: statusImage)
                .font(.headline)
                .foregroundStyle(statusColor)
                .frame(width: 34, height: 34)
                .background(statusColor.opacity(0.14), in: Circle())
            VStack(alignment: .leading, spacing: 4) {
                Text(action.title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(action.detail)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            Text(isActive ? "now" : isComplete ? "done" : "next")
                .font(.caption2.monospaced().weight(.semibold))
                .foregroundStyle(statusColor)
        }
        .padding(14)
        .dashboardCard(cornerRadius: 16)
    }

    private var statusImage: String {
        if isActive { return "arrow.trianglehead.2.clockwise.rotate.90" }
        if isComplete { return "checkmark.circle.fill" }
        return action.systemImage
    }

    private var statusColor: Color {
        if isActive { return Theme.accent }
        if isComplete { return .green }
        return Theme.textSecondary
    }
}

private struct DeveloperEvidenceLayerCard: View {
    let layerStatus: DeveloperEvidenceLayerStatus

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: layerStatus.layer.systemImage)
                .font(.title3)
                .foregroundStyle(tint)
                .frame(width: 36, height: 36)
                .background(tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Text(layerStatus.layer.title)
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    StatusPill(text: layerStatus.status, color: tint)
                }
                Text(layerStatus.detail)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(layerStatus.layer.privacySummary)
                    .font(.caption2)
                    .foregroundStyle(Theme.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .dashboardCard(cornerRadius: 16)
    }

    private var tint: Color {
        if layerStatus.layer.ownsLiveE2EScenarios { return .green }
        if layerStatus.isBlocking { return .red }
        return Theme.accent
    }
}

private struct DeveloperFindingCard: View {
    let finding: DeveloperFinding

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: finding.severity.systemImage)
                .font(.title3)
                .foregroundStyle(finding.severity.color)
                .frame(width: 34, height: 34)
            VStack(alignment: .leading, spacing: 5) {
                Text(finding.title)
                    .font(.subheadline.weight(.semibold))
                Text(finding.detail)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .dashboardCard(cornerRadius: 16)
    }
}

private struct DeveloperActionRow: View {
    let title: String
    let detail: String
    let systemImage: String
    let tint: Color
    var isRunning = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(tint.opacity(0.14))
                    if isRunning {
                        ProgressView()
                            .tint(tint)
                    } else {
                        Image(systemName: systemImage)
                            .font(.headline)
                            .foregroundStyle(tint)
                    }
                }
                .frame(width: 38, height: 38)

                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.textTertiary)
            }
            .padding(14)
            .dashboardCard(cornerRadius: 16)
        }
        .buttonStyle(.plain)
        .disabled(isRunning)
    }
}

private struct DeveloperNavigationRow: View {
    let title: String
    let detail: String
    let systemImage: String
    let tint: Color

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: systemImage)
                .font(.headline)
                .foregroundStyle(tint)
                .frame(width: 38, height: 38)
                .background(tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 10, style: .continuous))

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            Image(systemName: "chevron.right")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Theme.textTertiary)
        }
        .padding(14)
        .dashboardCard(cornerRadius: 16)
    }
}

private struct DeveloperReportButton: View {
    let title: String
    let subtitle: String
    let systemImage: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: systemImage)
                    .font(.title3)
                    .foregroundStyle(Theme.accent)
                    .frame(width: 36, height: 36)
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
            }
            .padding(14)
            .dashboardCard(cornerRadius: 16)
        }
        .buttonStyle(.plain)
    }
}

private struct MiniMetric: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(value)
                .font(.caption.monospacedDigit().weight(.bold))
                .foregroundStyle(Theme.textPrimary)
                .lineLimit(1)
            Text(title.uppercased())
                .font(.caption2.monospaced())
                .foregroundStyle(Theme.textTertiary)
                .lineLimit(1)
        }
        .frame(minWidth: 34, alignment: .leading)
    }
}

private struct DeveloperSurfaceTile: View {
    let title: String
    let subtitle: String
    let systemImage: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: systemImage)
                    .font(.headline)
                    .foregroundStyle(tint)
                    .frame(width: 32, height: 32)
                    .background(tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                Spacer()
                Image(systemName: "arrow.up.right")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(Theme.textTertiary)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                Text(subtitle)
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
                    .lineLimit(2)
            }
        }
        .padding(13)
        .frame(minHeight: 112, alignment: .topLeading)
        .dashboardCard(cornerRadius: 18)
    }
}

private struct DeveloperPrivacyRow: View {
    let title: String
    let value: String
    let systemImage: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: systemImage)
                .font(.headline)
                .foregroundStyle(Theme.accent)
                .frame(width: 32, height: 32)
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Text(value)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
        .padding(14)
        .dashboardCard(cornerRadius: 16)
    }
}

private struct DeveloperSectionHeader: View {
    let title: String
    let subtitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.headline.weight(.semibold))
            Text(subtitle)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.top, 2)
    }
}

private struct StatusPill: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text.uppercased())
            .font(.caption2.monospaced().weight(.bold))
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(color.opacity(0.13), in: Capsule())
            .lineLimit(1)
            .minimumScaleFactor(0.75)
    }
}

private struct DeveloperConsoleReportTextView: View {
    let title: String
    let bodyText: String
    let exportURL: URL?
    @State private var copiedLabel: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if let exportURL {
                    DeveloperExportHandoffCard(
                        title: exportURL.lastPathComponent,
                        subtitle: "Improve-loop evidence JSON",
                        url: exportURL
                    )
                    .padding(.horizontal)
                    .padding(.top)
                }

                Text(bodyText)
                    .font(.footnote.monospaced())
                    .foregroundStyle(Theme.textPrimary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal)

                if let copiedLabel {
                    Text(copiedLabel)
                        .font(.caption.monospaced())
                        .foregroundStyle(.green)
                        .padding(.horizontal)
                }
            }
            .padding(.bottom)
        }
        .background(AppBackground())
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Button {
                    UIPasteboard.general.string = bodyText
                    copiedLabel = "report copied"
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .accessibilityLabel("Copy report text")

                if let exportURL {
                    ShareLink(item: exportURL) {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .accessibilityLabel("Share export JSON")
                }
            }
        }
    }
}

private extension View {
    func dashboardCard(cornerRadius: CGFloat = 20) -> some View {
        self
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(Theme.surfaceHigh.opacity(0.76))
                    .overlay(
                        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                            .stroke(Theme.border.opacity(0.5), lineWidth: 0.8)
                    )
            )
    }
}
