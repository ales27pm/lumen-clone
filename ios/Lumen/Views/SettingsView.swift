import SwiftUI
import SwiftData

struct SettingsView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @State private var showDeveloperAlert = false
    @State private var developerAlertMessage = ""
    @State private var parseFailureSummary = "• Parse-failure traces: loading…"
    @State private var parseNoiseSummary = "• Recoverable noise traces: loading…"
    @State private var selectedModelFamily = LumenModelFamily.persistedSelected
    @State private var isSwitchingModelFamily = false

    var body: some View {
        @Bindable var state = appState

        NavigationStack {
            Form {
                Section("Prompt Presets") {
                    Picker("Preset", selection: Binding(
                        get: { state.selectedPresetID },
                        set: { id in
                            if let p = Presets.all.first(where: { $0.id == id }) {
                                state.applyPreset(p)
                            }
                        }
                    )) {
                        ForEach(Presets.all) { preset in
                            Label(preset.name, systemImage: preset.icon).tag(preset.id)
                        }
                    }
                }

                Section("Agent") {
                    Toggle("Agent mode", isOn: Binding(get: { state.agentModeEnabled }, set: { state.agentModeEnabled = $0 }))
                    Toggle("Show thinking by default", isOn: Binding(get: { state.showThinkingByDefault }, set: { state.showThinkingByDefault = $0 }))
                    Stepper(value: Binding(get: { state.maxAgentSteps }, set: { state.maxAgentSteps = $0 }), in: 1...10) {
                        HStack {
                            Text("Max steps")
                            Spacer()
                            Text("\(state.maxAgentSteps)")
                                .font(.callout.monospacedDigit())
                                .foregroundStyle(Theme.textSecondary)
                        }
                    }
                }

                Section {
                    Picker("Model family", selection: $selectedModelFamily) {
                        ForEach(LumenModelFamily.allCases) { family in
                            Text(family.displayName).tag(family)
                        }
                    }
                    .accessibilityIdentifier("settings.fleet.modelFamily")

                    VStack(alignment: .leading, spacing: 4) {
                        Text(selectedModelFamily.description)
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                        Text("First launch downloads only this family’s bootstrap chat and embedding artifacts.")
                            .font(.caption2)
                            .foregroundStyle(Theme.textTertiary)
                    }

                    Toggle("Auto-download fleet", isOn: Binding(get: { state.autoDownloadFleetModels }, set: { state.autoDownloadFleetModels = $0 }))
                    Toggle("Ask before fleet downloads", isOn: Binding(get: { state.confirmFleetDownloads }, set: { state.confirmFleetDownloads = $0 }))

                    Button {
                        switchModelFamily(selectedModelFamily)
                    } label: {
                        HStack {
                            Label(isSwitchingModelFamily ? "Switching…" : "Download / repair selected family", systemImage: "arrow.down.circle")
                            Spacer()
                            if isSwitchingModelFamily { ProgressView() }
                        }
                    }
                    .disabled(isSwitchingModelFamily)
                    .accessibilityIdentifier("settings.fleet.repairSelectedFamily")
                } header: {
                    Text("Fleet")
                } footer: {
                    Text("Qwen3 is the default bootstrap family. Switching resets active model IDs and downloads only the selected family artifacts instead of the entire historical catalog.")
                }

                Section("Voice") {
                    Toggle("Hands-free", isOn: Binding(get: { state.handsFree }, set: { state.handsFree = $0 }))
                    HStack {
                        Text("Speaking rate")
                        Spacer()
                        Text(String(format: "%.2f", state.speakingRate))
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(Theme.textSecondary)
                    }
                    Slider(value: Binding(get: { state.speakingRate }, set: { state.speakingRate = $0 }), in: 0.3...0.7)
                    NavigationLink {
                        VoicePickerList()
                    } label: {
                        HStack {
                            Text("Voice")
                            Spacer()
                            Text(currentVoiceName).foregroundStyle(Theme.textSecondary)
                        }
                    }
                }

                Section("System Prompt") {
                    TextEditor(text: Binding(get: { state.systemPrompt }, set: { state.systemPrompt = $0 }))
                        .frame(minHeight: 120)
                        .font(.footnote)
                }

                Section("Generation") {
                    sliderRow("Temperature", value: Binding(get: { state.temperature }, set: { state.temperature = $0 }), range: 0...2, format: "%.2f")
                    sliderRow("Top-P", value: Binding(get: { state.topP }, set: { state.topP = $0 }), range: 0...1, format: "%.2f")
                    sliderRow("Repetition penalty", value: Binding(get: { state.repetitionPenalty }, set: { state.repetitionPenalty = $0 }), range: 1...1.5, format: "%.2f")
                    Stepper(value: Binding(get: { state.contextSize }, set: { state.contextSize = $0 }), in: 1024...32768, step: 1024) {
                        HStack {
                            Text("Context size")
                            Spacer()
                            Text("\(state.contextSize)")
                                .font(.callout.monospacedDigit())
                                .foregroundStyle(Theme.textSecondary)
                        }
                    }
                    Stepper(value: Binding(get: { state.maxTokens }, set: { state.maxTokens = $0 }), in: 128...4096, step: 128) {
                        HStack {
                            Text("Max output")
                            Spacer()
                            Text("\(state.maxTokens)")
                                .font(.callout.monospacedDigit())
                                .foregroundStyle(Theme.textSecondary)
                        }
                    }
                }

                Section("Memory") {
                    Toggle("Auto-remember", isOn: Binding(get: { state.autoMemory }, set: { state.autoMemory = $0 }))
                }

                Section("Developer") {
                    Toggle("Developer trace mode", isOn: Binding(get: { state.developerTraceModeEnabled }, set: { state.developerTraceModeEnabled = $0 }))
                        .accessibilityIdentifier("settings.developer.traceMode")
                    Toggle("Capture reasoning", isOn: Binding(get: { state.developerReasoningCaptureEnabled }, set: { state.developerReasoningCaptureEnabled = $0 }))
                        .disabled(!state.developerTraceModeEnabled)
                        .accessibilityIdentifier("settings.developer.reasoningCapture")

                    NavigationLink {
                        AgentGroundingAuditView(registryProvider: LiveRuntimeToolRegistryProvider())
                    } label: {
                        Label("Agent grounding audit", systemImage: "checkmark.seal.text.page")
                    }
                    .accessibilityIdentifier("settings.developer.agentGroundingAudit")

                    NavigationLink {
                        E2ETestRunnerView()
                    } label: {
                        Label("End-to-end tests", systemImage: "testtube.2")
                    }
                    .accessibilityIdentifier("settings.developer.e2eTests")

                    Button {
                        runDeveloperChecks()
                    } label: {
                        Label("Run storage checks", systemImage: "checkmark.circle")
                    }
                    .accessibilityIdentifier("settings.developer.runTests")

                    NavigationLink {
                        DeveloperTextView(title: "Logs", bodyText: logsText)
                    } label: {
                        Label("Logs", systemImage: "doc.text.magnifyingglass")
                    }
                    .accessibilityIdentifier("settings.developer.logs")

                    NavigationLink {
                        DeveloperTextView(title: "Debug", bodyText: debugText)
                    } label: {
                        Label("Debug", systemImage: "ladybug")
                    }
                    .accessibilityIdentifier("settings.developer.debug")

                    NavigationLink {
                        DeveloperTextView(title: "Diagnostic", bodyText: diagnosticText)
                    } label: {
                        Label("Diagnostic", systemImage: "stethoscope")
                    }
                    .accessibilityIdentifier("settings.developer.diagnostic")
                }

                Section {
                    NavigationLink {
                        PermissionsView()
                    } label: {
                        Label("Permissions", systemImage: "hand.raised")
                    }
                } header: {
                    Text("Privacy")
                } footer: {
                    Text("Review which system features Lumen can access.")
                }

                Section("Diagnostics") {
                    NavigationLink {
                        DiagnosticsView()
                    } label: {
                        Label("Diagnostics", systemImage: "waveform.path.ecg")
                    }
                }

                Section("About") {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Lumen").font(.subheadline.weight(.semibold))
                        Text("Runs open-source language models locally via llama.cpp. Embeddings stored in local SQLite. Tool calls executed on-device. Nothing leaves your phone.")
                            .font(.footnote)
                            .foregroundStyle(Theme.textSecondary)
                    }
                    .padding(.vertical, 2)
                }
            }
            .navigationTitle("Settings")
            .task {
                selectedModelFamily = LumenModelFamily.persistedSelected
                await refreshParseFailureSummary()
            }
            .onChange(of: selectedModelFamily) { _, family in
                LumenModelFamily.persistedSelected = family
            }
            .alert("Run checks", isPresented: $showDeveloperAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(developerAlertMessage)
            }
        }
    }

    private var currentVoiceName: String {
        if let id = appState.voiceID,
           let v = VoiceCatalog.available().first(where: { $0.id == id }) {
            return v.name
        }
        return "System default"
    }

    private func switchModelFamily(_ family: LumenModelFamily) {
        guard !isSwitchingModelFamily else { return }
        isSwitchingModelFamily = true
        Task { @MainActor in
            await ModelLaunchBootstrap.switchFamily(family, appState: appState, context: modelContext)
            isSwitchingModelFamily = false
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        }
    }

    private func sliderRow(_ label: String, value: Binding<Double>, range: ClosedRange<Double>, format: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(Theme.textSecondary)
            }
            Slider(value: value, in: range)
        }
    }

    private var logsText: String {
        let modelsDirectory = ModelStorage.modelsDirectoryURL()
        let imported = FileStore.importedFiles()
        let modelFiles = (try? FileManager.default.contentsOfDirectory(at: modelsDirectory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
        return """
        Last launch diagnostics:
        • Imported files: \(imported.count)
        • Model files: \(modelFiles.count)
        • Models path: \(modelsDirectory.path)
        \(parseFailureSummary)
        """
    }

    private var debugText: String {
        """
        Runtime:
        • isGenerating: \(appState.isGenerating ? "true" : "false")
        • agentModeEnabled: \(appState.agentModeEnabled ? "true" : "false")
        • showThinkingByDefault: \(appState.showThinkingByDefault ? "true" : "false")
        • developerTraceModeEnabled: \(appState.developerTraceModeEnabled ? "true" : "false")
        • developerReasoningCaptureEnabled: \(appState.developerReasoningCaptureEnabled ? "true" : "false")
        • maxAgentSteps: \(appState.maxAgentSteps)

        Fleet:
        • selectedModelFamily: \(LumenModelFamily.persistedSelected.rawValue)
        • autoDownloadFleetModels: \(appState.autoDownloadFleetModels ? "true" : "false")
        • confirmFleetDownloads: \(appState.confirmFleetDownloads ? "true" : "false")

        Generation:
        • temperature: \(String(format: "%.2f", appState.temperature))
        • topP: \(String(format: "%.2f", appState.topP))
        • repetitionPenalty: \(String(format: "%.2f", appState.repetitionPenalty))
        • contextSize: \(appState.contextSize)
        • maxTokens: \(appState.maxTokens)
        """
    }

    private var diagnosticText: String {
        let permissions = PermissionKind.allCases
            .map { "\($0.title): \(PermissionsCenter.shared.state($0).label)" }
            .joined(separator: "\n")
        return """
        Permissions:
        \(permissions)

        Recoverable noise signatures:
        \(parseNoiseSummary)

        Latest E2E:
        \(E2ETestLogStore.latestText())
        """
    }

    private func runDeveloperChecks() {
        let fm = FileManager.default
        let modelsDirectory = ModelStorage.modelsDirectoryURL(fileManager: fm)
        let canReadModels = fm.isReadableFile(atPath: modelsDirectory.path)
        let canWriteModels = fm.isWritableFile(atPath: modelsDirectory.path)
        let importsDirectory = FileStore.importsDirectory
        let canReadImports = fm.isReadableFile(atPath: importsDirectory.path)
        let canWriteImports = fm.isWritableFile(atPath: importsDirectory.path)
        let e2eDirectory = try? E2ETestLogStore.reportsDirectory()
        let canWriteE2E = e2eDirectory.map { fm.isWritableFile(atPath: $0.path) } ?? false

        let checks: [(String, Bool)] = [
            ("Models folder readable", canReadModels),
            ("Models folder writable", canWriteModels),
            ("Imports folder readable", canReadImports),
            ("Imports folder writable", canWriteImports),
            ("E2E folder writable", canWriteE2E),
        ]

        let passed = checks.filter(\.1).count
        let summary = checks
            .map { check in "• \(check.0): \(check.1 ? "PASS" : "FAIL")" }
            .joined(separator: "\n")
        developerAlertMessage = "\(passed)/\(checks.count) checks passed\n\n\(summary)"
        showDeveloperAlert = true
    }

    @MainActor
    private func refreshParseFailureSummary() async {
        // Detached is intentional so diagnostics file parsing does not inherit
        // the caller's actor (SettingsView `.task` is main-actor bound).
        // The closure only reads snapshot files and returns Sendable Strings.
        let summary = await Task.detached(priority: .utility) {
            (
                AgentParseFailureSummaryLoader.developerText(topN: 5),
                AgentParseNoiseSummaryLoader.developerText(topN: 5)
            )
        }.value
        parseFailureSummary = summary.0
        parseNoiseSummary = summary.1
    }
}

private struct DeveloperTextView: View {
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

private struct E2ETestRunnerView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @State private var isRunning = false
    @State private var runMode: RunMode = .standard
    @State private var reportText = E2ETestLogStore.latestText()
    @State private var latestReport: E2ETestReport? = E2ETestLogStore.latestReport()
    @State private var liveResults: [E2ETestResult] = []
    @State private var liveEventBuffer: [E2ETestEvent] = []
    @State private var runStartedAt: Date?
    @State private var lastExportURL: URL?
    @State private var exportError: String?

    var body: some View {
        List {
            Section("Dashboard") {
                E2ETestDashboardView(
                    totalScenarioCount: runMode.scenarios.count,
                    results: dashboardResults,
                    report: latestReport,
                    isRunning: isRunning,
                    runStartedAt: runStartedAt
                )
                .accessibilityIdentifier("e2e.dashboard")
            }

            Section {
                Picker("Mode", selection: $runMode) {
                    ForEach(RunMode.allCases, id: \.self) { mode in
                        Text(mode.title).tag(mode)
                    }
                }

                Button {
                    run()
                } label: {
                    HStack {
                        Label(isRunning ? "Running…" : runMode.buttonTitle, systemImage: "play.circle")
                        Spacer()
                        if isRunning { ProgressView() }
                    }
                }
                .disabled(isRunning)

                Button {
                    reloadLatestReport()
                } label: {
                    Label("Reload latest report", systemImage: "arrow.clockwise")
                }

                Button {
                    exportLatestReport()
                } label: {
                    Label("Export Live E2E Report JSON", systemImage: "square.and.arrow.up")
                }
                .disabled(latestReport == nil)

                if let lastExportURL {
                    LabeledContent("Last E2E export", value: lastExportURL.lastPathComponent)
                        .font(.caption)
                    ShareLink(item: lastExportURL) {
                        Label("Share Live E2E JSON", systemImage: "square.and.arrow.up")
                    }
                }
            } footer: {
                Text(runMode.footerText)
            }

            if let exportError {
                Section("Export Error") {
                    Text(exportError).foregroundStyle(.red)
                }
            }

            if !failureBuckets.isEmpty {
                Section("Failure Breakdown") {
                    ForEach(failureBuckets) { bucket in
                        LabeledContent(bucket.name, value: "\(bucket.count)")
                    }
                }
            }

            if !dashboardResults.isEmpty {
                Section("Latest Results") {
                    ForEach(dashboardResults) { result in
                        E2ETestResultRow(result: result)
                    }
                }
            }

            if !eventLogEntries.isEmpty {
                Section {
                    E2ERealtimeLogView(entries: eventLogEntries, isRunning: isRunning)
                } header: {
                    Text("Real-time Logs")
                } footer: {
                    Text("Streaming event feed for each scenario run (intent, model readiness, tool steps, final hints, and final output).")
                }
            }

            if let accel = dashboardResults.last?.performanceMatrix?.accelerationDiagnostics {
                Section("Acceleration Diagnostics") {
                    LabeledContent("Metal device available", value: accel.metalDeviceAvailable ? "yes" : "no")
                    LabeledContent("Metal device", value: accel.metalDeviceName ?? "unknown")
                    LabeledContent("Backend requested", value: accel.backendRequested)
                    LabeledContent("Requested GPU layers", value: accel.requestedGpuLayers.map(String.init) ?? "nil")
                    LabeledContent("KQV offload requested", value: accel.requestedKQVOffload == true ? "true" : "false")
                    LabeledContent("Actual backend", value: accel.actualBackend ?? "unknown")
                    LabeledContent("Metal device used", value: accel.metalDeviceUsed ?? "unknown")
                    LabeledContent("Offloaded layers", value: "\(accel.actualOffloadedLayers.map(String.init) ?? "unknown") / \(accel.actualTotalLayers.map(String.init) ?? "unknown")")
                    LabeledContent("KQV/cache offload", value: accel.actualKQVOffload.map { $0 ? "true" : "false" } ?? "unknown")
                    LabeledContent("Prompt eval tok/s", value: accel.promptEvalTokensPerSecond.map { String(format: "%.1f", $0) } ?? "unknown")
                    LabeledContent("Decode tok/s", value: accel.decodeTokensPerSecond.map { String(format: "%.1f", $0) } ?? "unknown")
                    LabeledContent("Verification", value: accel.verificationLevel)
                    LabeledContent("ANE used by runtime", value: accel.aneUsedByCurrentRuntime ? "yes" : "no")
                    ForEach(accel.notes, id: \.self) { note in
                        Text("• \(note)")
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                    }
                }
            }

            Section("Scenarios") {
                ForEach(runMode.scenarios) { scenario in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(scenario.title)
                            .font(.subheadline.weight(.medium))
                        Text(scenario.prompt)
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                        Text("Intent: \(scenario.expectedIntent.rawValue) · \(scenario.kind.rawValue) · agent run: \(scenario.requiresAgentRun ? "yes" : "no")")
                            .font(.caption2.monospaced())
                            .foregroundStyle(Theme.textTertiary)
                    }
                    .padding(.vertical, 2)
                }
            }

            Section("Latest Report") {
                Text(reportText)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
            }
        }
        .navigationTitle("E2E Tests")
        .onChange(of: runMode) { _, _ in
            reportText = E2ETestLogStore.latestText()
            latestReport = nil
            liveResults = []
            runStartedAt = nil
            lastExportURL = nil
            exportError = nil
        }
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            reloadLatestReport()
        }
    }

    private var dashboardResults: [E2ETestResult] {
        if isRunning || !liveResults.isEmpty { return liveResults }
        return latestReport?.results ?? []
    }

    private var failureBuckets: [E2EFailureBucket] {
        let buckets = Dictionary(grouping: dashboardResults.flatMap(\.failures)) { failure in
            failureCategory(for: failure)
        }
        return ["intent", "tool-boundary", "response-quality", "runtime", "hygiene", "other"]
            .compactMap { key in
                guard let count = buckets[key]?.count else { return nil }
                return E2EFailureBucket(name: displayName(forFailureCategory: key), count: count)
            }
    }

    private func reloadLatestReport() {
        reportText = E2ETestLogStore.latestText()
        latestReport = E2ETestLogStore.latestReport()
        liveResults = []
        liveEventBuffer = []
        runStartedAt = nil
    }

    private func run() {
        guard !isRunning else { return }
        isRunning = true
        exportError = nil
        latestReport = nil
        liveResults = []
        liveEventBuffer = []
        runStartedAt = Date()
        reportText = runMode.runningLabel
        Task { @MainActor in
            let report: E2ETestReport
            switch runMode {
            case .standard:
                report = await E2ETestRunner.runStandard(appState: appState, context: modelContext, onResult: { result in
                    liveResults.append(result)
                    reportText = inProgressReportText(results: liveResults, total: runMode.scenarios.count)
                }, onEvent: { event in
                    liveEventBuffer.append(event)
                })
            case .trainingValidation:
                report = await E2ETestRunner.runTrainingValidation(appState: appState, context: modelContext, onResult: { result in
                    liveResults.append(result)
                    reportText = inProgressReportText(results: liveResults, total: runMode.scenarios.count)
                }, onEvent: { event in
                    liveEventBuffer.append(event)
                })
            }
            latestReport = report
            reportText = report.summaryText
            isRunning = false
            runStartedAt = nil
        }
    }

    private func exportLatestReport() {
        guard let latestReport else { return }
        do {
            let result = try EvidenceLayerExporter.writeLayer(
                payload: latestReport,
                filePrefix: "lumen-live-e2e-report",
                format: "live-e2e-test-report-json",
                sourceLayer: "e2eTestReport",
                ownsLiveE2EScenarios: true,
                includesDeterministicStaticScenarios: false,
                privacy: "Contains prompts, final outputs, failures, and event logs from the current local E2E run. Review before sharing outside the improve-loop.",
                notes: [
                    "This is the live E2E model/test layer export.",
                    "Scenarios with requiresAgentRun=true are intended to exercise the loaded SlotAgentService path.",
                    "If a scenario says no model loaded or routing-only checks completed, the offline ingester treats it as invalid E2E evidence."
                ]
            )
            lastExportURL = result.url
            exportError = nil
        } catch {
            exportError = "Live E2E report export failed: \(error.localizedDescription)"
        }
    }

    private func inProgressReportText(results: [E2ETestResult], total: Int) -> String {
        let passed = results.filter(\.passed).count
        let failed = results.count - passed
        return """
        Running \(runMode.title) E2E suite
        Completed: \(results.count)/\(total)
        Passed: \(passed)
        Failed: \(failed)
        """
    }

    private func failureCategory(for failure: String) -> String {
        if failure.contains("Intent mismatch") { return "intent" }
        if failure.contains("Forbidden tool") || failure.contains("Required tool not allowed") || failure.contains("Forbidden tool selected by agent") { return "tool-boundary" }
        if failure.contains("Required final hint") || failure.contains("Forbidden final hint") || failure.contains("RAG") { return "response-quality" }
        if failure.contains("Agent error") || failure.contains("No model loaded") { return "runtime" }
        if failure.contains("Raw output") || failure.contains("Sanitized output") || failure.contains("Sanitizer") || failure.contains("Final output empty") { return "hygiene" }
        return "other"
    }

    private func displayName(forFailureCategory category: String) -> String {
        switch category {
        case "intent": return "Intent"
        case "tool-boundary": return "Tool boundary"
        case "response-quality": return "Response quality"
        case "runtime": return "Runtime"
        case "hygiene": return "Output hygiene"
        default: return "Other"
        }
    }

    private var eventLogEntries: [E2ERealtimeLogEntry] {
        let scenariosByID = Dictionary(runMode.scenarios.map { ($0.id, $0.title) }, uniquingKeysWith: { first, _ in first })
        let streamingEvents = isRunning ? liveEventBuffer : []
        let reportEvents = (isRunning || !liveResults.isEmpty ? liveResults : (latestReport?.results ?? [])).flatMap(\.events)
        let sourceEvents = isRunning ? streamingEvents : reportEvents
        return sourceEvents.map { event in
                E2ERealtimeLogEntry(
                    id: event.id,
                    createdAt: event.createdAt,
                    scenarioTitle: scenariosByID[event.scenarioID] ?? event.scenarioID,
                    phase: event.phase,
                    message: event.message
                )
        }
        .sorted { $0.createdAt < $1.createdAt }
    }
}

private struct E2ETestDashboardView: View {
    let totalScenarioCount: Int
    let results: [E2ETestResult]
    let report: E2ETestReport?
    let isRunning: Bool
    let runStartedAt: Date?

    private var completedCount: Int { results.count }
    private var passedCount: Int { results.filter(\.passed).count }
    private var failedCount: Int { completedCount - passedCount }
    private var passRate: Double {
        guard completedCount > 0 else { return 0 }
        return Double(passedCount) / Double(completedCount)
    }
    private var progressFraction: Double {
        guard totalScenarioCount > 0 else { return 0 }
        return min(Double(completedCount) / Double(totalScenarioCount), 1)
    }
    private var elapsedSeconds: Double {
        if let report {
            return max(report.finishedAt.timeIntervalSince(report.startedAt), 0)
        }
        if let runStartedAt {
            return max(Date().timeIntervalSince(runStartedAt), 0)
        }
        return results.map { max($0.finishedAt.timeIntervalSince($0.startedAt), 0) }.reduce(0, +)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 118), spacing: 8)], spacing: 8) {
                E2ETestMetricTile(title: "Status", value: statusText, systemImage: statusIcon, tint: statusTint)
                E2ETestMetricTile(title: "Pass rate", value: percentText(passRate), systemImage: "gauge.with.dots.needle.bottom.50percent", tint: .blue)
                E2ETestMetricTile(title: "Passed", value: "\(passedCount)", systemImage: "checkmark.circle", tint: .green)
                E2ETestMetricTile(title: "Failed", value: "\(failedCount)", systemImage: "xmark.circle", tint: failedCount > 0 ? .red : .secondary)
                E2ETestMetricTile(title: "Elapsed", value: durationText(elapsedSeconds), systemImage: "timer", tint: .orange)
            }

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Progress")
                        .font(.caption.weight(.semibold))
                    Spacer()
                    Text("\(completedCount)/\(totalScenarioCount)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(Theme.textSecondary)
                }
                ProgressView(value: progressFraction)
                    .tint(statusTint)
            }

            if let latest = results.last {
                VStack(alignment: .leading, spacing: 4) {
                    Text(isRunning ? "Current signal" : "Last signal")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Theme.textSecondary)
                    HStack(spacing: 8) {
                        StatusDot(color: latest.passed ? .green : .red, size: 9)
                        Text(latest.title)
                            .font(.subheadline.weight(.medium))
                            .lineLimit(2)
                    }
                    if let firstFailure = latest.failures.first {
                        Text(firstFailure)
                            .font(.caption)
                            .foregroundStyle(.red)
                            .lineLimit(3)
                    }
                }
            } else {
                Text("No E2E report has been loaded yet.")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .padding(.vertical, 6)
    }

    private var statusText: String {
        if isRunning { return "Running" }
        if completedCount == 0 { return "Idle" }
        return failedCount == 0 ? "Passing" : "Failing"
    }

    private var statusIcon: String {
        if isRunning { return "play.circle" }
        if completedCount == 0 { return "circle.dashed" }
        return failedCount == 0 ? "checkmark.seal" : "exclamationmark.triangle"
    }

    private var statusTint: Color {
        if isRunning { return .blue }
        if completedCount == 0 { return .secondary }
        return failedCount == 0 ? .green : .red
    }
}

private struct E2ETestMetricTile: View {
    let title: String
    let value: String
    let systemImage: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: systemImage)
                    .foregroundStyle(tint)
                Spacer()
            }
            Text(value)
                .font(.headline.monospacedDigit())
                .lineLimit(1)
                .minimumScaleFactor(0.75)
            Text(title)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, minHeight: 82, alignment: .leading)
        .padding(10)
        .background(Theme.surface)
        .clipShape(.rect(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        }
    }
}

private struct E2ETestResultRow: View {
    let result: E2ETestResult

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                StatusDot(color: result.passed ? .green : .red, size: 9)
                Text(result.title)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(2)
                Spacer(minLength: 8)
                Text(durationText(max(result.finishedAt.timeIntervalSince(result.startedAt), 0)))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(Theme.textSecondary)
            }

            Text("Intent: \(result.actualIntent) / \(result.expectedIntent)")
                .font(.caption2.monospaced())
                .foregroundStyle(Theme.textTertiary)

            if let firstFailure = result.failures.first {
                Text(firstFailure)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(3)
            } else if !result.finalText.isEmpty {
                Text(result.finalText)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 3)
    }
}

private struct E2ERealtimeLogEntry: Identifiable {
    let id: UUID
    let createdAt: Date
    let scenarioTitle: String
    let phase: String
    let message: String
}

private struct E2ERealtimeLogView: View {
    let entries: [E2ERealtimeLogEntry]
    let isRunning: Bool

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(entries) { entry in
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(spacing: 8) {
                            Text(timeText(entry.createdAt))
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                            Text(entry.phase.uppercased())
                                .font(.caption2.monospaced())
                                .foregroundStyle(phaseColor(entry.phase))
                            Text(entry.scenarioTitle)
                                .font(.caption.weight(.medium))
                                .lineLimit(1)
                        }
                        Text(entry.message)
                            .font(.caption.monospaced())
                            .foregroundStyle(Theme.textSecondary)
                            .textSelection(.enabled)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 4)
                    .overlay(alignment: .bottom) {
                        Divider().opacity(0.35)
                    }
                }
            }
        }
        .frame(minHeight: isRunning ? 240 : 180, maxHeight: 320)
    }

    private func timeText(_ date: Date) -> String {
        Self.logTimeFormatter.string(from: date)
    }

    private func phaseColor(_ phase: String) -> Color {
        switch phase {
        case "error": return .red
        case "intent": return .blue
        case "models": return .orange
        case "step": return .purple
        case "final": return .green
        default: return Theme.textTertiary
        }
    }

    private static let logTimeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter
    }()
}

private struct E2EFailureBucket: Identifiable {
    let name: String
    let count: Int
    var id: String { name }
}

private func percentText(_ value: Double) -> String {
    String(format: "%.0f%%", value * 100)
}

private func durationText(_ seconds: Double) -> String {
    if seconds >= 60 {
        return String(format: "%.1fm", seconds / 60)
    }
    return String(format: "%.1fs", seconds)
}


private extension E2ETestRunnerView {
    enum RunMode: CaseIterable {
        case standard
        case trainingValidation

        var title: String {
            switch self {
            case .standard: return "Standard"
            case .trainingValidation: return "Training validation"
            }
        }

        var buttonTitle: String {
            switch self {
            case .standard: return "Run full E2E suite"
            case .trainingValidation: return "Run training validation"
            }
        }

        var runningLabel: String {
            switch self {
            case .standard: return "Running E2E suite…"
            case .trainingValidation: return "Running training validation…"
            }
        }

        var scenarios: [E2ETestScenario] {
            switch self {
            case .standard: return E2ETestScenario.standard
            case .trainingValidation: return E2ETestScenario.trainingValidation
            }
        }

        var footerText: String {
            switch self {
            case .standard:
                return "Runs deterministic routing checks plus live agent scenarios for tool boundaries, chat quality, stale-context regressions, and final-answer validation. Export creates a live E2E JSON layer with ownsLiveE2EScenarios=true."
            case .trainingValidation:
                return "Runs multi-scenario in-app validation using trained models in real agent flows, then summarizes failures as training signals for the next fine-tuning cycle. Export creates a live E2E JSON layer with ownsLiveE2EScenarios=true."
            }
        }
    }
}
