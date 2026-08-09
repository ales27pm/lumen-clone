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
    @State private var modelSetupError: String?

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

                #if DEBUG
                if LumenLaunchArguments.isUITesting {
                    developerSection
                }
                #endif

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
                        Text("Setup downloads this family’s verified chat, embedding, and required adapter artifacts after you confirm.")
                            .font(.caption2)
                            .foregroundStyle(Theme.textTertiary)
                    }

                    Text("Downloads start only after you tap the button below. The current family and model pair remain active unless the verified setup and runtime load both succeed.")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)

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
                    Text("Qwen3 is the default. A successful switch persists the chat and embedding selections together and restores them automatically on future launches.")
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

                #if DEBUG
                if !LumenLaunchArguments.isUITesting {
                    developerSection
                }
                #endif

                Section {
                    Toggle("Network tools", isOn: Binding(get: { state.networkToolsEnabled }, set: { state.networkToolsEnabled = $0 }))
                        .accessibilityIdentifier("settings.privacy.networkTools")

                    NavigationLink {
                        PermissionsView()
                    } label: {
                        Label("Permissions", systemImage: "hand.raised")
                    }
                } header: {
                    Text("Privacy")
                } footer: {
                    Text("Network tools allow explicit web and external data lookups. Individual tools still follow the Tools screen enablement and approval rules.")
                }

                Section("About") {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Lumen").font(.subheadline.weight(.semibold))
                        Text("Runs open-source language models locally via llama.cpp. Conversations, embeddings, and memory stay on-device by default. Network tools and connected services send only the data needed for actions you explicitly request, subject to their permissions and approvals.")
                            .font(.footnote)
                            .foregroundStyle(Theme.textSecondary)
                    }
                    .padding(.vertical, 2)

                    NavigationLink {
                        ThirdPartyNoticesView()
                    } label: {
                        Label("Open-source licenses", systemImage: "doc.text")
                    }
                    .accessibilityIdentifier("settings.about.openSourceLicenses")
                }
            }
            .scrollContentBackground(.hidden)
            .background(AppBackground())
            .navigationTitle("Settings")
            .task {
                selectedModelFamily = LumenModelFamily.persistedSelected
                await refreshParseFailureSummary()
            }
            .alert("Run checks", isPresented: $showDeveloperAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(developerAlertMessage)
            }
            .alert("Model setup failed", isPresented: Binding(
                get: { modelSetupError != nil },
                set: { if !$0 { modelSetupError = nil } }
            )) {
                Button("OK", role: .cancel) { modelSetupError = nil }
            } message: {
                Text(modelSetupError ?? "The selected model family could not be prepared.")
            }
        }
    }

    #if DEBUG
    @ViewBuilder
    private var developerSection: some View {
        Section("Developer") {
            #if DEBUG
            Toggle("Developer trace mode", isOn: Binding(get: { appState.developerTraceModeEnabled }, set: { appState.developerTraceModeEnabled = $0 }))
                .accessibilityIdentifier("settings.developer.traceMode")
            Toggle("Capture reasoning", isOn: Binding(get: { appState.developerReasoningCaptureEnabled }, set: { appState.developerReasoningCaptureEnabled = $0 }))
                .disabled(!appState.developerTraceModeEnabled)
                .accessibilityIdentifier("settings.developer.reasoningCapture")
            #endif

            NavigationLink {
                DeveloperConsoleView()
            } label: {
                Label("Developer Console", systemImage: "wrench.and.screwdriver")
            }
            .accessibilityIdentifier("settings.developer.console")
        }
    }
    #endif

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
            let result = await ModelLaunchBootstrap.switchFamily(family, appState: appState, context: modelContext)
            isSwitchingModelFamily = false
            if result.succeeded {
                selectedModelFamily = LumenModelFamily.persistedSelected
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            } else {
                modelSetupError = result.errorMessage
                    ?? "Only \(result.ready) of \(result.required) verified artifacts became ready."
                UINotificationFeedbackGenerator().notificationOccurred(.error)
            }
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
        let modelFilesResult = ModelStorage.modelFilesWithDiagnostics()
        let importedFilesResult = FileStore.importedFilesWithDiagnostics()
        return """
        Last launch diagnostics:
        • Imported files: \(importedFilesResult.files.count)
        • Imported files mode: \(importedFilesResult.mode)
        • Imported files diagnostic: \(importedFilesResult.diagnostic ?? "none")
        • Model files: \(modelFilesResult.files.count)
        • Model files mode: \(modelFilesResult.mode)
        • Model files diagnostic: \(modelFilesResult.diagnostic ?? "none")
        • Models path: \(modelFilesResult.directory.map(pathSummary) ?? "unavailable")
        \(parseFailureSummary)
        """
    }

    private func pathSummary(_ url: URL) -> String {
        "path_sha256=\(String(RuntimeFallbackLogger.promptHash(url.path).prefix(16)))"
    }

    private var debugText: String {
        """
        Runtime:
        • isGenerating: \(appState.isGenerating ? "true" : "false")
        • agentModeEnabled: \(appState.agentModeEnabled ? "true" : "false")
        • showThinkingByDefault: \(appState.showThinkingByDefault ? "true" : "false")
        • developerTraceModeEnabled: \(appState.developerTraceModeEnabled ? "true" : "false")
        • developerReasoningCaptureEnabled: \(appState.developerReasoningCaptureEnabled ? "true" : "false")
        • networkToolsEnabled: \(appState.networkToolsEnabled ? "true" : "false")
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
        let modelFilesResult = ModelStorage.modelFilesWithDiagnostics(fileManager: fm)
        let canReadModels = modelFilesResult.directory.map { fm.isReadableFile(atPath: $0.path) } ?? false
        let canWriteModels = modelFilesResult.directory.map { fm.isWritableFile(atPath: $0.path) } ?? false
        let importedFilesResult = FileStore.importedFilesWithDiagnostics(fileManager: fm)
        let canReadImports = importedFilesResult.directory.map { fm.isReadableFile(atPath: $0.path) } ?? false
        let canWriteImports = importedFilesResult.directory.map { fm.isWritableFile(atPath: $0.path) } ?? false
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
        developerAlertMessage = "\(passed)/\(checks.count) checks passed\n\n\(summary)\n\nModels diagnostic: \(modelFilesResult.diagnostic ?? modelFilesResult.mode)\nImports diagnostic: \(importedFilesResult.diagnostic ?? importedFilesResult.mode)"
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

struct E2ETestRunnerView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @State private var isRunning = false
    @State private var runMode: RunMode
    @State private var reportText = E2ETestLogStore.latestText()
    @State private var latestReport: E2ETestReport? = E2ETestLogStore.latestReport()
    @State private var liveResults: [E2ETestResult] = []
    @State private var liveEventBuffer: [E2ETestEvent] = []
    @State private var runStartedAt: Date?
    @State private var lastExportURL: URL?
    @State private var exportError: String?
    @State private var resourceSnapshot: ResourceBudgetGate.Snapshot?

    init(initialRunMode: RunMode = .standard) {
        _runMode = State(initialValue: initialRunMode)
    }

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

                if runMode == .trainingValidation {
                    LabeledContent("Thermal state", value: thermalStateForDisplay.rawValue)
                        .font(.caption)

                    if let blockedRunReason {
                        Label(blockedRunReason, systemImage: "thermometer.high")
                            .font(.caption)
                            .foregroundStyle(.orange)
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
                .disabled(isRunning || blockedRunReason != nil)

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
            refreshResourceSnapshot()
            reportText = E2ETestLogStore.latestText()
            latestReport = nil
            liveResults = []
            runStartedAt = nil
            lastExportURL = nil
            exportError = nil
        }
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            SceneTransitionCoordinator.shared.requestForegroundActivation()
            refreshResourceSnapshot()
            reloadLatestReport()
        }
    }

    private var thermalStateForDisplay: DeviceThermalState {
        resourceSnapshot?.thermalState ?? DeviceThermalState.from(processThermalState: ProcessInfo.processInfo.thermalState)
    }

    private var blockedRunReason: String? {
        Self.blockedRunReason(runMode: runMode, thermalState: thermalStateForDisplay)
    }

    private var dashboardResults: [E2ETestResult] {
        if isRunning || !liveResults.isEmpty { return liveResults }
        return latestReport?.results ?? []
    }

    private var failureBuckets: [E2EFailureBucket] {
        let actionableResults = dashboardResults.filter { !$0.isRuntimePreflightNonActionable }
        let buckets = Dictionary(grouping: actionableResults.flatMap(\.failures)) { failure in
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

    @MainActor
    private func run() {
        guard !isRunning else { return }
        SceneTransitionCoordinator.shared.requestForegroundActivation()
        refreshResourceSnapshot()
        if let blockedRunReason {
            exportError = blockedRunReason
            reportText = """
            Training validation blocked
            \(blockedRunReason)
            """
            return
        }

        let mode = runMode
        let totalScenarioCount = mode.scenarios.count
        let config = E2ERunConfig(appState: appState)

        isRunning = true
        exportError = nil
        latestReport = nil
        liveResults = []
        liveEventBuffer = []
        runStartedAt = Date()
        reportText = "Preparing live runtime artifacts…"

        Task { @MainActor in
            let artifactsReady = await ModelLaunchBootstrap.prepareLiveRuntimeArtifacts(appState: appState, context: modelContext)
            guard artifactsReady else {
                let readiness = await ModelLaunchBootstrap.liveRuntimeArtifactReadinessDetails(context: modelContext)
                let report = E2ETestRunner.liveRuntimeArtifactsBlockedReport(
                    startedAt: runStartedAt ?? Date(),
                    finishedAt: Date(),
                    readyArtifactCount: readiness.ready,
                    requiredArtifactCount: readiness.required,
                    missingAdapterSlots: readiness.missingAdapterSlots,
                    missingArtifactFileNames: readiness.missingArtifactFileNames,
                    diagnostic: readiness.diagnostic
                )
                E2ETestLogStore.writeLatest(report)
                latestReport = report
                reportText = report.summaryText
                isRunning = false
                runStartedAt = nil
                exportError = "Live runtime artifacts are not ready. Keep the app open until model downloads complete, then rerun E2E."
                return
            }
            let modelLoadSnapshotResult = ModelLoader.modelLoadSnapshot(appState: appState, context: modelContext)
            guard let modelLoadSnapshot = modelLoadSnapshotResult.snapshot else {
                let diagnostic = modelLoadSnapshotResult.diagnostic ?? "model_catalog_fetch_failed"
                let report = E2ETestRunner.liveModelCatalogFetchBlockedReport(
                    startedAt: runStartedAt ?? Date(),
                    finishedAt: Date(),
                    diagnostic: diagnostic
                )
                E2ETestLogStore.writeLatest(report)
                latestReport = report
                reportText = report.summaryText
                isRunning = false
                runStartedAt = nil
                exportError = "Live E2E model catalog fetch failed. Resolve the local model store issue, then rerun E2E."
                return
            }
            reportText = mode.runningLabel

            Task.detached(priority: .userInitiated) {
                let ensureChatLoaded: E2ETestRunner.EnsureChatLoaded = {
                    await Task.yield()
                    return await ModelLoader.ensureChatLoaded(snapshot: modelLoadSnapshot, intent: .userChat)
                }

                let onResult: E2ETestRunner.ResultCallback = { result in
                    await MainActor.run {
                        liveResults.append(result)
                        reportText = inProgressReportText(results: liveResults, total: totalScenarioCount)
                    }
                }

                let onEvent: E2ETestRunner.EventCallback = { event in
                    await MainActor.run {
                        liveEventBuffer.append(event)
                    }
                }

                let report: E2ETestReport
                switch mode {
                case .standard:
                    report = await E2ETestRunner.runStandard(config: config, ensureChatLoaded: ensureChatLoaded, onResult: onResult, onEvent: onEvent)
                case .trainingValidation:
                    report = await E2ETestRunner.runTrainingValidation(config: config, ensureChatLoaded: ensureChatLoaded, onResult: onResult, onEvent: onEvent)
                }

                await MainActor.run {
                    latestReport = report
                    reportText = report.summaryText
                    isRunning = false
                    runStartedAt = nil
                }
            }
        }
    }

    @MainActor
    private func refreshResourceSnapshot() {
        resourceSnapshot = ResourceBudgetGate.diagnosticSnapshot()
    }

    private func exportLatestReport() {
        guard let latestReport else { return }
        do {
            let result = try EvidenceLayerExporter.writeLiveE2EReport(latestReport)
            lastExportURL = result.url
            exportError = nil
        } catch {
            exportError = "Live E2E report export failed: \(error.localizedDescription)"
        }
    }

    private func inProgressReportText(results: [E2ETestResult], total: Int) -> String {
        let passed = results.filter(\.passed).count
        let nonActionable = results.filter { !$0.passed && $0.isRuntimePreflightNonActionable }.count
        let failed = results.count - passed - nonActionable
        return """
        Running \(runMode.title) E2E suite
        Completed: \(results.count)/\(total)
        Passed: \(passed)
        Failed: \(failed)
        Runtime preflight: \(nonActionable)
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
    private var runtimePreflightCount: Int { results.filter { !$0.passed && $0.isRuntimePreflightNonActionable }.count }
    private var failedCount: Int { completedCount - passedCount - runtimePreflightCount }
    private var passRate: Double {
        let actionableCompleted = completedCount - runtimePreflightCount
        guard actionableCompleted > 0 else { return 0 }
        return Double(passedCount) / Double(actionableCompleted)
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
                E2ETestMetricTile(title: "Preflight", value: "\(runtimePreflightCount)", systemImage: "thermometer.medium", tint: runtimePreflightCount > 0 ? .orange : .secondary)
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
                        StatusDot(color: latest.statusColor, size: 9)
                        Text(latest.title)
                            .font(.subheadline.weight(.medium))
                            .lineLimit(2)
                    }
                    if let firstFailure = latest.failures.first {
                        Text(firstFailure)
                            .font(.caption)
                            .foregroundStyle(latest.isRuntimePreflightNonActionable ? .orange : .red)
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
        if failedCount == 0, runtimePreflightCount > 0 { return "Preflight" }
        return failedCount == 0 ? "Passing" : "Failing"
    }

    private var statusIcon: String {
        if isRunning { return "play.circle" }
        if completedCount == 0 { return "circle.dashed" }
        if failedCount == 0, runtimePreflightCount > 0 { return "thermometer.medium" }
        return failedCount == 0 ? "checkmark.seal" : "exclamationmark.triangle"
    }

    private var statusTint: Color {
        if isRunning { return .blue }
        if completedCount == 0 { return .secondary }
        if failedCount == 0, runtimePreflightCount > 0 { return .orange }
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
                StatusDot(color: result.statusColor, size: 9)
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
                    .foregroundStyle(result.isRuntimePreflightNonActionable ? .orange : .red)
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


extension E2ETestRunnerView {
    nonisolated static func blockedRunReason(runMode: RunMode, thermalState: DeviceThermalState?) -> String? {
        guard runMode == .trainingValidation else { return nil }
        guard let thermalState else {
            return "thermal state unavailable; wait for device status and retry"
        }

        switch thermalState {
        case .nominal, .fair:
            return nil
        case .serious:
            return ResourceBudgetGate.seriousThermalRetryHint
        case .critical:
            return "device thermal state critical; cool device and retry"
        case .unknown:
            return "device thermal state unknown; wait for device status and retry"
        }
    }

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
            case .standard: return "Run standard E2E suite"
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
                return "Runs static routing/tool guard coverage plus live model-backed agent scenarios. Live scenarios must record fresh model runtime evidence before export is accepted as live E2E evidence."
            case .trainingValidation:
                return "Runs multi-scenario in-app validation using trained models in real agent flows, then summarizes failures as training signals for the next fine-tuning cycle. Export creates a live E2E JSON layer with ownsLiveE2EScenarios=true."
            }
        }
    }
}
