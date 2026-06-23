import Foundation
import Metal
import Observation
import SwiftUI

enum DeveloperConsoleTab: String, CaseIterable, Identifiable {
    case run
    case telemetry
    case reports

    var id: String { rawValue }

    var title: String {
        switch self {
        case .run: return "Run"
        case .telemetry: return "Telemetry"
        case .reports: return "Reports"
        }
    }
}

enum DeveloperEvidenceLayer: String, CaseIterable, Identifiable, Sendable {
    case agentGroundingRuntimeAudit
    case runtimeManifestAudit
    case agentModelBehaviorAuditor
    case runtimeScenarioRunnerStaticChecks
    case agentBehaviorTraceRecorder
    case e2eTestReport

    var id: String { rawValue }

    var sourceLayer: String {
        switch self {
        case .agentGroundingRuntimeAudit: return "agentGroundingRuntimeAudit"
        case .runtimeManifestAudit: return "runtimeManifestAudit"
        case .agentModelBehaviorAuditor: return "agentModelBehaviorAuditor"
        case .runtimeScenarioRunnerStaticChecks: return "runtimeScenarioRunner.staticChecks"
        case .agentBehaviorTraceRecorder: return "agentBehaviorTraceRecorder"
        case .e2eTestReport: return "e2eTestReport"
        }
    }

    var title: String {
        switch self {
        case .agentGroundingRuntimeAudit: return "Runtime Audit Package"
        case .runtimeManifestAudit: return "Runtime Registry"
        case .agentModelBehaviorAuditor: return "Model Behaviour"
        case .runtimeScenarioRunnerStaticChecks: return "Static Scenarios"
        case .agentBehaviorTraceRecorder: return "Runtime Traces"
        case .e2eTestReport: return "Live E2E"
        }
    }

    var systemImage: String {
        switch self {
        case .agentGroundingRuntimeAudit: return "checkmark.seal.text.page"
        case .runtimeManifestAudit: return "list.bullet.rectangle"
        case .agentModelBehaviorAuditor: return "brain.head.profile"
        case .runtimeScenarioRunnerStaticChecks: return "checklist.checked"
        case .agentBehaviorTraceRecorder: return "waveform.path.ecg"
        case .e2eTestReport: return "testtube.2"
        }
    }

    var ownsLiveE2EScenarios: Bool {
        self == .e2eTestReport
    }

    var trustRole: String {
        ownsLiveE2EScenarios ? "Scenario pass/fail owner" : "Diagnostic evidence"
    }

    var privacySummary: String {
        switch self {
        case .agentGroundingRuntimeAudit:
            return "Manifest audit failures, behavior violations, bounded trace prefixes, and gated improve-loop samples."
        case .runtimeManifestAudit:
            return "Runtime registry drift and missing/extra live tool evidence."
        case .agentModelBehaviorAuditor:
            return "Bounded model behavior violations and repair samples from recent app state."
        case .runtimeScenarioRunnerStaticChecks:
            return "Deterministic manifest checks only; no model execution proof."
        case .agentBehaviorTraceRecorder:
            return "Bounded recent model/tool trace metadata and parse/tool-selection counters."
        case .e2eTestReport:
            return "Live scenario prompts, final outputs, failures, and event logs; review before sharing."
        }
    }

    var nextAction: String {
        switch self {
        case .agentGroundingRuntimeAudit:
            return "Run Agent Grounding, then export the runtime audit package for the offline loop."
        case .runtimeManifestAudit:
            return "Export when debugging manifest/live tool registry drift."
        case .agentModelBehaviorAuditor:
            return "Export when diagnosing behavior drift without claiming live scenario results."
        case .runtimeScenarioRunnerStaticChecks:
            return "Use for manifest sanity only; do not feed as live E2E pass/fail."
        case .agentBehaviorTraceRecorder:
            return "Run real app interactions first; empty traces are a loop gap."
        case .e2eTestReport:
            return "Run E2E tests and export this JSON as the live scenario evidence layer."
        }
    }
}

struct DeveloperEvidenceLayerStatus: Identifiable, Equatable {
    let layer: DeveloperEvidenceLayer
    var status: String
    var detail: String
    var count: Int?
    var isBlocking: Bool

    var id: String { layer.id }

    static func baseline() -> [DeveloperEvidenceLayerStatus] {
        DeveloperEvidenceLayer.allCases.map { layer in
            DeveloperEvidenceLayerStatus(
                layer: layer,
                status: layer.ownsLiveE2EScenarios ? "live owner" : "diagnostic",
                detail: layer.nextAction,
                count: nil,
                isBlocking: false
            )
        }
    }
}

struct DeveloperDashboardMeter: Identifiable, Equatable {
    enum Level: Equatable {
        case nominal
        case warning
        case critical
        case inactive
    }

    let id: String
    let title: String
    let value: String
    let detail: String
    let progress: Double
    let systemImage: String
    let level: Level
}

struct DeveloperTelemetryMeter: Identifiable {
    enum MeterStyle: Equatable {
        case percent
        case value
        case availability
    }

    let id: String
    let title: String
    let value: String
    let unit: String
    let detail: String
    let progress: Double
    let systemImage: String
    let tint: Color
    let style: MeterStyle
}

struct DeveloperTelemetrySnapshot: Equatable {
    var generatedAt = Date()
    var residentMemoryMB: Double?
    var physicalMemoryMB: Double = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024)
    var cpuProxyPercent: Double = 0
    var processorSummary = "\(ProcessInfo.processInfo.activeProcessorCount)/\(ProcessInfo.processInfo.processorCount)"
    var thermalState = DeviceThermalState.from(processThermalState: ProcessInfo.processInfo.thermalState).rawValue
    var lowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
    var recentLatencyMs: Int?
    var firstTokenLatencyMs: Int?
    var tokensPerSecond: Double?
    var recentSuccessRate: Double?
    var metalAvailable = false
    var metalDeviceName: String?
    var recommendedWorkingSetMB: Double?
    var gpuVerification = "unverified"
    var gpuUtilizationPercent: Double?
    var gpuMemoryMB: Double?
    var coreMLAvailable = false
    var coreMLStatus = "unknown"
    var aneStatus = "Core ML path required"
    var aneUtilizationPercent: Double?
}

enum DeveloperWorkflowAction: String, CaseIterable, Identifiable {
    case collectDiagnostics
    case runAgentGrounding
    case runLiveTraceSmoke
    case runE2EStandard
    case runE2ETraining
    case runPersistentDiagnostics
    case exportRuntimeAudit
    case exportLiveE2E
    case exportRecentTraces

    var id: String { rawValue }

    var title: String {
        switch self {
        case .collectDiagnostics: return "Collect diagnostics"
        case .runAgentGrounding: return "Run Agent Grounding audit"
        case .runLiveTraceSmoke: return "Run live trace smoke test"
        case .runE2EStandard: return "Run standard E2E suite"
        case .runE2ETraining: return "Run training E2E suite"
        case .runPersistentDiagnostics: return "Run persistent diagnostics"
        case .exportRuntimeAudit: return "Export runtime audit package"
        case .exportLiveE2E: return "Export live E2E JSON"
        case .exportRecentTraces: return "Export recent runtime traces"
        }
    }

    var systemImage: String {
        switch self {
        case .collectDiagnostics: return "waveform.path.ecg"
        case .runAgentGrounding: return "checkmark.seal.text.page"
        case .runLiveTraceSmoke: return "bolt.heart"
        case .runE2EStandard: return "testtube.2"
        case .runE2ETraining: return "graduationcap"
        case .runPersistentDiagnostics: return "repeat.circle"
        case .exportRuntimeAudit: return "square.and.arrow.up"
        case .exportLiveE2E: return "arrow.up.doc"
        case .exportRecentTraces: return "waveform.path.ecg.rectangle"
        }
    }

    var detail: String {
        switch self {
        case .collectDiagnostics:
            return "Refresh runtime, permission, grounding, background, and privacy snapshots."
        case .runAgentGrounding:
            return "Open Agent Grounding and produce the runtime audit package."
        case .runLiveTraceSmoke:
            return "Keep monitoring active while you run one real chat/tool interaction."
        case .runE2EStandard:
            return "Run broad pass/fail coverage; this is the live scenario authority."
        case .runE2ETraining:
            return "Capture failed prompts and corrections for the next improvement loop."
        case .runPersistentDiagnostics:
            return "Run durable diagnostics and inspect storage, background, and tool drift."
        case .exportRuntimeAudit:
            return "Export package for `--runtime-audit` ingestion."
        case .exportLiveE2E:
            return "Export live E2E JSON; only this layer owns scenario pass/fail."
        case .exportRecentTraces:
            return "Export recent model/tool trace metadata after real app interactions."
        }
    }
}

struct DeveloperFinding: Identifiable, Equatable {
    enum Severity: String {
        case info
        case warning
        case error

        var color: Color {
            switch self {
            case .info: return .secondary
            case .warning: return .orange
            case .error: return .red
            }
        }

        var systemImage: String {
            switch self {
            case .info: return "info.circle"
            case .warning: return "exclamationmark.triangle"
            case .error: return "xmark.octagon"
            }
        }
    }

    let id = UUID()
    let severity: Severity
    let title: String
    let detail: String
}

@MainActor
@Observable
final class DeveloperConsoleModel {
    var selectedTab: DeveloperConsoleTab = .run
    var diagnosticsSnapshot: DiagnosticsSnapshot?
    var evidenceLayers = DeveloperEvidenceLayerStatus.baseline()
    var findings: [DeveloperFinding] = []
    var statusMessage = "Developer console ready."
    var isCollectingDiagnostics = false
    var isWorkflowRunning = false
    var workflowStartedAt: Date?
    var activeWorkflowAction: DeveloperWorkflowAction?
    var lastWorkflowSummary = "Not started"
    var storageCheckRatio: Double?
    var storageCheckSummary = "Not run"
    var workflowSnapshot = AgentWorkflowMonitor.shared.snapshot()
    var telemetry = DeveloperTelemetrySnapshot()
    var e2eLatestReport: E2ETestReport?
    var e2eReportText = E2ETestLogStore.latestText()
    var e2eLastExportURL: URL?
    var e2eExportError: String?

    private var diagnosticsProvider = DiagnosticsProvider()

    @ObservationIgnored
    private var liveRefreshTask: Task<Void, Never>?

    func loadCachedDiagnostics() {
        diagnosticsSnapshot = diagnosticsProvider.cachedSnapshot()
        loadLatestE2EReport()
        refreshWorkflowSnapshot()
        refreshFindings()
    }

    func loadLatestE2EReport() {
        e2eLatestReport = E2ETestLogStore.latestReport()
        e2eReportText = E2ETestLogStore.latestText()
    }

    func startLiveMeters() {
        liveRefreshTask?.cancel()
        liveRefreshTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                self?.refreshWorkflowSnapshot()
                await self?.refreshTelemetry()
                try? await Task.sleep(nanoseconds: 1_250_000_000)
            }
        }
    }

    func stopLiveMeters() {
        liveRefreshTask?.cancel()
        liveRefreshTask = nil
    }

    func collectDiagnostics() async {
        isCollectingDiagnostics = true
        diagnosticsSnapshot = await diagnosticsProvider.collect()
        isCollectingDiagnostics = false
        statusMessage = "Diagnostics refreshed."
        refreshWorkflowSnapshot()
        await refreshTelemetry()
        refreshFindings()
    }

    func startDeveloperWorkflow() async {
        guard !isWorkflowRunning else { return }
        isWorkflowRunning = true
        workflowStartedAt = Date()
        activeWorkflowAction = .collectDiagnostics
        lastWorkflowSummary = "Starting monitor and diagnostics refresh"
        statusMessage = "Developer workflow running."
        _ = AgentWorkflowMonitor.shared.start()
        refreshWorkflowSnapshot()

        await collectDiagnostics()

        activeWorkflowAction = .runPersistentDiagnostics
        _ = runStorageChecks()

        activeWorkflowAction = .runLiveTraceSmoke
        lastWorkflowSummary = "Monitoring active. Run real app interactions before exporting runtime evidence."
        statusMessage = "Workflow armed for live runtime evidence."
        refreshWorkflowSnapshot()
        isWorkflowRunning = false
    }

    func refreshWorkflowSnapshot() {
        workflowSnapshot = AgentWorkflowMonitor.shared.snapshot()
    }

    func refreshTelemetry() async {
        let processInfo = ProcessInfo.processInfo
        let recentMetrics = (try? await RuntimeMetricsStore.shared.recentMetrics(limit: 24)) ?? []
        let successful = recentMetrics.filter(\.success).count
        let latencies = recentMetrics.compactMap(\.latencyMs)
        let metricSpan = recentMetrics.last.map { last in
            max(last.timestamp.timeIntervalSince(recentMetrics.first?.timestamp ?? last.timestamp), 1)
        } ?? 1
        let workflowEvents = workflowSnapshot.events.suffix(60)
        let eventSpan = workflowEvents.last.map { last in
            max(last.createdAt.timeIntervalSince(workflowEvents.first?.createdAt ?? last.createdAt), 1)
        } ?? 1
        let metricDensity = Double(recentMetrics.count) / metricSpan
        let eventDensity = Double(workflowEvents.count) / eventSpan
        let cpuProxy = min(100, max(metricDensity, eventDensity) * 35)
        let firstToken = workflowEvents.compactMap(\.firstTokenLatencyMs).last
        let tps = workflowEvents.compactMap(\.tokensPerSecond).last
        let metalDevice = MTLCreateSystemDefaultDevice()
        let acceleration = await AppLlamaService.shared.currentAccelerationDiagnostics()
        let runtime = diagnosticsSnapshot?.runtime

        telemetry = DeveloperTelemetrySnapshot(
            generatedAt: Date(),
            residentMemoryMB: Self.residentMemoryUsageMB(),
            physicalMemoryMB: Double(processInfo.physicalMemory) / (1024 * 1024),
            cpuProxyPercent: cpuProxy,
            processorSummary: "\(processInfo.activeProcessorCount)/\(processInfo.processorCount)",
            thermalState: DeviceThermalState.from(processThermalState: processInfo.thermalState).rawValue,
            lowPowerMode: processInfo.isLowPowerModeEnabled,
            recentLatencyMs: latencies.isEmpty ? nil : latencies.reduce(0, +) / latencies.count,
            firstTokenLatencyMs: firstToken,
            tokensPerSecond: tps ?? acceleration.decodeTokensPerSecond,
            recentSuccessRate: recentMetrics.isEmpty ? nil : Double(successful) / Double(recentMetrics.count),
            metalAvailable: metalDevice != nil || runtime?.metalAvailable == true || acceleration.metalDeviceAvailable == true,
            metalDeviceName: acceleration.metalDeviceUsed ?? acceleration.metalDeviceName ?? metalDevice?.name,
            recommendedWorkingSetMB: acceleration.recommendedMaxWorkingSetSizeMB ?? metalDevice.map { Double($0.recommendedMaxWorkingSetSize) / (1024 * 1024) },
            gpuVerification: acceleration.verificationLevel.isEmpty ? (metalDevice == nil ? "unavailable" : "available") : acceleration.verificationLevel,
            gpuUtilizationPercent: acceleration.actualGpuUtilizationPercent,
            gpuMemoryMB: acceleration.actualGpuMemoryMB,
            coreMLAvailable: runtime?.coreMLAvailable == true,
            coreMLStatus: runtime?.coreMLStatus ?? "cached",
            aneStatus: acceleration.aneUsedByCurrentRuntime == true ? "active" : (runtime?.coreMLAvailable == true ? "available for Core ML" : "unavailable"),
            aneUtilizationPercent: acceleration.aneUtilizationPercent
        )
    }

    func runStorageChecks() -> String {
        let fm = FileManager.default
        let modelsDirectory = ModelStorage.modelsDirectoryURL(fileManager: fm)
        let importsDirectory = FileStore.importsDirectory
        let e2eDirectory = try? E2ETestLogStore.reportsDirectory()
        let checks: [(String, Bool)] = [
            ("Models folder readable", fm.isReadableFile(atPath: modelsDirectory.path)),
            ("Models folder writable", fm.isWritableFile(atPath: modelsDirectory.path)),
            ("Imports folder readable", fm.isReadableFile(atPath: importsDirectory.path)),
            ("Imports folder writable", fm.isWritableFile(atPath: importsDirectory.path)),
            ("E2E folder writable", e2eDirectory.map { fm.isWritableFile(atPath: $0.path) } ?? false),
        ]
        let passed = checks.filter(\.1).count
        storageCheckRatio = Double(passed) / Double(max(checks.count, 1))
        storageCheckSummary = "\(passed)/\(checks.count)"
        let summary = checks
            .map { "• \($0.0): \($0.1 ? "PASS" : "FAIL")" }
            .joined(separator: "\n")
        statusMessage = "\(passed)/\(checks.count) storage checks passed."
        refreshFindings(storageChecks: checks)
        return "\(passed)/\(checks.count) checks passed\n\n\(summary)"
    }

    @discardableResult
    func exportLatestE2EReport() -> String {
        loadLatestE2EReport()
        guard let e2eLatestReport else {
            e2eExportError = "No E2E report is available. Run the complete E2E runner first."
            activeWorkflowAction = .exportLiveE2E
            lastWorkflowSummary = "Live E2E export blocked: no report"
            return e2eExportError ?? "No E2E report is available."
        }

        do {
            let result = try EvidenceLayerExporter.writeLayer(
                payload: e2eLatestReport,
                filePrefix: "lumen-live-e2e-report",
                format: "live-e2e-test-report-json",
                sourceLayer: "e2eTestReport",
                ownsLiveE2EScenarios: true,
                includesDeterministicStaticScenarios: e2eLatestReport.results.contains { !$0.requiresAgentRun },
                privacy: "Contains prompts, final outputs, failures, and event logs from the current local E2E run. Review before sharing outside the improve-loop.",
                notes: [
                    "This is the live E2E model/test layer export.",
                    "Scenarios with requiresAgentRun=true must exercise AgentService's model-backed generation path and record fresh AgentBehaviorTrace modelTurn evidence.",
                    "Routing-only tool coverage scenarios are static guard checks; if a live scenario says no model loaded, routing-only checks completed, or has no model-evidence event, the offline ingester treats it as invalid E2E evidence."
                ]
            )
            e2eLastExportURL = result.url
            e2eExportError = nil
            activeWorkflowAction = .exportLiveE2E
            lastWorkflowSummary = "Live E2E JSON exported: \(result.url.lastPathComponent)"
            refreshFindings()
            return "Live E2E report exported:\n\(result.url.path)"
        } catch {
            let message = "Live E2E report export failed: \(error.localizedDescription)"
            e2eExportError = message
            activeWorkflowAction = .exportLiveE2E
            lastWorkflowSummary = message
            refreshFindings()
            return message
        }
    }

    func logsText() -> String {
        let modelsDirectory = ModelStorage.modelsDirectoryURL()
        let imported = FileStore.importedFiles()
        let modelFiles = (try? FileManager.default.contentsOfDirectory(at: modelsDirectory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
        return """
        Last launch diagnostics:
        • Imported files: \(imported.count)
        • Model files: \(modelFiles.count)
        • Models path: \(modelsDirectory.path)
        \(AgentParseFailureSummaryLoader.developerText(topN: 5))
        """
    }

    func debugText(appState: AppState) -> String {
        let runtimeContract = LumenTrainedModelRuntimeRegistry.selected
        return """
        Runtime:
        • isGenerating: \(appState.isGenerating ? "true" : "false")
        • agentModeEnabled: \(appState.agentModeEnabled ? "true" : "false")
        • showThinkingByDefault: \(appState.showThinkingByDefault ? "true" : "false")
        • developerTraceModeEnabled: \(appState.developerTraceModeEnabled ? "true" : "false")
        • developerReasoningCaptureEnabled: \(appState.developerReasoningCaptureEnabled ? "true" : "false")
        • maxAgentSteps: \(appState.maxAgentSteps)

        Fleet:
        • selectedModelFamily: \(LumenModelFamily.persistedSelected.rawValue)
        • trainedModelContract: \(runtimeContract.schemaVersion)
        • trainedBaseModelID: \(runtimeContract.sharedBaseModelID)
        • adapterRuntimeMode: \(runtimeContract.mode)
        • adapterRoles: \(runtimeContract.adapterRoleIDs.joined(separator: ","))
        • adapterFiles: \(runtimeContract.adapterFileNames.joined(separator: ","))
        • loadBaseModelOnce: \(runtimeContract.loadBaseModelOnce ? "true" : "false")
        • selectAdapterByAgentSlot: \(runtimeContract.selectAdapterByAgentSlot ? "true" : "false")
        • mergeAdaptersByDefault: \(runtimeContract.mergeAdaptersByDefault ? "true" : "false")
        • releaseBakeManualOnly: \(runtimeContract.releaseBakeManualOnly ? "true" : "false")
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

    func diagnosticText() -> String {
        let permissions = PermissionKind.allCases
            .map { "\($0.title): \(PermissionsCenter.shared.state($0).label)" }
            .joined(separator: "\n")
        return """
        Permissions:
        \(permissions)

        Recoverable noise signatures:
        \(AgentParseNoiseSummaryLoader.developerText(topN: 5))

        Latest E2E:
        \(E2ETestLogStore.latestText())
        """
    }

    func refreshFindings(storageChecks: [(String, Bool)] = []) {
        var next: [DeveloperFinding] = [
            DeveloperFinding(
                severity: .info,
                title: "Live E2E is authoritative",
                detail: "`e2eTestReport` is the only evidence layer that owns live scenario pass/fail."
            ),
            DeveloperFinding(
                severity: .warning,
                title: "Runtime traces required",
                detail: "Agent Grounding exports with empty recent traces become improve-loop gaps after real app interactions are expected."
            )
        ]

        if let diagnosticsSnapshot {
            if diagnosticsSnapshot.runtime.lowPowerModeEnabled {
                next.append(DeveloperFinding(severity: .warning, title: "Low Power Mode", detail: "Runtime validation may degrade while Low Power Mode is enabled."))
            }
            if diagnosticsSnapshot.runtime.memoryWarningCount > 0 {
                next.append(DeveloperFinding(severity: .warning, title: "Memory pressure", detail: "Recent memory warnings: \(diagnosticsSnapshot.runtime.memoryWarningCount)."))
            }
            if !diagnosticsSnapshot.background.entitlementWarnings.isEmpty {
                next.append(DeveloperFinding(severity: .warning, title: "Background entitlement warnings", detail: diagnosticsSnapshot.background.entitlementWarnings.joined(separator: "\n")))
            }
        }

        for check in storageChecks where !check.1 {
            next.append(DeveloperFinding(severity: .error, title: check.0, detail: "Storage check failed."))
        }

        if let e2eLatestReport {
            if e2eLatestReport.failed > 0 {
                next.append(DeveloperFinding(severity: .error, title: "E2E failures present", detail: "\(e2eLatestReport.failed) live E2E result(s) failed. Open the complete E2E runner for scenario details."))
            }
            if !e2eLatestReport.results.contains(where: \.requiresAgentRun) {
                next.append(DeveloperFinding(severity: .warning, title: "No live model E2E evidence", detail: "The latest E2E report contains only static guard scenarios; run live model-backed scenarios before release evidence export."))
            }
        } else {
            next.append(DeveloperFinding(severity: .warning, title: "E2E report missing", detail: "Run the complete standard E2E suite and export the live E2E JSON layer."))
        }

        findings = next
    }

    var monitoringEnabled: Bool {
        AgentWorkflowMonitor.shared.isMonitoring
    }

    var workflowProgress: Double {
        if isWorkflowRunning { return 0.62 }
        if monitoringEnabled { return workflowSnapshot.events.isEmpty ? 0.42 : 0.78 }
        return 0.18
    }

    var dashboardMeters: [DeveloperDashboardMeter] {
        let runtime = diagnosticsSnapshot?.runtime
        let runtimeChecks = [
            runtime?.foundationModelsAvailable == true,
            runtime?.coreMLAvailable == true,
            runtime?.metalAvailable == true,
        ]
        let runtimePassed = runtimeChecks.filter { $0 }.count
        let runtimeProgress = Double(runtimePassed) / Double(runtimeChecks.count)
        let memoryWarnings = runtime?.memoryWarningCount ?? 0
        let fallbackCount = workflowSnapshot.fallbackCount
        let errorCount = workflowSnapshot.errorCount
        let eventCount = workflowSnapshot.events.count
        let traceProgress = min(Double(max(eventCount - fallbackCount - errorCount, 0)) / 8.0, 1.0)
        let evidenceProgress = e2eLatestReport.map { report in
            report.failed == 0 && report.results.contains(where: \.requiresAgentRun) ? 1.0 : 0.58
        } ?? 0.25

        return [
            DeveloperDashboardMeter(
                id: "runtime",
                title: "Runtime",
                value: "\(runtimePassed)/3",
                detail: memoryWarnings > 0 ? "\(memoryWarnings) memory warnings" : (runtime?.thermalState ?? "cached"),
                progress: runtimeProgress,
                systemImage: "cpu",
                level: runtimePassed == 3 && memoryWarnings == 0 ? .nominal : .warning
            ),
            DeveloperDashboardMeter(
                id: "workflow",
                title: "Workflow",
                value: monitoringEnabled ? "Live" : "Idle",
                detail: "\(eventCount) events · \(workflowSnapshot.touchedSlots.count) slots",
                progress: workflowProgress,
                systemImage: monitoringEnabled ? "dot.radiowaves.left.and.right" : "pause.circle",
                level: monitoringEnabled ? .nominal : .inactive
            ),
            DeveloperDashboardMeter(
                id: "traces",
                title: "Traces",
                value: "\(eventCount)",
                detail: "\(fallbackCount) fallback · \(errorCount) error",
                progress: traceProgress,
                systemImage: "waveform.path.ecg",
                level: errorCount > 0 ? .critical : fallbackCount > 0 ? .warning : eventCount > 0 ? .nominal : .inactive
            ),
            DeveloperDashboardMeter(
                id: "evidence",
                title: "Evidence",
                value: e2eLatestReport.map { "\($0.passed)/\($0.results.count)" } ?? "E2E",
                detail: e2eLatestReport == nil ? "report missing" : "\(e2eLiveResultCount) live · \(e2eStaticResultCount) static",
                progress: evidenceProgress,
                systemImage: "testtube.2",
                level: e2eLatestReport == nil ? .warning : e2eLatestReport?.failed == 0 ? .nominal : .critical
            ),
            DeveloperDashboardMeter(
                id: "storage",
                title: "Storage",
                value: storageCheckSummary,
                detail: "models · imports · E2E",
                progress: storageCheckRatio ?? 0.2,
                systemImage: "externaldrive.badge.checkmark",
                level: (storageCheckRatio ?? 0) >= 1 ? .nominal : storageCheckRatio == nil ? .inactive : .warning
            ),
            DeveloperDashboardMeter(
                id: "findings",
                title: "Findings",
                value: "\(findings.count)",
                detail: "\(findings.filter { $0.severity == .error }.count) blocking",
                progress: findings.contains(where: { $0.severity == .error }) ? 0.32 : 0.82,
                systemImage: "exclamationmark.triangle",
                level: findings.contains(where: { $0.severity == .error }) ? .critical : .warning
            ),
        ]
    }

    var e2eLatestResults: [E2ETestResult] {
        e2eLatestReport?.results ?? []
    }

    var e2eLiveResultCount: Int {
        e2eLatestResults.filter(\.requiresAgentRun).count
    }

    var e2eStaticResultCount: Int {
        e2eLatestResults.filter { !$0.requiresAgentRun }.count
    }

    var e2ePassRate: Double {
        guard !e2eLatestResults.isEmpty else { return 0 }
        return Double(e2eLatestResults.filter(\.passed).count) / Double(e2eLatestResults.count)
    }

    var hasExportedLiveE2E: Bool {
        e2eLastExportURL != nil
    }

    func hasE2EReport(containing scenarios: [E2ETestScenario]) -> Bool {
        guard let e2eLatestReport else { return false }
        let reportedIDs = Set(e2eLatestReport.results.map(\.scenarioID))
        return scenarios.contains { reportedIDs.contains($0.id) }
    }

    var telemetryMeters: [DeveloperTelemetryMeter] {
        let memoryPercent = telemetry.residentMemoryMB.map { min(max($0 / max(telemetry.physicalMemoryMB, 1), 0), 1) } ?? 0
        let thermalProgress = Self.thermalProgress(telemetry.thermalState)
        let gpuProgress = telemetry.gpuUtilizationPercent.map { min(max($0 / 100, 0), 1) } ?? (telemetry.metalAvailable ? 0.62 : 0.08)
        let aneProgress = telemetry.aneUtilizationPercent.map { min(max($0 / 100, 0), 1) } ?? (telemetry.coreMLAvailable ? 0.42 : 0.08)
        let firstTokenProgress = telemetry.firstTokenLatencyMs.map { 1 - min(Double($0) / 6000, 1) } ?? 0.18
        let tpsProgress = telemetry.tokensPerSecond.map { min($0 / 32, 1) } ?? 0.12
        let successProgress = telemetry.recentSuccessRate ?? 0.24

        return [
            DeveloperTelemetryMeter(
                id: "cpu",
                title: "CPU",
                value: "\(Int(telemetry.cpuProxyPercent.rounded()))",
                unit: "%",
                detail: "event-density proxy · cores \(telemetry.processorSummary)",
                progress: telemetry.cpuProxyPercent / 100,
                systemImage: "cpu",
                tint: telemetry.cpuProxyPercent > 75 ? .orange : .cyan,
                style: .percent
            ),
            DeveloperTelemetryMeter(
                id: "ram",
                title: "RAM",
                value: telemetry.residentMemoryMB.map { "\(Int($0.rounded()))" } ?? "—",
                unit: "MB",
                detail: "resident / \(Int(telemetry.physicalMemoryMB / 1024))GB physical",
                progress: memoryPercent,
                systemImage: "memorychip",
                tint: memoryPercent > 0.72 ? .orange : .green,
                style: .value
            ),
            DeveloperTelemetryMeter(
                id: "gpu",
                title: "GPU",
                value: telemetry.gpuUtilizationPercent.map { "\(Int($0.rounded()))" } ?? (telemetry.metalAvailable ? "Metal" : "Off"),
                unit: telemetry.gpuUtilizationPercent == nil ? "" : "%",
                detail: "\(telemetry.gpuVerification) · \(telemetry.metalDeviceName ?? "no device")",
                progress: gpuProgress,
                systemImage: "gpu",
                tint: telemetry.metalAvailable ? .purple : Theme.textSecondary,
                style: telemetry.gpuUtilizationPercent == nil ? .availability : .percent
            ),
            DeveloperTelemetryMeter(
                id: "ane",
                title: "ANE",
                value: telemetry.aneUtilizationPercent.map { "\(Int($0.rounded()))" } ?? (telemetry.coreMLAvailable ? "Ready" : "Off"),
                unit: telemetry.aneUtilizationPercent == nil ? "" : "%",
                detail: telemetry.aneStatus,
                progress: aneProgress,
                systemImage: "bolt.badge.automatic",
                tint: telemetry.coreMLAvailable ? Theme.accent : Theme.textSecondary,
                style: telemetry.aneUtilizationPercent == nil ? .availability : .percent
            ),
            DeveloperTelemetryMeter(
                id: "thermal",
                title: "Thermal",
                value: telemetry.thermalState,
                unit: "",
                detail: telemetry.lowPowerMode ? "Low Power enabled" : "Power normal",
                progress: thermalProgress,
                systemImage: "thermometer.medium",
                tint: thermalProgress > 0.67 ? .orange : .green,
                style: .availability
            ),
            DeveloperTelemetryMeter(
                id: "latency",
                title: "First Token",
                value: telemetry.firstTokenLatencyMs.map(String.init) ?? "—",
                unit: "ms",
                detail: telemetry.recentLatencyMs.map { "avg runtime latency \($0)ms" } ?? "awaiting model turn",
                progress: firstTokenProgress,
                systemImage: "timer",
                tint: firstTokenProgress > 0.55 ? .green : .orange,
                style: .value
            ),
            DeveloperTelemetryMeter(
                id: "decode",
                title: "Decode",
                value: telemetry.tokensPerSecond.map { String(format: "%.1f", $0) } ?? "—",
                unit: "tok/s",
                detail: "latest model throughput",
                progress: tpsProgress,
                systemImage: "speedometer",
                tint: tpsProgress > 0.4 ? .green : Theme.accent,
                style: .value
            ),
            DeveloperTelemetryMeter(
                id: "success",
                title: "Success",
                value: telemetry.recentSuccessRate.map { "\(Int(($0 * 100).rounded()))" } ?? "—",
                unit: "%",
                detail: "recent runtime metrics",
                progress: successProgress,
                systemImage: "checkmark.seal",
                tint: successProgress > 0.9 ? .green : Theme.accent,
                style: .percent
            ),
        ]
    }

    private nonisolated static func thermalProgress(_ state: String) -> Double {
        switch state.lowercased() {
        case "nominal": return 0.22
        case "fair": return 0.45
        case "serious": return 0.72
        case "critical": return 1.0
        default: return 0.3
        }
    }

    private nonisolated static func residentMemoryUsageMB() -> Double? {
        #if canImport(Darwin)
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.stride)
        let result: kern_return_t = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPointer in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPointer, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return Double(info.phys_footprint) / (1024 * 1024)
        #else
        return nil
        #endif
    }
}
