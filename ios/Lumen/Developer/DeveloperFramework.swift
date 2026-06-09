import Foundation
import Observation
import SwiftUI

enum DeveloperConsoleTab: String, CaseIterable, Identifiable {
    case overview
    case evidence
    case workflows
    case reports
    case privacy

    var id: String { rawValue }

    var title: String {
        switch self {
        case .overview: return "Overview"
        case .evidence: return "Evidence"
        case .workflows: return "Workflow"
        case .reports: return "Reports"
        case .privacy: return "Privacy"
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
    var selectedTab: DeveloperConsoleTab = .overview
    var diagnosticsSnapshot: DiagnosticsSnapshot?
    var evidenceLayers = DeveloperEvidenceLayerStatus.baseline()
    var findings: [DeveloperFinding] = []
    var statusMessage = "Developer console ready."
    var isCollectingDiagnostics = false

    private var diagnosticsProvider = DiagnosticsProvider()

    func loadCachedDiagnostics() {
        diagnosticsSnapshot = diagnosticsProvider.cachedSnapshot()
        refreshFindings()
    }

    func collectDiagnostics() async {
        isCollectingDiagnostics = true
        diagnosticsSnapshot = await diagnosticsProvider.collect()
        isCollectingDiagnostics = false
        statusMessage = "Diagnostics refreshed."
        refreshFindings()
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
        let summary = checks
            .map { "• \($0.0): \($0.1 ? "PASS" : "FAIL")" }
            .joined(separator: "\n")
        statusMessage = "\(passed)/\(checks.count) storage checks passed."
        refreshFindings(storageChecks: checks)
        return "\(passed)/\(checks.count) checks passed\n\n\(summary)"
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

        findings = next
    }
}
