import Foundation
import SwiftUI
import UIKit

nonisolated enum PersistentDiagnosticAutomationPolicy: String, Codable, Sendable, CaseIterable {
    case automatic
    case manualOnly
    case disabled
}

nonisolated enum PersistentDiagnosticScenarioKind: String, Codable, Sendable, CaseIterable, Identifiable {
    case plainFastPrompt
    case plainDeveloperTraceBypass
    case agentFastPrompt
    case dryRunPromptBudgetOnly
    case sandboxedToolPlanOnly
    case liveAgentStream
    case agentToolPrompt
    case agentCancellation
    case lifecycleCancellation
    case diskWriteGate
    case swiftUIChurnProbe
    case groundingCostProbe
    case thermalResourceGate

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .plainFastPrompt: return "Plain fast prompt"
        case .plainDeveloperTraceBypass: return "Developer trace bypass"
        case .agentFastPrompt: return "Agent fast prompt"
        case .dryRunPromptBudgetOnly: return "Agent dry-run prompt budget"
        case .sandboxedToolPlanOnly: return "Agent sandboxed tool plan"
        case .liveAgentStream: return "Live agent stream"
        case .agentToolPrompt: return "Legacy live agent tool prompt"
        case .agentCancellation: return "Agent cancellation"
        case .lifecycleCancellation: return "Lifecycle cancellation"
        case .diskWriteGate: return "Disk write gate"
        case .swiftUIChurnProbe: return "SwiftUI churn probe"
        case .groundingCostProbe: return "Grounding cost probe"
        case .thermalResourceGate: return "Thermal resource gate"
        }
    }

    var automationPolicy: PersistentDiagnosticAutomationPolicy {
        switch self {
        case .lifecycleCancellation, .liveAgentStream, .agentToolPrompt:
            return .manualOnly
        case .plainFastPrompt, .plainDeveloperTraceBypass, .agentFastPrompt, .dryRunPromptBudgetOnly, .sandboxedToolPlanOnly, .agentCancellation, .diskWriteGate, .swiftUIChurnProbe, .groundingCostProbe, .thermalResourceGate:
            return .automatic
        }
    }

    static var automaticCases: [PersistentDiagnosticScenarioKind] {
        allCases.filter { $0.automationPolicy == .automatic }
    }

    var requiresExplicitUserRequest: Bool {
        automationPolicy == .manualOnly
    }

}

nonisolated enum PersistentDiagnosticStatus: String, Codable, Sendable {
    case pending
    case running
    case passed
    case failed
    case skipped
    case cancelled
    case interrupted
}

nonisolated struct PersistentDiagnosticCampaign: Codable, Sendable, Identifiable, Equatable {
    let id: UUID
    var createdAt: Date
    var updatedAt: Date
    var enabled: Bool
    var runContinuously: Bool
    var maxRunsPerScenario: Int
    var delayBetweenRunsSeconds: Double
    var scenarios: [PersistentDiagnosticScenarioKind]

    init(
        id: UUID = UUID(),
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        enabled: Bool = false,
        runContinuously: Bool = false,
        maxRunsPerScenario: Int = 1,
        delayBetweenRunsSeconds: Double = 5,
        scenarios: [PersistentDiagnosticScenarioKind] = PersistentDiagnosticScenarioKind.automaticCases
    ) {
        self.id = id
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.enabled = enabled
        self.runContinuously = runContinuously
        self.maxRunsPerScenario = max(1, maxRunsPerScenario)
        self.delayBetweenRunsSeconds = max(0.5, delayBetweenRunsSeconds)
        self.scenarios = scenarios.isEmpty ? [.plainFastPrompt, .agentFastPrompt] : scenarios
    }
}

nonisolated struct PersistentDiagnosticRunRecord: Codable, Sendable, Identifiable, Equatable {
    let id: UUID
    let campaignID: UUID
    let scenario: PersistentDiagnosticScenarioKind
    let startedAt: Date
    var finishedAt: Date?
    var status: PersistentDiagnosticStatus
    var metrics: PersistentDiagnosticMetrics
    var events: [PersistentDiagnosticEvent]
    var failureSummary: String?
    var remediationProposals: [PersistentDiagnosticRemediationProposal]?

    init(id: UUID = UUID(), campaignID: UUID, scenario: PersistentDiagnosticScenarioKind, startedAt: Date = Date(), status: PersistentDiagnosticStatus = .pending, metrics: PersistentDiagnosticMetrics = .init(), events: [PersistentDiagnosticEvent] = [], failureSummary: String? = nil) {
        self.id = id
        self.campaignID = campaignID
        self.scenario = scenario
        self.startedAt = startedAt
        self.finishedAt = nil
        self.status = status
        self.metrics = metrics
        self.events = events
        self.failureSummary = failureSummary
        self.remediationProposals = nil
    }
}

nonisolated struct PersistentDiagnosticMetrics: Codable, Sendable, Equatable {
    var scenePhase: String?
    var thermalState: String?
    var lowPowerMode: Bool?
    var memoryWarningCount: Int?
    var realScenePhase: String?
    var realThermalState: String?
    var realDenied: Bool?
    var simulatedScenePhase: String?
    var simulatedThermalState: String?
    var simulatedDenied: Bool?
    var cpuWatchdog: PersistentDiagnosticCPUWatchdogSnapshot?
    var diskWrite: PersistentDiagnosticDiskWriteSnapshot?
    var generationActive: Bool = false
    var promptLatencyClass: String?
    var promptInitialChars: Int?
    var promptFinalChars: Int?
    var estimatedPromptTokens: Int?
    var promptSHA256: String?
    var promptBodyBytes: Int?
    var promptRedactionMode: String?
    var firstTokenLatencyMs: Int?
    var generationElapsedMs: Int?
    var agentGroundingElapsedMs: Int?
    var groundingSectionCount: Int?
    var groundingChars: Int?
    var toolCount: Int?
    var inputToolCount: Int?
    var bridgedToolCount: Int?
    var memoryCount: Int?
    var didUseFastPath: Bool = false
    var didCancel: Bool = false
    var cancellationReason: String?
    var didFallback: Bool = false
    var fallbackReason: String?
    var uiUpdateCount: Int = 0
    var streamingUpdateCount: Int = 0
    var diskBytesBefore: Int64?
    var diskBytesAfter: Int64?
    var appBecameInactiveOrBackgroundDuringRun: Bool = false
    var errorCodes: [String] = []

    mutating func captureNonisolatedEnvironment() {
        thermalState = DeviceThermalState.from(processThermalState: ProcessInfo.processInfo.thermalState).rawValue
        lowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
        cpuWatchdog = PersistentDiagnosticCPUWatchdogSnapshot(CPUWatchdogGuard.shared.currentSnapshot())
        diskWrite = PersistentDiagnosticDiskWriteSnapshot(DiskWriteBudget.shared.snapshot())
        generationActive = DiskWriteBudget.shared.isGenerationActive()
    }

    static func sceneString(_ phase: ScenePhase?) -> String? {
        switch phase {
        case .active: return "active"
        case .inactive: return "inactive"
        case .background: return "background"
        case nil: return nil
        @unknown default: return "unknown"
        }
    }
}

nonisolated struct PersistentDiagnosticCPUWatchdogSnapshot: Codable, Sendable, Equatable {
    var degradedCategories: [String]
    var totalsByCategory: [String: Double]
    var activeCountsByCategory: [String: Int]

    init(_ snapshot: CPUWatchdogSnapshot) {
        degradedCategories = snapshot.degradedCategories.map(\.rawValue).sorted()
        totalsByCategory = Dictionary(uniqueKeysWithValues: snapshot.totalsByCategory.map { ($0.key.rawValue, $0.value) })
        activeCountsByCategory = Dictionary(uniqueKeysWithValues: snapshot.activeCountsByCategory.map { ($0.key.rawValue, $0.value) })
    }
}

nonisolated struct PersistentDiagnosticDiskWriteSnapshot: Codable, Sendable, Equatable {
    var bytes1Minute: Int64
    var bytes15Minutes: Int64
    var bytes24Hours: Int64
    var bytesByCategory24Hours: [String: Int64]
    var generationActive: Bool

    init(_ snapshot: DiskWriteBudgetSnapshot, generationActive: Bool = DiskWriteBudget.shared.isGenerationActive()) {
        bytes1Minute = snapshot.bytes1Minute
        bytes15Minutes = snapshot.bytes15Minutes
        bytes24Hours = snapshot.bytes24Hours
        bytesByCategory24Hours = Dictionary(uniqueKeysWithValues: snapshot.bytesByCategory24Hours.map { ($0.key.rawValue, $0.value) })
        self.generationActive = generationActive
    }
}

nonisolated struct PersistentDiagnosticEvent: Codable, Sendable, Identifiable, Equatable {
    let id: UUID
    let at: Date
    let code: String
    let message: String
    let values: [String: String]

    init(id: UUID = UUID(), at: Date = Date(), code: String, message: String, values: [String: String] = [:]) {
        self.id = id
        self.at = at
        self.code = PersistentRuntimeDiagnosticsRedactor.safeCode(code)
        self.message = PersistentRuntimeDiagnosticsRedactor.redact(message)
        self.values = PersistentRuntimeDiagnosticsRedactor.redact(values)
    }
}

nonisolated enum PersistentDiagnosticRemediationSeverity: String, Codable, Sendable, Equatable {
    case info
    case warning
    case critical
}

nonisolated struct PersistentDiagnosticRemediationProposal: Codable, Sendable, Equatable, Identifiable {
    let id: String
    let title: String
    let rationale: String
    let action: String
    let severity: PersistentDiagnosticRemediationSeverity
}

nonisolated enum PersistentDiagnosticRemediationAdvisor {
    static func proposals(
        for record: PersistentDiagnosticRunRecord,
        status: PersistentDiagnosticStatus,
        code: String
    ) -> [PersistentDiagnosticRemediationProposal] {
        guard status != .passed else { return [] }

        switch code {
        case "resource_gate_paused":
            return [proposal(id: "resource-gate-wait", title: "Wait for a safer resource window", rationale: "Diagnostics were paused by thermal, memory, or lifecycle resource policy.", action: "Run diagnostics again while the app is foreground, the device is cool, and no generation is active.", severity: .info)]
        case "manual_scenario_requires_explicit_request", "manual_live_agent_stream_required", "manual_probe_required":
            return [proposal(id: "manual-scenario-foreground", title: "Run the diagnostic from the foreground control", rationale: "This scenario requires explicit user action and should not run unattended.", action: "Open Runtime Diagnostics and start the matching manual probe from the foreground UI.", severity: .info)]
        case "skipped_no_model":
            return [proposal(id: "local-chat-model-required", title: "Select a local chat model", rationale: "Prompt budget checks ran, but live inference could not run without a loaded local chat runtime.", action: "Install or select a local GGUF/Core ML/FoundationModels-backed chat model, then rerun the scenario.", severity: .warning)]
        case "fast_prompt_too_large", "fast_tokens_too_large", "fast_latency_missing":
            return [proposal(id: "tighten-fast-prompt-budget", title: "Tighten fast prompt budgeting", rationale: "The fast prompt path exceeded its expected character, token, or latency class budget.", action: "Inspect prompt assembly, memory/RAG injection, and latency classification for the fast path.", severity: .warning)]
        case "developer_trace_bypass_missing":
            return [proposal(id: "restore-developer-trace-bypass", title: "Restore developer-trace bypass classification", rationale: "Developer trace mode should bypass normal fast-interactive routing.", action: "Check PromptLatencyClassifier developer-trace handling and regression coverage.", severity: .warning)]
        case "tool_prompt_used_fast_path":
            return [proposal(id: "route-tool-prompts-through-agent-grounding", title: "Route tool prompts through grounded agent planning", rationale: "A tool prompt incorrectly selected the fast answer path.", action: "Fix SlotAgentService.shouldUseFastAgentPath and prompt routing rules for tool-bearing requests.", severity: .critical)]
        case "agent_fast_path_unbounded":
            return [proposal(id: "bound-fast-agent-grounding", title: "Reduce fast-agent grounding payload", rationale: "Fast-agent grounding exceeded its bounded local context contract.", action: "Trim bridged tools, memory snippets, or grounding sections before fast-agent execution.", severity: .warning)]
        case "agent_tool_dry_run_unbounded", "sandboxed_tool_plan_unbounded", "grounding_cost_unbounded":
            return [proposal(id: "bound-grounding-and-tool-bridge", title: "Bound grounding and tool-bridge cost", rationale: "Grounding or tool planning exceeded the local diagnostic budget.", action: "Inspect LegacyTurnGroundingCoordinator, LegacyToolSchemaBridge, and context budgeting for this scenario.", severity: .warning)]
        case "disk_gate_unexpected":
            return [proposal(id: "audit-diagnostic-disk-write-gate", title: "Audit diagnostic disk-write gating", rationale: "Diagnostics writes did not obey the expected generation-time disk budget.", action: "Check DiskWriteBudget category handling and PersistentRuntimeDiagnosticsStore buffering.", severity: .warning)]
        case "ui_churn_excessive":
            return [proposal(id: "throttle-diagnostic-ui-updates", title: "Throttle diagnostic UI updates", rationale: "Synthetic UI churn exceeded the bounded update contract.", action: "Audit reducers and streaming update coalescing before widening diagnostic cadence.", severity: .warning)]
        case "resource_gate_policy_failed":
            return [proposal(id: "fix-resource-gate-policy-matrix", title: "Fix the resource-gate policy matrix", rationale: "Simulated resource states did not match expected foreground/background model-load policy.", action: "Inspect ResourceBudgetGate and ModelLoader policy checks for thermal, scene phase, and Low Power Mode drift.", severity: .critical)]
        case "not_manual_scenario":
            return [proposal(id: "choose-supported-diagnostic-scenario", title: "Choose a supported diagnostic scenario", rationale: "The requested manual runner path does not own this scenario.", action: "Run automatic scenarios through Run Once or choose a manual-only scenario from the foreground controls.", severity: .info)]
        case "crash_resume", "interrupted_or_terminated":
            return [proposal(id: "inspect-lifecycle-interruption", title: "Inspect lifecycle cancellation and crash-resume evidence", rationale: "A diagnostic run was interrupted or terminated before clean completion.", action: "Review scene-transition, cancellation-bus, and last-run events in the exported diagnostics package.", severity: .critical)]
        default:
            guard status == .failed || status == .interrupted else { return [] }
            return [proposal(id: "inspect-diagnostic-failure-\(code)", title: "Inspect diagnostic failure evidence", rationale: "The diagnostic failed with code \(code).", action: "Export persistent diagnostics and inspect the redacted events, metrics, and scenario-specific code path.", severity: status == .interrupted ? .critical : .warning)]
        }
    }

    private static func proposal(
        id: String,
        title: String,
        rationale: String,
        action: String,
        severity: PersistentDiagnosticRemediationSeverity
    ) -> PersistentDiagnosticRemediationProposal {
        PersistentDiagnosticRemediationProposal(id: id, title: title, rationale: rationale, action: action, severity: severity)
    }
}

nonisolated struct PersistentDiagnosticRunnerStatus: Codable, Sendable, Equatable {
    var isRunning: Bool = false
    var isPaused: Bool = false
    var latestScenario: PersistentDiagnosticScenarioKind?
    var passedCount: Int = 0
    var failedCount: Int = 0
    var skippedCount: Int = 0
    var lastFirstTokenLatencyMs: Int?
    var lastPromptFinalChars: Int?
    var lastCancellationReason: String?
    var lastCrashResumeStatus: String?
    var lastRemediationSummary: String?
    var lastUpdatedAt: Date = Date()
}

nonisolated struct PersistentDiagnosticState: Codable, Sendable, Equatable {
    static let maxCompletedRunIDs = 500

    var activeRunID: UUID?
    var activeCampaignID: UUID?
    var activeScenario: PersistentDiagnosticScenarioKind?
    var activeStartedAt: Date?
    var activeLaunchUUID: UUID?
    var cleanCancellationBeforeTermination: Bool = false
    var completedRunIDs: [UUID] = []
    var records: [PersistentDiagnosticRunRecord] = []
    var status: PersistentDiagnosticRunnerStatus = .init()

    mutating func markRunCompleted(_ id: UUID) {
        completedRunIDs.removeAll { $0 == id }
        completedRunIDs.append(id)
        trimCompletedRunIDs()
    }

    mutating func trimCompletedRunIDs(limit: Int = Self.maxCompletedRunIDs) {
        guard completedRunIDs.count > limit else { return }
        completedRunIDs.removeFirst(completedRunIDs.count - limit)
    }
}


nonisolated enum PersistentRuntimeDiagnosticsRedactor {
    static let maxEventMessageChars = 160
    static let maxValueChars = 96

    static func safeCode(_ code: String) -> String {
        let allowed = code.lowercased().map { ch in
            (ch.isLetter || ch.isNumber || ch == "_" || ch == "-" || ch == ".") ? ch : "_"
        }
        return String(String(allowed).prefix(64))
    }

    static func redact(_ text: String) -> String {
        String(redactWithoutTruncating(text).prefix(maxEventMessageChars))
    }

    static func redactWithoutTruncating(_ text: String) -> String {
        var output = text
        let patterns = [
            #"(?i)(prompt|memory|file|path|content|body|text)=([^,\n]+)"#,
            #"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"#,
            #"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"#,
            #"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"#,
            #"/[^\s,]+"#
        ]
        for pattern in patterns {
            output = output.replacingOccurrences(of: pattern, with: "[redacted]", options: [.regularExpression, .caseInsensitive])
        }
        return output
    }

    static func redact(_ values: [String: String]) -> [String: String] {
        Dictionary(uniqueKeysWithValues: values.map { key, value in
            let safeKey = safeCode(key)
            let redactedValue: String
            if isSensitiveValueKey(safeKey), !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                redactedValue = "[redacted]"
            } else {
                redactedValue = String(redact(value).prefix(maxValueChars))
            }
            return (safeKey, redactedValue)
        })
    }

    private static func isSensitiveValueKey(_ key: String) -> Bool {
        switch key {
        case "prompt", "memory", "file", "path", "content", "body", "text":
            return true
        default:
            return false
        }
    }
}

nonisolated enum PersistentRuntimeDiagnosticsAvailability {
    static var isDeveloperVisible: Bool {
        #if DEBUG
        return true
        #else
        return UserDefaults.standard.bool(forKey: "lumen.developer.persistentRuntimeDiagnostics.visible")
        #endif
    }

    static func enableForInternalDiagnostics() {
        UserDefaults.standard.set(true, forKey: "lumen.developer.persistentRuntimeDiagnostics.visible")
    }
}

nonisolated enum PersistentRuntimeDiagnosticSignalKind: String, Codable, Sendable {
    case llamaPromptBudget
    case llamaFirstToken
    case llamaEmptyOutput
    case llamaComplete
    case llamaCancel
    case llamaFailure
    case slotAgentStart
    case slotAgentPath
    case slotAgentFallback
    case slotAgentGroundingComplete
    case slotAgentEffectiveRequestBuilt
    case slotAgentDeterministicAnswerBuilt
    case slotAgentDoneYielded
    case slotAgentEnd
    case slotAgentEndEmitted
    case slotAgentContinuationFinished
    case slotAgentCancel
    case chatRuntimeTrace
    case groundingCost
    case uiUpdate
    case sceneTransition
    case metricKitPersistFailure
    case finalIntentCandidateReplaced
    case fallbackUsed
    case voiceStartupFailure
    case voiceAudioSessionEvent
}

nonisolated struct PersistentRuntimeDiagnosticSignal: Sendable {
    let kind: PersistentRuntimeDiagnosticSignalKind
    let at: Date
    let values: [String: String]

    init(kind: PersistentRuntimeDiagnosticSignalKind, values: [String: String] = [:], at: Date = Date()) {
        self.kind = kind
        self.at = at
        self.values = PersistentRuntimeDiagnosticsRedactor.redact(values)
    }
}

nonisolated final class PersistentRuntimeDiagnosticsObserver: @unchecked Sendable {
    static let shared = PersistentRuntimeDiagnosticsObserver()
    typealias Handler = @Sendable (PersistentRuntimeDiagnosticSignal) -> Void

    private let lock = NSLock()
    private var handlers: [UUID: Handler] = [:]

    private init() {}

    @discardableResult
    func addObserver(_ handler: @escaping Handler) -> UUID {
        let id = UUID()
        lock.lock()
        handlers[id] = handler
        lock.unlock()
        return id
    }

    func removeObserver(_ id: UUID) {
        lock.lock()
        handlers[id] = nil
        lock.unlock()
    }

    func emit(_ signal: PersistentRuntimeDiagnosticSignal) {
        let current: [Handler]
        lock.lock()
        current = Array(handlers.values)
        lock.unlock()
        current.forEach { $0(signal) }
    }
}

extension Bundle {
    var persistentDiagnosticsAppVersionSummary: String {
        let version = object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
        let build = object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown"
        return "\(version) (\(build))"
    }
}

nonisolated extension PersistentDiagnosticCampaign {
    var automaticScenarios: [PersistentDiagnosticScenarioKind] {
        scenarios.filter { $0.automationPolicy == .automatic }
    }

    func automaticOnly() -> PersistentDiagnosticCampaign {
        var copy = self
        copy.scenarios = automaticScenarios
        if copy.scenarios.isEmpty {
            copy.scenarios = [.dryRunPromptBudgetOnly, .sandboxedToolPlanOnly]
        }
        return copy
    }
}
