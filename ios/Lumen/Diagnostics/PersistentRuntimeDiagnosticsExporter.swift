import CryptoKit
import Foundation
import UIKit

actor PersistentRuntimeDiagnosticsExporter {
    typealias MetricKitPayloadProvider = @Sendable () async -> [PersistentMetricKitSourcePayload]

    static let shared = PersistentRuntimeDiagnosticsExporter()
    static let privacySchemaVersion = "2.0.0"
    static let privacySafeFilePrefix = "persistent-runtime-diagnostics-redacted-v2"

    private let store: PersistentRuntimeDiagnosticsStore
    private let fileManager: FileManager
    private let metricKitPayloadProvider: MetricKitPayloadProvider

    init(
        store: PersistentRuntimeDiagnosticsStore = .shared,
        fileManager: FileManager = .default,
        metricKitPayloadProvider: MetricKitPayloadProvider? = nil
    ) {
        self.store = store
        self.fileManager = fileManager
        self.metricKitPayloadProvider = metricKitPayloadProvider ?? {
            await PersistentRuntimeDiagnosticsExporter.metricKitPayloads()
        }
    }

    func export(includeFullHistory: Bool = false) async throws -> URL {
        let exportRoot = fileManager.temporaryDirectory.appendingPathComponent(
            "PersistentRuntimeDiagnosticsExport",
            isDirectory: true
        )

        let campaign = await store.loadCampaign()
        let state = Self.exportState(await store.loadState(), includeFullHistory: includeFullHistory)
        let logData = await store.readLogDataForExport(full: includeFullHistory)
        let decodedLogEntries = Self.decodeLogEntries(logData)
        let logEntries = includeFullHistory
            ? decodedLogEntries
            : Array(decodedLogEntries.suffix(500))
        let metricKitPayloads = await metricKitPayloadProvider()

        let device = await MainActor.run {
            (
                appVersion: Bundle.main.persistentDiagnosticsAppVersionSummary,
                deviceModel: UIDevice.current.model,
                systemName: UIDevice.current.systemName,
                systemVersion: UIDevice.current.systemVersion
            )
        }
        let projector = PersistentRuntimeDiagnosticsPrivacyProjector()
        let payload = try projector.project(
            exportedAt: Date(),
            appVersion: device.appVersion,
            sourceCommit: Self.sourceCommit(),
            deviceModel: device.deviceModel,
            systemName: device.systemName,
            systemVersion: device.systemVersion,
            campaign: campaign,
            state: state,
            logEntries: logEntries,
            metricKitPayloads: metricKitPayloads
        )

        try fileManager.createDirectory(at: exportRoot, withIntermediateDirectories: true)
        let directory = exportRoot.appendingPathComponent(payload.exportScope, isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let packageURL = directory.appendingPathComponent(
            "\(Self.privacySafeFilePrefix)-\(payload.exportScope).json"
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(payload)
        try data.write(to: packageURL, options: [.atomic, .completeFileProtection])

        guard Self.isPrivacySafeShareURL(packageURL) else {
            throw PersistentRuntimeDiagnosticsExportError.unsafeShareURL
        }
        pruneOldExports(in: exportRoot, preserving: directory)
        return packageURL
    }

    nonisolated static func isPrivacySafeShareURL(_ url: URL) -> Bool {
        url.pathExtension.lowercased() == "json"
            && url.lastPathComponent.hasPrefix("\(privacySafeFilePrefix)-export_v1_")
    }

    private func pruneOldExports(in root: URL, preserving current: URL) {
        let keys: Set<URLResourceKey> = [.contentModificationDateKey, .isDirectoryKey]
        guard let children = try? fileManager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles]
        ) else { return }
        let directories = children.compactMap { url -> (URL, Date)? in
            guard url != current,
                  let values = try? url.resourceValues(forKeys: keys),
                  values.isDirectory == true else { return nil }
            return (url, values.contentModificationDate ?? .distantPast)
        }
        for (url, _) in directories.sorted(by: { $0.1 > $1.1 }).dropFirst(7) {
            try? fileManager.removeItem(at: url)
        }
    }

    private static func exportState(
        _ state: PersistentDiagnosticState?,
        includeFullHistory: Bool
    ) -> PersistentDiagnosticState? {
        guard var state else { return nil }
        guard !includeFullHistory else { return state }
        if state.records.count > 500 {
            state.records.removeFirst(state.records.count - 500)
        }
        state.trimCompletedRunIDs()
        return state
    }

    private static func decodeLogEntries(_ data: Data) -> [PersistentDiagnosticLogEntry] {
        guard !data.isEmpty else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return data.split(separator: 0x0A).flatMap { line -> [PersistentDiagnosticLogEntry] in
            let lineData = Data(line)
            if let batch = try? decoder.decode([PersistentDiagnosticLogEntry].self, from: lineData) {
                return batch
            }
            if let entry = try? decoder.decode(PersistentDiagnosticLogEntry.self, from: lineData) {
                return [entry]
            }
            return []
        }
    }

    private static func metricKitPayloads() async -> [PersistentMetricKitSourcePayload] {
        let urls = await MetricKitDiagnosticsStore.shared.exportSummaryPayloadURLs()
        return urls.compactMap { url in
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
            return PersistentMetricKitSourcePayload(fileName: url.lastPathComponent, json: text)
        }
    }

    static func sourceCommit(infoDictionary: [String: Any] = Bundle.main.infoDictionary ?? [:]) -> String? {
        // Current builds inject LumenGitSHA. Keep the legacy key as a read-only
        // compatibility fallback for diagnostics exported by older internal builds.
        for key in ["LumenGitSHA", "GitCommit"] {
            guard let rawValue = infoDictionary[key] as? String else { continue }
            let value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty,
                  value.lowercased() != "unknown",
                  !value.hasPrefix("$(") else { continue }
            return String(value.prefix(128))
        }
        return nil
    }
}

nonisolated enum PersistentRuntimeDiagnosticsExportError: Error {
    case unsafeShareURL
}

nonisolated struct PersistentMetricKitSourcePayload: Sendable {
    let fileName: String
    let json: String
}

nonisolated struct PersistentRuntimeDiagnosticsExportPayload: Codable, Sendable {
    let schemaVersion: String
    let exportScope: String
    let exportedAt: Date
    let privacy: String
    let app: PersistentRuntimeDiagnosticsAppExport
    let campaign: PersistentDiagnosticCampaignExport?
    let state: PersistentDiagnosticStateExport?
    let ndjson: String
    let metricKitPayloads: [PersistentMetricKitExport]
}

nonisolated struct PersistentRuntimeDiagnosticsAppExport: Codable, Sendable {
    let appVersion: String
    let sourceRevisionToken: String?
    let deviceModel: String
    let systemName: String
    let systemVersion: String
}

nonisolated struct PersistentDiagnosticCampaignExport: Codable, Sendable {
    let id: String
    let createdAt: Date
    let updatedAt: Date
    let enabled: Bool
    let runContinuously: Bool
    let maxRunsPerScenario: Int
    let delayBetweenRunsSeconds: Double
    let scenarios: [String]
}

nonisolated struct PersistentDiagnosticStateExport: Codable, Sendable {
    let activeRunID: String?
    let activeCampaignID: String?
    let activeScenario: String?
    let activeStartedAt: Date?
    let activeLaunchID: String?
    let cleanCancellationBeforeTermination: Bool
    let completedRunIDs: [String]
    let records: [PersistentDiagnosticRunRecordExport]
    let status: PersistentDiagnosticRunnerStatusExport
}

nonisolated struct PersistentDiagnosticRunRecordExport: Codable, Sendable {
    let id: String
    let campaignID: String
    let scenario: String
    let startedAt: Date
    let finishedAt: Date?
    let status: String
    let metrics: PersistentDiagnosticMetricsExport
    let events: [PersistentDiagnosticEventExport]
    let failureSummaryToken: String?
    let failureSummaryCharacters: Int?
    let remediationProposals: [PersistentDiagnosticRemediationExport]
}

nonisolated struct PersistentDiagnosticMetricsExport: Codable, Sendable {
    let scenePhase: String?
    let thermalState: String?
    let lowPowerMode: Bool?
    let memoryWarningCount: Int?
    let realScenePhase: String?
    let realThermalState: String?
    let realDenied: Bool?
    let simulatedScenePhase: String?
    let simulatedThermalState: String?
    let simulatedDenied: Bool?
    let cpuWatchdog: PersistentDiagnosticCPUWatchdogExport?
    let diskWrite: PersistentDiagnosticDiskWriteExport?
    let generationActive: Bool
    let promptLatencyClass: String?
    let promptInitialChars: Int?
    let promptFinalChars: Int?
    let estimatedPromptTokens: Int?
    let promptToken: String?
    let promptBodyBytes: Int?
    let promptRedactionModeToken: String?
    let firstTokenLatencyMs: Int?
    let generationElapsedMs: Int?
    let agentGroundingElapsedMs: Int?
    let groundingSectionCount: Int?
    let groundingChars: Int?
    let toolCount: Int?
    let inputToolCount: Int?
    let bridgedToolCount: Int?
    let memoryCount: Int?
    let didUseFastPath: Bool
    let didCancel: Bool
    let cancellationReasonToken: String?
    let didFallback: Bool
    let fallbackReasonToken: String?
    let uiUpdateCount: Int
    let streamingUpdateCount: Int
    let diskBytesBefore: Int64?
    let diskBytesAfter: Int64?
    let appBecameInactiveOrBackgroundDuringRun: Bool
    let errorCodeTokens: [String]
}

nonisolated struct PersistentDiagnosticCPUWatchdogExport: Codable, Sendable {
    let degradedCategories: [String]
    let totalsByCategory: [String: Double]
    let activeCountsByCategory: [String: Int]
}

nonisolated struct PersistentDiagnosticDiskWriteExport: Codable, Sendable {
    let bytes1Minute: Int64
    let bytes15Minutes: Int64
    let bytes24Hours: Int64
    let bytesByCategory24Hours: [String: Int64]
    let generationActive: Bool
}

nonisolated struct PersistentDiagnosticEventExport: Codable, Sendable {
    let id: String
    let at: Date
    let code: String
    let messageToken: String?
    let messageCharacters: Int
    let values: [String: String]
}

nonisolated struct PersistentDiagnosticRemediationExport: Codable, Sendable {
    let id: String
    let severity: String
}

nonisolated struct PersistentDiagnosticRunnerStatusExport: Codable, Sendable {
    let isRunning: Bool
    let isPaused: Bool
    let latestScenario: String?
    let passedCount: Int
    let failedCount: Int
    let skippedCount: Int
    let lastFirstTokenLatencyMs: Int?
    let lastPromptFinalChars: Int?
    let lastCancellationReasonToken: String?
    let lastCrashResumeStatusToken: String?
    let lastRemediationSummaryToken: String?
    let lastUpdatedAt: Date
}

nonisolated struct PersistentDiagnosticLogEntryExport: Codable, Sendable {
    let kind: String
    let at: Date
    let recordID: String?
    let campaignID: String?
    let event: PersistentDiagnosticEventExport?
    let record: PersistentDiagnosticRunRecordExport?
}

nonisolated struct PersistentMetricKitExport: Codable, Sendable {
    let kind: String
    let fileToken: String
    let payloadBytes: Int
    let summaryToken: String?
}

private struct PersistentRuntimeDiagnosticsPrivacyProjector {
    private enum MetadataValuePolicy {
        case integer(ClosedRange<Int>)
        case boolean
        case category(Set<String>)
    }

    private static let forbiddenIdentityMetadataKeys: Set<String> = [
        "agentrunid", "conversationid", "correlationtoken", "currentlaunchuuid",
        "e2erunid", "previouslaunchuuid", "requestid", "runid", "turnid"
    ]
    private static let knownEventCodes: Set<String> = [
        "agent_cancel_clean", "agent_cancel_stream_completed", "agent_fast_path_bounded",
        "agent_fast_path_unbounded", "agent_run_cancelled", "agent_run_failed",
        "agent_run_passed", "agent_tool_dry_run_bounded", "agent_tool_dry_run_unbounded",
        "buffered_during_generation", "crash_resume", "developer_trace_bypass_expected",
        "developer_trace_bypass_missing", "diagnostic_remediation_proposal",
        "fast_latency_missing", "fast_prompt_too_large", "fast_tokens_too_large",
        "kernel_migration_probe_debug_only", "lifecycle_probe_armed",
        "lifecycle_probe_passed", "lifecycle_probe_skipped", "lifecycle_transition",
        "live_agent_stream_passed", "manual_live_agent_stream_required",
        "manual_probe_required", "manual_scenario_requires_explicit_request",
        "not_manual_scenario", "pass", "resource_gate_matrix", "resource_gate_paused",
        "resource_gate_policy_failed", "resource_gate_policy_passed", "sandboxed_tool_plan",
        "sandboxed_tool_plan_bounded", "sandboxed_tool_plan_unbounded", "skipped_no_model",
        "tester_action_required", "tool_prompt_used_fast_path",
        "llamapromptbudget", "llamafirsttoken", "llamaemptyoutput", "llamacomplete",
        "llamacancel", "llamafailure", "slotagentstart", "slotagentpath",
        "slotagentfallback", "slotagentgroundingcomplete", "slotagenteffectiverequestbuilt",
        "slotagentdeterministicanswerbuilt", "slotagentdoneyielded", "slotagentend",
        "slotagentendemitted", "slotagentcontinuationfinished", "slotagentcancel",
        "chatruntimetrace", "groundingcost", "uiupdate", "scenetransition",
        "metrickitpersistfailure", "finalintentcandidatereplaced", "fallbackused",
        "voicestartupfailure", "voiceaudiosessionevent"
    ]
    private static let sceneValues: Set<String> = ["active", "inactive", "background", "unknown"]
    private static let thermalValues: Set<String> = ["nominal", "fair", "serious", "critical", "unknown"]
    private static let latencyValues: Set<String> = [
        PromptLatencyClass.fastInteractive.rawValue,
        PromptLatencyClass.normalInteractive.rawValue,
        PromptLatencyClass.documentGrounded.rawValue,
        PromptLatencyClass.developerTrace.rawValue
    ]
    private static let structuralMetadataPolicies: [String: [String: MetadataValuePolicy]] = [
        "groundingcost": [
            "elapsedms": .integer(0...86_400_000),
            "sectioncount": .integer(0...10_000),
            "promptchars": .integer(0...10_000_000),
            "toolcount": .integer(0...10_000),
            "memorycount": .integer(0...10_000),
            "source": .category(["cache", "bridge", "degraded", "coordinator"])
        ],
        "lifecycle_transition": [
            "phase": .category(sceneValues)
        ],
        "llamacomplete": [
            "elapsedms": .integer(0...86_400_000),
            "firsttokenlatencyms": .integer(-1...86_400_000),
            "estimatedprompttokens": .integer(0...10_000_000),
            "finalpromptchars": .integer(0...10_000_000)
        ],
        "llamafirsttoken": [
            "latencyms": .integer(0...86_400_000)
        ],
        "llamapromptbudget": [
            "latencyclass": .category(latencyValues),
            "initialchars": .integer(0...10_000_000),
            "finalchars": .integer(0...10_000_000),
            "estimatedtokens": .integer(0...10_000_000)
        ],
        "resource_gate_matrix": [
            "seriousdenied": .boolean,
            "criticaldenied": .boolean,
            "backgrounddenied": .boolean,
            "lowpowerallowed": .boolean,
            "realexpectedallowed": .boolean,
            "realdenied": .boolean
        ],
        "sandboxed_tool_plan": [
            "toolcount": .integer(0...256),
            "maxsteps": .integer(0...64)
        ],
        "slotagentstart": [
            "promptchars": .integer(0...10_000_000),
            "toolcount": .integer(0...10_000),
            "memorycount": .integer(0...10_000)
        ],
        "uiupdate": [
            "surface": .category(["chat", "voice"]),
            "targethz": .integer(0...240)
        ]
    ]

    private let key = SymmetricKey(size: .bits256)

    var exportScope: String {
        opaqueToken(prefix: "export_v1", domain: "export", value: "scope")
    }

    func project(
        exportedAt: Date,
        appVersion: String,
        sourceCommit: String?,
        deviceModel: String,
        systemName: String,
        systemVersion: String,
        campaign: PersistentDiagnosticCampaign?,
        state: PersistentDiagnosticState?,
        logEntries: [PersistentDiagnosticLogEntry],
        metricKitPayloads: [PersistentMetricKitSourcePayload]
    ) throws -> PersistentRuntimeDiagnosticsExportPayload {
        let projectedLogEntries = logEntries.map(projectLogEntry)
        return PersistentRuntimeDiagnosticsExportPayload(
            schemaVersion: PersistentRuntimeDiagnosticsExporter.privacySchemaVersion,
            exportScope: exportScope,
            exportedAt: exportedAt,
            privacy: "Contains only per-export opaque identifiers, bounded structural categories, counts, booleans, and timing. Raw diagnostic text, correlation identifiers, arbitrary metadata keys, MetricKit payloads, and stable file identities are omitted.",
            app: PersistentRuntimeDiagnosticsAppExport(
                appVersion: boundedSystemCategory(appVersion),
                sourceRevisionToken: sourceCommit.map {
                    opaqueToken(prefix: "diagnostic_source_v1", domain: "sourceRevision", value: $0)
                },
                deviceModel: boundedSystemCategory(deviceModel),
                systemName: boundedSystemCategory(systemName),
                systemVersion: boundedSystemCategory(systemVersion)
            ),
            campaign: campaign.map(projectCampaign),
            state: state.map(projectState),
            ndjson: try encodeNDJSON(projectedLogEntries),
            metricKitPayloads: metricKitPayloads.map(projectMetricKitPayload)
        )
    }

    private func projectCampaign(_ campaign: PersistentDiagnosticCampaign) -> PersistentDiagnosticCampaignExport {
        PersistentDiagnosticCampaignExport(
            id: opaqueID(campaign.id, domain: "campaign"),
            createdAt: campaign.createdAt,
            updatedAt: campaign.updatedAt,
            enabled: campaign.enabled,
            runContinuously: campaign.runContinuously,
            maxRunsPerScenario: campaign.maxRunsPerScenario,
            delayBetweenRunsSeconds: campaign.delayBetweenRunsSeconds,
            scenarios: campaign.scenarios.map(\.rawValue)
        )
    }

    private func projectState(_ state: PersistentDiagnosticState) -> PersistentDiagnosticStateExport {
        PersistentDiagnosticStateExport(
            activeRunID: state.activeRunID.map { opaqueID($0, domain: "run") },
            activeCampaignID: state.activeCampaignID.map { opaqueID($0, domain: "campaign") },
            activeScenario: state.activeScenario?.rawValue,
            activeStartedAt: state.activeStartedAt,
            activeLaunchID: state.activeLaunchUUID.map { opaqueID($0, domain: "launch") },
            cleanCancellationBeforeTermination: state.cleanCancellationBeforeTermination,
            completedRunIDs: state.completedRunIDs.map { opaqueID($0, domain: "run") },
            records: state.records.map(projectRecord),
            status: projectStatus(state.status)
        )
    }

    private func projectRecord(_ record: PersistentDiagnosticRunRecord) -> PersistentDiagnosticRunRecordExport {
        PersistentDiagnosticRunRecordExport(
            id: opaqueID(record.id, domain: "run"),
            campaignID: opaqueID(record.campaignID, domain: "campaign"),
            scenario: record.scenario.rawValue,
            startedAt: record.startedAt,
            finishedAt: record.finishedAt,
            status: record.status.rawValue,
            metrics: projectMetrics(record.metrics),
            events: record.events.map(projectEvent),
            failureSummaryToken: record.failureSummary.map {
                opaqueToken(prefix: "diagnostic_value_v1", domain: "failureSummary", value: $0)
            },
            failureSummaryCharacters: record.failureSummary?.count,
            remediationProposals: (record.remediationProposals ?? []).map(projectRemediation)
        )
    }

    private func projectMetrics(_ metrics: PersistentDiagnosticMetrics) -> PersistentDiagnosticMetricsExport {
        PersistentDiagnosticMetricsExport(
            scenePhase: allowlistedCategory(metrics.scenePhase, allowed: Self.sceneValues, domain: "scenePhase"),
            thermalState: allowlistedCategory(metrics.thermalState, allowed: Self.thermalValues, domain: "thermalState"),
            lowPowerMode: metrics.lowPowerMode,
            memoryWarningCount: metrics.memoryWarningCount,
            realScenePhase: allowlistedCategory(metrics.realScenePhase, allowed: Self.sceneValues, domain: "realScenePhase"),
            realThermalState: allowlistedCategory(metrics.realThermalState, allowed: Self.thermalValues, domain: "realThermalState"),
            realDenied: metrics.realDenied,
            simulatedScenePhase: allowlistedCategory(metrics.simulatedScenePhase, allowed: Self.sceneValues, domain: "simulatedScenePhase"),
            simulatedThermalState: allowlistedCategory(metrics.simulatedThermalState, allowed: Self.thermalValues, domain: "simulatedThermalState"),
            simulatedDenied: metrics.simulatedDenied,
            cpuWatchdog: metrics.cpuWatchdog.map(projectCPUWatchdog),
            diskWrite: metrics.diskWrite.map(projectDiskWrite),
            generationActive: metrics.generationActive,
            promptLatencyClass: allowlistedCategory(metrics.promptLatencyClass, allowed: Self.latencyValues, domain: "promptLatencyClass"),
            promptInitialChars: metrics.promptInitialChars,
            promptFinalChars: metrics.promptFinalChars,
            estimatedPromptTokens: metrics.estimatedPromptTokens,
            promptToken: metrics.promptSHA256.map {
                opaqueToken(prefix: "diagnostic_value_v1", domain: "promptDigest", value: $0)
            },
            promptBodyBytes: metrics.promptBodyBytes,
            promptRedactionModeToken: metrics.promptRedactionMode.map {
                opaqueToken(prefix: "diagnostic_value_v1", domain: "promptRedactionMode", value: $0)
            },
            firstTokenLatencyMs: metrics.firstTokenLatencyMs,
            generationElapsedMs: metrics.generationElapsedMs,
            agentGroundingElapsedMs: metrics.agentGroundingElapsedMs,
            groundingSectionCount: metrics.groundingSectionCount,
            groundingChars: metrics.groundingChars,
            toolCount: metrics.toolCount,
            inputToolCount: metrics.inputToolCount,
            bridgedToolCount: metrics.bridgedToolCount,
            memoryCount: metrics.memoryCount,
            didUseFastPath: metrics.didUseFastPath,
            didCancel: metrics.didCancel,
            cancellationReasonToken: metrics.cancellationReason.map {
                opaqueToken(prefix: "diagnostic_value_v1", domain: "cancellationReason", value: $0)
            },
            didFallback: metrics.didFallback,
            fallbackReasonToken: metrics.fallbackReason.map {
                opaqueToken(prefix: "diagnostic_value_v1", domain: "fallbackReason", value: $0)
            },
            uiUpdateCount: metrics.uiUpdateCount,
            streamingUpdateCount: metrics.streamingUpdateCount,
            diskBytesBefore: metrics.diskBytesBefore,
            diskBytesAfter: metrics.diskBytesAfter,
            appBecameInactiveOrBackgroundDuringRun: metrics.appBecameInactiveOrBackgroundDuringRun,
            errorCodeTokens: metrics.errorCodes.map {
                opaqueToken(prefix: "diagnostic_value_v1", domain: "errorCode", value: $0)
            }
        )
    }

    private func projectCPUWatchdog(
        _ snapshot: PersistentDiagnosticCPUWatchdogSnapshot
    ) -> PersistentDiagnosticCPUWatchdogExport {
        let allowed = Set(CPUWatchdogCategory.allCases.map(\.rawValue))
        return PersistentDiagnosticCPUWatchdogExport(
            degradedCategories: snapshot.degradedCategories.compactMap { allowed.contains($0) ? $0 : nil },
            totalsByCategory: snapshot.totalsByCategory.filter { allowed.contains($0.key) },
            activeCountsByCategory: snapshot.activeCountsByCategory.filter { allowed.contains($0.key) }
        )
    }

    private func projectDiskWrite(
        _ snapshot: PersistentDiagnosticDiskWriteSnapshot
    ) -> PersistentDiagnosticDiskWriteExport {
        let allowed = Set(DiskWriteCategory.allCases.map(\.rawValue))
        return PersistentDiagnosticDiskWriteExport(
            bytes1Minute: snapshot.bytes1Minute,
            bytes15Minutes: snapshot.bytes15Minutes,
            bytes24Hours: snapshot.bytes24Hours,
            bytesByCategory24Hours: snapshot.bytesByCategory24Hours.filter { allowed.contains($0.key) },
            generationActive: snapshot.generationActive
        )
    }

    private func projectEvent(_ event: PersistentDiagnosticEvent) -> PersistentDiagnosticEventExport {
        let trimmedMessage = event.message.trimmingCharacters(in: .whitespacesAndNewlines)
        let eventCode = projectEventCode(event.code)
        return PersistentDiagnosticEventExport(
            id: opaqueID(event.id, domain: "event"),
            at: event.at,
            code: eventCode,
            messageToken: trimmedMessage.isEmpty ? nil : opaqueToken(
                prefix: "diagnostic_value_v1",
                domain: "eventMessage",
                value: trimmedMessage
            ),
            messageCharacters: trimmedMessage.count,
            values: projectMetadata(event.values, eventCode: eventCode)
        )
    }

    private func projectRemediation(
        _ proposal: PersistentDiagnosticRemediationProposal
    ) -> PersistentDiagnosticRemediationExport {
        PersistentDiagnosticRemediationExport(
            id: opaqueToken(prefix: "diagnostic_remediation_v1", domain: "remediation", value: proposal.id),
            severity: proposal.severity.rawValue
        )
    }

    private func projectStatus(_ status: PersistentDiagnosticRunnerStatus) -> PersistentDiagnosticRunnerStatusExport {
        PersistentDiagnosticRunnerStatusExport(
            isRunning: status.isRunning,
            isPaused: status.isPaused,
            latestScenario: status.latestScenario?.rawValue,
            passedCount: status.passedCount,
            failedCount: status.failedCount,
            skippedCount: status.skippedCount,
            lastFirstTokenLatencyMs: status.lastFirstTokenLatencyMs,
            lastPromptFinalChars: status.lastPromptFinalChars,
            lastCancellationReasonToken: status.lastCancellationReason.map {
                opaqueToken(prefix: "diagnostic_value_v1", domain: "statusCancellationReason", value: $0)
            },
            lastCrashResumeStatusToken: status.lastCrashResumeStatus.map {
                opaqueToken(prefix: "diagnostic_value_v1", domain: "statusCrashResume", value: $0)
            },
            lastRemediationSummaryToken: status.lastRemediationSummary.map {
                opaqueToken(prefix: "diagnostic_value_v1", domain: "statusRemediation", value: $0)
            },
            lastUpdatedAt: status.lastUpdatedAt
        )
    }

    private func projectLogEntry(_ entry: PersistentDiagnosticLogEntry) -> PersistentDiagnosticLogEntryExport {
        PersistentDiagnosticLogEntryExport(
            kind: ["event", "run"].contains(entry.kind) ? entry.kind : "unknown",
            at: entry.at,
            recordID: entry.recordID.map { opaqueID($0, domain: "run") },
            campaignID: entry.campaignID.map { opaqueID($0, domain: "campaign") },
            event: entry.event.map(projectEvent),
            record: entry.record.map(projectRecord)
        )
    }

    private func projectMetricKitPayload(_ payload: PersistentMetricKitSourcePayload) -> PersistentMetricKitExport {
        let loweredName = payload.fileName.lowercased()
        let kind: String
        if loweredName.hasPrefix("mxmetric-") {
            kind = "metric"
        } else if loweredName.hasPrefix("mxdiagnostic-") {
            kind = "diagnostic"
        } else {
            kind = "unknown"
        }
        let trimmedJSON = payload.json.trimmingCharacters(in: .whitespacesAndNewlines)
        return PersistentMetricKitExport(
            kind: kind,
            fileToken: opaqueToken(prefix: "diagnostic_metric_v1", domain: "metricFile", value: payload.fileName),
            payloadBytes: payload.json.lengthOfBytes(using: .utf8),
            summaryToken: trimmedJSON.isEmpty ? nil : opaqueToken(
                prefix: "diagnostic_value_v1",
                domain: "metricSummary",
                value: trimmedJSON
            )
        )
    }

    private func projectMetadata(
        _ values: [String: String],
        eventCode: String
    ) -> [String: String] {
        values.keys.sorted().reduce(into: [:]) { result, rawKey in
            let normalizedKey = PersistentRuntimeDiagnosticsRedactor.safeCode(rawKey)
            guard !isIdentityMetadataKey(normalizedKey) else { return }
            let rawValue = values[rawKey] ?? ""
            if let policy = Self.structuralMetadataPolicies[eventCode]?[normalizedKey] {
                result[normalizedKey] = projectStructuralMetadataValue(
                    rawValue,
                    key: normalizedKey,
                    policy: policy
                )
            } else {
                result[opaqueMetadataKey(normalizedKey)] = opaqueToken(
                    prefix: "diagnostic_value_v1",
                    domain: "metadata.\(normalizedKey)",
                    value: rawValue
                )
            }
        }
    }

    private func projectStructuralMetadataValue(
        _ value: String,
        key: String,
        policy: MetadataValuePolicy
    ) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        switch policy {
        case .integer(let allowed):
            if let number = Int(trimmed), allowed.contains(number) {
                return String(number)
            }
        case .boolean:
            let lowered = trimmed.lowercased()
            if lowered == "true" || lowered == "false" {
                return lowered
            }
        case .category(let allowed):
            let lowered = trimmed.lowercased()
            if allowed.contains(lowered) {
                return lowered
            }
        }
        return opaqueToken(prefix: "diagnostic_value_v1", domain: "metadata.\(key)", value: trimmed)
    }

    private func encodeNDJSON(_ entries: [PersistentDiagnosticLogEntryExport]) throws -> String {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        return try entries.map { entry in
            String(decoding: try encoder.encode(entry), as: UTF8.self)
        }.joined(separator: "\n")
    }

    private func allowlistedCategory(
        _ value: String?,
        allowed: Set<String>,
        domain: String
    ) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if allowed.contains(trimmed) {
            return trimmed
        }
        return opaqueToken(prefix: "diagnostic_value_v1", domain: domain, value: trimmed)
    }

    private func boundedSystemCategory(_ value: String) -> String {
        let normalized = PersistentRuntimeDiagnosticsRedactor.safeCode(value)
        return normalized.isEmpty ? "unknown" : String(normalized.prefix(64))
    }

    private func projectEventCode(_ value: String) -> String {
        let normalized = PersistentRuntimeDiagnosticsRedactor.safeCode(value)
        return Self.knownEventCodes.contains(normalized) ? normalized : "other"
    }

    private func isIdentityMetadataKey(_ key: String) -> Bool {
        Self.forbiddenIdentityMetadataKeys.contains(key)
            || key.contains("correlation")
            || key.contains("uuid")
            || key.hasSuffix("id")
            || key.hasSuffix("token")
    }

    private func opaqueMetadataKey(_ value: String) -> String {
        let digest = hmac(domain: "metadataKey", value: value)
        return "metadata_" + digest.prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    private func opaqueID(_ value: UUID, domain: String) -> String {
        opaqueToken(prefix: "diagnostic_\(domain)_v1", domain: domain, value: value.uuidString)
    }

    private func opaqueToken(prefix: String, domain: String, value: String) -> String {
        let digest = hmac(domain: domain, value: value)
        return prefix + "_" + digest.prefix(16).map { String(format: "%02x", $0) }.joined()
    }

    private func hmac(domain: String, value: String) -> HMAC<SHA256>.MAC {
        HMAC<SHA256>.authenticationCode(
            for: Data("\(domain)|\(value)".utf8),
            using: key
        )
    }
}
