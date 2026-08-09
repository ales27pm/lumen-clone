import CryptoKit
import Foundation

nonisolated struct LumenInAppDatasetPackage: Codable, Sendable {
    let schemaVersion: String
    let generatedAt: Date
    let exportKind: String
    let app: InAppDatasetAppInfo
    let testFlight: TestFlightAgentGroundingExportInfo
    let manifestSource: String
    let usedRuntimeFallback: Bool
    let runtimeManifestAudit: RuntimeAgentManifestAuditReport?
    let behaviorAudit: AgentBehaviorAuditReport?
    let scenarioResults: [RuntimeScenarioResult]
    let recentTraces: [InAppDatasetTraceExport]
    let liveE2EReport: InAppDatasetLiveE2EReportExport?
    let traceSelectedToolAllowedCount: Int
    let traceParseErrorCount: Int
    let exportQualityFailures: [InAppDatasetExportQualityFailure]?
    let improveLoop: ImproveLoopDataset
    let exportPolicy: InAppDatasetExportPolicy
}

nonisolated struct InAppDatasetTraceExport: Codable, Sendable, Hashable {
    let id: UUID
    let createdAt: Date
    let event: AgentBehaviorTrace.Event
    let slot: String
    let stage: String
    let scenarioID: String?
    let correlationToken: String?
    let intent: String?
    let promptPrefix: String
    let rawOutputPrefix: String
    let selectedToolID: String?
    let toolArguments: [String: String]
    let allowedToolIDs: [String]
    let requiresApproval: Bool?
    let approvalMode: String?
    let parseError: String?
    let emittedFinalInActionTurn: Bool
    let modelFamily: String?
    let baseModelPath: String?
    let adapterID: String?
    let adapterSlot: String?
    let adapterPath: String?
    let adapterApplied: Bool?
    let adapterScale: Float?
    let adapterFailureReason: String?
    let generationElapsedMs: Int?
    let firstTokenLatencyMs: Int?
    let outputTokenCount: Int?
    let estimatedPromptTokenCount: Int?
    let preFirstTokenMs: Int?
    let messageBuildMs: Int?
    let decodeMs: Int?
    let tokensPerSecond: Double?
    let ensureReadyMs: Int?
    let adapterActivationMs: Int?
    let runtimePath: String?
    let activeAdapterSlot: String?
    let maxTokensRequested: Int?
    let maxTokensEffective: Int?
    let promptCharCount: Int?
    let accelerationDiagnostic: String?
    let accelerationDiagnostics: RuntimeAccelerationDiagnostics?
    let emptyOutputReason: String?
    let streamStarted: Bool?
    let selectedRuntime: String?
    let selectedAdapter: String?
    let modelIdentifier: String?
    let modelLoaded: Bool?
    let stopSequences: [String]
    let temperature: Double?
    let topP: Double?
    let cancellationStateBeforeStream: String?
    let firstChunkReceived: Bool?
    let textChunkCount: Int?
    let finalChunkReceived: Bool?
    let streamTerminationReason: String?
    let successfulObservationCount: Int?
    let finalizerAccepted: Bool?
    let finalizerRejectionReason: String?
    let finalValidatorAcceptedCandidate: Bool?
    let finalValidatorReplacementSource: String?
    let finalValidatorRejectionReason: String?
    let selfModel: AgentBehaviorTrace.SelfModelDecisionSummary?
}

nonisolated struct InAppDatasetLiveE2EReportExport: Codable, Sendable {
    let schemaVersion: String
    let generatedAt: Date
    let app: InAppDatasetAppInfo
    let exportPolicy: EvidenceLayerExportPolicy
    let payload: E2ETestReport
    let correlatedTraceCount: Int
    let modelBackedCorrelatedTraceCount: Int
    let modelBackedCorrelatedScenarioCount: Int
    let deterministicCompatibilityTraceCount: Int
    let traceSidecarField: String
}

nonisolated struct InAppDatasetExportQualityFailure: Codable, Sendable, Hashable, Identifiable {
    var id: String { [type, agent ?? "", actual ?? "", problem].joined(separator: "|") }
    let type: String
    let agent: String?
    let expected: [String]
    let actual: String?
    let scenario: String?
    let problem: String
    let sourceLayer: String
}

nonisolated struct InAppDatasetAppInfo: Codable, Sendable, Hashable {
    let name: String
    let bundleIdentifier: String?
    let shortVersion: String?
    let buildNumber: String?
    let sourceRevision: String?

    static func current(
        infoDictionary: [String: Any] = Bundle.main.infoDictionary ?? [:],
        bundleIdentifier: String? = Bundle.main.bundleIdentifier
    ) -> InAppDatasetAppInfo {
        InAppDatasetAppInfo(
            name: normalizedString(infoDictionary["CFBundleName"]) ?? "Lumen",
            bundleIdentifier: bundleIdentifier,
            shortVersion: normalizedString(infoDictionary["CFBundleShortVersionString"]),
            buildNumber: normalizedString(infoDictionary["CFBundleVersion"]),
            sourceRevision: sourceRevision(infoDictionary["LumenGitSHA"])
        )
    }

    private static func normalizedString(_ value: Any?) -> String? {
        guard let value = value as? String else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func sourceRevision(_ value: Any?) -> String? {
        guard let revision = normalizedString(value),
              revision.lowercased() != "unknown",
              !revision.hasPrefix("$(") else {
            return nil
        }
        return revision
    }
}

nonisolated struct TestFlightAgentGroundingExportInfo: Codable, Sendable, Hashable {
    let sourceAction: String
    let filePrefix: String
    let distributionChannel: String
    let sandboxReceipt: Bool
    let appShortVersion: String?
    let appBuildNumber: String?
    let liveE2EReportIncluded: Bool
    let expectedIngestArgument: String
}

nonisolated struct InAppDatasetExportPolicy: Codable, Sendable, Hashable {
    let format: String
    let privacy: String
    let promptPolicy: String
    let traceLimit: Int
    let source: String
    let sourceLayer: String
    let ownsLiveE2EScenarios: Bool
    let includesDeterministicStaticScenarios: Bool
    let deterministicScenarioPolicy: String
}

nonisolated struct InAppDatasetPackageExportResult: Sendable {
    let url: URL
    let package: LumenInAppDatasetPackage
}

nonisolated enum InAppDatasetPackageExporter {
    private struct ExportCorrelationContext {
        let key: SymmetricKey
        let resultTokens: [UUID: String]
        let traceTokens: [UUID: String]
        let traceResultIDs: [UUID: UUID]
    }

    static let schemaVersion = "2.0.0"
    static let exportKind = "testflight-agent-grounding-runtime-export"
    static let sourceAction = "Agent Grounding > Export TestFlight + Agent Grounding Package"
    static let filePrefix = "lumen-testflight-agent-grounding-redacted-v1"
    static let defaultIncludesScenarioResults = false
    static let slowModelTurnThresholdMs = 30_000
    static let severeModelTurnThresholdMs = 120_000
    private static let directoryName = "LumenDatasetExports"

    private static func exportCorrelationContext(
        report: E2ETestReport?,
        traces: [AgentBehaviorTrace]
    ) -> ExportCorrelationContext {
        let key = SymmetricKey(size: .bits256)
        guard let report else {
            return ExportCorrelationContext(key: key, resultTokens: [:], traceTokens: [:], traceResultIDs: [:])
        }

        var traceResultIDs: [UUID: UUID] = [:]
        for trace in traces {
            let matchingResults = report.results.filter { traceMatches(result: $0, trace: trace) }
            // A subset of identifiers can match multiple results; ambiguous traces prove neither run.
            guard matchingResults.count == 1, let result = matchingResults.first else { continue }
            traceResultIDs[trace.id] = result.id
        }

        var resultTokens: [UUID: String] = [:]
        var traceTokens: [UUID: String] = [:]
        for result in report.results {
            let matchingTraceIDs = traceResultIDs.compactMap { traceID, resultID in
                resultID == result.id ? traceID : nil
            }
            guard !matchingTraceIDs.isEmpty else { continue }
            let token = opaqueCorrelationToken(for: result, key: key)
            resultTokens[result.id] = token
            for traceID in matchingTraceIDs {
                traceTokens[traceID] = token
            }
        }
        return ExportCorrelationContext(
            key: key,
            resultTokens: resultTokens,
            traceTokens: traceTokens,
            traceResultIDs: traceResultIDs
        )
    }

    private static func opaqueCorrelationToken(for result: E2ETestResult, key: SymmetricKey) -> String {
        let seed = [
            result.scenarioID,
            result.e2eRunID?.uuidString ?? "",
            result.agentRunID?.uuidString ?? "",
            result.conversationID?.uuidString ?? "",
            result.turnID?.uuidString ?? ""
        ].joined(separator: "|")
        let digest = HMAC<SHA256>.authenticationCode(for: Data(seed.utf8), using: key)
        return "corr_v1_" + digest.prefix(16).map { String(format: "%02x", $0) }.joined()
    }

    private static func redactedLiveE2EReport(
        _ report: E2ETestReport,
        correlationContext: ExportCorrelationContext
    ) -> E2ETestReport {
        let privacySafeReport = EvidenceLayerExporter.privacySafeE2EReportForExport(report)
        return E2ETestReport(
            id: redactedUUID(report.id, key: correlationContext.key, domain: "report"),
            startedAt: report.startedAt,
            finishedAt: report.finishedAt,
            passed: report.passed,
            failed: report.failed,
            results: zip(report.results, privacySafeReport.results).map { result, privacySafeResult in
                return E2ETestResult(
                    id: redactedUUID(result.id, key: correlationContext.key, domain: "result"),
                    scenarioID: privacySafeResult.scenarioID,
                    kind: privacySafeResult.kind,
                    title: privacySafeResult.title,
                    prompt: privacySafeResult.prompt,
                    expectedIntent: privacySafeResult.expectedIntent,
                    actualIntent: privacySafeResult.actualIntent,
                    e2eRunID: nil,
                    agentRunID: nil,
                    conversationID: nil,
                    turnID: nil,
                    correlationToken: correlationContext.resultTokens[result.id],
                    requiresAgentRun: privacySafeResult.requiresAgentRun,
                    evidenceMode: privacySafeResult.evidenceMode,
                    passed: privacySafeResult.passed,
                    failures: privacySafeResult.failures,
                    finalText: privacySafeResult.finalText,
                    missingHints: privacySafeResult.missingHints,
                    rewriteAttempted: privacySafeResult.rewriteAttempted,
                    rewriteSuccess: privacySafeResult.rewriteSuccess,
                    events: zip(result.events, privacySafeResult.events).map { event, privacySafeEvent in
                        E2ETestEvent(
                            id: redactedUUID(event.id, key: correlationContext.key, domain: "event"),
                            createdAt: privacySafeEvent.createdAt,
                            scenarioID: privacySafeEvent.scenarioID,
                            phase: privacySafeEvent.phase,
                            message: privacySafeEvent.message
                        )
                    },
                    startedAt: privacySafeResult.startedAt,
                    finishedAt: privacySafeResult.finishedAt,
                    rawFinalPrefix: privacySafeResult.rawFinalPrefix,
                    sanitizedFinalPrefix: privacySafeResult.sanitizedFinalPrefix,
                    rawFinalHadUnsafeLeakage: privacySafeResult.rawFinalHadUnsafeLeakage,
                    sanitizedFinalRemovedArtifacts: privacySafeResult.sanitizedFinalRemovedArtifacts,
                    outputHygieneFailures: privacySafeResult.outputHygieneFailures,
                    performanceMatrix: privacySafeResult.performanceMatrix,
                    metadata: privacySafeResult.metadata
                )
            }
        )
    }

    private static func redactedUUID(_ value: UUID, key: SymmetricKey, domain: String) -> UUID {
        let digest = HMAC<SHA256>.authenticationCode(
            for: Data("\(domain)|\(value.uuidString)".utf8),
            using: key
        )
        var bytes = Array(digest.prefix(16))
        bytes[6] = (bytes[6] & 0x0f) | 0x40
        bytes[8] = (bytes[8] & 0x3f) | 0x80
        return UUID(uuid: (
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15]
        ))
    }

    /// Assembles a complete TestFlight + Agent Grounding package incorporating audit reports and recent traces.
    /// - Parameters:
    ///   - manifestSource: Identifier for the manifest audit source.
    ///   - includeScenarioResults: If `true`, includes scenario results in the package; otherwise omits them.
    /// - Returns: A dataset package containing app metadata, audits, trace statistics, scenario results (if included), and export policy.
    static func makePackage(
        manifestSource: String,
        usedRuntimeFallback: Bool,
        runtimeManifestAudit: RuntimeAgentManifestAuditReport?,
        behaviorAudit: AgentBehaviorAuditReport?,
        scenarioResults: [RuntimeScenarioResult],
        liveE2EReport: E2ETestReport? = nil,
        traceLimit: Int = 200,
        includeScenarioResults: Bool = defaultIncludesScenarioResults
    ) -> LumenInAppDatasetPackage {
        let traces = AgentBehaviorTraceRecorder.recent(limit: traceLimit)
        return makePackage(
            manifestSource: manifestSource,
            usedRuntimeFallback: usedRuntimeFallback,
            runtimeManifestAudit: runtimeManifestAudit,
            behaviorAudit: behaviorAudit,
            scenarioResults: scenarioResults,
            liveE2EReport: liveE2EReport,
            traceLimit: traceLimit,
            includeScenarioResults: includeScenarioResults,
            traces: traces
        )
    }

    private static func makePackage(
        manifestSource: String,
        usedRuntimeFallback: Bool,
        runtimeManifestAudit: RuntimeAgentManifestAuditReport?,
        behaviorAudit: AgentBehaviorAuditReport?,
        scenarioResults: [RuntimeScenarioResult],
        liveE2EReport: E2ETestReport?,
        traceLimit: Int,
        includeScenarioResults: Bool,
        traces: [AgentBehaviorTrace]
    ) -> LumenInAppDatasetPackage {
        // Correlate while the current in-memory trace still carries ephemeral
        // run identifiers. The persisted/shareable trace copy below drops them.
        let correlationContext = exportCorrelationContext(report: liveE2EReport, traces: traces)
        let privacySafeTraces = traces.map { $0.redactedForPersistentDiagnostics() }
        let mergedBehaviorAudit = mergedBehaviorAuditWithRuntimeTraceViolations(behaviorAudit, traces: privacySafeTraces)
        let exportedBehaviorAudit = redactedBehaviorAudit(
            mergedBehaviorAudit,
            identityKey: correlationContext.key
        )
        let exportedRuntimeManifestAudit = redactedRuntimeManifestAudit(runtimeManifestAudit)
        let exportedScenarioResults = includeScenarioResults
            ? scenarioResults.map(redactedScenarioResult)
            : []
        let exportedTraces = zip(traces, privacySafeTraces).map { sourceTrace, privacySafeTrace in
            exportTrace(
                privacySafeTrace,
                id: redactedUUID(sourceTrace.id, key: correlationContext.key, domain: "trace"),
                correlationToken: correlationContext.traceTokens[sourceTrace.id]
            )
        }
        let app = appInfo()
        let liveReportExport = liveE2EReport.map { report in
            liveE2EReportExport(
                from: report,
                generatedAt: Date(),
                traces: traces,
                correlationContext: correlationContext
            )
        }
        let qualityFailures = exportQualityFailures(
            from: exportedTraces,
            liveE2EReport: liveReportExport,
            rawLiveE2EReport: liveE2EReport,
            rawTraces: traces,
            correlationContext: correlationContext
        )
        // A shareable diagnostics package must never double as a training-data
        // transport. Even privacy hashes are not useful supervised examples, so
        // keep the stable dataset shape while exporting no sample records.
        let improveLoop = emptyImproveLoopDataset()
        return LumenInAppDatasetPackage(
            schemaVersion: schemaVersion,
            generatedAt: Date(),
            exportKind: exportKind,
            app: app,
            testFlight: testFlightExportInfo(app: app, liveE2EReportIncluded: liveReportExport != nil),
            manifestSource: manifestSource,
            usedRuntimeFallback: usedRuntimeFallback,
            runtimeManifestAudit: exportedRuntimeManifestAudit,
            behaviorAudit: exportedBehaviorAudit,
            scenarioResults: exportedScenarioResults,
            recentTraces: exportedTraces,
            liveE2EReport: liveReportExport,
            traceSelectedToolAllowedCount: privacySafeTraces.reduce(into: 0) { count, trace in
                guard let selectedToolID = trace.selectedToolID else { return }
                let selected = ToolRouteGuard.canonicalToolID(selectedToolID)
                let allowed = Set(trace.allowedToolIDs.map(ToolRouteGuard.canonicalToolID))
                if allowed.contains(selected) {
                    count += 1
                }
            },
            traceParseErrorCount: privacySafeTraces.reduce(into: 0) { count, trace in
                if traceHasActionParseError(trace) {
                    count += 1
                }
            },
            exportQualityFailures: qualityFailures,
            improveLoop: improveLoop,
            exportPolicy: InAppDatasetExportPolicy(
                format: "testflight-agent-grounding-runtime-json-package",
                privacy: "Contains metrics, safe categories, counts, one-way hash summaries, opaque per-export correlation tokens, and no improve-loop sample records. Raw prompts, model outputs, audit prose, scenario text, conversations, contacts, calendar bodies, files, photos, trace identifiers, correlation UUIDs, local paths, and tool payload bodies are omitted.",
                promptPolicy: "All prompt, output, audit, repair-sample, and scenario free-form fields are replaced by one-way character-count and SHA-256 summaries; improve-loop arrays are intentionally empty.",
                traceLimit: traceLimit,
                source: "TestFlight app runtime + RuntimeManifestAuditor + AgentModelBehaviorAuditor + AgentBehaviorTraceRecorder",
                sourceLayer: "agentGroundingRuntimeAudit",
                ownsLiveE2EScenarios: false,
                includesDeterministicStaticScenarios: includeScenarioResults,
                deterministicScenarioPolicy: includeScenarioResults
                    ? "Static manifest scenario checks were explicitly included; they are not proof of live model execution and must not be treated as E2E model runs. If liveE2EReport is present, its embedded e2eTestReport envelope is the only layer that owns live scenario pass/fail."
                    : "Static manifest scenario checks are displayed in-app only and omitted from the dataset export; E2ETestRunner owns live model scenario results through the embedded liveE2EReport envelope when available."
            )
        )
    }

    static func makePackageForTests(
        liveE2EReport: E2ETestReport,
        traces: [AgentBehaviorTrace]
    ) -> LumenInAppDatasetPackage {
        makePackage(
            manifestSource: "test-manifest",
            usedRuntimeFallback: false,
            runtimeManifestAudit: nil,
            behaviorAudit: nil,
            scenarioResults: [],
            liveE2EReport: liveE2EReport,
            traceLimit: traces.count,
            includeScenarioResults: false,
            traces: traces
        )
    }

    private static func liveE2EReportExport(
        from report: E2ETestReport,
        generatedAt: Date,
        traces: [AgentBehaviorTrace],
        correlationContext: ExportCorrelationContext
    ) -> InAppDatasetLiveE2EReportExport {
        InAppDatasetLiveE2EReportExport(
            schemaVersion: EvidenceLayerExporter.schemaVersion,
            generatedAt: generatedAt,
            app: appInfo(),
            exportPolicy: EvidenceLayerExportPolicy(
                format: "live-e2e-test-report-json",
                sourceLayer: "e2eTestReport",
                ownsLiveE2EScenarios: true,
                includesDeterministicStaticScenarios: report.results.contains { !$0.requiresAgentRun },
                privacy: "Privacy-redacted live E2E metrics, safe categories, hashes, counts, and bounded AgentBehaviorTrace sidecars joined by opaque per-export correlation tokens. Raw free-form content and correlation UUIDs are omitted.",
                notes: [
                    "Embedded TestFlight/live E2E layer exported from the app.",
                    "The parent Agent Grounding package does not own live E2E pass/fail.",
                    "Offline ingestion must validate each scenario against recentTraces by correlationToken according to evidenceMode."
                ]
            ),
            payload: redactedLiveE2EReport(report, correlationContext: correlationContext),
            correlatedTraceCount: correlatedTraceCount(correlationContext: correlationContext),
            modelBackedCorrelatedTraceCount: modelBackedCorrelatedTraceCount(
                report: report,
                traces: traces,
                correlationContext: correlationContext
            ),
            modelBackedCorrelatedScenarioCount: modelBackedCorrelatedScenarioCount(
                report: report,
                traces: traces,
                correlationContext: correlationContext
            ),
            deterministicCompatibilityTraceCount: deterministicCompatibilityTraceCount(
                report: report,
                traces: traces,
                correlationContext: correlationContext
            ),
            traceSidecarField: "recentTraces"
        )
    }

    private static func appInfo() -> InAppDatasetAppInfo {
        .current()
    }

    private static func testFlightExportInfo(
        app: InAppDatasetAppInfo,
        liveE2EReportIncluded: Bool
    ) -> TestFlightAgentGroundingExportInfo {
        return TestFlightAgentGroundingExportInfo(
            sourceAction: sourceAction,
            filePrefix: filePrefix,
            distributionChannel: "testflight_or_unknown",
            sandboxReceipt: false,
            appShortVersion: app.shortVersion,
            appBuildNumber: app.buildNumber,
            liveE2EReportIncluded: liveE2EReportIncluded,
            expectedIngestArgument: "--runtime-audit <exported-testflight-json>"
        )
    }

    private static func correlatedTraceCount(correlationContext: ExportCorrelationContext) -> Int {
        correlationContext.traceResultIDs.count
    }

    private static func modelBackedCorrelatedTraceCount(
        report: E2ETestReport,
        traces: [AgentBehaviorTrace],
        correlationContext: ExportCorrelationContext
    ) -> Int {
        traces.reduce(into: 0) { count, trace in
            guard let resultID = correlationContext.traceResultIDs[trace.id],
                  let result = report.results.first(where: { $0.id == resultID }),
                  result.requiresAgentRun,
                  isModelBackedLiveEvidenceTrace(trace, for: result) else { return }
            count += 1
        }
    }

    private static func modelBackedCorrelatedScenarioCount(
        report: E2ETestReport,
        traces: [AgentBehaviorTrace],
        correlationContext: ExportCorrelationContext
    ) -> Int {
        report.results.reduce(into: 0) { count, result in
            guard result.requiresAgentRun,
                  result.evidenceMode == E2EEvidenceMode.modelBackedRequired.rawValue,
                  traces.contains(where: { trace in
                      isModelBackedLiveEvidenceTrace(trace, for: result)
                          && correlationContext.traceResultIDs[trace.id] == result.id
                  }) else {
                return
            }
            count += 1
        }
    }

    private static func deterministicCompatibilityTraceCount(
        report: E2ETestReport,
        traces: [AgentBehaviorTrace],
        correlationContext: ExportCorrelationContext
    ) -> Int {
        traces.reduce(into: 0) { count, trace in
            guard let resultID = correlationContext.traceResultIDs[trace.id],
                  report.results.contains(where: { $0.id == resultID && $0.requiresAgentRun }),
                  trace.runtimePath == "deterministic-compatibility" else { return }
            count += 1
        }
    }

    private static func isModelBackedLiveEvidenceTrace(
        _ trace: AgentBehaviorTrace,
        for result: E2ETestResult
    ) -> Bool {
        guard trace.event == .modelTurn,
              trace.runtimePath != "deterministic-compatibility",
              trace.parseError == nil,
              !trace.rawOutputPrefix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return false
        }
        if !trace.stage.hasPrefix("agent-json") {
            return plainModelEvidenceStageIsAccepted(
                trace.stage,
                requiresAgentRun: result.requiresAgentRun,
                kind: result.kind,
                expectedIntent: result.expectedIntent,
                actualIntent: result.actualIntent
            )
        }
        guard trace.runtimePath == "agent-model",
              trace.streamStarted == true,
              trace.modelLoaded == true,
              trace.firstChunkReceived == true,
              (trace.textChunkCount ?? 0) > 0,
              trace.finalChunkReceived == true else {
            return false
        }
        if let selectedToolID = trace.selectedToolID, !selectedToolID.isEmpty {
            let canonicalToolID = ToolRouteGuard.canonicalToolID(selectedToolID)
            return !trace.emittedFinalInActionTurn
                && trace.allowedToolIDs.map(ToolRouteGuard.canonicalToolID).contains(canonicalToolID)
        }
        guard trace.emittedFinalInActionTurn,
              trace.finalizerAccepted == true else {
            return false
        }
        return !traceIntentRequiresTool(trace.intent, allowedToolIDs: trace.allowedToolIDs)
            || (trace.successfulObservationCount ?? 0) > 0
    }

    private static func plainModelEvidenceStageIsAccepted(
        _ stage: String,
        requiresAgentRun: Bool,
        kind: String,
        expectedIntent: String,
        actualIntent: String
    ) -> Bool {
        stage == "chat-text-turn"
            && requiresAgentRun
            && kind.caseInsensitiveCompare(E2ETestKind.chat.rawValue) == .orderedSame
            && expectedIntent.caseInsensitiveCompare(UserIntent.chat.rawValue) == .orderedSame
            && actualIntent.caseInsensitiveCompare(UserIntent.chat.rawValue) == .orderedSame
    }

    static func plainModelEvidenceStageIsAcceptedForTests(
        _ stage: String,
        requiresAgentRun: Bool,
        kind: String,
        expectedIntent: String,
        actualIntent: String
    ) -> Bool {
        plainModelEvidenceStageIsAccepted(
            stage,
            requiresAgentRun: requiresAgentRun,
            kind: kind,
            expectedIntent: expectedIntent,
            actualIntent: actualIntent
        )
    }

    private static func traceIntentRequiresTool(_ rawIntent: String?, allowedToolIDs: [String]) -> Bool {
        guard let rawIntent,
              let intent = UserIntent(rawValue: rawIntent) else {
            return !allowedToolIDs.isEmpty
        }
        return IntentRouter.intentRequiresTool(IntentRoutingDecision(
            intent: intent,
            allowedToolIDs: Set(allowedToolIDs),
            requiresClarification: false,
            clarificationPrompt: nil
        ))
    }

    private static func traceMatches(result: E2ETestResult, trace: AgentBehaviorTrace) -> Bool {
        let redactedScenarioID = privacySummary(label: "scenarioID", text: result.scenarioID)
        guard trace.scenarioID == result.scenarioID || trace.scenarioID == redactedScenarioID else { return false }
        guard result.e2eRunID != nil
                || result.agentRunID != nil
                || result.conversationID != nil
                || result.turnID != nil else {
            return false
        }
        if let e2eRunID = result.e2eRunID {
            guard trace.e2eRunID == e2eRunID else { return false }
        }
        if let agentRunID = result.agentRunID {
            guard trace.agentRunID == agentRunID else { return false }
        }
        if let conversationID = result.conversationID {
            guard trace.conversationID == conversationID else { return false }
        }
        if let turnID = result.turnID {
            guard trace.turnID == turnID else { return false }
        }
        return true
    }

    private static func exportQualityFailures(
        from traces: [InAppDatasetTraceExport],
        liveE2EReport: InAppDatasetLiveE2EReportExport?,
        rawLiveE2EReport: E2ETestReport?,
        rawTraces: [AgentBehaviorTrace],
        correlationContext: ExportCorrelationContext
    ) -> [InAppDatasetExportQualityFailure] {
        var failures: [InAppDatasetExportQualityFailure] = []
        if traces.isEmpty {
            failures.append(InAppDatasetExportQualityFailure(
                type: "agent_grounding_no_recent_model_traces",
                agent: "runtime",
                expected: ["Agent Grounding export should include recent model/tool traces captured from real in-app execution."],
                actual: "recentTraces is empty",
                scenario: sourceAction,
                problem: "The TestFlight + Agent Grounding package exported no recent traces. This usually means AgentBehaviorTraceRecorder.record is not wired into the live model path, or the app audit was exported before exercising real model interactions.",
                sourceLayer: "agentGroundingRuntimeAudit.exportQuality"
            ))
        }

        if let liveE2EReport,
           let rawLiveE2EReport,
           let failure = liveE2EModelBackedTraceGapFailure(
            rawLiveE2EReport,
            traces: rawTraces,
            exportSummary: liveE2EReport,
            correlationContext: correlationContext
           ) {
            failures.append(failure)
        }

        for trace in traces where requiresStructuredModelTraceCompleteness(trace) {
            let missing = missingStructuredModelTraceFields(trace)
            guard !missing.isEmpty else { continue }
            failures.append(InAppDatasetExportQualityFailure(
                type: "agent_grounding_model_trace_incomplete",
                agent: trace.slot,
                expected: [
                    "Structured model-turn traces must include runtimePath, selectedRuntime, modelLoaded, outputTokenCount, stream state, and either text-chunk proof or a precise empty/failure reason."
                ],
                actual: [
                    "missing=\(missing.joined(separator: ","))",
                    "stage=\(trace.stage)",
                    "runtimePath=\(trace.runtimePath ?? "nil")",
                    "parseError=\(trace.parseError ?? "nil")",
                    "emptyOutputReason=\(trace.emptyOutputReason ?? "nil")",
                    "streamTerminationReason=\(trace.streamTerminationReason ?? "nil")"
                ].joined(separator: "; "),
                scenario: trace.promptPrefix,
                problem: "A structured model-turn trace does not carry the minimum runtime evidence needed to distinguish real generation from deterministic fallback or pre-stream failure.",
                sourceLayer: "agentGroundingRuntimeAudit.exportQuality"
            ))
        }
        for trace in traces where trace.event == .finalAnswer && trace.finalValidatorAcceptedCandidate == false {
            failures.append(InAppDatasetExportQualityFailure(
                type: "agent_grounding_final_validator_replaced_candidate",
                agent: trace.slot,
                expected: [
                    "Final answer traces should preserve whether ToolObservationFinalizer and FinalIntentValidator accepted the observed candidate before it became user-visible output."
                ],
                actual: [
                    "stage=\(trace.stage)",
                    "selectedToolID=\(trace.selectedToolID ?? "nil")",
                    "replacementSource=\(trace.finalValidatorReplacementSource ?? "nil")",
                    "rejectionReason=\(trace.finalValidatorRejectionReason ?? "nil")",
                    "finalizerAccepted=\(trace.finalizerAccepted.map { $0 ? "true" : "false" } ?? "nil")",
                    "finalizerRejectionReason=\(trace.finalizerRejectionReason ?? "nil")"
                ].joined(separator: "; "),
                scenario: trace.promptPrefix,
                problem: "The final validator replaced the candidate response. Treat this as runtime/finalization feedback, not as proof that the model produced a valid final answer.",
                sourceLayer: "agentGroundingRuntimeAudit.exportQuality"
            ))
        }
        return failures
    }

    private static func liveE2EModelBackedTraceGapFailure(
        _ liveE2EReport: E2ETestReport,
        traces: [AgentBehaviorTrace],
        exportSummary: InAppDatasetLiveE2EReportExport,
        correlationContext: ExportCorrelationContext
    ) -> InAppDatasetExportQualityFailure? {
        let evidenceRequired = liveE2EReport.results.filter {
            $0.requiresAgentRun
                && $0.evidenceMode != E2EEvidenceMode.routingOnly.rawValue
        }
        let missing = evidenceRequired.filter { result in
            let correlated = traces.filter { correlationContext.traceResultIDs[$0.id] == result.id }
            if result.evidenceMode == E2EEvidenceMode.policyFirstAllowed.rawValue {
                return !correlated.contains(where: {
                    isModelBackedLiveEvidenceTrace($0, for: result)
                        || isDeterministicCompatibilityEvidenceTrace($0)
                })
            }
            return !correlated.contains(where: { isModelBackedLiveEvidenceTrace($0, for: result) })
        }
        guard !evidenceRequired.isEmpty, !missing.isEmpty else {
            return nil
        }
        return InAppDatasetExportQualityFailure(
            type: "agent_grounding_live_e2e_model_backed_trace_gap",
            agent: "runtime",
            expected: [
                "modelBackedRequired scenarios need correlated AssistantKernel model-backed structured generation evidence; policyFirstAllowed scenarios may use correlated model-backed or deterministic policy-first evidence; routingOnly scenarios need no runtime evidence."
            ],
            actual: [
                "evidenceRequiredScenarioCount=\(evidenceRequired.count)",
                "missingEvidenceScenarioCount=\(missing.count)",
                "modelBackedCorrelatedTraceCount=\(exportSummary.modelBackedCorrelatedTraceCount)",
                "modelBackedCorrelatedScenarioCount=\(exportSummary.modelBackedCorrelatedScenarioCount)",
                "correlatedTraceCount=\(exportSummary.correlatedTraceCount)",
                "deterministicCompatibilityTraceCount=\(exportSummary.deterministicCompatibilityTraceCount)"
            ].joined(separator: "; "),
            scenario: "Agent Grounding > E2E Test Runner > Export TestFlight + Agent Grounding Package",
            problem: "The embedded live E2E report is missing correlated evidence required by each scenario's evidenceMode.",
            sourceLayer: "agentGroundingRuntimeAudit.exportQuality"
        )
    }

    private static func isDeterministicCompatibilityEvidenceTrace(_ trace: AgentBehaviorTrace) -> Bool {
        guard trace.runtimePath == "deterministic-compatibility" else { return false }
        return trace.event == .toolAction || trace.event == .finalAnswer
    }

    private static func requiresStructuredModelTraceCompleteness(_ trace: InAppDatasetTraceExport) -> Bool {
        guard trace.event == .modelTurn else { return false }
        if trace.runtimePath == "deterministic-compatibility" { return false }
        let stage = trace.stage.lowercased()
        if stage == "agent-json" || stage.hasPrefix("agent-json-") { return true }
        return stage.contains("executor-json")
    }

    private static func missingStructuredModelTraceFields(_ trace: InAppDatasetTraceExport) -> [String] {
        var missing: [String] = []
        if (trace.runtimePath ?? "").isEmpty { missing.append("runtimePath") }
        if (trace.selectedRuntime ?? "").isEmpty { missing.append("selectedRuntime") }
        if trace.modelLoaded == nil { missing.append("modelLoaded") }
        if trace.outputTokenCount == nil { missing.append("outputTokenCount") }

        if trace.streamStarted == nil {
            missing.append("streamStarted")
            return missing
        }

        if trace.streamStarted == true {
            if trace.firstChunkReceived == nil { missing.append("firstChunkReceived") }
            if trace.textChunkCount == nil { missing.append("textChunkCount") }
            if trace.finalChunkReceived == nil { missing.append("finalChunkReceived") }

            let hasTextEvidence = trace.firstChunkReceived == true
                || (trace.textChunkCount ?? 0) > 0
                || (trace.outputTokenCount ?? 0) > 0
                || !trace.rawOutputPrefix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            if !hasTextEvidence && !hasPreciseRuntimeFailureReason(trace) {
                missing.append("emptyOutputReasonOrParseError")
            }
            if trace.finalChunkReceived == false && (trace.streamTerminationReason ?? "").isEmpty && !hasPreciseRuntimeFailureReason(trace) {
                missing.append("streamTerminationReason")
            }
            if trace.emittedFinalInActionTurn {
                if trace.finalizerAccepted == nil { missing.append("finalizerAccepted") }
                if traceIntentRequiresTool(trace.intent, allowedToolIDs: trace.allowedToolIDs),
                   (trace.successfulObservationCount ?? 0) <= 0 {
                    missing.append("successfulObservationCount")
                }
            }
        } else if !hasPreciseRuntimeFailureReason(trace) {
            missing.append("emptyOutputReasonOrParseError")
        }
        return missing
    }

    private static func hasPreciseRuntimeFailureReason(_ trace: InAppDatasetTraceExport) -> Bool {
        if !(trace.emptyOutputReason ?? "").isEmpty { return true }
        if !(trace.parseError ?? "").isEmpty { return true }
        if !(trace.streamTerminationReason ?? "").isEmpty { return true }
        if !(trace.cancellationStateBeforeStream ?? "").isEmpty { return true }
        return false
    }

    private static func exportTrace(
        _ trace: AgentBehaviorTrace,
        id: UUID,
        correlationToken: String?
    ) -> InAppDatasetTraceExport {
        let privacySafeSelectedToolID = AgentDiagnosticFileRedactor.privacySafeSelectedToolID(trace.selectedToolID)
        return InAppDatasetTraceExport(
            id: id,
            createdAt: trace.createdAt,
            event: trace.event,
            slot: safeCode(trace.slot),
            stage: safeCode(trace.stage),
            scenarioID: trace.scenarioID.map { privacySummary(label: "scenarioID", text: $0) },
            correlationToken: correlationToken,
            intent: trace.intent.map(safeCode),
            promptPrefix: sanitizedSnippet(trace.promptPrefix),
            rawOutputPrefix: sanitizedSnippet(trace.rawOutputPrefix),
            selectedToolID: privacySafeSelectedToolID,
            toolArguments: AgentDiagnosticFileRedactor.privacySafeToolArguments(
                trace.toolArguments,
                selectedToolID: trace.selectedToolID
            ),
            allowedToolIDs: AgentDiagnosticFileRedactor.privacySafeAllowedToolIDs(trace.allowedToolIDs),
            requiresApproval: trace.requiresApproval,
            approvalMode: trace.approvalMode.map(safeCode),
            parseError: actionTraceParseError(trace) ?? trace.parseError.map(safeCode),
            emittedFinalInActionTurn: trace.emittedFinalInActionTurn,
            modelFamily: trace.modelFamily.map(safeModelIdentifier),
            baseModelPath: trace.baseModelPath.map(pathLeaf),
            adapterID: trace.adapterID.map(safeModelIdentifier),
            adapterSlot: trace.adapterSlot.map(safeCode),
            adapterPath: trace.adapterPath.map(pathLeaf),
            adapterApplied: trace.adapterApplied,
            adapterScale: trace.adapterScale,
            adapterFailureReason: trace.adapterFailureReason.map { sanitizedSnippet($0, limit: 240) },
            generationElapsedMs: trace.generationElapsedMs,
            firstTokenLatencyMs: trace.firstTokenLatencyMs,
            outputTokenCount: trace.outputTokenCount,
            estimatedPromptTokenCount: trace.estimatedPromptTokenCount,
            preFirstTokenMs: trace.preFirstTokenMs,
            messageBuildMs: trace.messageBuildMs,
            decodeMs: trace.decodeMs,
            tokensPerSecond: trace.tokensPerSecond,
            ensureReadyMs: trace.ensureReadyMs,
            adapterActivationMs: trace.adapterActivationMs,
            runtimePath: trace.runtimePath.map(safeIdentifier),
            activeAdapterSlot: trace.activeAdapterSlot.map(safeCode),
            maxTokensRequested: trace.maxTokensRequested,
            maxTokensEffective: trace.maxTokensEffective,
            promptCharCount: trace.promptCharCount,
            accelerationDiagnostic: trace.accelerationDiagnostic.map { sanitizedSnippet($0, limit: 240) },
            accelerationDiagnostics: trace.accelerationDiagnostics,
            emptyOutputReason: trace.emptyOutputReason.map { sanitizedSnippet($0, limit: 240) },
            streamStarted: trace.streamStarted,
            selectedRuntime: trace.selectedRuntime.map(safeIdentifier),
            selectedAdapter: trace.selectedAdapter.map(safeModelIdentifier),
            modelIdentifier: trace.modelIdentifier.map(safeModelIdentifier),
            modelLoaded: trace.modelLoaded,
            stopSequences: trace.stopSequences.map { sanitizedSnippet($0, limit: 80) },
            temperature: trace.temperature,
            topP: trace.topP,
            cancellationStateBeforeStream: trace.cancellationStateBeforeStream.map(safeCode),
            firstChunkReceived: trace.firstChunkReceived,
            textChunkCount: trace.textChunkCount,
            finalChunkReceived: trace.finalChunkReceived,
            streamTerminationReason: trace.streamTerminationReason.map { sanitizedSnippet($0, limit: 160) },
            successfulObservationCount: trace.successfulObservationCount,
            finalizerAccepted: trace.finalizerAccepted,
            finalizerRejectionReason: trace.finalizerRejectionReason.map(safeCode),
            finalValidatorAcceptedCandidate: trace.finalValidatorAcceptedCandidate,
            finalValidatorReplacementSource: trace.finalValidatorReplacementSource.map(safeIdentifier),
            finalValidatorRejectionReason: trace.finalValidatorRejectionReason.map(safeCode),
            selfModel: redactedSelfModelSummary(trace.selfModel)
        )
    }

    private static func redactedSelfModelSummary(_ summary: AgentBehaviorTrace.SelfModelDecisionSummary?) -> AgentBehaviorTrace.SelfModelDecisionSummary? {
        summary?.redactedForPersistentDiagnostics()
    }

    private static func redactedBehaviorAudit(
        _ audit: AgentBehaviorAuditReport?,
        identityKey: SymmetricKey
    ) -> AgentBehaviorAuditReport? {
        guard let audit else { return nil }
        return AgentBehaviorAuditReport(
            passed: audit.passed,
            score: audit.score,
            generatedAt: audit.generatedAt,
            traceCount: audit.traceCount,
            violationCount: audit.violationCount,
            sourceCommit: audit.sourceCommit,
            violations: audit.violations.map {
                redactedViolation($0, identityKey: identityKey)
            },
            recommendations: audit.recommendations.map { privacySummary(label: "recommendation", text: $0) },
            repairSamples: audit.repairSamples.map {
                redactedRepairSample($0, identityKey: identityKey)
            }
        )
    }

    private static func redactedRuntimeManifestAudit(
        _ audit: RuntimeAgentManifestAuditReport?
    ) -> RuntimeAgentManifestAuditReport? {
        guard let audit else { return nil }
        return privacySafeRuntimeManifestAuditForExport(audit)
    }

    static func privacySafeRuntimeManifestAuditForExport(
        _ audit: RuntimeAgentManifestAuditReport
    ) -> RuntimeAgentManifestAuditReport {
        return RuntimeAgentManifestAuditReport(
            passed: audit.passed,
            score: audit.score,
            failures: audit.failures.map(redactedRuntimeManifestFailure),
            generatedAt: audit.generatedAt,
            recommendedDatasetRepairs: audit.recommendedDatasetRepairs.map {
                privacySummary(label: "repairRecommendation", text: $0)
            }
        )
    }

    private static func redactedRuntimeManifestFailure(
        _ failure: RuntimeManifestFailure
    ) -> RuntimeManifestFailure {
        RuntimeManifestFailure(
            type: safeCode(failure.type),
            agent: failure.agent.map(safeCode),
            expected: failure.expected.enumerated().map { index, value in
                privacySummary(label: "expected\(index)", text: value)
            },
            actual: failure.actual.map { privacySummary(label: "actual", text: $0) },
            scenario: failure.scenario.map { privacySummary(label: "scenario", text: $0) },
            problem: privacySummary(label: "problem", text: failure.problem)
        )
    }

    private static func redactedScenarioResult(_ result: RuntimeScenarioResult) -> RuntimeScenarioResult {
        RuntimeScenarioResult(
            id: privacySummary(label: "scenarioID", text: result.id),
            scenario: RuntimeScenario(
                id: privacySummary(label: "scenarioID", text: result.scenario.id),
                intent: safeCode(result.scenario.intent),
                expectedToolID: ToolRouteGuard.canonicalToolID(result.scenario.expectedToolID),
                requiresApproval: result.scenario.requiresApproval,
                prompt: privacySummary(label: "prompt", text: result.scenario.prompt)
            ),
            passed: result.passed,
            failures: result.failures.map(redactedRuntimeManifestFailure)
        )
    }

    static func privacySafeScenarioResultsForExport(
        _ results: [RuntimeScenarioResult]
    ) -> [RuntimeScenarioResult] {
        results.map(redactedScenarioResult)
    }

    private static func emptyImproveLoopDataset() -> ImproveLoopDataset {
        ImproveLoopDataset(
            schemaVersion: ImproveLoopSampleGate.schemaVersion,
            generatedAt: Date(),
            acceptedTraining: [],
            quarantinedSamples: [],
            regressionTests: [],
            counters: ImproveLoopDatasetCounters(
                accepted: 0,
                quarantined: 0,
                regression: 0,
                staleTraceRejected: 0,
                legacyToolNamespaceRejected: 0,
                architectureFailureRejected: 0,
                resourceFallbackRejected: 0
            )
        )
    }

    private static func redactedViolation(
        _ violation: AgentBehaviorViolation,
        identityKey: SymmetricKey
    ) -> AgentBehaviorViolation {
        AgentBehaviorViolation(
            id: redactedUUID(violation.id, key: identityKey, domain: "behaviorViolation"),
            createdAt: violation.createdAt,
            severity: violation.severity,
            code: safeCode(violation.code),
            agent: safeCode(violation.agent),
            expected: privacySummary(label: "expected", text: violation.expected),
            actual: privacySummary(label: "actual", text: violation.actual),
            promptPrefix: privacySummary(label: "prompt", text: violation.promptPrefix),
            problem: privacySummary(label: "problem", text: violation.problem)
        )
    }

    private static func redactedRepairSample(
        _ sample: AgentBehaviorRepairSample,
        identityKey: SymmetricKey
    ) -> AgentBehaviorRepairSample {
        AgentBehaviorRepairSample(
            id: redactedUUID(sample.id, key: identityKey, domain: "behaviorRepairSample"),
            createdAt: sample.createdAt,
            agent: safeCode(sample.agent),
            violationCode: safeCode(sample.violationCode),
            promptPrefix: privacySummary(label: "prompt", text: sample.promptPrefix),
            expected: privacySummary(label: "expected", text: sample.expected),
            badOutput: privacySummary(label: "badOutput", text: sample.badOutput),
            correctedOutput: privacySummary(label: "correctedOutput", text: sample.correctedOutput),
            lesson: privacySummary(label: "lesson", text: sample.lesson),
            curriculum: privacySummary(label: "curriculum", text: sample.curriculum)
        )
    }

    static func writePackage(
        manifestSource: String,
        usedRuntimeFallback: Bool,
        runtimeManifestAudit: RuntimeAgentManifestAuditReport?,
        behaviorAudit: AgentBehaviorAuditReport?,
        scenarioResults: [RuntimeScenarioResult],
        liveE2EReport: E2ETestReport? = E2ETestLogStore.latestReport(),
        traceLimit: Int = 200,
        includeScenarioResults: Bool = defaultIncludesScenarioResults
    ) throws -> InAppDatasetPackageExportResult {
        let package = makePackage(
            manifestSource: manifestSource,
            usedRuntimeFallback: usedRuntimeFallback,
            runtimeManifestAudit: runtimeManifestAudit,
            behaviorAudit: behaviorAudit,
            scenarioResults: scenarioResults,
            liveE2EReport: liveE2EReport,
            traceLimit: traceLimit,
            includeScenarioResults: includeScenarioResults
        )
        let directory = try exportDirectory()
        try purgeLegacyUnsafeArtifacts(in: directory)
        let fileName = "\(filePrefix)-\(Self.safeTimestamp(package.generatedAt))-\(UUID().uuidString.lowercased()).json"
        let url = directory.appendingPathComponent(fileName, isDirectory: false)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(package)
        try data.write(to: url, options: [.atomic, .completeFileProtection])
        try writeImproveLoopJSONL(package.improveLoop, directory: directory, timestamp: Self.safeTimestamp(package.generatedAt))
        return InAppDatasetPackageExportResult(url: url, package: package)
    }

    static func exportDirectory() throws -> URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first ?? FileManager.default.temporaryDirectory
        let directory = base
            .appendingPathComponent("Diagnostics", isDirectory: true)
            .appendingPathComponent(directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    static func purgeLegacyUnsafeArtifacts() throws {
        try purgeLegacyUnsafeArtifacts(in: exportDirectory())
    }

    static func purgeLegacyUnsafeArtifacts(in directory: URL) throws {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        let safePrefixes = [
            filePrefix + "-",
            "accepted_training-redacted-v1-",
            "quarantined_samples-redacted-v1-",
            "regression_tests-redacted-v1-"
        ]
        let legacyPrefixes = [
            "lumen-testflight-agent-grounding-",
            "accepted_training-",
            "quarantined_samples-",
            "regression_tests-"
        ]
        for url in files {
            let name = url.lastPathComponent
            guard legacyPrefixes.contains(where: name.hasPrefix),
                  !safePrefixes.contains(where: name.hasPrefix) else {
                continue
            }
            try FileManager.default.removeItem(at: url)
        }
    }

    private static func mergedBehaviorAuditWithRuntimeTraceViolations(
        _ baseAudit: AgentBehaviorAuditReport?,
        traces: [AgentBehaviorTrace]
    ) -> AgentBehaviorAuditReport? {
        let traceViolations = runtimeTraceViolations(from: traces)
        guard !traceViolations.isEmpty else { return baseAudit }

        let existingViolations = baseAudit?.violations ?? []
        let violations = (existingViolations + traceViolations).sorted { lhs, rhs in
            if lhs.severity.weight == rhs.severity.weight { return lhs.createdAt > rhs.createdAt }
            return lhs.severity.weight > rhs.severity.weight
        }

        let existingRecommendations = baseAudit?.recommendations ?? []
        let latencyRecommendation = "Investigate model runtime latency: keep shared base/adapters resident, verify acceleration path, reduce mouth prompt size, and cap per-stage token budgets."
        let recommendations = Array(Set(existingRecommendations + [latencyRecommendation])).sorted()
        let baseTraceCount = baseAudit?.traceCount ?? 0
        let auditedTraceCount = max(baseTraceCount, traces.count)
        let weightedPenalty = violations.reduce(0.0) { $0 + $1.severity.weight }
        let denominator = max(1.0, Double(max(1, auditedTraceCount)) * 2.0)
        let score = max(0.0, min(1.0, 1.0 - weightedPenalty / denominator))

        return AgentBehaviorAuditReport(
            passed: violations.allSatisfy { $0.severity == .warning },
            score: score,
            generatedAt: baseAudit?.generatedAt ?? Date(),
            traceCount: auditedTraceCount,
            violationCount: violations.count,
            sourceCommit: baseAudit?.sourceCommit,
            violations: violations,
            recommendations: recommendations,
            repairSamples: baseAudit?.repairSamples ?? []
        )
    }

    /// Computes runtime-derived behavior violations from traces.
    /// - Returns: An array of violations based on action parse errors and model turn latency thresholds.
    private static func runtimeTraceViolations(from traces: [AgentBehaviorTrace]) -> [AgentBehaviorViolation] {
        traces.compactMap { trace in
            if let parseError = actionTraceParseError(trace) {
                return AgentBehaviorViolation(
                    id: UUID(),
                    createdAt: Date(),
                    severity: .error,
                    code: "structured_action_trace_parse_error",
                    agent: trace.slot,
                    expected: "Executor/tool-action traces must be strict structured JSON parsable as an action turn.",
                    actual: "stage=\(safeCode(trace.stage)); parseError=\(safeCode(parseError)); selectedToolID=\(trace.selectedToolID.map(ToolRouteGuard.canonicalToolID) ?? "nil"); rawOutputPrefix=\(sanitizedSnippet(trace.rawOutputPrefix))",
                    promptPrefix: sanitizedSnippet(trace.promptPrefix),
                    problem: "A tool-action trace did not contain a parseable structured action object."
                )
            }

            guard trace.event == .modelTurn, let elapsed = trace.generationElapsedMs else { return nil }
            let severity: AgentBehaviorViolation.Severity
            let code: String
            let problem: String
            if elapsed > severeModelTurnThresholdMs {
                severity = .critical
                code = "model_turn_latency_severe"
                problem = "A model turn exceeded the severe latency threshold."
            } else if elapsed > slowModelTurnThresholdMs {
                severity = .error
                code = "model_turn_too_slow"
                problem = "A model turn exceeded the acceptable live-agent latency threshold."
            } else {
                return nil
            }

            return AgentBehaviorViolation(
                id: UUID(),
                createdAt: Date(),
                severity: severity,
                code: code,
                agent: trace.slot,
                expected: "Model turn latency <= \(slowModelTurnThresholdMs) ms; severe latency threshold <= \(severeModelTurnThresholdMs) ms.",
                actual: "stage=\(safeCode(trace.stage)); elapsedMs=\(elapsed); firstTokenMs=\(trace.firstTokenLatencyMs.map(String.init) ?? "nil"); estimatedPromptTokens=\(trace.estimatedPromptTokenCount.map(String.init) ?? "nil"); outputTokens=\(trace.outputTokenCount.map(String.init) ?? "nil"); tps=\(trace.tokensPerSecond.map { String(format: "%.2f", $0) } ?? "nil"); promptChars=\(trace.promptCharCount.map(String.init) ?? "nil"); modelFile=\(trace.baseModelPath.map(pathLeaf) ?? "nil"); adapterFile=\(trace.adapterPath.map(pathLeaf) ?? "nil"); accel=\(trace.accelerationDiagnostic.map { sanitizedSnippet($0, limit: 240) } ?? "unknown")",
                promptPrefix: sanitizedSnippet(trace.promptPrefix),
                problem: problem
            )
        }
    }

    /// Determines if a trace has an action parse error.
    /// - Returns: `true` if the trace has an action parse error, `false` otherwise.
    private static func traceHasActionParseError(_ trace: AgentBehaviorTrace) -> Bool {
        actionTraceParseError(trace) != nil
    }

    /// Retrieves the parse error for a structured action trace.
    /// - Parameters:
    ///   - trace: The behavior trace to examine.
    /// - Returns: The parse error string if the trace is in a structured action stage and a parse error exists, `nil` otherwise.
    private static func actionTraceParseError(_ trace: AgentBehaviorTrace) -> String? {
        guard isActionStructuredStage(trace) else { return nil }
        return trace.parseError
    }

    /// Determines whether a trace represents a structured action stage.
    /// - Returns: `true` if the trace expects structured JSON action output, `false` otherwise.
    private static func isActionStructuredStage(_ trace: AgentBehaviorTrace) -> Bool {
        if trace.event == .toolAction { return true }
        guard trace.event == .modelTurn else { return false }
        let stage = trace.stage.lowercased()
        if stage == "agent-json" { return true }
        if stage.contains("mouth") || stage.contains("final") || stage.contains("direct") {
            return false
        }
        return stage.contains("executor-json")
    }

    /// Writes improve-loop dataset samples as three timestamped JSONL files.
    /// - Parameters:
    ///   - dataset: The improve-loop dataset containing accepted training samples, quarantined samples, and regression tests.
    /// - Throws: Errors from encoding or file write operations.
    private static func writeImproveLoopJSONL(_ dataset: ImproveLoopDataset, directory: URL, timestamp: String) throws {
        try writeJSONL(dataset.acceptedTraining, to: directory.appendingPathComponent("accepted_training-redacted-v1-\(timestamp).jsonl", isDirectory: false))
        try writeJSONL(dataset.quarantinedSamples, to: directory.appendingPathComponent("quarantined_samples-redacted-v1-\(timestamp).jsonl", isDirectory: false))
        try writeJSONL(dataset.regressionTests, to: directory.appendingPathComponent("regression_tests-redacted-v1-\(timestamp).jsonl", isDirectory: false))
    }

    private static func writeJSONL<T: Encodable>(_ records: [T], to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var data = Data()
        for record in records {
            data.append(try encoder.encode(record))
            data.append(0x0A)
        }
        try data.write(to: url, options: [.atomic, .completeFileProtection])
    }

    private static func safeTimestamp(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
            .replacingOccurrences(of: ":", with: "-")
            .replacingOccurrences(of: ".", with: "-")
    }

    private static func sanitizedSnippet(_ text: String, limit: Int = 1_200) -> String {
        let withoutHiddenReasoning = ModelOutputSanitizer.stripHiddenBlocks(text)
        let redacted = PersistentRuntimeDiagnosticsRedactor.redactWithoutTruncating(withoutHiddenReasoning)
        return String(redacted.prefix(max(0, limit)))
    }

    private static func privacySummary(label: String, text: String) -> String {
        AgentDiagnosticFileRedactor.summary(label: label, text: text)
    }

    private static func pathLeaf(_ path: String) -> String {
        let leaf = URL(fileURLWithPath: path).lastPathComponent
        let nonEmpty = leaf.isEmpty ? "[redacted-path]" : leaf
        return sanitizedSnippet(nonEmpty, limit: 160)
    }

    private static func safeModelIdentifier(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.contains("/Users/"),
              !trimmed.contains("/private/"),
              !trimmed.contains("/var/"),
              !trimmed.hasPrefix("/") else {
            return pathLeaf(trimmed)
        }
        return safeIdentifier(trimmed)
    }

    private static func safeCode(_ value: String) -> String {
        PersistentRuntimeDiagnosticsRedactor.safeCode(value)
    }

    private static func safeIdentifier(_ value: String) -> String {
        let stripped = ModelOutputSanitizer.stripHiddenBlocks(value)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let withoutEmails = stripped.replacingOccurrences(
            of: #"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"#,
            with: "[redacted]",
            options: [.regularExpression, .caseInsensitive]
        )
        let withoutUUIDs = withoutEmails.replacingOccurrences(
            of: #"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"#,
            with: "[redacted]",
            options: [.regularExpression, .caseInsensitive]
        )
        return String(withoutUUIDs.prefix(160))
    }
}
