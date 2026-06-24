import Foundation

nonisolated struct LumenInAppDatasetPackage: Codable, Sendable {
    let schemaVersion: String
    let generatedAt: Date
    let app: InAppDatasetAppInfo
    let manifestSource: String
    let usedRuntimeFallback: Bool
    let runtimeManifestAudit: RuntimeAgentManifestAuditReport?
    let behaviorAudit: AgentBehaviorAuditReport?
    let scenarioResults: [RuntimeScenarioResult]
    let recentTraces: [InAppDatasetTraceExport]
    let traceSelectedToolAllowedCount: Int
    let traceParseErrorCount: Int
    let improveLoop: ImproveLoopDataset
    let exportPolicy: InAppDatasetExportPolicy
}

nonisolated struct InAppDatasetTraceExport: Codable, Sendable, Hashable {
    let createdAt: Date
    let event: AgentBehaviorTrace.Event
    let slot: String
    let stage: String
    let scenarioID: String?
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
}

nonisolated struct InAppDatasetAppInfo: Codable, Sendable, Hashable {
    let name: String
    let bundleIdentifier: String?
    let shortVersion: String?
    let buildNumber: String?
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
    static let schemaVersion = "1.3.0"
    static let defaultIncludesScenarioResults = false
    static let slowModelTurnThresholdMs = 30_000
    static let severeModelTurnThresholdMs = 120_000
    private static let directoryName = "LumenDatasetExports"

    /// Assembles a complete in-app dataset package incorporating audit reports and recent traces.
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
        traceLimit: Int = 200,
        includeScenarioResults: Bool = defaultIncludesScenarioResults
    ) -> LumenInAppDatasetPackage {
        let traces = AgentBehaviorTraceRecorder.recent(limit: traceLimit)
        let mergedBehaviorAudit = mergedBehaviorAuditWithRuntimeTraceViolations(behaviorAudit, traces: traces)
        let exportedBehaviorAudit = redactedBehaviorAudit(mergedBehaviorAudit)
        let improveLoop = ImproveLoopSampleGate.buildDataset(
            behaviorAudit: exportedBehaviorAudit,
            traces: traces,
            scenarioResults: includeScenarioResults ? scenarioResults : [],
            sourceCommit: exportedBehaviorAudit?.sourceCommit
        )
        let exportedTraces = traces.map(exportTrace)
        return LumenInAppDatasetPackage(
            schemaVersion: schemaVersion,
            generatedAt: Date(),
            app: InAppDatasetAppInfo(
                name: Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String ?? "Lumen",
                bundleIdentifier: Bundle.main.bundleIdentifier,
                shortVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String,
                buildNumber: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String
            ),
            manifestSource: manifestSource,
            usedRuntimeFallback: usedRuntimeFallback,
            runtimeManifestAudit: runtimeManifestAudit,
            behaviorAudit: exportedBehaviorAudit,
            scenarioResults: includeScenarioResults ? scenarioResults : [],
            recentTraces: exportedTraces,
            traceSelectedToolAllowedCount: traces.reduce(into: 0) { count, trace in
                guard let selectedToolID = trace.selectedToolID else { return }
                let selected = ToolRouteGuard.canonicalToolID(selectedToolID)
                let allowed = Set(trace.allowedToolIDs.map(ToolRouteGuard.canonicalToolID))
                if allowed.contains(selected) {
                    count += 1
                }
            },
            traceParseErrorCount: traces.reduce(into: 0) { count, trace in
                if traceHasActionParseError(trace) {
                    count += 1
                }
            },
            improveLoop: improveLoop,
            exportPolicy: InAppDatasetExportPolicy(
                format: "agent-grounding-runtime-json-package",
                privacy: "contains only manifest audit failures, behavior violations, redacted bounded runtime trace prefixes, and gated improve-loop samples; no full conversations, contacts, calendar bodies, files, photos, trace identifiers, local paths, or tool payload bodies are exported",
                promptPolicy: "promptPrefix and rawOutputPrefix fields are redacted, hidden-reasoning-stripped, bounded diagnostic snippets only",
                traceLimit: traceLimit,
                source: "RuntimeManifestAuditor + AgentModelBehaviorAuditor + AgentBehaviorTraceRecorder",
                sourceLayer: "agentGroundingRuntimeAudit",
                ownsLiveE2EScenarios: false,
                includesDeterministicStaticScenarios: includeScenarioResults,
                deterministicScenarioPolicy: includeScenarioResults
                    ? "Static manifest scenario checks were explicitly included; they are not proof of live model execution and must not be treated as E2E model runs."
                    : "Static manifest scenario checks are displayed in-app only and omitted from the dataset export; E2ETestRunner owns live model scenario results."
            )
        )
    }

    private static func exportTrace(_ trace: AgentBehaviorTrace) -> InAppDatasetTraceExport {
        InAppDatasetTraceExport(
            createdAt: trace.createdAt,
            event: trace.event,
            slot: safeCode(trace.slot),
            stage: safeCode(trace.stage),
            scenarioID: trace.scenarioID.map { sanitizedSnippet($0, limit: 160) },
            intent: trace.intent.map(safeCode),
            promptPrefix: sanitizedSnippet(trace.promptPrefix),
            rawOutputPrefix: sanitizedSnippet(trace.rawOutputPrefix),
            selectedToolID: trace.selectedToolID.map(ToolRouteGuard.canonicalToolID),
            toolArguments: redactedToolArguments(trace.toolArguments),
            allowedToolIDs: Array(Set(trace.allowedToolIDs.map(ToolRouteGuard.canonicalToolID))).sorted(),
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
            streamTerminationReason: trace.streamTerminationReason.map { sanitizedSnippet($0, limit: 160) }
        )
    }

    private static func redactedBehaviorAudit(_ audit: AgentBehaviorAuditReport?) -> AgentBehaviorAuditReport? {
        guard let audit else { return nil }
        return AgentBehaviorAuditReport(
            passed: audit.passed,
            score: audit.score,
            generatedAt: audit.generatedAt,
            traceCount: audit.traceCount,
            violationCount: audit.violationCount,
            sourceCommit: audit.sourceCommit,
            violations: audit.violations.map(redactedViolation),
            recommendations: audit.recommendations.map { sanitizedSnippet($0, limit: 500) },
            repairSamples: audit.repairSamples.map(redactedRepairSample)
        )
    }

    private static func redactedViolation(_ violation: AgentBehaviorViolation) -> AgentBehaviorViolation {
        AgentBehaviorViolation(
            id: violation.id,
            createdAt: violation.createdAt,
            severity: violation.severity,
            code: safeCode(violation.code),
            agent: safeCode(violation.agent),
            expected: sanitizedSnippet(violation.expected, limit: 500),
            actual: sanitizedSnippet(violation.actual, limit: 800),
            promptPrefix: sanitizedSnippet(violation.promptPrefix),
            problem: sanitizedSnippet(violation.problem, limit: 500)
        )
    }

    private static func redactedRepairSample(_ sample: AgentBehaviorRepairSample) -> AgentBehaviorRepairSample {
        AgentBehaviorRepairSample(
            id: sample.id,
            createdAt: sample.createdAt,
            agent: safeCode(sample.agent),
            violationCode: safeCode(sample.violationCode),
            promptPrefix: sanitizedSnippet(sample.promptPrefix),
            expected: sanitizedSnippet(sample.expected, limit: 500),
            badOutput: sanitizedSnippet(sample.badOutput),
            correctedOutput: sanitizedSnippet(sample.correctedOutput),
            lesson: sanitizedSnippet(sample.lesson, limit: 500),
            curriculum: sanitizedSnippet(sample.curriculum, limit: 240)
        )
    }

    static func writePackage(
        manifestSource: String,
        usedRuntimeFallback: Bool,
        runtimeManifestAudit: RuntimeAgentManifestAuditReport?,
        behaviorAudit: AgentBehaviorAuditReport?,
        scenarioResults: [RuntimeScenarioResult],
        traceLimit: Int = 200,
        includeScenarioResults: Bool = defaultIncludesScenarioResults
    ) throws -> InAppDatasetPackageExportResult {
        let package = makePackage(
            manifestSource: manifestSource,
            usedRuntimeFallback: usedRuntimeFallback,
            runtimeManifestAudit: runtimeManifestAudit,
            behaviorAudit: behaviorAudit,
            scenarioResults: scenarioResults,
            traceLimit: traceLimit,
            includeScenarioResults: includeScenarioResults
        )
        let directory = try exportDirectory()
        let fileName = "lumen-agent-grounding-audit-\(Self.safeTimestamp(package.generatedAt))-\(UUID().uuidString.lowercased()).json"
        let url = directory.appendingPathComponent(fileName, isDirectory: false)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(package)
        try data.write(to: url, options: [.atomic])
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
        if let parseError = trace.parseError { return parseError }
        return AgentTurnParser.parse(trace.rawOutputPrefix).parseError?.rawValue
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
        try writeJSONL(dataset.acceptedTraining, to: directory.appendingPathComponent("accepted_training-\(timestamp).jsonl", isDirectory: false))
        try writeJSONL(dataset.quarantinedSamples, to: directory.appendingPathComponent("quarantined_samples-\(timestamp).jsonl", isDirectory: false))
        try writeJSONL(dataset.regressionTests, to: directory.appendingPathComponent("regression_tests-\(timestamp).jsonl", isDirectory: false))
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
        try data.write(to: url, options: [.atomic])
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

    private static func redactedToolArguments(_ arguments: [String: String]) -> [String: String] {
        arguments.reduce(into: [:]) { result, element in
            let (key, value) = element
            let redactedValue = value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "" : "[redacted]"
            result[safeCode(key)] = redactedValue
        }
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
