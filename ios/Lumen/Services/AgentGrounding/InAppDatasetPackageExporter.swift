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
    let recentTraces: [AgentBehaviorTrace]
    let traceSelectedToolAllowedCount: Int
    let traceParseErrorCount: Int
    let improveLoop: ImproveLoopDataset
    let exportPolicy: InAppDatasetExportPolicy
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
    static let schemaVersion = "1.2.0"
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
        let improveLoop = ImproveLoopSampleGate.buildDataset(
            behaviorAudit: mergedBehaviorAudit,
            traces: traces,
            scenarioResults: includeScenarioResults ? scenarioResults : [],
            sourceCommit: mergedBehaviorAudit?.sourceCommit
        )
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
            behaviorAudit: mergedBehaviorAudit,
            scenarioResults: includeScenarioResults ? scenarioResults : [],
            recentTraces: traces,
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
                privacy: "contains only manifest audit failures, behavior violations, bounded runtime trace prefixes, and gated improve-loop samples; no full conversations, contacts, calendar bodies, files, photos, or tool payload bodies are exported",
                promptPolicy: "promptPrefix fields are bounded and should be treated as diagnostic snippets only",
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
                    actual: "stage=\(trace.stage); parseError=\(parseError); selectedToolID=\(trace.selectedToolID ?? "nil"); rawOutputPrefix=\(trace.rawOutputPrefix)",
                    promptPrefix: trace.promptPrefix,
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
                actual: "stage=\(trace.stage); elapsedMs=\(elapsed); firstTokenMs=\(trace.firstTokenLatencyMs.map(String.init) ?? "nil"); estimatedPromptTokens=\(trace.estimatedPromptTokenCount.map(String.init) ?? "nil"); outputTokens=\(trace.outputTokenCount.map(String.init) ?? "nil"); tps=\(trace.tokensPerSecond.map { String(format: "%.2f", $0) } ?? "nil"); promptChars=\(trace.promptCharCount.map(String.init) ?? "nil"); modelPath=\(trace.baseModelPath ?? "nil"); adapterPath=\(trace.adapterPath ?? "nil"); accel=\(trace.accelerationDiagnostic ?? "unknown")",
                promptPrefix: trace.promptPrefix,
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
}
