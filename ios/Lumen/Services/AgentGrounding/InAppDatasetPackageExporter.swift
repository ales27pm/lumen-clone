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
    static let schemaVersion = "1.1.0"
    static let defaultIncludesScenarioResults = false
    static let slowModelTurnThresholdMs = 30_000
    static let severeModelTurnThresholdMs = 120_000
    private static let directoryName = "LumenDatasetExports"

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
                if trace.allowedToolIDs.contains(selectedToolID) {
                    count += 1
                }
            },
            traceParseErrorCount: traces.reduce(into: 0) { count, trace in
                if isActionStructuredStage(trace), trace.parseError != nil {
                    count += 1
                }
            },
            exportPolicy: InAppDatasetExportPolicy(
                format: "agent-grounding-runtime-json-package",
                privacy: "contains only manifest audit failures, behavior violations, and bounded runtime trace prefixes; no full conversations, contacts, calendar bodies, files, photos, or tool payload bodies are exported",
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

    private static func runtimeTraceViolations(from traces: [AgentBehaviorTrace]) -> [AgentBehaviorViolation] {
        traces.compactMap { trace in
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

    private static func isActionStructuredStage(_ trace: AgentBehaviorTrace) -> Bool {
        let stage = trace.stage.lowercased()
        if stage.contains("mouth") || stage.contains("final") || stage.contains("direct") {
            return false
        }
        return stage.contains("json") || stage.contains("orchestrator") || stage.contains("executor") || trace.slot.lowercased() == "cortex"
    }

    private static func safeTimestamp(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
            .replacingOccurrences(of: ":", with: "-")
            .replacingOccurrences(of: ".", with: "-")
    }
}
