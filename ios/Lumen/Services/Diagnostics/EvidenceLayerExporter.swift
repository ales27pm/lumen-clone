import Foundation
import CryptoKit

nonisolated struct EvidenceLayerExportPolicy: Codable, Sendable, Hashable {
    let format: String
    let sourceLayer: String
    let ownsLiveE2EScenarios: Bool
    let includesDeterministicStaticScenarios: Bool
    let privacy: String
    let notes: [String]
}

nonisolated struct EvidenceLayerEnvelope<Payload: Encodable>: Encodable {
    let schemaVersion: String
    let generatedAt: Date
    let app: InAppDatasetAppInfo
    let exportPolicy: EvidenceLayerExportPolicy
    let payload: Payload
}

nonisolated struct EvidenceLayerExportResult<Payload: Encodable> {
    let url: URL
    let envelope: EvidenceLayerEnvelope<Payload>
}

nonisolated enum EvidenceLayerExporter {
    static let schemaVersion = "1.0.0"
    private static let directoryName = "LumenEvidenceLayerExports"

    static func writeLiveE2EReport(
        _ report: E2ETestReport
    ) throws -> EvidenceLayerExportResult<E2ETestReport> {
        try purgeLegacyLiveE2EExports()
        let redactedReport = privacySafeE2EReportForExport(report)
        return try writeLayer(
            payload: redactedReport,
            filePrefix: "lumen-live-e2e-report-redacted-v1",
            format: "live-e2e-test-report-json",
            sourceLayer: "e2eTestReport",
            ownsLiveE2EScenarios: true,
            includesDeterministicStaticScenarios: report.results.contains { !$0.requiresAgentRun },
            privacy: "Privacy-redacted live E2E evidence. Free-form content is replaced by SHA-256 digests and character counts before export.",
            notes: [
                "This is the privacy-safe live E2E model/test layer export.",
                "Raw prompts, final outputs, failures, event messages, output prefixes, and arbitrary metadata values are not exported.",
                "modelBackedRequired scenarios must exercise the AssistantKernel model-backed structured generation path and record fresh AgentBehaviorTrace modelTurn evidence.",
                "Routing-only tool coverage scenarios are static guard checks; missing model evidence remains invalid E2E evidence."
            ]
        )
    }

    static func writeBehaviorAuditReport(
        _ report: AgentBehaviorAuditReport
    ) throws -> EvidenceLayerExportResult<AgentBehaviorAuditReport> {
        let redactedReport = privacySafeBehaviorAuditForExport(report)
        return try writeLayer(
            payload: redactedReport,
            filePrefix: "lumen-model-behaviour-audit-redacted-v1",
            format: "agent-model-behaviour-audit-json",
            sourceLayer: "agentModelBehaviorAuditor",
            ownsLiveE2EScenarios: false,
            includesDeterministicStaticScenarios: false,
            privacy: "Privacy-redacted behavior-audit evidence. Message-derived and other free-form fields are replaced by SHA-256 digests and character counts before export.",
            notes: [
                "Audits recent persisted app messages and model behavior violations without exporting raw ChatMessage content.",
                "Expected, actual, prompt, problem, recommendation, and repair-sample text is one-way summarized.",
                "Use this for drift metrics and privacy-safe diagnostics, not as training data or an E2E scenario result."
            ]
        )
    }

    static func writeRuntimeManifestAuditReport(
        _ report: RuntimeAgentManifestAuditReport
    ) throws -> EvidenceLayerExportResult<RuntimeAgentManifestAuditReport> {
        let redactedReport = InAppDatasetPackageExporter.privacySafeRuntimeManifestAuditForExport(report)
        return try writeLayer(
            payload: redactedReport,
            filePrefix: "lumen-runtime-registry-audit-redacted-v1",
            format: "runtime-registry-audit-json",
            sourceLayer: "runtimeManifestAudit",
            ownsLiveE2EScenarios: false,
            includesDeterministicStaticScenarios: false,
            privacy: "Privacy-redacted runtime registry audit. Free-form failure and repair fields are replaced by SHA-256 summaries before export.",
            notes: [
                "Compares AgentBehaviorManifest.json against the live runtime tool registry.",
                "Failure details and repair recommendations are structural hash summaries, not training data.",
                "Does not run model scenarios."
            ]
        )
    }

    static func writeStaticScenarioResults(
        _ results: [RuntimeScenarioResult]
    ) throws -> EvidenceLayerExportResult<[RuntimeScenarioResult]> {
        let redactedResults = InAppDatasetPackageExporter.privacySafeScenarioResultsForExport(results)
        return try writeLayer(
            payload: redactedResults,
            filePrefix: "lumen-static-scenario-checks-redacted-v1",
            format: "deterministic-static-scenario-checks-json",
            sourceLayer: "runtimeScenarioRunner.staticChecks",
            ownsLiveE2EScenarios: false,
            includesDeterministicStaticScenarios: true,
            privacy: "Privacy-redacted deterministic scenario checks. Prompts, scenario identifiers, and failure details are replaced by SHA-256 summaries before export.",
            notes: [
                "Deterministic manifest sanity checks only.",
                "Does not run the model and must not be treated as live E2E evidence."
            ]
        )
    }

    static func writeAgentBehaviorTraces(
        _ traces: [AgentBehaviorTrace]
    ) throws -> EvidenceLayerExportResult<[AgentBehaviorTrace]> {
        let redactedTraces = traces.map { $0.redactedForPersistentDiagnostics() }
        return try writeLayer(
            payload: redactedTraces,
            filePrefix: "lumen-agent-runtime-traces-redacted-v1",
            format: "agent-runtime-traces-json",
            sourceLayer: "agentBehaviorTraceRecorder",
            ownsLiveE2EScenarios: false,
            includesDeterministicStaticScenarios: false,
            privacy: "Privacy-redacted runtime traces. Free-form content and raw run, agent, conversation, turn, and scenario identifiers are omitted or one-way summarized.",
            notes: [
                "Bounded recent traces captured by AgentBehaviorTraceRecorder.",
                "Empty exports indicate the recorder is not wired or no real model interactions were exercised."
            ]
        )
    }

    private static func writeLayer<Payload: Encodable>(
        payload: Payload,
        filePrefix: String,
        format: String,
        sourceLayer: String,
        ownsLiveE2EScenarios: Bool,
        includesDeterministicStaticScenarios: Bool,
        privacy: String,
        notes: [String]
    ) throws -> EvidenceLayerExportResult<Payload> {
        let envelope = EvidenceLayerEnvelope(
            schemaVersion: schemaVersion,
            generatedAt: Date(),
            app: appInfo(),
            exportPolicy: EvidenceLayerExportPolicy(
                format: format,
                sourceLayer: sourceLayer,
                ownsLiveE2EScenarios: ownsLiveE2EScenarios,
                includesDeterministicStaticScenarios: includesDeterministicStaticScenarios,
                privacy: privacy,
                notes: notes
            ),
            payload: payload
        )
        let directory = try exportDirectory()
        try purgeLegacyUnsafeArtifacts(in: directory)
        let safePrefix = sanitizeFilePrefix(filePrefix)
        let url = directory.appendingPathComponent("\(safePrefix)-\(safeTimestamp(envelope.generatedAt))-\(UUID().uuidString.lowercased()).json", isDirectory: false)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(envelope).write(to: url, options: [.atomic, .completeFileProtection])
        return EvidenceLayerExportResult(url: url, envelope: envelope)
    }

    static func exportDirectory() throws -> URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first ?? FileManager.default.temporaryDirectory
        let directory = base
            .appendingPathComponent("Diagnostics", isDirectory: true)
            .appendingPathComponent(directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func appInfo() -> InAppDatasetAppInfo {
        .current()
    }

    private static func sanitizeFilePrefix(_ value: String) -> String {
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        let scalars = value.unicodeScalars.map { allowed.contains($0) ? Character($0) : "-" }
        let collapsed = String(scalars)
            .replacingOccurrences(of: "--+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-_"))
        return collapsed.isEmpty ? "lumen-evidence-layer" : collapsed.lowercased()
    }

    private static func safeTimestamp(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
            .replacingOccurrences(of: ":", with: "-")
            .replacingOccurrences(of: ".", with: "-")
    }

    static func privacySafeE2EReportForExport(_ report: E2ETestReport) -> E2ETestReport {
        let identityKey = SymmetricKey(size: .bits256)
        return E2ETestReport(
            id: privacySafeUUID(report.id, key: identityKey, domain: "report"),
            startedAt: report.startedAt,
            finishedAt: report.finishedAt,
            passed: report.passed,
            failed: report.failed,
            results: report.results.map {
                privacySafeE2EResultForExport($0, identityKey: identityKey)
            }
        )
    }

    static func privacySafeBehaviorAuditForExport(
        _ report: AgentBehaviorAuditReport
    ) -> AgentBehaviorAuditReport {
        AgentBehaviorAuditReport(
            passed: report.passed,
            score: report.score,
            generatedAt: report.generatedAt,
            traceCount: report.traceCount,
            violationCount: report.violationCount,
            sourceCommit: report.sourceCommit.map(privacySafeRevision),
            violations: report.violations.map { violation in
                AgentBehaviorViolation(
                    id: UUID(),
                    createdAt: violation.createdAt,
                    severity: violation.severity,
                    code: safeIdentifier(violation.code),
                    agent: safeIdentifier(violation.agent),
                    expected: redactedText(violation.expected),
                    actual: redactedText(violation.actual),
                    promptPrefix: redactedText(violation.promptPrefix),
                    problem: redactedText(violation.problem)
                )
            },
            recommendations: report.recommendations.map(redactedText),
            repairSamples: report.repairSamples.map { sample in
                AgentBehaviorRepairSample(
                    id: UUID(),
                    createdAt: sample.createdAt,
                    agent: safeIdentifier(sample.agent),
                    violationCode: safeIdentifier(sample.violationCode),
                    promptPrefix: redactedText(sample.promptPrefix),
                    expected: redactedText(sample.expected),
                    badOutput: redactedText(sample.badOutput),
                    correctedOutput: redactedText(sample.correctedOutput),
                    lesson: redactedText(sample.lesson),
                    curriculum: redactedText(sample.curriculum)
                )
            }
        )
    }

    static func privacySafeE2EResultForExport(_ result: E2ETestResult) -> E2ETestResult {
        privacySafeE2EResultForExport(
            result,
            identityKey: SymmetricKey(size: .bits256)
        )
    }

    private static func privacySafeE2EResultForExport(
        _ result: E2ETestResult,
        identityKey: SymmetricKey
    ) -> E2ETestResult {
        E2ETestResult(
            id: privacySafeUUID(result.id, key: identityKey, domain: "result"),
            scenarioID: safeIdentifier(result.scenarioID),
            kind: safeIdentifier(result.kind),
            title: redactedText(result.title),
            prompt: redactedText(result.prompt),
            expectedIntent: safeIdentifier(result.expectedIntent),
            actualIntent: safeIdentifier(result.actualIntent),
            e2eRunID: nil,
            agentRunID: nil,
            conversationID: nil,
            turnID: nil,
            correlationToken: privacySafeCorrelationToken(result, key: identityKey),
            requiresAgentRun: result.requiresAgentRun,
            evidenceMode: safeIdentifier(result.evidenceMode),
            passed: result.passed,
            failures: result.failures.map(redactedText),
            finalText: redactedText(result.finalText),
            missingHints: result.missingHints.map(redactedText),
            rewriteAttempted: result.rewriteAttempted,
            rewriteSuccess: result.rewriteSuccess,
            events: result.events.map {
                E2ETestEvent(
                    id: privacySafeUUID($0.id, key: identityKey, domain: "event"),
                    createdAt: $0.createdAt,
                    scenarioID: safeIdentifier($0.scenarioID),
                    phase: safeIdentifier($0.phase),
                    message: redactedText($0.message)
                )
            },
            startedAt: result.startedAt,
            finishedAt: result.finishedAt,
            rawFinalPrefix: redactedText(result.rawFinalPrefix),
            sanitizedFinalPrefix: redactedText(result.sanitizedFinalPrefix),
            rawFinalHadUnsafeLeakage: result.rawFinalHadUnsafeLeakage,
            sanitizedFinalRemovedArtifacts: result.sanitizedFinalRemovedArtifacts.map(safeIdentifier),
            outputHygieneFailures: result.outputHygieneFailures.map(redactedText),
            performanceMatrix: result.performanceMatrix.map(privacySafePerformanceMatrix),
            metadata: privacySafeMetadata(result.metadata, identityKey: identityKey)
        )
    }

    private static func privacySafePerformanceMatrix(_ matrix: E2EPerformanceMatrix) -> E2EPerformanceMatrix {
        E2EPerformanceMatrix(
            aneUtilizationPercent: matrix.aneUtilizationPercent,
            eventDensityCPUProxyPercent: matrix.eventDensityCPUProxyPercent,
            gpuUtilizationPercent: matrix.gpuUtilizationPercent,
            peakRAMMB: matrix.peakRAMMB,
            averageRAMMB: matrix.averageRAMMB,
            sampleCount: matrix.sampleCount,
            notes: matrix.notes.map(redactedText),
            accelerationDiagnostics: nil
        )
    }

    private static func privacySafeMetadata(
        _ metadata: [String: String],
        identityKey: SymmetricKey
    ) -> [String: String] {
        let safeCategoryKeys: Set<String> = [
            "actionable",
            "expectedToolID",
            "failureKind",
            "missingAdapterSlots",
            "readyArtifactCount",
            "remediationApplied",
            "requiredArtifactCount",
            "requiredSlots",
            "runtimeEvidence",
            "scenarioBankKind",
            "toolFailureCode",
            "trainingSignal"
        ]
        var redacted: [String: String] = ["privacyRedacted": "true"]
        for (key, value) in metadata {
            if safeCategoryKeys.contains(key) {
                redacted[key] = safeIdentifier(value)
            } else {
                let opaqueKey = opaqueMetadataKey(key, identityKey: identityKey)
                redacted[opaqueKey] = redactedText(value)
            }
        }
        return redacted
    }

    private static func safeIdentifier(_ value: String) -> String {
        guard !containsSensitiveIdentifier(value) else {
            return redactedText(value)
        }
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-,")
        let normalized = String(value.unicodeScalars.map { allowed.contains($0) ? Character($0) : "_" })
        return String(normalized.prefix(160))
    }

    private static func containsSensitiveIdentifier(_ value: String) -> Bool {
        let patterns = [
            #"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"#,
            #"\b\d{3}-\d{2}-\d{4}\b"#,
            #"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"#,
            #"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"#
        ]
        return patterns.contains { pattern in
            value.range(of: pattern, options: [.regularExpression, .caseInsensitive]) != nil
        }
    }

    private static func opaqueMetadataKey(
        _ value: String,
        identityKey: SymmetricKey
    ) -> String {
        let digest = HMAC<SHA256>.authenticationCode(
            for: Data(value.utf8),
            using: identityKey
        )
        return "metadata_" + digest.prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    private static func redactedText(_ value: String) -> String {
        guard !value.isEmpty else { return "" }
        return "[redacted sha256=\(RuntimeFallbackLogger.promptHash(value)) chars=\(value.count)]"
    }

    private static func privacySafeRevision(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        let isHexRevision = (7...64).contains(trimmed.count) && trimmed.allSatisfy(\.isHexDigit)
        return isHexRevision ? trimmed.lowercased() : redactedText(trimmed)
    }

    private static func privacySafeCorrelationToken(
        _ result: E2ETestResult,
        key: SymmetricKey
    ) -> String? {
        // Never trust or preserve a caller-provided token. Only the raw,
        // high-entropy in-memory run identifiers may seed this per-write HMAC.
        let seed = [result.e2eRunID, result.agentRunID, result.conversationID, result.turnID]
            .compactMap { $0?.uuidString }
            .joined(separator: "|")
        guard !seed.isEmpty else { return nil }
        let digest = HMAC<SHA256>.authenticationCode(for: Data(seed.utf8), using: key)
        return "corr_hash_v2_" + digest.prefix(16).map { String(format: "%02x", $0) }.joined()
    }

    private static func privacySafeUUID(
        _ value: UUID,
        key: SymmetricKey,
        domain: String
    ) -> UUID {
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

    static func purgeLegacyLiveE2EExports() throws {
        let directory = try exportDirectory()
        try purgeLegacyUnsafeArtifacts(in: directory)
    }

    private static func purgeLegacyUnsafeArtifacts(in directory: URL) throws {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        let rules: [(legacyPrefix: String, safePrefix: String)] = [
            ("lumen-live-e2e-report-", "lumen-live-e2e-report-redacted-v1-"),
            ("lumen-model-behaviour-audit-", "lumen-model-behaviour-audit-redacted-v1-"),
            ("lumen-agent-runtime-traces-", "lumen-agent-runtime-traces-redacted-v1-"),
            ("lumen-runtime-registry-audit-", "lumen-runtime-registry-audit-redacted-v1-"),
            ("lumen-static-scenario-checks-", "lumen-static-scenario-checks-redacted-v1-")
        ]
        for url in files {
            let name = url.lastPathComponent
            for rule in rules where name.hasPrefix(rule.legacyPrefix) && !name.hasPrefix(rule.safePrefix) {
                try FileManager.default.removeItem(at: url)
                break
            }
        }
    }
}
