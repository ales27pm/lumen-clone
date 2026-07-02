import Foundation

nonisolated enum ImproveLoopSampleDisposition: String, Codable, Sendable, Hashable {
    case acceptedTraining = "accepted_training"
    case quarantined = "quarantined"
    case regressionTest = "regression_test"
}

nonisolated enum ImproveLoopSampleType: String, Codable, Sendable, Hashable {
    case cortexJSONContract = "cortex_json_contract"
    case executorArgumentExtraction = "executor_argument_extraction"
    case mouthObservationFinal = "mouth_observation_final"
    case remTraceCompression = "rem_trace_compression"
    case deterministicRegression = "deterministic_regression"
    case rejectedStaleTrace = "rejected_stale_trace"
    case rejectedLegacyToolNamespace = "rejected_legacy_tool_namespace"
    case rejectedArchitectureFailure = "rejected_architecture_failure"
    case rejectedResourceFallback = "rejected_resource_fallback"
    case rejectedLatencyOnly = "rejected_latency_only"
    case rejectedUncorrectedModelOutput = "rejected_uncorrected_model_output"
}

nonisolated enum ImproveLoopSampleSource: String, Codable, Sendable, Hashable {
    case behaviorRepairSample = "behavior_repair_sample"
    case behaviorTrace = "behavior_trace"
    case staticScenario = "static_scenario"
    case runtimeViolation = "runtime_violation"
}

nonisolated enum ImproveLoopSampleAuthority: String, Codable, Sendable, Hashable {
    case deterministic = "deterministic"
    case modelGenerated = "model_generated"
    case userConfirmed = "user_confirmed"
    case auditDerived = "audit_derived"
}

nonisolated struct ImproveLoopTrainingSample: Codable, Sendable, Identifiable, Hashable {
    let id: UUID
    let createdAt: Date
    let disposition: ImproveLoopSampleDisposition
    let sampleType: ImproveLoopSampleType
    let source: ImproveLoopSampleSource
    let authority: ImproveLoopSampleAuthority
    let slot: String
    let intent: String?
    let promptPrefix: String
    let expected: String
    let badOutput: String
    let correctedOutput: String
    let lesson: String
    let rejectionReason: String?
    let canonicalToolID: String?
    let allowedToolIDs: [String]
    let sourceCommit: String?
    let sourceTraceID: UUID?
}

nonisolated struct ImproveLoopDatasetCounters: Codable, Sendable, Hashable {
    let accepted: Int
    let quarantined: Int
    let regression: Int
    let staleTraceRejected: Int
    let legacyToolNamespaceRejected: Int
    let architectureFailureRejected: Int
    let resourceFallbackRejected: Int
}

nonisolated struct ImproveLoopDataset: Codable, Sendable, Hashable {
    let schemaVersion: String
    let generatedAt: Date
    let acceptedTraining: [ImproveLoopTrainingSample]
    let quarantinedSamples: [ImproveLoopTrainingSample]
    let regressionTests: [ImproveLoopTrainingSample]
    let counters: ImproveLoopDatasetCounters
}

nonisolated enum ImproveLoopSampleGate {
    static let schemaVersion = "2026.06.07-policy-first-adapter-specialized"

    static func buildDataset(
        behaviorAudit: AgentBehaviorAuditReport?,
        traces: [AgentBehaviorTrace],
        scenarioResults: [RuntimeScenarioResult],
        sourceCommit: String?
    ) -> ImproveLoopDataset {
        var accepted: [ImproveLoopTrainingSample] = []
        var quarantined: [ImproveLoopTrainingSample] = []
        var regressions: [ImproveLoopTrainingSample] = []

        for sample in behaviorAudit?.repairSamples ?? [] {
            let gated = gateRepairSample(sample, sourceCommit: behaviorAudit?.sourceCommit ?? sourceCommit)
            append(gated, accepted: &accepted, quarantined: &quarantined, regressions: &regressions)
        }

        for violation in behaviorAudit?.violations ?? [] {
            let gated = gateViolation(violation, sourceCommit: behaviorAudit?.sourceCommit ?? sourceCommit)
            append(gated, accepted: &accepted, quarantined: &quarantined, regressions: &regressions)
        }

        for trace in traces {
            let gated = gateTrace(trace, sourceCommit: sourceCommit)
            append(gated, accepted: &accepted, quarantined: &quarantined, regressions: &regressions)
        }

        for result in scenarioResults where !result.passed {
            let gated = gateStaticScenario(result, sourceCommit: sourceCommit)
            append(gated, accepted: &accepted, quarantined: &quarantined, regressions: &regressions)
        }

        let counters = ImproveLoopDatasetCounters(
            accepted: accepted.count,
            quarantined: quarantined.count,
            regression: regressions.count,
            staleTraceRejected: quarantined.filter { $0.sampleType == .rejectedStaleTrace }.count,
            legacyToolNamespaceRejected: quarantined.filter { $0.sampleType == .rejectedLegacyToolNamespace }.count,
            architectureFailureRejected: quarantined.filter { $0.sampleType == .rejectedArchitectureFailure }.count,
            resourceFallbackRejected: quarantined.filter { $0.sampleType == .rejectedResourceFallback }.count
        )

        return ImproveLoopDataset(
            schemaVersion: schemaVersion,
            generatedAt: Date(),
            acceptedTraining: accepted.sorted(by: sortSamples),
            quarantinedSamples: quarantined.sorted(by: sortSamples),
            regressionTests: regressions.sorted(by: sortSamples),
            counters: counters
        )
    }

    private static func append(
        _ sample: ImproveLoopTrainingSample,
        accepted: inout [ImproveLoopTrainingSample],
        quarantined: inout [ImproveLoopTrainingSample],
        regressions: inout [ImproveLoopTrainingSample]
    ) {
        switch sample.disposition {
        case .acceptedTraining:
            accepted.append(sample)
        case .quarantined:
            quarantined.append(sample)
        case .regressionTest:
            regressions.append(sample)
        }
    }

    private static func gateRepairSample(_ sample: AgentBehaviorRepairSample, sourceCommit: String?) -> ImproveLoopTrainingSample {
        let combined = [sample.promptPrefix, sample.expected, sample.badOutput, sample.correctedOutput, sample.lesson, sample.curriculum].joined(separator: "\n")
        if let rejection = rejectionReason(for: combined) {
            return makeSample(
                disposition: .quarantined,
                sampleType: sampleType(forRejection: rejection),
                source: .behaviorRepairSample,
                authority: .auditDerived,
                slot: sample.agent,
                intent: nil,
                promptPrefix: sample.promptPrefix,
                expected: sample.expected,
                badOutput: sample.badOutput,
                correctedOutput: sample.correctedOutput,
                lesson: sample.lesson,
                rejectionReason: rejection,
                canonicalToolID: canonicalToolID(from: sample.correctedOutput),
                allowedToolIDs: [],
                sourceCommit: sourceCommit,
                sourceTraceID: nil
            )
        }

        if sample.correctedOutput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return makeSample(
                disposition: .quarantined,
                sampleType: .rejectedUncorrectedModelOutput,
                source: .behaviorRepairSample,
                authority: .auditDerived,
                slot: sample.agent,
                intent: nil,
                promptPrefix: sample.promptPrefix,
                expected: sample.expected,
                badOutput: sample.badOutput,
                correctedOutput: "",
                lesson: sample.lesson,
                rejectionReason: "repair sample has no corrected output",
                canonicalToolID: nil,
                allowedToolIDs: [],
                sourceCommit: sourceCommit,
                sourceTraceID: nil
            )
        }

        return makeSample(
            disposition: .acceptedTraining,
            sampleType: sampleType(forAgent: sample.agent, violationCode: sample.violationCode),
            source: .behaviorRepairSample,
            authority: .auditDerived,
            slot: sample.agent,
            intent: nil,
            promptPrefix: sample.promptPrefix,
            expected: sample.expected,
            badOutput: sample.badOutput,
            correctedOutput: canonicalizeToolAliases(in: sample.correctedOutput),
            lesson: sample.lesson,
            rejectionReason: nil,
            canonicalToolID: canonicalToolID(from: sample.correctedOutput),
            allowedToolIDs: [],
            sourceCommit: sourceCommit,
            sourceTraceID: nil
        )
    }

    private static func gateViolation(_ violation: AgentBehaviorViolation, sourceCommit: String?) -> ImproveLoopTrainingSample {
        let combined = [violation.promptPrefix, violation.expected, violation.actual, violation.problem].joined(separator: "\n")
        let rejection = rejectionReason(for: combined)
        let isArchitectureFailure = violation.code == "missing_required_tool_action"
            || violation.code.hasPrefix("tool_not_allowed")
            || violation.problem.lowercased().contains("old model-loop")
        let disposition: ImproveLoopSampleDisposition = rejection == nil && !isArchitectureFailure ? .regressionTest : .quarantined
        let type: ImproveLoopSampleType = rejection.map(sampleType(forRejection:))
            ?? (isArchitectureFailure ? .rejectedArchitectureFailure : .deterministicRegression)

        return makeSample(
            disposition: disposition,
            sampleType: type,
            source: .runtimeViolation,
            authority: .auditDerived,
            slot: violation.agent,
            intent: nil,
            promptPrefix: violation.promptPrefix,
            expected: violation.expected,
            badOutput: violation.actual,
            correctedOutput: "",
            lesson: violation.problem,
            rejectionReason: disposition == .quarantined ? (rejection ?? "architecture/runtime failure; convert to regression before training") : nil,
            canonicalToolID: canonicalToolID(from: violation.expected),
            allowedToolIDs: canonicalToolIDs(in: violation.expected),
            sourceCommit: sourceCommit,
            sourceTraceID: nil
        )
    }

    private static func gateTrace(_ trace: AgentBehaviorTrace, sourceCommit: String?) -> ImproveLoopTrainingSample {
        let combined = [trace.promptPrefix, trace.rawOutputPrefix, trace.selectedToolID ?? "", trace.parseError ?? ""].joined(separator: "\n")
        if let rejection = rejectionReason(for: combined) {
            return makeTraceSample(trace, disposition: .quarantined, type: sampleType(forRejection: rejection), rejectionReason: rejection, sourceCommit: sourceCommit)
        }
        if isResourceFallback(combined) {
            return makeTraceSample(trace, disposition: .quarantined, type: .rejectedResourceFallback, rejectionReason: "resource fallback is runtime policy, not model training data", sourceCommit: sourceCommit)
        }
        if trace.parseError != nil {
            return makeTraceSample(trace, disposition: .regressionTest, type: .cortexJSONContract, rejectionReason: nil, sourceCommit: sourceCommit)
        }
        if trace.event == .toolAction {
            return makeTraceSample(trace, disposition: .regressionTest, type: .deterministicRegression, rejectionReason: nil, sourceCommit: sourceCommit)
        }
        return makeTraceSample(trace, disposition: .quarantined, type: .rejectedUncorrectedModelOutput, rejectionReason: "raw trace has no trusted corrected output", sourceCommit: sourceCommit)
    }

    private static func makeTraceSample(
        _ trace: AgentBehaviorTrace,
        disposition: ImproveLoopSampleDisposition,
        type: ImproveLoopSampleType,
        rejectionReason: String?,
        sourceCommit: String?
    ) -> ImproveLoopTrainingSample {
        makeSample(
            disposition: disposition,
            sampleType: type,
            source: .behaviorTrace,
            authority: .modelGenerated,
            slot: trace.slot,
            intent: trace.intent,
            promptPrefix: trace.promptPrefix,
            expected: trace.allowedToolIDs.map(ToolRouteGuard.canonicalToolID).sorted().joined(separator: ","),
            badOutput: trace.rawOutputPrefix,
            correctedOutput: "",
            lesson: trace.parseError ?? trace.stage,
            rejectionReason: rejectionReason,
            canonicalToolID: trace.selectedToolID.map(ToolRouteGuard.canonicalToolID),
            allowedToolIDs: trace.allowedToolIDs.map(ToolRouteGuard.canonicalToolID).sorted(),
            sourceCommit: sourceCommit,
            sourceTraceID: trace.id
        )
    }

    private static func gateStaticScenario(_ result: RuntimeScenarioResult, sourceCommit: String?) -> ImproveLoopTrainingSample {
        let failures = result.failures.map { "\($0.type): \($0.problem)" }.joined(separator: "\n")
        return makeSample(
            disposition: .regressionTest,
            sampleType: .deterministicRegression,
            source: .staticScenario,
            authority: .deterministic,
            slot: "manifest",
            intent: result.scenario.intent,
            promptPrefix: result.scenario.prompt,
            expected: result.scenario.expectedToolID,
            badOutput: failures,
            correctedOutput: "",
            lesson: "Preserve deterministic manifest scenario failure as regression input.",
            rejectionReason: nil,
            canonicalToolID: ToolRouteGuard.canonicalToolID(result.scenario.expectedToolID),
            allowedToolIDs: [ToolRouteGuard.canonicalToolID(result.scenario.expectedToolID)],
            sourceCommit: sourceCommit,
            sourceTraceID: nil
        )
    }

    private static func makeSample(
        disposition: ImproveLoopSampleDisposition,
        sampleType: ImproveLoopSampleType,
        source: ImproveLoopSampleSource,
        authority: ImproveLoopSampleAuthority,
        slot: String,
        intent: String?,
        promptPrefix: String,
        expected: String,
        badOutput: String,
        correctedOutput: String,
        lesson: String,
        rejectionReason: String?,
        canonicalToolID: String?,
        allowedToolIDs: [String],
        sourceCommit: String?,
        sourceTraceID: UUID?
    ) -> ImproveLoopTrainingSample {
        ImproveLoopTrainingSample(
            id: UUID(),
            createdAt: Date(),
            disposition: disposition,
            sampleType: sampleType,
            source: source,
            authority: authority,
            slot: slot,
            intent: intent,
            promptPrefix: canonicalizeToolAliases(in: bounded(promptPrefix)),
            expected: canonicalizeToolAliases(in: bounded(expected)),
            badOutput: canonicalizeToolAliases(in: bounded(badOutput)),
            correctedOutput: canonicalizeToolAliases(in: bounded(correctedOutput)),
            lesson: bounded(lesson),
            rejectionReason: rejectionReason,
            canonicalToolID: canonicalToolID.map(ToolRouteGuard.canonicalToolID),
            allowedToolIDs: Array(Set(allowedToolIDs.map(ToolRouteGuard.canonicalToolID))).sorted(),
            sourceCommit: sourceCommit,
            sourceTraceID: nil
        )
    }

    private static func rejectionReason(for text: String) -> String? {
        let lower = text.lowercased()
        if containsLegacyToolAlias(lower) {
            return "legacy tool namespace present; canonicalize or reject before training"
        }
        if lower.contains("lumen_grounding_v1") {
            return "stale generated grounding marker present in training candidate"
        }
        if isResourceFallback(lower) {
            return "resource fallback is runtime policy, not model behavior"
        }
        if lower.contains("tool output could not be validated") {
            return "final validator fallback is architecture/runtime feedback, not direct fine-tuning data"
        }
        if isRAGEmptyRetrieval(lower) {
            return "empty local retrieval index is fixture/runtime state, not direct fine-tuning data"
        }
        if containsInternalRoutingJSON(lower) {
            return "internal routing JSON leakage is architecture/runtime feedback, not direct fine-tuning data"
        }
        return nil
    }

    private static func sampleType(forRejection reason: String) -> ImproveLoopSampleType {
        if reason.contains("legacy tool namespace") { return .rejectedLegacyToolNamespace }
        if reason.contains("stale generated grounding") { return .rejectedStaleTrace }
        if reason.contains("resource fallback") { return .rejectedResourceFallback }
        return .rejectedArchitectureFailure
    }

    private static func sampleType(forAgent agent: String, violationCode: String) -> ImproveLoopSampleType {
        let lowerAgent = agent.lowercased()
        let lowerCode = violationCode.lowercased()
        if lowerAgent.contains("mouth") || lowerCode.contains("final") { return .mouthObservationFinal }
        if lowerAgent.contains("executor") { return .executorArgumentExtraction }
        if lowerAgent.contains("rem") { return .remTraceCompression }
        return .cortexJSONContract
    }

    private static func canonicalToolID(from text: String) -> String? {
        canonicalToolIDs(in: text).first
    }

    private static func canonicalToolIDs(in text: String) -> [String] {
        let tokens = text
            .components(separatedBy: CharacterSet(charactersIn: " \n\t,;()[]{}<>\"'"))
            .filter { $0.contains(".") }
            .map { ToolRouteGuard.canonicalToolID($0.trimmingCharacters(in: .punctuationCharacters)) }
        return Array(Set(tokens)).sorted()
    }

    static func canonicalizeToolAliases(in text: String) -> String {
        legacyToolAliasMap.reduce(text) { partial, entry in
            partial.replacingOccurrences(of: entry.key, with: entry.value)
        }
    }

    private static func containsLegacyToolAlias(_ lower: String) -> Bool {
        legacyToolAliasMap.keys.contains { lower.contains($0) }
    }

    private static func isResourceFallback(_ text: String) -> Bool {
        let lower = text.lowercased()
        return lower.contains("can’t safely start the full agent pipeline")
            || lower.contains("can't safely start the full agent pipeline")
            || lower.contains("device has cooled down")
            || lower.contains("resource-budget-denied-before-prompt-eval")
            || lower.contains("cpu-watchdog-degraded")
            || lower.contains("liveruntimecpuwatchdogdegraded")
            || lower.contains("live e2e preflight blocked model-backed generation before prompt evaluation")
            || lower.contains("alarmkit runtime unavailable")
            || lower.contains("alarmkit availability: unavailable")
            || lower.contains("device-runtime evidence required")
            || lower.contains("scenephase=inactive")
            || lower.contains("scenephase=background")
            || lower.contains("thermalstate=serious")
            || lower.contains("thermalstate=critical")
    }

    private static func isRAGEmptyRetrieval(_ lower: String) -> Bool {
        lower.contains("no matching files found")
            || lower.contains("local index appears empty")
            || lower.contains("no matching local snippets")
            || lower.contains("import or create local files")
    }

    private static func containsInternalRoutingJSON(_ lower: String) -> Bool {
        RoutingJSONLeakDetector.containsInternalRoutingJSON(lower)
    }

    private static let legacyToolAliasMap: [String: String] = [
        "contacts.lookup": "contacts.search",
        "memory.search": "memory.recall",
        "calendar.read": "calendar.list",
        "rag.search.secure": "rag.search"
    ]

    private static func bounded(_ value: String, limit: Int = 1_200) -> String {
        let stripped = ModelOutputSanitizer.stripHiddenBlocks(value)
        let redacted = PersistentRuntimeDiagnosticsRedactor.redactWithoutTruncating(stripped)
        let trimmed = redacted.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > limit else { return trimmed }
        return String(trimmed.prefix(limit))
    }

    private static func sortSamples(_ lhs: ImproveLoopTrainingSample, _ rhs: ImproveLoopTrainingSample) -> Bool {
        if lhs.sampleType.rawValue == rhs.sampleType.rawValue {
            return lhs.createdAt > rhs.createdAt
        }
        return lhs.sampleType.rawValue < rhs.sampleType.rawValue
    }
}
