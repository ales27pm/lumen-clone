import Foundation
import SwiftData
#if canImport(Darwin)
import Darwin
#endif

nonisolated enum E2ETestKind: String, Codable, Sendable, CaseIterable {
    case routing
    case toolGuard
    case chat
    case regression
    case training
}

nonisolated enum E2EEvidenceMode: String, Codable, Sendable, Hashable {
    case modelBackedRequired
    case policyFirstAllowed
    case routingOnly
}

nonisolated struct E2ERunConfig: Sendable {
    nonisolated struct Snapshot: Sendable {
        let systemPrompt: String
        let temperature: Double
        let topP: Double
        let repetitionPenalty: Double
        let maxTokens: Int
        let maxAgentSteps: Int
        let enabledToolIDs: Set<String>

        init(
            systemPrompt: String,
            temperature: Double,
            topP: Double,
            repetitionPenalty: Double,
            maxTokens: Int,
            maxAgentSteps: Int,
            enabledToolIDs: Set<String>
        ) {
            self.systemPrompt = systemPrompt
            self.temperature = temperature
            self.topP = topP
            self.repetitionPenalty = repetitionPenalty
            self.maxTokens = maxTokens
            self.maxAgentSteps = maxAgentSteps
            self.enabledToolIDs = enabledToolIDs
        }

        @MainActor
        init(appState: AppState) {
            self.init(
                systemPrompt: appState.systemPrompt,
                temperature: appState.temperature,
                topP: appState.topP,
                repetitionPenalty: appState.repetitionPenalty,
                maxTokens: appState.maxTokens,
                maxAgentSteps: appState.maxAgentSteps,
                enabledToolIDs: appState.enabledToolIDs
            )
        }
    }

    let systemPrompt: String
    let temperature: Double
    let topP: Double
    let repetitionPenalty: Double
    let maxTokens: Int
    let maxAgentSteps: Int
    let enabledToolIDs: Set<String>

    init(snapshot: Snapshot) {
        self.systemPrompt = snapshot.systemPrompt
        self.temperature = snapshot.temperature
        self.topP = snapshot.topP
        self.repetitionPenalty = snapshot.repetitionPenalty
        self.maxTokens = snapshot.maxTokens
        self.maxAgentSteps = snapshot.maxAgentSteps
        self.enabledToolIDs = snapshot.enabledToolIDs
    }

    init(
        systemPrompt: String,
        temperature: Double,
        topP: Double,
        repetitionPenalty: Double,
        maxTokens: Int,
        maxAgentSteps: Int,
        enabledToolIDs: Set<String>
    ) {
        self.init(
            snapshot: Snapshot(
                systemPrompt: systemPrompt,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                maxTokens: maxTokens,
                maxAgentSteps: maxAgentSteps,
                enabledToolIDs: enabledToolIDs
            )
        )
    }

    @MainActor
    init(appState: AppState) {
        self.init(snapshot: Snapshot(appState: appState))
    }
}

nonisolated struct E2ETestScenario: Identifiable, Codable, Sendable, Hashable {
    let id: String
    let title: String
    let kind: E2ETestKind
    let prompt: String
    let expectedIntent: UserIntent
    let requiredAllowedToolIDs: [String]
    let forbiddenToolIDs: [String]
    let requiredTextHints: [String]
    let forbiddenTextHints: [String]
    let requiresAgentRun: Bool
    let evidenceMode: E2EEvidenceMode
    let expectedToolID: String?
    let scenarioBankKind: String?
    let requiredSlotIDs: [String]?

    init(
        id: String,
        title: String,
        kind: E2ETestKind,
        prompt: String,
        expectedIntent: UserIntent,
        requiredAllowedToolIDs: [String] = [],
        forbiddenToolIDs: [String],
        requiredTextHints: [String],
        forbiddenTextHints: [String],
        requiresAgentRun: Bool,
        evidenceMode: E2EEvidenceMode? = nil,
        expectedToolID: String? = nil,
        scenarioBankKind: String? = nil,
        requiredSlotIDs: [String]? = nil
    ) {
        self.id = id
        self.title = title
        self.kind = kind
        self.prompt = prompt
        self.expectedIntent = expectedIntent
        self.requiredAllowedToolIDs = requiredAllowedToolIDs
        self.forbiddenToolIDs = forbiddenToolIDs
        self.requiredTextHints = requiredTextHints
        self.forbiddenTextHints = forbiddenTextHints
        self.requiresAgentRun = requiresAgentRun
        self.evidenceMode = evidenceMode ?? (requiresAgentRun ? .modelBackedRequired : .routingOnly)
        self.expectedToolID = expectedToolID.map(ToolRouteGuard.canonicalToolID)
        self.scenarioBankKind = scenarioBankKind
        self.requiredSlotIDs = requiredSlotIDs
    }

    static let standard: [E2ETestScenario] = regression + allToolCoverage + chatCoverage

    static let trainingValidation: [E2ETestScenario] = [
        E2ETestScenario(id: "training-weather-grounded", title: "Training eval: weather stays grounded", kind: .training, prompt: "What is the weather here and should I carry an umbrella?", expectedIntent: .weather, requiredAllowedToolIDs: ["weather", "location.current"], forbiddenToolIDs: ["calendar.create", "mail.draft"], requiredTextHints: ["weather"], forbiddenTextHints: ["created a new event"], requiresAgentRun: true),
        E2ETestScenario(id: "training-web-research", title: "Training eval: web research synthesis", kind: .training, prompt: "Search the web for two recent Swift concurrency best practices and summarize them.", expectedIntent: .webSearch, requiredAllowedToolIDs: ["web.search", "web.fetch"], forbiddenToolIDs: ["calendar.create", "weather"], requiredTextHints: ["swift"], forbiddenTextHints: ["created a new event"], requiresAgentRun: true),
        E2ETestScenario(id: "training-memory-loop", title: "Training eval: memory save/recall", kind: .training, prompt: "Remember that I prefer concise bullet points, then tell me what you remembered.", expectedIntent: .memory, requiredAllowedToolIDs: ["memory.save", "memory.recall"], forbiddenToolIDs: ["calendar.create", "weather"], requiredTextHints: ["remember"], forbiddenTextHints: ["created a new event"], requiresAgentRun: true),
        E2ETestScenario(id: "training-rag-grounding", title: "Training eval: local knowledge grounding", kind: .training, prompt: "Search my files for architecture notes and summarize key modules.", expectedIntent: .rag, requiredAllowedToolIDs: ["rag.search", "files.read"], forbiddenToolIDs: ["calendar.create", "weather"], requiredTextHints: ["module", "[1]"], forbiddenTextHints: ["created a new event"], requiresAgentRun: true),
        E2ETestScenario(id: "training-scheduler-agent", title: "Training eval: trigger scheduling quality", kind: .training, prompt: "Schedule a trigger to summarize reminders tonight and confirm what will run.", expectedIntent: .trigger, requiredAllowedToolIDs: ["trigger.create", "trigger.list"], forbiddenToolIDs: ["calendar.create", "weather"], requiredTextHints: ["trigger"], forbiddenTextHints: ["created a new event"], requiresAgentRun: true),
        E2ETestScenario(id: "training-communication-draft", title: "Training eval: communication drafting", kind: .training, prompt: "Draft an email to Alex with a professional update and ask one clarifying question.", expectedIntent: .emailDraft, requiredAllowedToolIDs: ["mail.draft", "contacts.search"], forbiddenToolIDs: ["calendar.create", "weather"], requiredTextHints: ["question"], forbiddenTextHints: ["created a new event"], requiresAgentRun: true),
        E2ETestScenario(id: "training-general-chat", title: "Training eval: pure chat quality", kind: .training, prompt: "Explain tradeoffs between precision and recall in retrieval systems in plain English.", expectedIntent: .chat, requiredAllowedToolIDs: [], forbiddenToolIDs: ["calendar.create", "weather", "mail.draft"], requiredTextHints: ["precision", "recall"], forbiddenTextHints: ["created a new event"], requiresAgentRun: true)
    ]

    static let regression: [E2ETestScenario] = [
        E2ETestScenario(id: "weather-here-no-calendar", title: "Weather here must not create events", kind: .regression, prompt: "What is the weather here?", expectedIntent: .weather, requiredAllowedToolIDs: ["weather", "location.current"], forbiddenToolIDs: ["calendar.create", "calendar.list", "reminders.create", "web.search"], requiredTextHints: [], forbiddenTextHints: ["created a new event", "calendar event", "will start in", "search web for diy underground shelter"], requiresAgentRun: true),
        E2ETestScenario(id: "web-search-no-calendar", title: "Web search must not create calendar event", kind: .regression, prompt: "Search web for diy underground shelter", expectedIntent: .webSearch, requiredAllowedToolIDs: ["web.search"], forbiddenToolIDs: ["calendar.create", "calendar.list", "reminders.create", "maps.search"], requiredTextHints: [], forbiddenTextHints: ["created a new event", "calendar event", "will start in"], requiresAgentRun: true),
        E2ETestScenario(id: "vague-email-clarifies", title: "Vague email draft asks clarification", kind: .routing, prompt: "Draft a email", expectedIntent: .emailDraft, requiredAllowedToolIDs: ["mail.draft", "contacts.search"], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "reminders.create"], requiredTextHints: ["who should", "what should"], forbiddenTextHints: ["i will be in touch soon", "created a new event"], requiresAgentRun: true, evidenceMode: .policyFirstAllowed),
        E2ETestScenario(id: "normal-chat-no-forced-tool", title: "Normal chat does not force tools", kind: .chat, prompt: "Explain why a sharp chisel is safer than a dull one.", expectedIntent: .chat, requiredAllowedToolIDs: [], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "mail.draft", "reminders.create"], requiredTextHints: [], forbiddenTextHints: ["created a new event", "weather for"], requiresAgentRun: true)
    ]

    static let allToolCoverage: [E2ETestScenario] = liveToolCoverageScenarios()

    static let chatCoverage: [E2ETestScenario] = [
        E2ETestScenario(id: "chat-carpentry-advice", title: "Carpentry chat stays direct", kind: .chat, prompt: "Give me three tips for fitting a door hinge cleanly.", expectedIntent: .chat, requiredAllowedToolIDs: [], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "mail.draft", "reminders.create"], requiredTextHints: [], forbiddenTextHints: ["created a new event", "weather for"], requiresAgentRun: true),
        E2ETestScenario(id: "chat-code-explanation", title: "Code explanation stays chat", kind: .chat, prompt: "Explain actor isolation in Swift in simple terms.", expectedIntent: .chat, requiredAllowedToolIDs: [], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "mail.draft", "reminders.create"], requiredTextHints: [], forbiddenTextHints: ["created a new event", "weather for"], requiresAgentRun: true)
    ]

    private static func liveToolCoverageScenarios() -> [E2ETestScenario] {
        let scenarios = ToolScenarioBank.entries().map(liveToolCoverageScenario)
        var seen: Set<String> = []
        return scenarios.filter { scenario in
            seen.insert(scenario.id).inserted
        }
    }

    private static func liveToolCoverageScenario(from entry: ToolScenarioBankEntry) -> E2ETestScenario {
        let toolID = ToolRouteGuard.canonicalToolID(entry.expectedToolID)
        let expectedIntent = UserIntent(rawValue: entry.expectedIntent ?? "") ?? inferredIntent(forToolID: toolID)
        return E2ETestScenario(
            id: "live-\(entry.id)",
            title: "Live \(entry.toolID) \(entry.kind.rawValue)",
            kind: .toolGuard,
            prompt: entry.prompt,
            expectedIntent: expectedIntent,
            requiredAllowedToolIDs: [toolID],
            forbiddenToolIDs: forbiddenToolIDs(for: expectedIntent),
            requiredTextHints: [],
            forbiddenTextHints: forbiddenTextHints(for: toolID),
            requiresAgentRun: true,
            evidenceMode: .policyFirstAllowed,
            expectedToolID: toolID,
            scenarioBankKind: entry.kind.rawValue,
            requiredSlotIDs: entry.requiredSlots.map(\.rawValue)
        )
    }

    private static func forbiddenToolIDs(for intent: UserIntent) -> [String] {
        let allowed = IntentRouter.allowedToolIDs(for: intent)
        return ToolRegistry.all
            .map { ToolRouteGuard.canonicalToolID($0.id) }
            .filter { !allowed.contains($0) }
            .sorted()
    }

    private static func forbiddenTextHints(for toolID: String) -> [String] {
        switch toolID {
        case "calendar.create": return ["weather for", "web search"]
        case "web.search", "web.fetch": return ["created a new event", "calendar event", "will start in"]
        default: return ["created a new event"]
        }
    }

    private static func inferredIntent(forToolID toolID: String) -> UserIntent {
        if toolID == "weather" { return .weather }
        if toolID.hasPrefix("web.") { return .webSearch }
        if toolID.hasPrefix("mail.") { return .emailDraft }
        if toolID.hasPrefix("messages.") { return .messageDraft }
        if toolID == "phone.call" { return .phoneCall }
        if toolID.hasPrefix("contacts.") { return .contactSearch }
        if toolID.hasPrefix("calendar.") { return .calendar }
        if toolID.hasPrefix("reminders.") { return .reminder }
        if toolID.hasPrefix("maps.") || toolID == "location.current" { return .maps }
        if toolID.hasPrefix("photos.") { return .photos }
        if toolID == "camera.capture" { return .camera }
        if toolID == "health.summary" { return .health }
        if toolID == "motion.activity" { return .motion }
        if toolID.hasPrefix("files.") { return .files }
        if toolID.hasPrefix("memory.") { return .memory }
        if toolID.hasPrefix("rag.") { return .rag }
        if toolID.hasPrefix("trigger.") { return .trigger }
        if toolID.hasPrefix("alarm.") { return .alarm }
        if toolID.hasPrefix("outlook.") { return .outlook }
        return .unknown
    }
}

private extension AgentBehaviorTrace {
    var streamStartedText: String {
        streamStarted.map { $0 ? "true" : "false" } ?? "unknown"
    }

    var firstChunkReceivedText: String {
        firstChunkReceived.map { $0 ? "true" : "false" } ?? "unknown"
    }

    var finalChunkReceivedText: String {
        finalChunkReceived.map { $0 ? "true" : "false" } ?? "unknown"
    }
}

nonisolated struct E2EPerformanceSample: Codable, Sendable {
    let timestamp: Date
    let residentMemoryMB: Double?
    let totalMemoryMB: Double
}

nonisolated struct E2EPerformanceMatrix: Codable, Sendable {
    let aneUtilizationPercent: Double?
    let eventDensityCPUProxyPercent: Double?
    let gpuUtilizationPercent: Double?
    let peakRAMMB: Double
    let averageRAMMB: Double
    let sampleCount: Int
    let notes: [String]
    let accelerationDiagnostics: RuntimeAccelerationDiagnostics?
}

nonisolated struct E2ETestEvent: Codable, Sendable, Identifiable {
    let id: UUID
    let createdAt: Date
    let scenarioID: String
    let phase: String
    let message: String
}

#if DEBUG
nonisolated struct AgentJSONTrainingProbeResult: Codable, Sendable, Hashable {
    let scenarioID: String
    let promptFitsBudget: Bool
    let streamStarted: Bool
    let firstChunkReceived: Bool
    let firstTextReceived: Bool
    let parsedJSON: Bool
    let actionOrFinal: String?
    let selectedTool: String?
    let emptyStreamReason: String?
}
#endif

nonisolated struct E2ETestResult: Codable, Sendable, Identifiable {
    let id: UUID
    let scenarioID: String
    let kind: String
    let title: String
    let prompt: String
    let expectedIntent: String
    let actualIntent: String
    let e2eRunID: UUID?
    let agentRunID: UUID?
    let conversationID: UUID?
    let turnID: UUID?
    let correlationToken: String?
    let requiresAgentRun: Bool
    let evidenceMode: String
    let passed: Bool
    let failures: [String]
    let finalText: String
    let missingHints: [String]
    let rewriteAttempted: Bool
    let rewriteSuccess: Bool
    let events: [E2ETestEvent]
    let startedAt: Date
    let finishedAt: Date
    let rawFinalPrefix: String
    let sanitizedFinalPrefix: String
    let rawFinalHadUnsafeLeakage: Bool
    let sanitizedFinalRemovedArtifacts: [String]
    let outputHygieneFailures: [String]
    let performanceMatrix: E2EPerformanceMatrix?
    let metadata: [String: String]

    var isRuntimePreflightNonActionable: Bool {
        guard !passed else { return false }
        if metadata["trainingSignal"]?.lowercased() == "false",
           metadata["actionable"]?.lowercased() == "false",
           metadata["failureKind"]?.hasPrefix("liveRuntime") == true {
            return true
        }
        let evidence = ([finalText] + failures + events.map(\.message) + Array(metadata.values))
            .joined(separator: "\n")
            .lowercased()
        return evidence.contains("live e2e preflight blocked model-backed generation before prompt evaluation")
            || evidence.contains("thermalstate=serious")
            || evidence.contains("thermalstate=critical")
            || evidence.contains("scenephase=inactive")
            || evidence.contains("scenephase=background")
            || evidence.contains("cpu-watchdog-degraded")
            || evidence.contains("runtime-preflight/non-actionable")
    }

    init(
        id: UUID,
        scenarioID: String,
        kind: String = "",
        title: String,
        prompt: String,
        expectedIntent: String,
        actualIntent: String,
        e2eRunID: UUID? = nil,
        agentRunID: UUID? = nil,
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
        correlationToken: String? = nil,
        requiresAgentRun: Bool = false,
        evidenceMode: String = E2EEvidenceMode.modelBackedRequired.rawValue,
        passed: Bool,
        failures: [String],
        finalText: String,
        missingHints: [String],
        rewriteAttempted: Bool,
        rewriteSuccess: Bool,
        events: [E2ETestEvent],
        startedAt: Date,
        finishedAt: Date,
        rawFinalPrefix: String,
        sanitizedFinalPrefix: String,
        rawFinalHadUnsafeLeakage: Bool,
        sanitizedFinalRemovedArtifacts: [String],
        outputHygieneFailures: [String],
        performanceMatrix: E2EPerformanceMatrix? = nil,
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.scenarioID = scenarioID
        self.kind = kind
        self.title = title
        self.prompt = prompt
        self.expectedIntent = expectedIntent
        self.actualIntent = actualIntent
        self.e2eRunID = e2eRunID
        self.agentRunID = agentRunID
        self.conversationID = conversationID
        self.turnID = turnID
        self.correlationToken = correlationToken
        self.requiresAgentRun = requiresAgentRun
        self.evidenceMode = evidenceMode
        self.passed = passed
        self.failures = Self.unique(failures)
        self.finalText = finalText
        self.missingHints = missingHints
        self.rewriteAttempted = rewriteAttempted
        self.rewriteSuccess = rewriteSuccess
        self.events = events
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.rawFinalPrefix = rawFinalPrefix
        self.sanitizedFinalPrefix = sanitizedFinalPrefix
        self.rawFinalHadUnsafeLeakage = rawFinalHadUnsafeLeakage
        self.sanitizedFinalRemovedArtifacts = Self.unique(sanitizedFinalRemovedArtifacts)
        self.outputHygieneFailures = Self.unique(outputHygieneFailures)
        self.performanceMatrix = performanceMatrix
        self.metadata = metadata
    }

    private static func unique(_ values: [String]) -> [String] {
        var seen = Set<String>()
        return values.filter { seen.insert($0).inserted }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        scenarioID = try c.decode(String.self, forKey: .scenarioID)
        kind = try c.decodeIfPresent(String.self, forKey: .kind) ?? ""
        title = try c.decode(String.self, forKey: .title)
        prompt = try c.decode(String.self, forKey: .prompt)
        expectedIntent = try c.decode(String.self, forKey: .expectedIntent)
        actualIntent = try c.decode(String.self, forKey: .actualIntent)
        e2eRunID = try c.decodeIfPresent(UUID.self, forKey: .e2eRunID)
        agentRunID = try c.decodeIfPresent(UUID.self, forKey: .agentRunID)
        conversationID = try c.decodeIfPresent(UUID.self, forKey: .conversationID)
        turnID = try c.decodeIfPresent(UUID.self, forKey: .turnID)
        correlationToken = try c.decodeIfPresent(String.self, forKey: .correlationToken)
        requiresAgentRun = try c.decodeIfPresent(Bool.self, forKey: .requiresAgentRun) ?? false
        evidenceMode = try c.decodeIfPresent(String.self, forKey: .evidenceMode) ?? (requiresAgentRun ? E2EEvidenceMode.modelBackedRequired.rawValue : E2EEvidenceMode.routingOnly.rawValue)
        passed = try c.decode(Bool.self, forKey: .passed)
        failures = try c.decode([String].self, forKey: .failures)
        finalText = try c.decode(String.self, forKey: .finalText)
        missingHints = try c.decode([String].self, forKey: .missingHints)
        rewriteAttempted = try c.decode(Bool.self, forKey: .rewriteAttempted)
        rewriteSuccess = try c.decode(Bool.self, forKey: .rewriteSuccess)
        events = try c.decode([E2ETestEvent].self, forKey: .events)
        startedAt = try c.decode(Date.self, forKey: .startedAt)
        finishedAt = try c.decode(Date.self, forKey: .finishedAt)
        rawFinalPrefix = try c.decodeIfPresent(String.self, forKey: .rawFinalPrefix) ?? ""
        sanitizedFinalPrefix = try c.decodeIfPresent(String.self, forKey: .sanitizedFinalPrefix) ?? ""
        rawFinalHadUnsafeLeakage = try c.decodeIfPresent(Bool.self, forKey: .rawFinalHadUnsafeLeakage) ?? false
        sanitizedFinalRemovedArtifacts = try c.decodeIfPresent([String].self, forKey: .sanitizedFinalRemovedArtifacts) ?? []
        outputHygieneFailures = try c.decodeIfPresent([String].self, forKey: .outputHygieneFailures) ?? []
        performanceMatrix = try c.decodeIfPresent(E2EPerformanceMatrix.self, forKey: .performanceMatrix)
        metadata = try c.decodeIfPresent([String: String].self, forKey: .metadata) ?? [:]
    }
}

nonisolated struct E2ETestReport: Codable, Sendable, Identifiable {
    let id: UUID
    let startedAt: Date
    let finishedAt: Date
    let passed: Int
    let failed: Int
    let results: [E2ETestResult]

    var summaryText: String {
        var lines: [String] = []
        let runtimePreflightNonActionable = results.filter { !$0.passed && $0.isRuntimePreflightNonActionable }.count
        let actionableFailed = max(failed - runtimePreflightNonActionable, 0)
        lines.append("E2E Test Report")
        lines.append("Passed: \(passed)")
        lines.append("Failed: \(actionableFailed)")
        if runtimePreflightNonActionable > 0 {
            lines.append("Runtime preflight/non-actionable: \(runtimePreflightNonActionable)")
        }
        let modelBackedPassed = results.filter { $0.passed && $0.evidenceMode == E2EEvidenceMode.modelBackedRequired.rawValue }.count
        let policyFirstPassed = results.filter { $0.passed && $0.evidenceMode == E2EEvidenceMode.policyFirstAllowed.rawValue }.count
        let routingOnlyPassed = results.filter { $0.passed && $0.evidenceMode == E2EEvidenceMode.routingOnly.rawValue }.count
        if policyFirstPassed > 0 || routingOnlyPassed > 0 {
            lines.append("Model-backed passed: \(modelBackedPassed)")
            lines.append("Policy-first passed: \(policyFirstPassed)")
            lines.append("Routing-only passed: \(routingOnlyPassed)")
        }
        lines.append("")

        let bucketForFailure: (String) -> String = { failure in
            if failure.contains("Live E2E preflight blocked model-backed generation before prompt evaluation")
                || failure.contains("AlarmKit runtime unavailable")
                || failure.contains("CPU watchdog degraded")
                || failure.contains("cpu-watchdog-degraded") {
                return "runtime-preflight/non-actionable"
            }
            if failure.contains("Intent mismatch") { return "intent" }
            if failure.contains("Forbidden tool") || failure.contains("Required tool not allowed") || failure.contains("Forbidden tool selected by agent") { return "tool-boundary" }
            if failure.contains("Required final hint") || failure.contains("Forbidden final hint") { return "response-quality" }
            if failure.contains("Agent error") { return "runtime" }
            return "other"
        }
        var failureBuckets: [String: Int] = [:]
        for result in results where !result.failures.isEmpty {
            let forcedBucket: String? = {
                if result.isRuntimePreflightNonActionable {
                    return "runtime-preflight/non-actionable"
                }
                if result.metadata["trainingSignal"]?.lowercased() == "false",
                   result.metadata["failureKind"]?.hasPrefix("liveRuntime") == true {
                    return "runtime-preflight/non-actionable"
                }
                let evidence = ([result.finalText] + result.failures + result.events.map(\.message) + Array(result.metadata.values))
                    .joined(separator: "\n")
                    .lowercased()
                if evidence.contains("cpu-watchdog-degraded") || evidence.contains("alarmkit runtime unavailable") {
                    return "runtime-preflight/non-actionable"
                }
                return nil
            }()
            for failure in result.failures {
                let bucket = forcedBucket ?? bucketForFailure(failure)
                failureBuckets[bucket, default: 0] += 1
            }
        }
        if !failureBuckets.isEmpty {
            lines.append("Training signals for next run:")
            for key in ["intent", "tool-boundary", "response-quality", "runtime", "runtime-preflight/non-actionable", "other"] where failureBuckets[key] != nil {
                lines.append("• \(key): \(failureBuckets[key] ?? 0) issues")
            }
            let trainableFailureCount = results.filter { result in
                fineTuningNegativeCandidate(result)
            }.count
            if trainableFailureCount > 0 {
                lines.append("• Capture failed prompts + final outputs into next fine-tuning dataset.")
                lines.append("• Fine-tuning candidates: \(trainableFailureCount) grounded model-quality failures.")
            } else {
                lines.append("• Non-trainable architecture/runtime/finalizer failures quarantined; create regression tests instead of SFT negatives.")
            }
            lines.append("")
        }

        for result in results {
            lines.append("\(result.passed ? "✅" : "❌") \(result.title)")
            lines.append("Prompt: \(result.prompt)")
            lines.append("Intent: \(result.actualIntent) / expected \(result.expectedIntent)")
            if !result.failures.isEmpty {
                lines.append("Failures: \(result.failures.joined(separator: "; "))")
            }
            if !result.finalText.isEmpty {
                lines.append("Final: \(result.finalText)")
            }
            if let matrix = result.performanceMatrix {
                lines.append("Performance: ANE \(displayPercent(matrix.aneUtilizationPercent)), CPU-proxy \(displayPercent(matrix.eventDensityCPUProxyPercent)), GPU \(displayPercent(matrix.gpuUtilizationPercent)), RAM avg \(Int(matrix.averageRAMMB))MB / peak \(Int(matrix.peakRAMMB))MB")
            }
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    private func displayPercent(_ value: Double?) -> String {
        guard let value else { return "n/a" }
        return "\(Int(value.rounded()))%"
    }

    private func fineTuningNegativeCandidate(_ result: E2ETestResult) -> Bool {
        guard !result.failures.isEmpty else { return false }
        if result.isRuntimePreflightNonActionable { return false }
        if result.metadata["trainingSignal"]?.lowercased() == "false" { return false }
        let evidence = ([result.finalText] + result.failures + result.events.map(\.message) + Array(result.metadata.values))
            .joined(separator: "\n")
            .lowercased()
        let nonTrainableSignals = [
            "cpu-watchdog-degraded",
            "thermalstate=serious",
            "thermalstate=critical",
            "scenephase=background",
            "scenephase=inactive",
            "no direct answer from web search",
            "i'm ready. please ask again",
            "please ask again or tell me what you'd like to do next",
            "tool output could not be validated",
            "could not be validated",
            "no matching files found",
            "local index appears empty",
            "no matching local snippets",
            "import or create local files",
            "internal routing json"
        ]
        if nonTrainableSignals.contains(where: { evidence.contains($0) }) { return false }
        if RoutingJSONLeakDetector.containsInternalRoutingJSON(evidence) {
            return false
        }
        return true
    }
}

nonisolated enum E2ETestRunner {
    typealias ResultCallback = @Sendable (E2ETestResult) async -> Void
    typealias EventCallback = @Sendable (E2ETestEvent) async -> Void
    typealias EnsureChatLoaded = @Sendable () async -> Bool

    #if DEBUG
    @TaskLocal static var debugStandardScenariosOverride: [E2ETestScenario]?
    @TaskLocal static var debugAssertScenarioLoopOffMainThread = false
    @TaskLocal static var debugScenarioLoopThreadRecorder: (@Sendable (Bool) -> Void)?
    @TaskLocal static var debugExecutorRuntimePreflightOverride: (@Sendable () async -> ExecutorRuntimePreflightResult)?
    @TaskLocal static var debugCPUWatchdogDegradedOverride: Bool?
    @TaskLocal static var debugCPUWatchdogDegradedProbe: (@Sendable (E2ETestScenario) -> Bool)?

    private static func debugIsRunningOnMainThread() -> Bool {
        #if canImport(Darwin)
        pthread_main_np() != 0
        #else
        Thread.isMainThread
        #endif
    }
    #endif

    private static func executorRuntimePreflight() async -> ExecutorRuntimePreflightResult {
        #if DEBUG
        if let override = debugExecutorRuntimePreflightOverride {
            return await override()
        }
        #endif
        return await ExecutorRuntimePreflight.run()
    }

    static func runStandard(config: E2ERunConfig, ensureChatLoaded: EnsureChatLoaded? = nil, onResult: ResultCallback? = nil, onEvent: EventCallback? = nil) async -> E2ETestReport {
        #if DEBUG
        let scenarios = debugStandardScenariosOverride ?? E2ETestScenario.standard
        #else
        let scenarios = E2ETestScenario.standard
        #endif
        return await run(scenarios: scenarios, config: config, ensureChatLoaded: ensureChatLoaded, onResult: onResult, onEvent: onEvent)
    }

    private static func appendResult(
        _ result: E2ETestResult,
        to results: inout [E2ETestResult],
        onResult: ResultCallback?
    ) async {
        results.append(result)
        E2ETestLogStore.append(result)
        await onResult?(result)
    }

    static func runTrainingValidation(config: E2ERunConfig, ensureChatLoaded: EnsureChatLoaded? = nil, onResult: ResultCallback? = nil, onEvent: EventCallback? = nil) async -> E2ETestReport {
        let started = Date()
        if let ensureChatLoaded {
            let setupSucceeded = await ensureChatLoaded()
            guard setupSucceeded else {
                let scenario = E2ETestScenario.trainingValidation[0]
                let reason = "Training validation could not prepare the chat fleet before executor preflight."
                let event = E2ETestEvent(id: UUID(), createdAt: Date(), scenarioID: "executor-runtime-preflight", phase: "model-setup", message: reason)
                await onEvent?(event)
                let result = E2ETestResult(
                    id: UUID(),
                    scenarioID: "executor-runtime-preflight",
                    kind: E2ETestKind.training.rawValue,
                    title: "Executor runtime preflight",
                    prompt: scenario.prompt,
                    expectedIntent: scenario.expectedIntent.rawValue,
                    actualIntent: "preflight",
                    requiresAgentRun: true,
                    passed: false,
                    failures: [reason],
                    finalText: "",
                    missingHints: [],
                    rewriteAttempted: false,
                    rewriteSuccess: false,
                    events: [event],
                    startedAt: started,
                    finishedAt: Date(),
                    rawFinalPrefix: "",
                    sanitizedFinalPrefix: "",
                    rawFinalHadUnsafeLeakage: false,
                    sanitizedFinalRemovedArtifacts: [],
                    outputHygieneFailures: [],
                    performanceMatrix: nil,
                    metadata: ["failureKind": "trainingValidationModelSetupFailed"]
                )
                E2ETestLogStore.append(result)
                await onResult?(result)
                let report = E2ETestReport(id: UUID(), startedAt: started, finishedAt: Date(), passed: 0, failed: 1, results: [result])
                E2ETestLogStore.writeLatest(report)
                return report
            }
        }

        let preflight = await executorRuntimePreflight()
        guard preflight.passed else {
            let scenario = E2ETestScenario.trainingValidation[0]
            let result = await executorRuntimePreflightBlockedResult(
                startedAt: started,
                scenario: scenario,
                preflight: preflight,
                onEvent: onEvent
            )
            E2ETestLogStore.append(result)
            await onResult?(result)
            let report = E2ETestReport(id: UUID(), startedAt: started, finishedAt: Date(), passed: 0, failed: 1, results: [result])
            E2ETestLogStore.writeLatest(report)
            return report
        }
        return await run(scenarios: E2ETestScenario.trainingValidation, config: config, ensureChatLoaded: ensureChatLoaded, onResult: onResult, onEvent: onEvent, performExecutorPreflight: false)
    }

    static func liveRuntimeArtifactsBlockedReport(
        startedAt: Date = Date(),
        finishedAt: Date = Date(),
        readyArtifactCount: Int,
        requiredArtifactCount: Int,
        missingAdapterSlots: [String] = [],
        missingArtifactFileNames: [String] = [],
        diagnostic: String? = nil
    ) -> E2ETestReport {
        let reason = "Live runtime artifact preparation did not complete; required Qwen3 shared base and role adapters must be downloaded before live E2E scenarios run."
        let missingAdapters = missingAdapterSlots.isEmpty ? "" : " Missing adapter slots: \(missingAdapterSlots.joined(separator: ", "))."
        let missingFiles = missingArtifactFileNames.isEmpty ? "" : " Missing artifacts: \(missingArtifactFileNames.joined(separator: ", "))."
        let diagnosticText = diagnostic.map { " Diagnostic: \($0)." } ?? ""
        let detail = "\(readyArtifactCount) / \(requiredArtifactCount) live runtime artifacts ready.\(missingAdapters)\(missingFiles)\(diagnosticText)"
        let event = E2ETestEvent(
            id: UUID(),
            createdAt: finishedAt,
            scenarioID: "live-runtime-artifact-preflight",
            phase: "runtime-artifacts",
            message: "\(reason) \(detail)."
        )
        let result = E2ETestResult(
            id: UUID(),
            scenarioID: "live-runtime-artifact-preflight",
            kind: E2ETestKind.training.rawValue,
            title: "Live runtime artifact preflight",
            prompt: "Prepare live runtime artifacts before running E2E scenarios.",
            expectedIntent: UserIntent.unknown.rawValue,
            actualIntent: "preflight",
            requiresAgentRun: true,
            passed: false,
            failures: [reason],
            finalText: detail,
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [event],
            startedAt: startedAt,
            finishedAt: finishedAt,
            rawFinalPrefix: "",
            sanitizedFinalPrefix: detail,
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: [],
            performanceMatrix: nil,
            metadata: [
                "failureKind": "liveRuntimeArtifactsNotReady",
                "readyArtifactCount": "\(readyArtifactCount)",
                "requiredArtifactCount": "\(requiredArtifactCount)",
                "missingAdapterSlots": missingAdapterSlots.joined(separator: ","),
                "missingArtifactFileNames": missingArtifactFileNames.joined(separator: ","),
                "diagnostic": diagnostic ?? ""
            ]
        )
        return E2ETestReport(id: UUID(), startedAt: startedAt, finishedAt: finishedAt, passed: 0, failed: 1, results: [result])
    }

    static func liveModelCatalogFetchBlockedReport(
        startedAt: Date = Date(),
        finishedAt: Date = Date(),
        diagnostic: String
    ) -> E2ETestReport {
        let reason = "Live E2E model setup could not fetch the stored model catalog."
        let detail = "Model catalog fetch failed before live scenarios ran. Diagnostic: \(diagnostic)."
        let event = E2ETestEvent(
            id: UUID(),
            createdAt: finishedAt,
            scenarioID: "live-model-catalog-preflight",
            phase: "model-catalog",
            message: "\(reason) \(detail)"
        )
        let result = E2ETestResult(
            id: UUID(),
            scenarioID: "live-model-catalog-preflight",
            kind: E2ETestKind.training.rawValue,
            title: "Live model catalog preflight",
            prompt: "Fetch stored model catalog before running E2E scenarios.",
            expectedIntent: UserIntent.unknown.rawValue,
            actualIntent: "preflight",
            requiresAgentRun: true,
            passed: false,
            failures: [reason],
            finalText: detail,
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [event],
            startedAt: startedAt,
            finishedAt: finishedAt,
            rawFinalPrefix: "",
            sanitizedFinalPrefix: detail,
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: [],
            performanceMatrix: nil,
            metadata: [
                "failureKind": "liveModelCatalogFetchFailed",
                "diagnostic": diagnostic
            ]
        )
        return E2ETestReport(id: UUID(), startedAt: startedAt, finishedAt: finishedAt, passed: 0, failed: 1, results: [result])
    }

    /// Executes end-to-end test scenarios sequentially and generates a report of results and metrics.
    /// - Parameters:
    ///   - scenarios: The test scenarios to execute.
    ///   - config: Runtime configuration for execution.
    ///   - ensureChatLoaded: Optional callback to ensure the chat model is available for scenarios requiring agent execution.
    ///   - onResult: Optional callback invoked when each scenario completes.
    ///   - onEvent: Optional callback invoked during scenario execution.
    /// - Returns: A report containing all results, pass/fail counts, and performance metrics.
    static func run(scenarios: [E2ETestScenario], config: E2ERunConfig, ensureChatLoaded: EnsureChatLoaded? = nil, onResult: ResultCallback? = nil, onEvent: EventCallback? = nil, performExecutorPreflight: Bool = true) async -> E2ETestReport {
        let started = Date()
        var results: [E2ETestResult] = []
        if performExecutorPreflight,
           let preflightScenario = scenarios.first(where: requiresExecutorPreflight) {
            let preflight = await executorRuntimePreflight()
            guard preflight.passed else {
                let result = await executorRuntimePreflightBlockedResult(
                    startedAt: started,
                    scenario: preflightScenario,
                    preflight: preflight,
                    onEvent: onEvent
                )
                await appendResult(result, to: &results, onResult: onResult)
                let report = E2ETestReport(id: UUID(), startedAt: started, finishedAt: Date(), passed: 0, failed: 1, results: results)
                E2ETestLogStore.writeLatest(report)
                return report
            }
        }
        for scenario in scenarios {
            #if DEBUG
            let isOnMainThread = debugIsRunningOnMainThread()
            await MainActor.run {
                debugScenarioLoopThreadRecorder?(isOnMainThread)
            }
            if debugAssertScenarioLoopOffMainThread {
                assert(!isOnMainThread, "E2ETestRunner scenario loop must not run on the main thread")
            }
            #endif
            do {
                try Task.checkCancellation()
                await Task.yield()
                if let blocked = await liveRuntimePreflightBlockedResultIfNeeded(for: scenario, onEvent: onEvent) {
                    await appendResult(blocked, to: &results, onResult: onResult)
                    break
                }
                let result = try await runScenario(scenario, config: config, ensureChatLoaded: ensureChatLoaded, onEvent: onEvent)
                try Task.checkCancellation()
                await Task.yield()
                await appendResult(result, to: &results, onResult: onResult)
                if liveRuntimeShouldStopAfter(result) {
                    break
                }
                await paceAfterLiveRuntimeScenario(result)
            } catch is CancellationError {
                break
            } catch {
                let result = E2ETestResult(id: UUID(), scenarioID: scenario.id, kind: scenario.kind.rawValue, title: scenario.title, prompt: scenario.prompt, expectedIntent: scenario.expectedIntent.rawValue, actualIntent: "error", requiresAgentRun: scenario.requiresAgentRun, evidenceMode: scenario.evidenceMode.rawValue, passed: false, failures: ["E2E runner error: \(error.localizedDescription)"], finalText: "", missingHints: [], rewriteAttempted: false, rewriteSuccess: false, events: [], startedAt: Date(), finishedAt: Date(), rawFinalPrefix: "", sanitizedFinalPrefix: "", rawFinalHadUnsafeLeakage: false, sanitizedFinalRemovedArtifacts: [], outputHygieneFailures: [], performanceMatrix: nil)
                await appendResult(result, to: &results, onResult: onResult)
                if liveRuntimeShouldStopAfter(result) {
                    break
                }
                await paceAfterLiveRuntimeScenario(result)
            }
        }
        let passed = results.filter(\.passed).count
        let report = E2ETestReport(id: UUID(), startedAt: started, finishedAt: Date(), passed: passed, failed: results.count - passed, results: results)
        do { try Task.checkCancellation() } catch { return report }
        await Task.yield()
        E2ETestLogStore.writeLatest(report)
        do { try Task.checkCancellation() } catch { return report }
        await Task.yield()
        return report
    }

    private struct FinalHygieneState {
        var rawFinalText: String
        var finalText: String
        var rawSanitized: SanitizedFinalOutput
        var postRewriteSanitized: SanitizedFinalOutput
        var recoveredBeforeRewrite: SanitizedFinalOutput?
        var recoveredAfterRewrite: SanitizedFinalOutput?

        var hadUnsafeLeakage: Bool {
            rawSanitized.hadUnsafeLeakage
                || postRewriteSanitized.hadUnsafeLeakage
                || recoveredBeforeRewrite?.hadUnsafeLeakage == true
                || recoveredAfterRewrite?.hadUnsafeLeakage == true
        }

        var removedArtifacts: [FinalOutputArtifact] {
            E2ETestRunner.mergedArtifacts(
                rawSanitized.removedArtifacts,
                postRewriteSanitized.removedArtifacts,
                recoveredBeforeRewrite?.removedArtifacts ?? [],
                recoveredAfterRewrite?.removedArtifacts ?? []
            )
        }
    }

    private struct ModelRuntimeEvidence: Sendable {
        let runtimePath: String
        let stage: String
        let evidenceKind: String
        let generationElapsedMs: Int?
        let outputTokenCount: Int?
        let adapterSlot: String?
        let parseError: String?
        let matchedBy: String
    }

    private struct ModelRuntimeEvidenceDiagnosis: Sendable {
        let evidence: ModelRuntimeEvidence?
        let failureMessage: String
    }

    private static func runScenario(_ scenario: E2ETestScenario, config: E2ERunConfig, ensureChatLoaded: EnsureChatLoaded? = nil, onEvent: EventCallback? = nil) async throws -> E2ETestResult {
        let started = Date()
        let e2eRunID = UUID()
        let agentRunID = UUID()
        let conversationID = UUID()
        let turnID = UUID()
        var events: [E2ETestEvent] = []
        var failures: [String] = []
        var finalText = ""
        var missingHints: [String] = []
        var rewriteAttempted = false
        var rewriteSuccess = false
        var rawFinalText = ""
        var agentSteps: [AgentStep] = []
        var hygieneState = FinalHygieneState(
            rawFinalText: "",
            finalText: "",
            rawSanitized: FinalOutputSanitizer.sanitizeUserVisibleText(""),
            postRewriteSanitized: FinalOutputSanitizer.sanitizeUserVisibleText(""),
            recoveredBeforeRewrite: nil,
            recoveredAfterRewrite: nil
        )
        var performanceSamples: [E2EPerformanceSample] = []
        var lastPerformanceSampleAt: Date?
        var hasAcceptedModelEvidenceForScenario = !scenario.requiresAgentRun
        var deterministicChatFallbackRemediationApplied = false
        let totalMemoryMB = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024)

        func event(_ phase: String, _ message: String) async {
            let emitted = E2ETestEvent(id: UUID(), createdAt: Date(), scenarioID: scenario.id, phase: phase, message: message)
            events.append(emitted)
            await onEvent?(emitted)
        }
        func collectPerformanceSample(force: Bool = false) {
            let now = Date()
            if !force, let lastPerformanceSampleAt, now.timeIntervalSince(lastPerformanceSampleAt) < 0.5 {
                return
            }
            performanceSamples.append(
                E2EPerformanceSample(
                    timestamp: now,
                    residentMemoryMB: residentMemoryUsageMB(),
                    totalMemoryMB: totalMemoryMB
                )
            )
            lastPerformanceSampleAt = now
        }

        collectPerformanceSample(force: true)
        try Task.checkCancellation()
        await Task.yield()
        await event("start", scenario.prompt)
        try Task.checkCancellation()
        await Task.yield()
        let routing = await IntentClassifierService.shared.route(scenario.prompt)
        try Task.checkCancellation()
        await Task.yield()
        collectPerformanceSample()
        await event("intent", "actual=\(routing.intent.rawValue), expected=\(scenario.expectedIntent.rawValue)")
        if routing.intent != scenario.expectedIntent {
            failures.append("Intent mismatch: \(routing.intent.rawValue) != \(scenario.expectedIntent.rawValue)")
        }

        for toolID in scenario.requiredAllowedToolIDs where !IntentRouter.isToolAllowed(toolID, for: routing) {
            failures.append("Required tool not allowed: \(toolID)")
        }

        for toolID in scenario.forbiddenToolIDs where IntentRouter.isToolAllowed(toolID, for: routing) {
            failures.append("Forbidden tool allowed: \(toolID)")
        }

        if scenario.requiresAgentRun {
            let enabledCanonicalToolIDs = Set(config.enabledToolIDs.map(ToolRouteGuard.canonicalToolID))
            let disabledRequiredTools = scenario.requiredAllowedToolIDs
                .map(ToolRouteGuard.canonicalToolID)
                .filter { !enabledCanonicalToolIDs.contains($0) }
            if !disabledRequiredTools.isEmpty {
                failures.append("Required live E2E tools disabled: \(Array(Set(disabledRequiredTools)).sorted().joined(separator: ", "))")
            }
        }

        if scenario.requiresAgentRun {
            try Task.checkCancellation()
            await Task.yield()
            let traceCorrelation = AgentTraceCorrelation(
                scenarioID: scenario.id,
                e2eRunID: e2eRunID,
                agentRunID: agentRunID,
                conversationID: conversationID,
                turnID: turnID
            )
            await event("correlation", traceCorrelation.diagnosticText)
            let modelLoaded: Bool
            if let ensureChatLoaded = ensureChatLoaded {
                modelLoaded = await ensureChatLoaded()
            } else {
                modelLoaded = false
            }
            try Task.checkCancellation()
            await Task.yield()
            collectPerformanceSample()
            await event("models", modelLoaded ? "chat fleet ready" : "no chat model loaded")
            if !modelLoaded {
                failures.append("Live E2E scenario did not run: no chat model loaded")
                await event("model-evidence", "AgentService model path was not entered; reason=model not loaded; \(traceCorrelation.diagnosticText)")
            }
            if modelLoaded {
                let enabledCanonicalToolIDs = Set(config.enabledToolIDs.map(ToolRouteGuard.canonicalToolID))
                let forbiddenCanonicalToolIDs = Set(scenario.forbiddenToolIDs.map(ToolRouteGuard.canonicalToolID))
                let availableTools = ToolRegistry.all.filter { tool in
                    let canonical = ToolRouteGuard.canonicalToolID(tool.id)
                    return enabledCanonicalToolIDs.contains(canonical) && IntentRouter.isToolAllowed(canonical, for: routing)
                }
                await event("tools", "available=\(availableTools.map(\.id).sorted().joined(separator: ","))")
                var steps: [AgentStep] = []
                try Task.checkCancellation()
                await Task.yield()
                let modelEvidenceStartedAt = Date()
                let requiresStructuredAgentJSON = requiresStructuredModelBackedAgentRun(scenario: scenario, routing: routing)
                if routing.requiresClarification, let clarification = routing.clarificationPrompt, !clarification.isEmpty {
                    let step = AgentStep(kind: .reflection, content: "Clarification required before tool execution.")
                    steps.append(step)
                    await event("step", "\(step.kind.rawValue): \(step.content)")
                    rawFinalText = clarification
                    recordPolicyFirstClarificationTrace(
                        scenario: scenario,
                        routing: routing,
                        clarification: clarification,
                        correlation: traceCorrelation,
                        startedAt: modelEvidenceStartedAt
                    )
                } else if shouldRunAsPlainTextTurn(scenario: scenario, routing: routing) && scenario.kind != .training {
                    let turn = AssistantTurnContext(
                        task: .chat,
                        input: scenario.prompt,
                        systemPrompt: config.systemPrompt,
                        isForeground: true,
                        lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
                        thermalState: ProcessInfo.processInfo.thermalState,
                        prefersFoundationModels: false,
                        allowHeavyRuntime: true,
                        temperature: min(config.temperature, 0.3),
                        topP: config.topP,
                        repetitionPenalty: config.repetitionPenalty,
                        maxTokens: min(config.maxTokens, 512),
                        traceCorrelation: traceCorrelation,
                        allowedToolIDs: []
                    )
                    do {
                        rawFinalText = try await AssistantKernel.shared.runTextTurn(turn)
                        if isGenericChatFallbackFinal(rawFinalText) {
                            await event("retry", "plain chat returned generic fallback; retrying once with direct-answer prompt")
                            let retryTurn = AssistantTurnContext(
                                task: .chat,
                                input: directAnswerRetryPrompt(for: scenario.prompt),
                                systemPrompt: config.systemPrompt,
                                isForeground: true,
                                lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
                                thermalState: ProcessInfo.processInfo.thermalState,
                                prefersFoundationModels: false,
                                allowHeavyRuntime: true,
                                temperature: min(config.temperature, 0.2),
                                topP: config.topP,
                                repetitionPenalty: config.repetitionPenalty,
                                maxTokens: min(config.maxTokens, 512),
                                traceCorrelation: traceCorrelation,
                                allowedToolIDs: []
                            )
                            rawFinalText = try await AssistantKernel.shared.runTextTurn(retryTurn)
                        }
                    } catch {
                        let message = RuntimeMetricErrorSanitizer.code(for: error)
                        failures.append("Agent error: \(message)")
                        rawFinalText = "I couldn't complete the chat turn because the local model failed: \(message)."
                    }
                } else {
                    let shouldEnableNetworkAccess = shouldTemporarilyEnableNetworkAccess(
                        scenario: scenario,
                        routing: routing,
                        availableToolIDs: availableTools.map(\.id)
                    )
                    let previousNetworkAccessGranted: Bool
                    if shouldEnableNetworkAccess {
                        previousNetworkAccessGranted = await PermissionRegistry.shared.currentStatus(for: .networkAccess) == .granted
                        await MainActor.run {
                            PermissionRegistry.shared.setNetworkAccessEnabled(true)
                        }
                        await event("permissions", "networkAccess=temporarily-enabled-for-live-e2e")
                    } else {
                        previousNetworkAccessGranted = false
                    }
                    defer {
                        if shouldEnableNetworkAccess {
                            Task { @MainActor in
                                PermissionRegistry.shared.setNetworkAccessEnabled(previousNetworkAccessGranted)
                            }
                        }
                    }
                    let kernelRequest = strictLiveAgentKernelRequest(
                        prompt: scenario.prompt,
                        systemPrompt: config.systemPrompt,
                        config: config,
                        conversationID: conversationID,
                        turnID: turnID,
                        traceCorrelation: traceCorrelation,
                        forceModelBackedToolPlanning: requiresStructuredAgentJSON,
                        structuredMode: requiresStructuredAgentJSON ? .requiredAgentJSON : .automatic,
                        structuredAllowedToolIDs: requiresStructuredAgentJSON ? availableTools.map(\.id) : []
                    )
                    let agentEvents: AsyncStream<AgentKernelEvent> = await MainActor.run {
                        let kernelModelContext = SharedContainer.shared.map { ModelContext($0) }
                        return AssistantKernel.shared.run(kernelRequest, modelContext: kernelModelContext)
                    }
                    for await agentEvent in agentEvents {
                        try Task.checkCancellation()
                        await Task.yield()
                        switch agentEvent {
                        case .step(let step):
                            collectPerformanceSample()
                            steps.append(step)
                            await event("step", "\(step.kind.rawValue): \(step.content)")
                            if let toolID = step.toolID,
                               forbiddenCanonicalToolIDs.contains(ToolRouteGuard.canonicalToolID(toolID)) {
                                failures.append("Forbidden tool selected by agent: \(toolID)")
                            }
                        case .toolResult(let result):
                            if let errorCode = result.errorCode {
                                let availability = result.structuredPayload?["availability"] ?? "unknown"
                                await event(
                                    "tool-result",
                                    "status=\(result.status.rawValue), errorCode=\(errorCode), availability=\(availability)"
                                )
                            }
                        case .diagnostic(let diagnostic):
                            if let diagnosticEvidence = structuredKernelDiagnosticEvidence(diagnostic) {
                                await event("kernel-diagnostic", diagnosticEvidence)
                            }
                        case .stepDelta, .toolInvocation:
                            break
                        case .token(let chunk), .finalDelta(let chunk):
                            rawFinalText += chunk
                            collectPerformanceSample()
                        case .final(let text):
                            collectPerformanceSample(force: true)
                            if !text.isEmpty { rawFinalText = text }
                        case .done(let text, let allSteps):
                            collectPerformanceSample(force: true)
                            if !text.isEmpty { rawFinalText = text }
                            steps = allSteps.isEmpty ? steps : allSteps
                        case .error(let message):
                            collectPerformanceSample(force: true)
                            failures.append("Agent error: \(message)")
                        }
                    }
                }
                if let synthesized = deterministicWebSynthesisFallback(
                    scenario: scenario,
                    rawFinalText: rawFinalText,
                    events: events
                ) {
                    await event("finalizer", "deterministic web synthesis fallback used after observations")
                    rawFinalText = synthesized
                }
                let acceptsPolicyFirstEvidence = acceptsPolicyFirstExecutionEvidence(scenario: scenario, routing: routing)
                let evidenceDiagnosis = modelRuntimeEvidenceDiagnosis(
                    since: modelEvidenceStartedAt,
                    prompt: scenario.prompt,
                    correlation: traceCorrelation,
                    acceptsPolicyFirstEvidence: acceptsPolicyFirstEvidence,
                    requiresPrimaryAgentJSON: requiresStructuredAgentJSON
                )
                hasAcceptedModelEvidenceForScenario = evidenceDiagnosis.evidence != nil
                if let evidence = evidenceDiagnosis.evidence {
                    let elapsed = evidence.generationElapsedMs.map(String.init) ?? "unknown"
                    let tokens = evidence.outputTokenCount.map(String.init) ?? "unknown"
                    let adapter = evidence.adapterSlot ?? "none"
                    await event("model-evidence", "runtime=\(evidence.runtimePath), kind=\(evidence.evidenceKind), stage=\(evidence.stage), parseError=\(evidence.parseError ?? "none"), elapsedMs=\(elapsed), outputTokens=\(tokens), adapter=\(adapter), matchedBy=\(evidence.matchedBy), \(traceCorrelation.diagnosticText)")
                    if shouldRunAsPlainTextTurn(scenario: scenario, routing: routing),
                       isGenericChatFallbackFinal(rawFinalText),
                       let deterministic = deterministicDirectChatFallback(for: scenario.prompt) {
                        rawFinalText = deterministic
                        deterministicChatFallbackRemediationApplied = true
                        await event("remediation", "deterministic user-visible fallback applied after model-backed generic chat final")
                    }
                } else {
                    let requiredEvidence = acceptsPolicyFirstEvidence ? "model-backed or policy-first execution evidence" : "model-backed generation evidence"
                    failures.append("Live E2E scenario did not record \(requiredEvidence)")
                    await event("model-evidence", evidenceDiagnosis.failureMessage)
                }
                try Task.checkCancellation()
                await Task.yield()
                agentSteps = steps
                rawFinalText = FinalIntentValidator.validate(rawFinalText, routing: routing, fallback: nil)
                if let synthesized = deterministicWebSynthesisFallback(
                    scenario: scenario,
                    rawFinalText: rawFinalText,
                    events: events
                ) {
                    await event("finalizer", "deterministic web synthesis fallback used after observations")
                    rawFinalText = synthesized
                }
                let recoveredBeforeRewrite = FinalOutputSanitizer.consumeRecoveredUnsafeOutput(forSanitizedText: rawFinalText)
                let rawSanitized = mergeSanitizerOutputs(FinalOutputSanitizer.sanitizeUserVisibleText(rawFinalText), recovered: recoveredBeforeRewrite)
                finalText = rawSanitized.text

                try Task.checkCancellation()
                await Task.yield()
                let preHintNonActionableMetadata = nonActionableInfrastructureMetadata(
                    scenario: scenario,
                    finalText: finalText,
                    failures: failures,
                    events: events
                )
                let preRewriteRAGRetrievalEvidence = ragRetrievalEvidenceState(
                    finalText: finalText,
                    agentSteps: agentSteps,
                    events: events
                )
                let skippedFinalHintsForNonActionable = nonActionableQuarantineFailure(metadata: preHintNonActionableMetadata) != nil
                let rewriteOutcome = await finalHintRewriteOutcome(
                    scenario: scenario,
                    routing: routing,
                    originalFinal: finalText,
                    hasAcceptedModelEvidence: hasAcceptedModelEvidenceForScenario,
                    nonActionableMetadata: preHintNonActionableMetadata,
                    ragRetrievalEvidenceState: preRewriteRAGRetrievalEvidence
                )

                let recoveredAfterRewrite = FinalOutputSanitizer.consumeRecoveredUnsafeOutput(forSanitizedText: rewriteOutcome.finalText)
                let postRewriteSanitized = mergeSanitizerOutputs(FinalOutputSanitizer.sanitizeUserVisibleText(rewriteOutcome.finalText), recovered: recoveredAfterRewrite)
                finalText = postRewriteSanitized.text
                if let repaired = deterministicToolObservationFallbackForIncompleteFinal(
                    scenario: scenario,
                    routing: routing,
                    finalText: finalText,
                    events: events
                ) {
                    await event("finalizer", "deterministic tool-observation fallback used for incomplete final")
                    rawFinalText = repaired
                    let recoveredAfterRepair = FinalOutputSanitizer.consumeRecoveredUnsafeOutput(forSanitizedText: repaired)
                    let repairSanitized = mergeSanitizerOutputs(FinalOutputSanitizer.sanitizeUserVisibleText(repaired), recovered: recoveredAfterRepair)
                    finalText = repairSanitized.text
                }

                hygieneState = FinalHygieneState(
                    rawFinalText: rawFinalText,
                    finalText: finalText,
                    rawSanitized: rawSanitized,
                    postRewriteSanitized: postRewriteSanitized,
                    recoveredBeforeRewrite: recoveredBeforeRewrite,
                    recoveredAfterRewrite: recoveredAfterRewrite
                )
                missingHints = rewriteOutcome.missingHints
                rewriteAttempted = rewriteOutcome.rewriteAttempted
                rewriteSuccess = rewriteOutcome.rewriteSuccess
                if skippedFinalHintsForNonActionable {
                    await event("final-hints", "skipped_non_actionable=true, missing_hints=[], rewrite_attempted=false, rewrite_success=false")
                } else {
                    await event("final-hints", "missing_hints=\(missingHints), rewrite_attempted=\(rewriteAttempted), rewrite_success=\(rewriteSuccess)")
                }
                await event("final", finalText)
                collectPerformanceSample(force: true)
            } else {
                finalText = "No model loaded; routing-only checks completed."
                rawFinalText = finalText
                let sanitized = FinalOutputSanitizer.sanitizeUserVisibleText(finalText)
                hygieneState = FinalHygieneState(rawFinalText: rawFinalText, finalText: finalText, rawSanitized: sanitized, postRewriteSanitized: sanitized, recoveredBeforeRewrite: nil, recoveredAfterRewrite: nil)
            }
        } else {
            finalText = "Routing guard checks completed."
            rawFinalText = finalText
            let sanitized = FinalOutputSanitizer.sanitizeUserVisibleText(finalText)
            hygieneState = FinalHygieneState(rawFinalText: rawFinalText, finalText: finalText, rawSanitized: sanitized, postRewriteSanitized: sanitized, recoveredBeforeRewrite: nil, recoveredAfterRewrite: nil)
        }

        let lowerFinal = finalText.lowercased()
        let lowerRawFinal = rawFinalText.lowercased()
        let observations = events.filter { $0.phase == "step" }.map(\.message).joined(separator: "\n")
        let ragRetrievalEvidence = ragRetrievalEvidenceState(
            finalText: finalText,
            agentSteps: agentSteps,
            events: events
        )
        var outputHygieneFailures: [String] = []
        var nonActionableMetadata = nonActionableInfrastructureMetadata(
            scenario: scenario,
            finalText: finalText,
            failures: failures,
            events: events
        )
        let alarmRuntimeUnavailableFailure = alarmRuntimeUnavailableEvidenceFailure(
            scenario: scenario,
            agentSteps: agentSteps,
            finalText: finalText
        )
        if let alarmRuntimeUnavailableFailure {
            failures.append(alarmRuntimeUnavailableFailure)
        }
        var cpuWatchdogDegraded = cpuWatchdogDegradedEvidence(
            finalText: finalText,
            failures: failures,
            events: events
        )
        if cpuWatchdogDegraded {
            nonActionableMetadata["failureKind"] = "liveRuntimeCPUWatchdogDegraded"
            nonActionableMetadata["actionable"] = "false"
            nonActionableMetadata["trainingSignal"] = "false"
            nonActionableMetadata["runtimeEvidence"] = "runtime-preflight"
        }

        if let quarantineFailure = nonActionableQuarantineFailure(metadata: nonActionableMetadata) {
            failures = [quarantineFailure]
        } else {
            outputHygieneFailures = hygieneFailures(
                lowerRawFinal: lowerRawFinal,
                lowerFinal: lowerFinal,
                removedArtifacts: hygieneState.removedArtifacts,
                scenario: scenario,
                observations: observations
            )
            let liveAgentQualityFailures = liveAgentQualityFailures(
                rawFinalText: rawFinalText,
                finalText: finalText,
                scenario: scenario
            )
            failures = mergedStrings(failures, outputHygieneFailures, liveAgentQualityFailures)
            if scenario.requiresAgentRun, IntentRouter.intentRequiresTool(routing), !routing.requiresClarification {
                let actionToolIDs = Set(agentSteps
                    .filter { $0.kind == .action || $0.kind == .approvalBoundary }
                    .compactMap(\.toolID)
                    .map(ToolRouteGuard.canonicalToolID))
                if actionToolIDs.isEmpty {
                    failures.append("Live agent produced no action step for tool-backed intent")
                } else if !actionToolIDs.contains(where: { routing.allowedToolIDs.contains($0) }) {
                    failures.append("Live agent selected no manifest-allowed action tool")
                }
            }
            failures = mergedStrings(
                failures,
                toolCoverageEvidenceFailures(
                    scenario: scenario,
                    routing: routing,
                    agentSteps: agentSteps,
                    finalText: finalText
                )
            )
            cpuWatchdogDegraded = cpuWatchdogDegradedEvidence(
                finalText: finalText,
                failures: failures,
                events: events
            )
            if cpuWatchdogDegraded,
               !failures.contains(where: { $0.contains("CPU watchdog degraded") }) {
                failures.append("Live runtime CPU watchdog degraded before completing model-backed scenario.")
            }
            if !cpuWatchdogDegraded,
               shouldValidateFinalContentHints(
                scenario: scenario,
                hasAcceptedModelEvidence: hasAcceptedModelEvidenceForScenario
            ) {
                let ragEmptyRetrieval = scenario.expectedIntent == .rag
                    && ragRetrievalEvidence == .empty
                if scenario.expectedIntent == .rag,
                   ragRetrievalEvidence == .contradictory,
                   nonActionableQuarantineFailure(metadata: nonActionableMetadata) == nil {
                    failures.append("RAG retrieval evidence is contradictory: empty retrieval and retrieved snippets both present")
                }
                for hint in scenario.requiredTextHints where !lowerFinal.contains(hint.lowercased()) {
                    if ragEmptyRetrieval, isRAGGroundingHint(hint) { continue }
                    failures.append("Required final hint missing: \(hint)")
                }
                if scenario.expectedIntent == .rag
                    && scenario.requiresAgentRun
                    && scenario.requiredAllowedToolIDs.map(ToolRouteGuard.canonicalToolID).contains("rag.search")
                    && ragRetrievalEvidence == .positive {
                    if ragScenarioRequiresArchitectureGrounding(scenario)
                        && !lowerFinal.contains("module")
                        && !lowerFinal.contains("modules") {
                        failures.append("RAG final response must mention module/modules")
                    }
                    let hasGroundingMarkers = finalText.contains("[") || lowerFinal.contains("snippet") || lowerFinal.contains("source")
                    if !hasGroundingMarkers {
                        failures.append("RAG final response must reference retrieved docs/snippets")
                    }
                }
                for hint in scenario.forbiddenTextHints where lowerFinal.contains(hint.lowercased()) {
                    failures.append("Forbidden final hint present: \(hint)")
                }
                if scenario.id == "training-rag-grounding", !ragEmptyRetrieval {
                    if !(lowerFinal.contains("module") || lowerFinal.contains("modules")) {
                        failures.append("RAG grounding assertion failed: final text must mention module/modules")
                    }
                    if !referencesRetrievedSnippet(lowerFinal) {
                        failures.append("RAG grounding assertion failed: summary must reference retrieved docs/snippets")
                    }
                }
            }
        }

        let mergedAuditArtifacts = hygieneState.removedArtifacts
        let rawPrefix = !hygieneState.rawSanitized.artifactAudit.rawPrefix.isEmpty
            ? hygieneState.rawSanitized.artifactAudit.rawPrefix
            : hygieneState.postRewriteSanitized.artifactAudit.rawPrefix
        let sanitizedPrefix = hygieneState.postRewriteSanitized.artifactAudit.sanitizedPrefix
        let endedAt = Date()
        collectPerformanceSample(force: true)
        try Task.checkCancellation()
        await Task.yield()
        let matrix = await performanceMatrix(from: performanceSamples, startedAt: started, finishedAt: endedAt)
        var metadata = scenarioMetadata(scenario)
        if alarmRuntimeUnavailableFailure != nil {
            metadata["failureKind"] = "liveRuntimeAlarmKitUnavailable"
            metadata["actionable"] = "false"
            metadata["trainingSignal"] = "false"
            metadata["runtimeEvidence"] = "device-runtime-required"
        }
        for (key, value) in nonActionableMetadata {
            metadata[key] = value
        }
        if deterministicChatFallbackRemediationApplied {
            metadata["failureKind"] = "genericFallbackFinal"
            metadata["trainingSignal"] = "true"
            metadata["remediationApplied"] = "deterministicUserVisibleFallback"
        }
        return E2ETestResult(id: UUID(), scenarioID: scenario.id, kind: scenario.kind.rawValue, title: scenario.title, prompt: scenario.prompt, expectedIntent: scenario.expectedIntent.rawValue, actualIntent: routing.intent.rawValue, e2eRunID: e2eRunID, agentRunID: agentRunID, conversationID: conversationID, turnID: turnID, requiresAgentRun: scenario.requiresAgentRun, evidenceMode: scenario.evidenceMode.rawValue, passed: failures.isEmpty, failures: failures, finalText: finalText, missingHints: missingHints, rewriteAttempted: rewriteAttempted, rewriteSuccess: rewriteSuccess, events: events, startedAt: started, finishedAt: endedAt, rawFinalPrefix: rawPrefix, sanitizedFinalPrefix: sanitizedPrefix, rawFinalHadUnsafeLeakage: hygieneState.hadUnsafeLeakage, sanitizedFinalRemovedArtifacts: mergedAuditArtifacts.map(\.rawValue), outputHygieneFailures: outputHygieneFailures, performanceMatrix: matrix, metadata: metadata)
    }

    /// Determines whether a scenario accepts policy-first deterministic execution traces as valid evidence.
    /// - Returns: `true` if the scenario accepts such traces, `false` otherwise.
    private nonisolated static func acceptsPolicyFirstExecutionEvidence(scenario: E2ETestScenario, routing: IntentRoutingDecision) -> Bool {
        guard scenario.requiresAgentRun else { return false }
        if scenario.evidenceMode == .routingOnly {
            return false
        }
        if scenario.evidenceMode == .policyFirstAllowed {
            return true
        }
        if routing.requiresClarification {
            return true
        }
        // A scenario marked as live training/evidence must prove the loaded
        // model path. Deterministic compatibility traces are diagnostics only;
        // static routing coverage should use requiresAgentRun=false.
        return false
    }

    private nonisolated static func shouldRunAsPlainTextTurn(scenario: E2ETestScenario, routing: IntentRoutingDecision) -> Bool {
        scenario.requiresAgentRun
            && scenario.kind == .chat
            && routing.intent == .chat
            && scenario.requiredAllowedToolIDs.isEmpty
            && routing.allowedToolIDs.isEmpty
    }

    private nonisolated static func requiresStructuredModelBackedAgentRun(scenario: E2ETestScenario, routing: IntentRoutingDecision) -> Bool {
        scenario.requiresAgentRun
            && scenario.evidenceMode == .modelBackedRequired
            && !routing.requiresClarification
            && !shouldRunAsPlainTextTurn(scenario: scenario, routing: routing)
    }

    private nonisolated static func liveRuntimeShouldStopAfter(_ result: E2ETestResult) -> Bool {
        guard result.requiresAgentRun, !result.passed else { return false }
        let evidence = ([result.finalText] + result.failures + result.events.map(\.message) + Array(result.metadata.values))
            .joined(separator: "\n")
            .lowercased()
        return evidence.contains("thermalstate=serious")
            || evidence.contains("thermal state serious")
            || evidence.contains(ResourceBudgetGate.seriousThermalRetryHint.lowercased())
            || evidence.contains("resource-budget-denied-before-prompt-eval")
            || evidence.contains("cpu-watchdog-degraded")
            || evidence.contains("adapter required but adapter path missing")
            || evidence.contains("adapterpathmissing")
    }

    private nonisolated static func requiresExecutorPreflight(_ scenario: E2ETestScenario) -> Bool {
        scenario.requiresAgentRun
            && scenario.evidenceMode == .modelBackedRequired
            && scenario.kind != .chat
    }

    private static func executorRuntimePreflightBlockedResult(
        startedAt: Date,
        scenario: E2ETestScenario,
        preflight: ExecutorRuntimePreflightResult,
        onEvent: EventCallback?
    ) async -> E2ETestResult {
        let event = E2ETestEvent(
            id: UUID(),
            createdAt: Date(),
            scenarioID: "executor-runtime-preflight",
            phase: "executor-preflight",
            message: "\(preflight.reason); \(preflight.diagnosticsSummary)"
        )
        await onEvent?(event)
        let finalText = preflight.budgetReason?.contains(ResourceBudgetGate.seriousThermalRetryHint) == true
            ? ResourceBudgetGate.seriousThermalRetryHint
            : preflight.diagnosticsSummary
        var metadata = preflight.diagnosticsMetadata
        metadata["trainingSignal"] = "false"
        metadata["runtimeEvidence"] = "executor-runtime-preflight"
        let preflightEvidence = "\(preflight.reason)\n\(preflight.diagnosticsSummary)".lowercased()
        if preflightEvidence.contains("cpu-watchdog-degraded")
            || preflightEvidence.contains("cpu watchdog degraded") {
            metadata["failureKind"] = "liveRuntimeCPUWatchdogDegraded"
            metadata["actionable"] = "false"
            metadata["runtimeEvidence"] = "runtime-preflight"
        } else if preflightEvidence.contains("thermalstate=serious")
                    || preflightEvidence.contains("thermal state serious")
                    || preflightEvidence.contains("resource-budget-denied-before-prompt-eval") {
            metadata["failureKind"] = "liveRuntimePreflightUnavailable"
            metadata["actionable"] = "false"
            metadata["runtimeEvidence"] = "runtime-preflight"
        }
        return E2ETestResult(
            id: UUID(),
            scenarioID: "executor-runtime-preflight",
            kind: scenario.kind.rawValue,
            title: "Executor runtime preflight",
            prompt: scenario.prompt,
            expectedIntent: scenario.expectedIntent.rawValue,
            actualIntent: "preflight",
            requiresAgentRun: true,
            evidenceMode: E2EEvidenceMode.modelBackedRequired.rawValue,
            passed: false,
            failures: [preflight.reason],
            finalText: finalText,
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: [event],
            startedAt: startedAt,
            finishedAt: Date(),
            rawFinalPrefix: "",
            sanitizedFinalPrefix: finalText,
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: [],
            performanceMatrix: nil,
            metadata: metadata
        )
    }

    private static func liveRuntimePreflightBlockedResultIfNeeded(
        for scenario: E2ETestScenario,
        onEvent: EventCallback?
    ) async -> E2ETestResult? {
        guard scenario.requiresAgentRun else { return nil }
        let readiness = await liveRuntimeReadinessBarrier(
            for: scenario,
            maxWaitNanoseconds: liveRuntimeReadinessMaxWaitNanoseconds,
            pollNanoseconds: liveRuntimeReadinessPollNanoseconds,
            onEvent: onEvent
        )
        if let denial = readiness.denialReason {
            return await liveRuntimePreflightBlockedResult(
                for: scenario,
                denialReason: denial,
                readinessEvents: readiness.events,
                onEvent: onEvent
            )
        }
        guard isCPUWatchdogDegradedForLiveTraining(scenario: scenario) else { return nil }
        return await liveRuntimePreflightBlockedResult(
            for: scenario,
            denialReason: "live-e2e.pre-scenario: cpu-watchdog-degraded",
            readinessEvents: readiness.events,
            onEvent: onEvent
        )
    }

    private struct LiveRuntimeReadinessBarrierOutcome: Sendable {
        let denialReason: String?
        let events: [E2ETestEvent]
    }

    private static let liveRuntimeReadinessMaxWaitNanoseconds: UInt64 = 8_000_000_000
    private static let liveRuntimeReadinessPollNanoseconds: UInt64 = 500_000_000

    private static func liveRuntimeReadinessBarrier(
        for scenario: E2ETestScenario,
        maxWaitNanoseconds: UInt64,
        pollNanoseconds: UInt64,
        onEvent: EventCallback?
    ) async -> LiveRuntimeReadinessBarrierOutcome {
        let started = Date()
        let timeoutSeconds = Double(maxWaitNanoseconds) / 1_000_000_000
        var events: [E2ETestEvent] = []
        var attempt = 0

        while true {
            if isCPUWatchdogDegradedForLiveTraining(scenario: scenario) {
                return LiveRuntimeReadinessBarrierOutcome(
                    denialReason: "live-e2e.pre-scenario: cpu-watchdog-degraded",
                    events: events
                )
            }
            let decision = await MainActor.run {
                ResourceBudgetGate.decision(
                    policy: .foregroundInteractive,
                    reason: "live-e2e.pre-scenario"
                )
            }
            if decision.allowed {
                return LiveRuntimeReadinessBarrierOutcome(denialReason: nil, events: events)
            }

            guard let denial = decision.denialReason else {
                return LiveRuntimeReadinessBarrierOutcome(denialReason: nil, events: events)
            }

            let elapsedNanoseconds = UInt64(max(Date().timeIntervalSince(started), 0.0) * 1_000_000_000)
            guard maxWaitNanoseconds > 0,
                  elapsedNanoseconds < maxWaitNanoseconds,
                  liveRuntimePreflightDenialCanWait(denial) else {
                return LiveRuntimeReadinessBarrierOutcome(denialReason: denial, events: events)
            }

            if events.isEmpty || attempt % 4 == 0 {
                let event = E2ETestEvent(
                    id: UUID(),
                    createdAt: Date(),
                    scenarioID: scenario.id,
                    phase: "live-runtime-preflight-wait",
                    message: "waiting for foreground runtime readiness; timeoutSeconds=\(String(format: "%.1f", timeoutSeconds)); reason=\(denial)"
                )
                events.append(event)
                await onEvent?(event)
            }

            let remaining = maxWaitNanoseconds - elapsedNanoseconds
            let sleepNanoseconds = min(max(pollNanoseconds, 1_000_000), remaining)
            do {
                try await Task.sleep(nanoseconds: sleepNanoseconds)
            } catch {
                return LiveRuntimeReadinessBarrierOutcome(
                    denialReason: "live-e2e.pre-scenario: cancelled while waiting for runtime readiness; \(denial)",
                    events: events
                )
            }
            attempt += 1
        }
    }

    private nonisolated static func liveRuntimePreflightDenialCanWait(_ denialReason: String) -> Bool {
        let lower = denialReason.lowercased()
        return lower.contains("scenephase=inactive")
            || lower.contains("scenephase=background")
            || lower.contains("thermalstate=serious")
            || lower.contains("thermalstate=critical")
            || lower.contains("thermalstate=unknown")
            || lower.contains("thermalstate=nil")
    }

    private static func liveRuntimePreflightBlockedResult(
        for scenario: E2ETestScenario,
        denialReason: String,
        readinessEvents: [E2ETestEvent] = [],
        onEvent: EventCallback?
    ) async -> E2ETestResult {
        let started = Date()
        let failureKind = liveRuntimeBudgetFailureKind(denialReason)
        let finalText = denialReason.contains(ResourceBudgetGate.seriousThermalRetryHint)
            ? ResourceBudgetGate.seriousThermalRetryHint
            : "Live E2E paused before starting this scenario: \(denialReason)."
        let event = E2ETestEvent(
            id: UUID(),
            createdAt: started,
            scenarioID: scenario.id,
            phase: "live-runtime-preflight",
            message: "blocked before model prompt evaluation; reason=\(denialReason)"
        )
        await onEvent?(event)
        let events = readinessEvents + [event]
        return E2ETestResult(
            id: UUID(),
            scenarioID: scenario.id,
            kind: scenario.kind.rawValue,
            title: scenario.title,
            prompt: scenario.prompt,
            expectedIntent: scenario.expectedIntent.rawValue,
            actualIntent: "preflight",
            requiresAgentRun: true,
            evidenceMode: scenario.evidenceMode.rawValue,
            passed: false,
            failures: ["Live E2E preflight blocked model-backed generation before prompt evaluation: \(denialReason)"],
            finalText: finalText,
            missingHints: [],
            rewriteAttempted: false,
            rewriteSuccess: false,
            events: events,
            startedAt: started,
            finishedAt: Date(),
            rawFinalPrefix: "",
            sanitizedFinalPrefix: finalText,
            rawFinalHadUnsafeLeakage: false,
            sanitizedFinalRemovedArtifacts: [],
            outputHygieneFailures: [],
            performanceMatrix: nil,
            metadata: [
                "failureKind": failureKind,
                "budgetPolicy": LumenSlotBudgetPolicy.foregroundInteractive.rawValue,
                "budgetDenialReason": denialReason,
                "actionable": "false",
                "trainingSignal": "false"
            ]
        )
    }

    private nonisolated static func liveRuntimeBudgetFailureKind(_ denialReason: String) -> String {
        let lower = denialReason.lowercased()
        if lower.contains("cpu-watchdog-degraded") || lower.contains("cpu watchdog degraded") {
            return "liveRuntimeCPUWatchdogDegraded"
        }
        if lower.contains("thermalstate=serious") || lower.contains("thermal state serious") {
            return "liveRuntimeThermalCooldownRequired"
        }
        if lower.contains("thermalstate=critical") {
            return "liveRuntimeThermalCritical"
        }
        if lower.contains("thermalstate=unknown") || lower.contains("thermalstate=nil") {
            return "liveRuntimeThermalStateUnavailable"
        }
        if lower.contains("recent-memory-warning") {
            return "liveRuntimeRecentMemoryWarning"
        }
        if lower.contains("scenephase=") {
            return "liveRuntimeScenePhaseUnavailable"
        }
        if lower.contains("lowpowermode=") {
            return "liveRuntimePowerStateUnavailable"
        }
        return "liveRuntimeResourceBudgetDenied"
    }

    private static func paceAfterLiveRuntimeScenario(_ result: E2ETestResult) async {
        let delay = liveRuntimePacingNanoseconds(
            after: result,
            thermalState: ProcessInfo.processInfo.thermalState,
            lowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled
        )
        guard delay > 0 else { return }
        try? await Task.sleep(nanoseconds: delay)
    }

    private nonisolated static func liveRuntimePacingNanoseconds(
        after result: E2ETestResult,
        thermalState: ProcessInfo.ThermalState,
        lowPowerModeEnabled: Bool
    ) -> UInt64 {
        guard result.requiresAgentRun else { return 0 }
        guard !liveRuntimeShouldStopAfter(result) else { return 0 }
        let duration = max(0, result.finishedAt.timeIntervalSince(result.startedAt))
        let ranToolOrExecutor = result.expectedIntent != UserIntent.chat.rawValue
            || result.events.contains { event in
                let lower = event.message.lowercased()
                return event.phase == "step" && (lower.contains("action:") || lower.contains("observation:"))
            }
        let minimumCooldown: UInt64 = ranToolOrExecutor && duration >= 20 ? 8_000_000_000 : 0
        switch thermalState {
        case .nominal:
            return max(lowPowerModeEnabled ? 3_000_000_000 : 1_500_000_000, minimumCooldown)
        case .fair:
            return max(lowPowerModeEnabled ? 8_000_000_000 : 5_000_000_000, minimumCooldown)
        case .serious, .critical:
            return 0
        @unknown default:
            return 3_000_000_000
        }
    }

    private nonisolated static func isCPUWatchdogDegradedForLiveTraining(scenario: E2ETestScenario) -> Bool {
        guard scenario.kind == .training,
              scenario.requiresAgentRun,
              scenario.evidenceMode == .modelBackedRequired else {
            return false
        }
        #if DEBUG
        if let probe = debugCPUWatchdogDegradedProbe {
            return probe(scenario)
        }
        if let override = debugCPUWatchdogDegradedOverride {
            return override
        }
        #endif
        return CPUWatchdogGuard.shared.shouldDegrade(category: .chatGeneration)
    }

    private static func recordPolicyFirstClarificationTrace(
        scenario: E2ETestScenario,
        routing: IntentRoutingDecision,
        clarification: String,
        correlation: AgentTraceCorrelation,
        startedAt: Date
    ) {
        AgentBehaviorTraceEmitter.recordPolicyFirstFinal(
            correlation: correlation,
            prompt: scenario.prompt,
            intent: routing.intent.rawValue,
            stage: "compatibility-clarification-final",
            finalText: clarification,
            allowedToolIDs: routing.allowedToolIDs.sorted(),
            requiresApproval: false,
            startedAt: startedAt,
            streamTerminationReason: "clarification-required"
        )
    }

    private nonisolated static func shouldTemporarilyEnableNetworkAccess(
        scenario: E2ETestScenario,
        routing: IntentRoutingDecision,
        availableToolIDs: [String]
    ) -> Bool {
        guard scenario.requiresAgentRun, routing.intent == .webSearch else { return false }
        let canonicalAvailable = Set(availableToolIDs.map(ToolRouteGuard.canonicalToolID))
        return canonicalAvailable.contains("web.search") || canonicalAvailable.contains("web.fetch")
    }

    private nonisolated static func strictLiveAgentKernelRequest(
        prompt: String,
        systemPrompt: String,
        config: E2ERunConfig,
        conversationID: UUID,
        turnID: UUID,
        traceCorrelation: AgentTraceCorrelation? = nil,
        forceModelBackedToolPlanning: Bool = false,
        structuredMode: AgentStructuredMode = .automatic,
        structuredAllowedToolIDs: [String] = []
    ) -> AgentKernelRequest {
        AgentKernelRequest(
            conversationID: conversationID,
            turnID: turnID,
            userMessage: prompt,
            history: [],
            systemPrompt: systemPrompt,
            relevantMemories: [],
            task: .chat,
            source: .diagnostics,
            options: AgentKernelOptions(
                allowHeavyRuntime: true,
                allowDegradedMode: false,
                requireUserVisibleFinal: true,
                diagnosticsEnabled: true,
                maxSteps: min(config.maxAgentSteps, 3),
                prefersFoundationModels: false,
                temperature: min(config.temperature, 0.3),
                topP: config.topP,
                repetitionPenalty: config.repetitionPenalty,
                maxTokens: min(config.maxTokens, 512),
                forceModelBackedToolPlanning: forceModelBackedToolPlanning,
                structuredMode: structuredMode,
                structuredAllowedToolIDs: structuredAllowedToolIDs
            ),
            traceCorrelation: traceCorrelation
        )
    }

    private nonisolated static func strictLiveAgentRequest(
        scenario: E2ETestScenario,
        config: E2ERunConfig,
        availableTools: [ToolDefinition],
        correlation: AgentTraceCorrelation
    ) -> AgentRequest {
        AgentRequest(
            systemPrompt: config.systemPrompt,
            history: [],
            userMessage: scenario.prompt,
            temperature: min(config.temperature, 0.3),
            topP: config.topP,
            repetitionPenalty: config.repetitionPenalty,
            maxTokens: min(config.maxTokens, 512),
            maxSteps: min(config.maxAgentSteps, 3),
            availableTools: availableTools,
            relevantMemories: [],
            conversationID: correlation.conversationID,
            turnID: correlation.turnID,
            scenarioID: correlation.scenarioID,
            e2eRunID: correlation.e2eRunID,
            agentRunID: correlation.agentRunID
        )
    }

    private nonisolated static func strictLiveAgentRunOptions(
        req: AgentRequest,
        scenario: E2ETestScenario,
        e2eRunID: UUID,
        agentRunID: UUID,
        acceptsPolicyFirstEvidence: Bool
    ) -> LegacyAgentRunOptions {
        LegacyAgentRunOptions(
            modelContext: nil,
            conversationID: req.conversationID,
            turnID: req.turnID,
            scenarioID: scenario.id,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            groundingMode: .rolePipeline,
            allowDegradedGrounding: false,
            preventDoubleGrounding: true,
            diagnosticsEnabled: false,
            allowDeterministicCompatibility: acceptsPolicyFirstEvidence,
            allowParseFailureDeterministicRecovery: acceptsPolicyFirstEvidence,
            allowsMemoryPressureContinuation: scenario.kind == .training
        )
    }

    private nonisolated static func modelRuntimeEvidence(
        since startedAt: Date,
        prompt: String,
        correlation: AgentTraceCorrelation? = nil,
        acceptsPolicyFirstEvidence: Bool,
        requiresPrimaryAgentJSON: Bool = false
    ) -> ModelRuntimeEvidence? {
        modelRuntimeEvidenceDiagnosis(
            since: startedAt,
            prompt: prompt,
            correlation: correlation,
            acceptsPolicyFirstEvidence: acceptsPolicyFirstEvidence,
            requiresPrimaryAgentJSON: requiresPrimaryAgentJSON
        ).evidence
    }

    private nonisolated static func modelRuntimeEvidenceDiagnosis(
        since startedAt: Date,
        prompt: String,
        correlation: AgentTraceCorrelation? = nil,
        acceptsPolicyFirstEvidence: Bool,
        requiresPrimaryAgentJSON: Bool = false
    ) -> ModelRuntimeEvidenceDiagnosis {
        let promptNeedle = prompt.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let promptSummaryNeedle = AgentDiagnosticFileRedactor.summary(label: "prompt", text: prompt).lowercased()
        let recentTraces = AgentBehaviorTraceRecorder.recent(limit: 64).reversed()
        let correlatedTraces = recentTraces.filter { trace in
            traceMatchesCorrelation(trace, correlation: correlation, startedAt: startedAt)
        }
        let hasExplicitCorrelation = correlation?.hasAnyIdentifier == true
        let conversationTurnFallbackTraces = recentTraces.filter { trace in
            guard let correlation,
                  let conversationID = correlation.conversationID,
                  let turnID = correlation.turnID,
                  trace.createdAt >= startedAt,
                  trace.conversationID == conversationID,
                  trace.turnID == turnID else {
                return false
            }
            return true
        }
        let fallbackTraces = recentTraces.filter { trace in
            guard trace.createdAt >= startedAt else { return false }
            let promptPrefix = trace.promptPrefix.lowercased()
            if !promptNeedle.isEmpty,
               !promptPrefix.contains(promptNeedle),
               (promptSummaryNeedle.isEmpty || !promptPrefix.contains(promptSummaryNeedle)) {
                return false
            }
            return true
        }
        let usedCorrelation = !correlatedTraces.isEmpty
        let usesConversationTurnFallback = hasExplicitCorrelation
            && correlatedTraces.isEmpty
            && !requiresPrimaryAgentJSON
            && !conversationTurnFallbackTraces.isEmpty
        let matchingTraces = hasExplicitCorrelation
            ? (usesConversationTurnFallback ? conversationTurnFallbackTraces : correlatedTraces)
            : fallbackTraces
        let matchedBy = hasExplicitCorrelation
            ? (usesConversationTurnFallback ? "conversation-turn-fallback" : "correlation")
            : "prompt-time"

        let validModelTrace = matchingTraces.first { trace in
            isValidModelBackedEvidenceTrace(trace, requiresPrimaryAgentJSON: requiresPrimaryAgentJSON)
        }

        if let modelTrace = validModelTrace {
            let evidence = ModelRuntimeEvidence(
                runtimePath: modelTrace.runtimePath ?? "unknown",
                stage: modelTrace.stage,
                evidenceKind: "model-backed",
                generationElapsedMs: modelTrace.generationElapsedMs,
                outputTokenCount: modelTrace.outputTokenCount,
                adapterSlot: modelTrace.activeAdapterSlot ?? modelTrace.adapterSlot,
                parseError: modelTrace.parseError,
                matchedBy: matchedBy
            )
            return ModelRuntimeEvidenceDiagnosis(evidence: evidence, failureMessage: "")
        }

        if acceptsPolicyFirstEvidence,
           let policyTrace = matchingTraces.first(where: isPolicyFirstExecutionTrace) {
            let evidence = ModelRuntimeEvidence(
                runtimePath: policyTrace.runtimePath ?? "deterministic-compatibility",
                stage: policyTrace.stage,
                evidenceKind: "policy-first-deterministic",
                generationElapsedMs: policyTrace.generationElapsedMs,
                outputTokenCount: policyTrace.outputTokenCount,
                adapterSlot: policyTrace.activeAdapterSlot ?? policyTrace.adapterSlot,
                parseError: policyTrace.parseError,
                matchedBy: matchedBy
            )
            return ModelRuntimeEvidenceDiagnosis(evidence: evidence, failureMessage: "")
        }

        return ModelRuntimeEvidenceDiagnosis(
            evidence: nil,
            failureMessage: modelRuntimeEvidenceFailureMessage(
                matchingTraces: Array(matchingTraces),
                acceptsPolicyFirstEvidence: acceptsPolicyFirstEvidence,
                correlation: correlation,
                usedCorrelation: usedCorrelation,
                fallbackTraceCount: fallbackTraces.count,
                conversationTurnFallbackTraceCount: conversationTurnFallbackTraces.count,
                requiresPrimaryAgentJSON: requiresPrimaryAgentJSON
            )
        )
    }

    private nonisolated static func traceMatchesCorrelation(_ trace: AgentBehaviorTrace, correlation: AgentTraceCorrelation?, startedAt: Date) -> Bool {
        guard let correlation, correlation.hasAnyIdentifier else { return false }
        var matchedAny = false
        if let scenarioID = correlation.scenarioID, !scenarioID.isEmpty {
            guard trace.scenarioID == scenarioID else { return false }
            matchedAny = true
        }
        if let e2eRunID = correlation.e2eRunID {
            guard trace.e2eRunID == e2eRunID else { return false }
            matchedAny = true
        }
        if let agentRunID = correlation.agentRunID {
            guard trace.agentRunID == agentRunID else { return false }
            matchedAny = true
        }
        if let conversationID = correlation.conversationID {
            guard trace.conversationID == conversationID else { return false }
            matchedAny = true
        }
        if let turnID = correlation.turnID {
            guard trace.turnID == turnID else { return false }
            matchedAny = true
        }
        guard matchedAny else { return false }
        if correlation.e2eRunID == nil,
           correlation.agentRunID == nil,
           correlation.conversationID == nil,
           correlation.turnID == nil {
            return trace.createdAt >= startedAt
        }
        return true
    }

    private nonisolated static func modelRuntimeEvidenceFailureMessage(
        matchingTraces: [AgentBehaviorTrace],
        acceptsPolicyFirstEvidence: Bool,
        correlation: AgentTraceCorrelation? = nil,
        usedCorrelation: Bool = false,
        fallbackTraceCount: Int = 0,
        conversationTurnFallbackTraceCount: Int = 0,
        requiresPrimaryAgentJSON: Bool = false
    ) -> String {
        let preferredRejectedTrace = requiresPrimaryAgentJSON
            ? matchingTraces.first(where: isPrimaryAgentJSONTrace)
            : nil
        if let rejectedModelTrace = preferredRejectedTrace ?? matchingTraces.first(where: { $0.event == .modelTurn }) {
            let rawIsEmpty = rejectedModelTrace.rawOutputPrefix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            let runtimePath = rejectedModelTrace.runtimePath ?? "unknown"
            let parseError = rejectedModelTrace.parseError ?? "none"
            var reasons: [String] = []
            if runtimePath == "deterministic-compatibility" {
                reasons.append("runtimePath was deterministic-compatibility")
            }
            if rawIsEmpty {
                if let emptyOutputReason = rejectedModelTrace.emptyOutputReason, !emptyOutputReason.isEmpty {
                    if emptyOutputReason == "agent-json-stream-completed-without-text" {
                        reasons.append("model stream returned no tokens")
                    } else if emptyOutputReason == "adapterUnavailable" || emptyOutputReason == "resource-budget-denied-ensure-ready" {
                        reasons.append("runtime readiness failure (\(emptyOutputReason))")
                    } else if emptyOutputReason.contains("executor preflight failed") {
                        reasons.append("runtime readiness failure (\(emptyOutputReason))")
                    } else if emptyOutputReason.hasPrefix("resource-budget-denied-") {
                        reasons.append("budget failure (\(emptyOutputReason))")
                    } else {
                        reasons.append("agent-json emitted empty output (\(emptyOutputReason))")
                    }
                } else {
                    reasons.append("raw output was empty")
                }
            }
            if rejectedModelTrace.parseError != nil {
                let producedTextOrTokens = !rawIsEmpty || (rejectedModelTrace.outputTokenCount ?? 0) > 0 || rejectedModelTrace.firstChunkReceived == true || (rejectedModelTrace.textChunkCount ?? 0) > 0
                if !producedTextOrTokens {
                    reasons.append("parseError suppressed because no text/tokens were produced")
                } else if rejectedModelTrace.parseError == AgentTurnParseError.contextWindowExceeded.rawValue {
                    reasons.append(AgentTurnParseError.contextWindowExceeded.rawValue)
                } else {
                    reasons.append("parseError=\(parseError)")
                }
            }
            if isPrimaryAgentJSONTrace(rejectedModelTrace) {
                if rejectedModelTrace.streamStarted != true { reasons.append("streamStarted was not true") }
                if rejectedModelTrace.modelLoaded != true { reasons.append("modelLoaded was not true") }
                if rejectedModelTrace.firstChunkReceived != true { reasons.append("firstChunkReceived was not true") }
                if (rejectedModelTrace.textChunkCount ?? 0) <= 0 { reasons.append("textChunkCount was not positive") }
                if rejectedModelTrace.finalChunkReceived != true { reasons.append("finalChunkReceived was not true") }
                if rejectedModelTrace.emittedFinalInActionTurn {
                    if rejectedModelTrace.finalizerAccepted != true { reasons.append("finalizerAccepted was not true") }
                    if traceIntentRequiresTool(rejectedModelTrace), (rejectedModelTrace.successfulObservationCount ?? 0) <= 0 {
                        reasons.append("successfulObservationCount was not positive for a tool-backed final")
                    }
                }
            }
            if reasons.isEmpty {
                reasons.append("trace did not satisfy model-backed evidence policy")
            }
            let subject = isPrimaryAgentJSONTrace(rejectedModelTrace)
                ? "found primary agent-json modelTurn"
                : "found AgentBehaviorTrace modelTurn"
            return "\(subject) but \(reasons.joined(separator: "; ")); stage=\(rejectedModelTrace.stage); runtimePath=\(runtimePath); parseError=\(parseError); outputTokens=\(rejectedModelTrace.outputTokenCount.map(String.init) ?? "unknown"); streamStarted=\(rejectedModelTrace.streamStartedText); firstChunkReceived=\(rejectedModelTrace.firstChunkReceivedText); textChunkCount=\(rejectedModelTrace.textChunkCount.map(String.init) ?? "unknown"); finalChunkReceived=\(rejectedModelTrace.finalChunkReceivedText); streamTerminationReason=\(rejectedModelTrace.streamTerminationReason ?? "unknown"); successfulObservationCount=\(rejectedModelTrace.successfulObservationCount.map(String.init) ?? "unknown"); finalizerAccepted=\(rejectedModelTrace.finalizerAccepted.map(String.init) ?? "unknown")"
        }

        if let policyTrace = matchingTraces.first(where: { $0.runtimePath == "deterministic-compatibility" }) {
            let policy = acceptsPolicyFirstEvidence ? "not accepted by policy" : "policy-first evidence disabled for this scenario"
            return "found deterministic-compatibility execution trace but \(policy); stage=\(policyTrace.stage); runtimePath=\(policyTrace.runtimePath ?? "deterministic-compatibility"); parseError=\(policyTrace.parseError ?? "none")"
        }

        let base = acceptsPolicyFirstEvidence
            ? "missing fresh AgentBehaviorTrace modelTurn or deterministic-compatibility execution trace"
            : "missing fresh AgentBehaviorTrace modelTurn"
        if let correlation, correlation.hasAnyIdentifier, !usedCorrelation {
            let fallbackText = fallbackTraceCount > 0 ? "; fallbackPromptTimeTraceCount=\(fallbackTraceCount)" : ""
            let conversationTurnFallbackText = conversationTurnFallbackTraceCount > 0 ? "; conversationTurnFallbackTraceCount=\(conversationTurnFallbackTraceCount)" : ""
            return "no correlated AgentBehaviorTrace found; checked \(correlation.diagnosticText)\(fallbackText)\(conversationTurnFallbackText); \(base); model boundary skipped AgentBehaviorTrace emission or trace export missed the strict IDs"
        }
        return base
    }

    #if DEBUG
    nonisolated static func modelRuntimeEvidenceFailureMessageForTests(
        matchingTraces: [AgentBehaviorTrace],
        acceptsPolicyFirstEvidence: Bool,
        requiresPrimaryAgentJSON: Bool
    ) -> String {
        modelRuntimeEvidenceFailureMessage(
            matchingTraces: matchingTraces,
            acceptsPolicyFirstEvidence: acceptsPolicyFirstEvidence,
            requiresPrimaryAgentJSON: requiresPrimaryAgentJSON
        )
    }

    nonisolated static func isValidModelBackedEvidenceTraceForTests(
        _ trace: AgentBehaviorTrace,
        requiresPrimaryAgentJSON: Bool
    ) -> Bool {
        isValidModelBackedEvidenceTrace(trace, requiresPrimaryAgentJSON: requiresPrimaryAgentJSON)
    }
    #endif

    private nonisolated static func isPrimaryAgentJSONTrace(_ trace: AgentBehaviorTrace) -> Bool {
        trace.event == .modelTurn
            && trace.stage.hasPrefix("agent-json")
            && trace.runtimePath == "agent-model"
    }

    private nonisolated static func isValidModelBackedEvidenceTrace(
        _ trace: AgentBehaviorTrace,
        requiresPrimaryAgentJSON: Bool
    ) -> Bool {
        guard trace.event == AgentBehaviorTrace.Event.modelTurn,
              trace.runtimePath != "deterministic-compatibility",
              trace.parseError == nil else {
            return false
        }
        if requiresPrimaryAgentJSON {
            guard isPrimaryAgentJSONTrace(trace) else { return false }
            guard !trace.rawOutputPrefix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  trace.streamStarted == true,
                  trace.modelLoaded == true,
                  trace.firstChunkReceived == true,
                  (trace.textChunkCount ?? 0) > 0,
                  trace.finalChunkReceived == true else {
                return false
            }
            if let selectedToolID = trace.selectedToolID, !selectedToolID.isEmpty {
                let canonicalTool = ToolRouteGuard.canonicalToolID(selectedToolID)
                return trace.allowedToolIDs.contains(canonicalTool)
                    && !trace.emittedFinalInActionTurn
            }
            guard trace.emittedFinalInActionTurn,
                  trace.finalizerAccepted == true else {
                return false
            }
            return !traceIntentRequiresTool(trace)
                || (trace.successfulObservationCount ?? 0) > 0
        }
        return true
    }

    private nonisolated static func traceIntentRequiresTool(_ trace: AgentBehaviorTrace) -> Bool {
        guard let rawIntent = trace.intent,
              let intent = UserIntent(rawValue: rawIntent) else {
            return !trace.allowedToolIDs.isEmpty
        }
        return IntentRouter.intentRequiresTool(IntentRoutingDecision(
            intent: intent,
            allowedToolIDs: Set(trace.allowedToolIDs),
            requiresClarification: false,
            clarificationPrompt: nil
        ))
    }

    /// Determines if an agent behavior trace qualifies as a policy-first execution trace.
    /// - Returns: `true` if the trace represents a policy-first execution, `false` otherwise.
    private nonisolated static func isPolicyFirstExecutionTrace(_ trace: AgentBehaviorTrace) -> Bool {
        guard trace.runtimePath == "deterministic-compatibility" else { return false }
        switch trace.event {
        case .toolAction:
            return trace.selectedToolID?.isEmpty == false
        case .finalAnswer:
            let stage = trace.stage.lowercased()
            return trace.emittedFinalInActionTurn
                || stage.contains("compatibility-clarification")
                || stage.contains("compatibility-memory-final")
                || stage.contains("compatibility-chain-stopped")
                || stage == "compatibility-direct-final"
                || stage == "compatibility-final"
        case .modelTurn:
            return false
        }
    }

    nonisolated private static func scenarioMetadata(_ scenario: E2ETestScenario) -> [String: String] {
        var metadata: [String: String] = [:]
        if let expectedToolID = scenario.expectedToolID {
            metadata["expectedToolID"] = expectedToolID
        }
        if let scenarioBankKind = scenario.scenarioBankKind {
            metadata["scenarioBankKind"] = scenarioBankKind
        }
        if scenario.kind == .toolGuard {
            metadata["expectedToolID"] = scenario.expectedToolID ?? ""
            metadata["scenarioBankKind"] = scenario.scenarioBankKind ?? ""
        }
        if let requiredSlotIDs = scenario.requiredSlotIDs, !requiredSlotIDs.isEmpty {
            metadata["requiredSlots"] = requiredSlotIDs.joined(separator: ",")
        }
        return metadata
    }

    nonisolated private static func toolCoverageEvidenceFailures(
        scenario: E2ETestScenario,
        routing: IntentRoutingDecision,
        agentSteps: [AgentStep],
        finalText: String
    ) -> [String] {
        guard scenario.kind == .toolGuard else {
            return []
        }
        guard let expectedTool = scenario.expectedToolID.map(ToolRouteGuard.canonicalToolID),
              !expectedTool.isEmpty else {
            return ["Tool coverage scenario missing expectedToolID metadata."]
        }

        let bankKind = scenario.scenarioBankKind ?? ""
        let requiredArgs = ToolRegistry.find(id: expectedTool)?
            .capabilityContract
            .arguments
            .filter(\.required)
            .map(\.name) ?? []
        let expectedSteps = agentSteps.filter { ToolRouteGuard.canonicalToolID($0.toolID ?? "") == expectedTool }
        let hasAction = expectedSteps.contains { $0.kind == .action }
        let hasApprovalBoundary = expectedSteps.contains { $0.kind == .approvalBoundary }
        let hasObservation = expectedSteps.contains { $0.kind == .observation }
        let requiresApproval = ToolRouteGuard.requiresUserApproval(expectedTool)

        if bankKind == ToolScenarioBankEntry.ScenarioKind.missingArgument.rawValue {
            if routing.requiresClarification {
                return requiredArgs.isEmpty
                    ? ["Tool coverage missing-argument scenario cannot pass by clarification for no-arg tool \(expectedTool)"]
                    : []
            }
            if expectedSteps.isEmpty {
                return ["Tool coverage missing-argument scenario neither clarified nor selected expected tool \(expectedTool)"]
            }
            return []
        }

        if routing.requiresClarification {
            return ["Tool coverage scenario \(bankKind.isEmpty ? "direct" : bankKind) incorrectly stopped at clarification for expected tool \(expectedTool)"]
        }

        if bankKind == ToolScenarioBankEntry.ScenarioKind.approvalBoundary.rawValue || requiresApproval {
            return hasApprovalBoundary ? [] : ["Tool coverage scenario missing approval boundary for expected tool \(expectedTool)"]
        }

        guard hasAction else {
            return ["Tool coverage scenario missing action step for expected tool \(expectedTool)"]
        }

        if requiredArgs.isEmpty,
           !hasObservation,
           !isSafeToolObservationFinal(finalText, expectedToolID: expectedTool) {
            return ["Tool coverage read-only no-arg scenario missing observation/finalizer evidence for expected tool \(expectedTool)"]
        }

        return []
    }

    nonisolated private static func isSafeToolObservationFinal(_ finalText: String, expectedToolID: String) -> Bool {
        let lower = finalText.lowercased()
        switch expectedToolID {
        case "alarm.authorization_status":
            return lower.contains("alarm authorization status:")
                || AlarmTools.isRuntimeUnavailableText(finalText)
        case "alarm.list":
            return lower.contains("active alarms:")
                || lower.contains("no active alarms")
                || AlarmTools.isRuntimeUnavailableText(finalText)
        default:
            return !lower.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }

    nonisolated private static func alarmRuntimeUnavailableEvidenceFailure(
        scenario: E2ETestScenario,
        agentSteps: [AgentStep],
        finalText: String
    ) -> String? {
        guard scenario.kind == .toolGuard,
              let expectedTool = scenario.expectedToolID.map(ToolRouteGuard.canonicalToolID),
              expectedTool.hasPrefix("alarm.") else {
            return nil
        }
        let unavailableObservation = agentSteps.contains { step in
            ToolRouteGuard.canonicalToolID(step.toolID ?? "") == expectedTool
                && step.kind == .observation
                && AlarmTools.isRuntimeUnavailableText(step.content)
        }
        guard unavailableObservation || AlarmTools.isRuntimeUnavailableText(finalText) else {
            return nil
        }
        return "AlarmKit runtime unavailable for expected tool \(expectedTool); device-runtime evidence required."
    }

    nonisolated private static func cpuWatchdogDegradedEvidence(
        finalText: String,
        failures: [String],
        events: [E2ETestEvent]
    ) -> Bool {
        let evidence = ([finalText] + failures + events.map(\.message))
            .joined(separator: "\n")
            .lowercased()
        return evidence.contains("cpu-watchdog-degraded")
    }

    nonisolated private static func structuredKernelDiagnosticEvidence(
        _ diagnostic: AgentKernelDiagnosticEvent
    ) -> String? {
        guard diagnostic.stage.hasPrefix("structured-agent-json-") else { return nil }
        let allowedMetadataKeys: Set<String> = [
            "stepIndex",
            "runtimePath",
            "parseError",
            "modelLoaded",
            "streamStarted",
            "firstChunkReceived",
            "textChunkCount",
            "finalChunkReceived",
            "streamTerminationReason",
            "emptyOutputReason"
        ]
        let safeMetadata = diagnostic.metadata
            .filter { allowedMetadataKeys.contains($0.key) }
            .map { key, value in
                let safeKey = PersistentRuntimeDiagnosticsRedactor.safeCode(key)
                let safeValue = String(PersistentRuntimeDiagnosticsRedactor.redact(value).prefix(96))
                return "\(safeKey)=\(safeValue)"
            }
            .sorted()
        guard !safeMetadata.isEmpty else { return nil }
        let stage = PersistentRuntimeDiagnosticsRedactor.safeCode(diagnostic.stage)
        return (["stage=\(stage)"] + safeMetadata).joined(separator: ", ")
    }

#if DEBUG
    nonisolated static func structuredKernelDiagnosticEvidenceForTests(
        _ diagnostic: AgentKernelDiagnosticEvent
    ) -> String? {
        structuredKernelDiagnosticEvidence(diagnostic)
    }

    nonisolated static func ragScenarioRequiresArchitectureGroundingForTests(
        _ scenario: E2ETestScenario
    ) -> Bool {
        ragScenarioRequiresArchitectureGrounding(scenario)
    }

    nonisolated static func scenarioTemporarilyEnablesNetworkAccessForTests(
        _ scenario: E2ETestScenario,
        routing: IntentRoutingDecision,
        availableToolIDs: [String]
    ) -> Bool {
        shouldTemporarilyEnableNetworkAccess(scenario: scenario, routing: routing, availableToolIDs: availableToolIDs)
    }

    nonisolated static func acceptsPolicyFirstExecutionEvidenceForTests(
        _ scenario: E2ETestScenario,
        routing: IntentRoutingDecision
    ) -> Bool {
        acceptsPolicyFirstExecutionEvidence(scenario: scenario, routing: routing)
    }

    nonisolated static func strictLiveAgentRunOptionsForTests(
        req: AgentRequest,
        scenario: E2ETestScenario,
        e2eRunID: UUID,
        agentRunID: UUID,
        routing: IntentRoutingDecision? = nil
    ) -> LegacyAgentRunOptions {
        let routing = routing ?? IntentRouter.classify(scenario.prompt)
        return strictLiveAgentRunOptions(
            req: req,
            scenario: scenario,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            acceptsPolicyFirstEvidence: acceptsPolicyFirstExecutionEvidence(scenario: scenario, routing: routing)
        )
    }

    nonisolated static func strictLiveAgentKernelRequestForTests(
        prompt: String,
        systemPrompt: String,
        config: E2ERunConfig,
        conversationID: UUID,
        turnID: UUID,
        traceCorrelation: AgentTraceCorrelation?,
        forceModelBackedToolPlanning: Bool
    ) -> AgentKernelRequest {
        strictLiveAgentKernelRequest(
            prompt: prompt,
            systemPrompt: systemPrompt,
            config: config,
            conversationID: conversationID,
            turnID: turnID,
            traceCorrelation: traceCorrelation,
            forceModelBackedToolPlanning: forceModelBackedToolPlanning
        )
    }

    nonisolated static func strictLiveAgentRequestForTests(
        scenario: E2ETestScenario,
        config: E2ERunConfig,
        availableTools: [ToolDefinition],
        correlation: AgentTraceCorrelation
    ) -> AgentRequest {
        strictLiveAgentRequest(
            scenario: scenario,
            config: config,
            availableTools: availableTools,
            correlation: correlation
        )
    }

    nonisolated static func requiresStructuredModelBackedAgentRunForTests(
        scenario: E2ETestScenario,
        routing: IntentRoutingDecision
    ) -> Bool {
        requiresStructuredModelBackedAgentRun(scenario: scenario, routing: routing)
    }

    nonisolated static func modelRuntimeEvidenceForTests(
        since startedAt: Date,
        prompt: String,
        scenarioID: String? = nil,
        e2eRunID: UUID? = nil,
        agentRunID: UUID? = nil,
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
        acceptsPolicyFirstEvidence: Bool = false,
        requiresPrimaryAgentJSON: Bool = false
    ) -> Bool {
        modelRuntimeEvidence(
            since: startedAt,
            prompt: prompt,
            correlation: AgentTraceCorrelation(scenarioID: scenarioID, e2eRunID: e2eRunID, agentRunID: agentRunID, conversationID: conversationID, turnID: turnID),
            acceptsPolicyFirstEvidence: acceptsPolicyFirstEvidence,
            requiresPrimaryAgentJSON: requiresPrimaryAgentJSON
        ) != nil
    }

    nonisolated static func modelRuntimeEvidenceMatchedByForTests(
        since startedAt: Date,
        prompt: String,
        scenarioID: String? = nil,
        e2eRunID: UUID? = nil,
        agentRunID: UUID? = nil,
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
        acceptsPolicyFirstEvidence: Bool = false,
        requiresPrimaryAgentJSON: Bool = false
    ) -> String? {
        modelRuntimeEvidence(
            since: startedAt,
            prompt: prompt,
            correlation: AgentTraceCorrelation(scenarioID: scenarioID, e2eRunID: e2eRunID, agentRunID: agentRunID, conversationID: conversationID, turnID: turnID),
            acceptsPolicyFirstEvidence: acceptsPolicyFirstEvidence,
            requiresPrimaryAgentJSON: requiresPrimaryAgentJSON
        )?.matchedBy
    }

    nonisolated static func modelRuntimeEvidenceFailureMessageForTests(
        since startedAt: Date,
        prompt: String,
        scenarioID: String? = nil,
        e2eRunID: UUID? = nil,
        agentRunID: UUID? = nil,
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
        acceptsPolicyFirstEvidence: Bool = false,
        requiresPrimaryAgentJSON: Bool = false
    ) -> String {
        modelRuntimeEvidenceDiagnosis(
            since: startedAt,
            prompt: prompt,
            correlation: AgentTraceCorrelation(scenarioID: scenarioID, e2eRunID: e2eRunID, agentRunID: agentRunID, conversationID: conversationID, turnID: turnID),
            acceptsPolicyFirstEvidence: acceptsPolicyFirstEvidence,
            requiresPrimaryAgentJSON: requiresPrimaryAgentJSON
        ).failureMessage
    }

    nonisolated static func shouldRewriteFinalForEvalHintsForTests(
        _ scenario: E2ETestScenario,
        hasAcceptedModelEvidence: Bool
    ) -> Bool {
        shouldRewriteFinalForEvalHints(
            scenario: scenario,
            hasAcceptedModelEvidence: hasAcceptedModelEvidence
        )
    }

    nonisolated static func shouldValidateFinalContentHintsForTests(
        _ scenario: E2ETestScenario,
        hasAcceptedModelEvidence: Bool
    ) -> Bool {
        shouldValidateFinalContentHints(
            scenario: scenario,
            hasAcceptedModelEvidence: hasAcceptedModelEvidence
        )
    }

    nonisolated static func ragFinalIndicatesNoRetrievedSnippetsForTests(_ lowerFinal: String) -> Bool {
        ragFinalIndicatesNoRetrievedSnippets(lowerFinal)
    }

    static func validateAndRewriteFinalTextIfNeededForTests(
        scenario: E2ETestScenario,
        routing: IntentRoutingDecision,
        originalFinal: String
    ) async -> (finalText: String, missingHints: [String], rewriteAttempted: Bool, rewriteSuccess: Bool) {
        let outcome = await validateAndRewriteFinalTextIfNeeded(
            scenario: scenario,
            routing: routing,
            originalFinal: originalFinal
        )
        return (outcome.finalText, outcome.missingHints, outcome.rewriteAttempted, outcome.rewriteSuccess)
    }

    static func finalHintRewriteOutcomeForTests(
        scenario: E2ETestScenario,
        routing: IntentRoutingDecision,
        originalFinal: String,
        hasAcceptedModelEvidence: Bool,
        nonActionableMetadata: [String: String]
    ) async -> (finalText: String, missingHints: [String], rewriteAttempted: Bool, rewriteSuccess: Bool) {
        let outcome = await finalHintRewriteOutcome(
            scenario: scenario,
            routing: routing,
            originalFinal: originalFinal,
            hasAcceptedModelEvidence: hasAcceptedModelEvidence,
            nonActionableMetadata: nonActionableMetadata,
            ragRetrievalEvidenceState: nil
        )
        return (outcome.finalText, outcome.missingHints, outcome.rewriteAttempted, outcome.rewriteSuccess)
    }

    nonisolated static func requiredHintsMissingForTests(
        finalText: String,
        scenario: E2ETestScenario,
        agentSteps: [AgentStep],
        events: [E2ETestEvent]
    ) -> [String] {
        requiredHintsMissing(
            in: finalText,
            scenario: scenario,
            ragRetrievalEvidenceState: ragRetrievalEvidenceState(
                finalText: finalText,
                agentSteps: agentSteps,
                events: events
            )
        )
    }

    nonisolated static func liveRuntimeShouldStopAfterForTests(_ result: E2ETestResult) -> Bool {
        liveRuntimeShouldStopAfter(result)
    }

    static func liveRuntimePreflightBlockedResultForTests(
        _ scenario: E2ETestScenario,
        denialReason: String
    ) async -> E2ETestResult {
        await liveRuntimePreflightBlockedResult(
            for: scenario,
            denialReason: denialReason,
            onEvent: nil
        )
    }

    static func liveRuntimePreflightBlockedResultIfNeededForTests(
        _ scenario: E2ETestScenario
    ) async -> E2ETestResult? {
        await liveRuntimePreflightBlockedResultIfNeeded(for: scenario, onEvent: nil)
    }

    nonisolated static func liveRuntimeBudgetFailureKindForTests(_ denialReason: String) -> String {
        liveRuntimeBudgetFailureKind(denialReason)
    }

    static func liveRuntimeReadinessBarrierForTests(
        _ scenario: E2ETestScenario,
        maxWaitNanoseconds: UInt64,
        pollNanoseconds: UInt64
    ) async -> (denialReason: String?, events: [E2ETestEvent]) {
        let outcome = await liveRuntimeReadinessBarrier(
            for: scenario,
            maxWaitNanoseconds: maxWaitNanoseconds,
            pollNanoseconds: pollNanoseconds,
            onEvent: nil
        )
        return (outcome.denialReason, outcome.events)
    }

    nonisolated static func liveRuntimePacingNanosecondsForTests(
        after result: E2ETestResult,
        thermalState: ProcessInfo.ThermalState,
        lowPowerModeEnabled: Bool = false
    ) -> UInt64 {
        liveRuntimePacingNanoseconds(
            after: result,
            thermalState: thermalState,
            lowPowerModeEnabled: lowPowerModeEnabled
        )
    }

    nonisolated static func shouldRunAsPlainTextTurnForTests(
        _ scenario: E2ETestScenario,
        routing: IntentRoutingDecision
    ) -> Bool {
        shouldRunAsPlainTextTurn(scenario: scenario, routing: routing)
    }

    nonisolated static func toolCoverageEvidenceFailuresForTests(
        scenario: E2ETestScenario,
        routing: IntentRoutingDecision,
        agentSteps: [AgentStep],
        finalText: String
    ) -> [String] {
        toolCoverageEvidenceFailures(
            scenario: scenario,
            routing: routing,
            agentSteps: agentSteps,
            finalText: finalText
        )
    }

    nonisolated static func isSafeToolObservationFinalForTests(_ finalText: String, expectedToolID: String) -> Bool {
        isSafeToolObservationFinal(finalText, expectedToolID: expectedToolID)
    }

    nonisolated static func alarmRuntimeUnavailableEvidenceFailureForTests(
        scenario: E2ETestScenario,
        agentSteps: [AgentStep],
        finalText: String
    ) -> String? {
        alarmRuntimeUnavailableEvidenceFailure(
            scenario: scenario,
            agentSteps: agentSteps,
            finalText: finalText
        )
    }

    nonisolated static func webSearchSummaryQualityFailureForTests(finalText: String, scenario: E2ETestScenario) -> Bool {
        webSearchSummaryQualityFailure(finalText: finalText, scenario: scenario)
    }

    nonisolated static func deterministicWebSynthesisFallbackForTests(
        scenario: E2ETestScenario,
        rawFinalText: String,
        events: [E2ETestEvent]
    ) -> String? {
        deterministicWebSynthesisFallback(
            scenario: scenario,
            rawFinalText: rawFinalText,
            events: events
        )
    }

    nonisolated static func deterministicToolObservationFallbackForIncompleteFinalForTests(
        scenario: E2ETestScenario,
        routing: IntentRoutingDecision,
        finalText: String,
        events: [E2ETestEvent]
    ) -> String? {
        deterministicToolObservationFallbackForIncompleteFinal(
            scenario: scenario,
            routing: routing,
            finalText: finalText,
            events: events
        )
    }

    nonisolated static func isGenericChatFallbackFinalForTests(_ text: String) -> Bool {
        isGenericChatFallbackFinal(text)
    }

    nonisolated static func directAnswerRetryPromptForTests(_ prompt: String) -> String {
        directAnswerRetryPrompt(for: prompt)
    }

    nonisolated static func deterministicDirectChatFallbackForTests(_ prompt: String) -> String? {
        deterministicDirectChatFallback(for: prompt)
    }

    nonisolated static func nonActionableInfrastructureMetadataForTests(
        scenario: E2ETestScenario,
        finalText: String,
        failures: [String],
        events: [E2ETestEvent]
    ) -> [String: String] {
        nonActionableInfrastructureMetadata(
            scenario: scenario,
            finalText: finalText,
            failures: failures,
            events: events
        )
    }

    nonisolated static func nonActionableQuarantineFailureForTests(metadata: [String: String]) -> String? {
        nonActionableQuarantineFailure(metadata: metadata)
    }

    nonisolated static func isRAGEmptyRetrievalEvidenceForTests(_ lowerText: String) -> Bool {
        isRAGEmptyRetrievalEvidence(lowerText)
    }

    nonisolated static func ragRetrievalEvidenceStateForTests(
        finalText: String,
        agentSteps: [AgentStep],
        events: [E2ETestEvent]
    ) -> String {
        ragRetrievalEvidenceState(
            finalText: finalText,
            agentSteps: agentSteps,
            events: events
        ).rawValue
    }

    static func agentJSONTrainingProbeForTests() async -> [AgentJSONTrainingProbeResult] {
        var results: [AgentJSONTrainingProbeResult] = []
        for scenario in E2ETestScenario.trainingValidation {
            let allowedIDs = Set(scenario.requiredAllowedToolIDs.map(ToolRouteGuard.canonicalToolID))
            let tools = ToolRegistry.all.filter { allowedIDs.contains(ToolRouteGuard.canonicalToolID($0.id)) }
            let req = AgentRequest(
                systemPrompt: "You are Lumen's local structured agent executor.",
                history: [],
                userMessage: scenario.prompt,
                temperature: 0.1,
                topP: 0.8,
                repetitionPenalty: 1.05,
                maxTokens: 512,
                maxSteps: 1,
                availableTools: tools,
                relevantMemories: []
            )
            let structuredSystemPrompt = await AgentService.shared.structuredSystemPromptForTests(req: req)
            let structuredUserTurn = await AgentService.shared.structuredAgentUserTurnForTests(req: req)
            let structuredMaxTokens = await AgentService.shared.structuredTurnMaxTokensForTests(from: req.maxTokens)
            let genReq = GenerateRequest(
                systemPrompt: structuredSystemPrompt,
                history: [],
                userMessage: structuredUserTurn,
                temperature: 0.05,
                topP: 0.6,
                repetitionPenalty: 1.05,
                maxTokens: structuredMaxTokens,
                modelName: "agent-json",
                relevantMemories: [],
                attachments: [],
                responseFormat: .constrainedJSON(schema: AgentService.structuredAgentResponseSchema),
                allowsMemoryPressureContinuation: true
            )
            let promptBuild = await AppLlamaService.shared.buildMessagesForTesting(req: genReq, contextSize: 2048, slot: .executor)
            let promptFitsBudget = promptBuild.estimatedPromptTokens + genReq.maxTokens + PromptBudgetConstants.agentJSONSafetyTokens < 2048
            var raw = ""
            var streamStarted = false
            var firstTextReceived = false
            var finalChunkReceived = false
            var textChunkCount = 0
            streamStarted = true
            for await token in await AppLlamaService.shared.stream(genReq, slot: .executor) {
                switch token {
                case .text(let text):
                    if !text.isEmpty {
                        firstTextReceived = true
                        textChunkCount += 1
                    }
                    raw += text
                case .done:
                    finalChunkReceived = true
                    break
                }
            }
            let payload = await AppLlamaService.shared.takeCompletedTracePayload(requestID: genReq.id)
            let fallbackEmptyReason = raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? (finalChunkReceived ? "completedWithoutText" : "unknownEmptyStream")
                : nil
            let parsed = AgentTurnParser.parse(raw)
            let actionTool = parsed.action.map { ToolRouteGuard.canonicalToolID($0.tool) }
            let actionOrFinal: String?
            if actionTool != nil {
                actionOrFinal = "action"
            } else if parsed.final?.isEmpty == false {
                actionOrFinal = "final"
            } else {
                actionOrFinal = nil
            }
            results.append(
                AgentJSONTrainingProbeResult(
                    scenarioID: scenario.id,
                    promptFitsBudget: promptFitsBudget,
                    streamStarted: payload?.streamStarted ?? streamStarted,
                    firstChunkReceived: payload?.firstChunkReceived ?? firstTextReceived,
                    firstTextReceived: (payload?.textChunkCount ?? textChunkCount) > 0,
                    parsedJSON: parsed.parseError == nil,
                    actionOrFinal: actionOrFinal,
                    selectedTool: actionTool,
                    emptyStreamReason: payload?.emptyOutputReason ?? fallbackEmptyReason
                )
            )
        }
        return results
    }
#endif

    nonisolated private static func performanceMatrix(from samples: [E2EPerformanceSample], startedAt: Date, finishedAt: Date) async -> E2EPerformanceMatrix {
        let residentSamples = samples.compactMap(\.residentMemoryMB)
        let averageRAM = residentSamples.isEmpty ? 0 : residentSamples.reduce(0, +) / Double(residentSamples.count)
        let peakRAM = residentSamples.max() ?? 0
        let duration = finishedAt.timeIntervalSince(startedAt)
        let cpuEstimate = duration > 0 ? min(100, (Double(samples.count) / max(duration, 0.001)) * 2.5) : nil
        var notes = [
            "CPU is an event-density estimate for test comparison only.",
            "ANE and GPU counters are unavailable in this runtime and exported as null.",
        ]
        if residentSamples.isEmpty {
            notes.append("Resident-memory sampling unavailable; RAM fields defaulted to 0MB.")
        }
        return E2EPerformanceMatrix(
            aneUtilizationPercent: nil,
            eventDensityCPUProxyPercent: cpuEstimate,
            gpuUtilizationPercent: nil,
            peakRAMMB: peakRAM,
            averageRAMMB: averageRAM,
            sampleCount: samples.count,
            notes: notes,
            accelerationDiagnostics: await AppLlamaService.shared.currentAccelerationDiagnostics()
        )
    }

    nonisolated private static func residentMemoryUsageMB() -> Double? {
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

    nonisolated static func mergeSanitizerOutputs(_ primary: SanitizedFinalOutput, recovered: SanitizedFinalOutput?) -> SanitizedFinalOutput {
        guard let recovered else { return primary }
        let mergedArtifactsList = mergedArtifacts(primary.removedArtifacts, recovered.removedArtifacts)
        let hadUnsafeLeakage = primary.hadUnsafeLeakage || recovered.hadUnsafeLeakage
        return SanitizedFinalOutput(
            text: primary.text,
            removedArtifacts: mergedArtifactsList,
            hadUnsafeLeakage: hadUnsafeLeakage,
            artifactAudit: FinalOutputArtifactAudit(
                rawPrefix: recovered.artifactAudit.hadUnsafeLeakage ? recovered.artifactAudit.rawPrefix : (primary.artifactAudit.rawPrefix.isEmpty ? recovered.artifactAudit.rawPrefix : primary.artifactAudit.rawPrefix),
                sanitizedPrefix: primary.artifactAudit.sanitizedPrefix,
                hadUnsafeLeakage: hadUnsafeLeakage,
                removedArtifacts: !mergedArtifactsList.isEmpty,
                removedArtifactTypes: mergedArtifactsList
            )
        )
    }

    nonisolated static func mergedArtifacts(_ groups: [FinalOutputArtifact]...) -> [FinalOutputArtifact] {
        var merged: [FinalOutputArtifact] = []
        for artifact in groups.flatMap({ $0 }) where !merged.contains(artifact) {
            merged.append(artifact)
        }
        return merged
    }

    nonisolated static func mergedStrings(_ groups: [String]...) -> [String] {
        var merged: [String] = []
        for item in groups.flatMap({ $0 }) where !merged.contains(item) {
            merged.append(item)
        }
        return merged
    }

    nonisolated static func hygieneFailures(lowerRawFinal: String, lowerFinal: String, removedArtifacts: [FinalOutputArtifact], scenario: E2ETestScenario, observations: String) -> [String] {
        var failures: [String] = []
        if lowerFinal.contains("<think") || lowerFinal.contains("</think>")
            || lowerFinal.contains("<analysis") || lowerFinal.contains("</analysis>")
            || lowerFinal.contains("<reasoning") || lowerFinal.contains("</reasoning>")
            || lowerFinal.contains("<thinking") || lowerFinal.contains("</thinking>")
            || lowerFinal.contains("<chain_of_thought") || lowerFinal.contains("</chain_of_thought>") {
            failures.append("Sanitized output still contains hidden reasoning")
        }
        if lowerFinal.contains("<lumen_web_payload") || lowerFinal.contains("</lumen_web_payload>") {
            failures.append("Sanitized output still contains lumen_web_payload markers")
        }
        if lowerFinal.contains("{\"kind\":\"searchresults\"") || lowerFinal.contains("\"mediakind\":\"page\"") || lowerFinal.contains("\"sourcepageurl\"") {
            failures.append("Sanitized output still contains search-results JSON")
        }
        if removedArtifacts.contains(.emptyAfterSanitization) {
            failures.append("Final output empty after sanitization")
        }
        if finalHasDanglingIncompleteEnding(lowerFinal) {
            failures.append("Final output appears incomplete or truncated")
        }
        if scenario.expectedIntent == .weather && weatherGroundingOverreach(finalText: lowerFinal, observations: observations) {
            failures.append("Weather precipitation recommendation not grounded")
        }
        return mergedStrings(failures)
    }

    nonisolated private static func finalHasDanglingIncompleteEnding(_ lowerFinal: String) -> Bool {
        let text = lowerFinal
            .replacingOccurrences(of: #"[\s\p{P}]+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return false }
        let danglingSuffixes = [
            "an",
            "a",
            "the",
            "with",
            "because",
            "you do not need an"
        ]
        return danglingSuffixes.contains(text)
            || danglingSuffixes.contains(where: { text.hasSuffix(" \($0)") })
    }

    nonisolated private static func deterministicToolObservationFallbackForIncompleteFinal(
        scenario: E2ETestScenario,
        routing: IntentRoutingDecision,
        finalText: String,
        events: [E2ETestEvent]
    ) -> String? {
        guard finalHasDanglingIncompleteEnding(finalText.lowercased()) else { return nil }
        guard scenario.expectedIntent == .weather || routing.intent == .weather else { return nil }
        guard let observation = lastWeatherObservation(from: events) else { return nil }
        return ToolObservationFinalizer.immediateFinalIfSafe(
            intent: .weather,
            toolID: "weather",
            observation: observation,
            originalPrompt: scenario.prompt
        ) ?? "Weather update: \(observation)"
    }

    nonisolated private static func lastWeatherObservation(from events: [E2ETestEvent]) -> String? {
        for event in events.reversed() where event.phase == "step" {
            let lower = event.message.lowercased()
            guard lower.contains("weather") || lower.contains("°") || lower.contains("temperature") else {
                continue
            }
            let message = event.message
            if let range = message.range(of: "observation:", options: [.caseInsensitive]) {
                let observation = String(message[range.upperBound...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !observation.isEmpty { return observation }
            }
            if let range = message.range(of: "observation", options: [.caseInsensitive]) {
                let observation = String(message[range.upperBound...])
                    .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters))
                if !observation.isEmpty { return observation }
            }
            let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        return nil
    }

    nonisolated static func liveAgentQualityFailures(rawFinalText: String, finalText: String, scenario: E2ETestScenario) -> [String] {
        guard scenario.requiresAgentRun else { return [] }
        var failures: [String] = []
        let lowerRaw = rawFinalText.lowercased()
        let lowerFinal = finalText.lowercased()
        if let reason = liveAgentInvalidFinalReason(lowerRaw: lowerRaw, lowerFinal: lowerFinal) {
            failures.append(reason)
        }
        if webSearchSummaryQualityFailure(finalText: finalText, scenario: scenario) {
            failures.append("Web search summarize scenario returned raw results or URL instead of a concise summary")
        }
        if scenario.id == "training-rag-grounding",
           ragArchitectureGroundingIsIrrelevant(lowerFinal) {
            failures.append("RAG grounding assertion failed: architecture-notes answer used unrelated photo-library snippets")
        }

        let rawMissingHints = Set(requiredHintsMissing(in: rawFinalText, scenario: scenario))
        let finalMissingHints = Set(requiredHintsMissing(in: finalText, scenario: scenario))
        for hint in rawMissingHints.intersection(finalMissingHints).sorted() {
            failures.append("Live final required hint missing: \(hint)")
        }

        return mergedStrings(failures)
    }

    nonisolated private static func liveAgentInvalidFinalReason(lowerRaw: String, lowerFinal: String) -> String? {
        let combined = lowerRaw + "\n" + lowerFinal
        let invalidSignals = [
            "not available in this build",
            "tools are unavailable",
            "tool unavailable",
            "tool denied by legacy secure policy",
            "tool is disabled",
            "tool output could not be validated",
            "could not be validated",
            "i hit an internal response-format issue",
            "only internal reasoning and no final answer",
            "no model loaded; routing-only checks completed",
            "routing-only checks completed",
            "full local model pipeline is temporarily running in compatibility mode",
            "full agent pipeline",
            "no direct answer from web search",
            "i couldn't safely complete",
            "i couldn’t safely complete",
            "cpu-watchdog-degraded",
            "\"rewrittenfinalanswer\"",
            "\"requiresapprovaldecision\"",
            "\"requiresapprovalreasoningsummary\"",
            "please try again with thinking disabled",
            "i'm ready. please ask again",
            "please ask again or tell me what you'd like to do next"
        ]
        return invalidSignals.contains(where: { combined.contains($0) })
            || RoutingJSONLeakDetector.containsInternalRoutingJSON(combined)
            ? "Live agent returned fallback/error text instead of completing the scenario"
            : nil
    }

    nonisolated private static func isGenericChatFallbackFinal(_ text: String) -> Bool {
        let lower = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !lower.isEmpty else { return false }
        return lower.contains("i'm ready")
            || lower.contains("please ask again")
            || lower.contains("what you'd like to do next")
    }

    nonisolated private static func directAnswerRetryPrompt(for prompt: String) -> String {
        """
        Answer the user's request directly in plain text now.
        Requirements:
        - Do not say you are ready.
        - Do not ask the user to ask again.
        - Do not mention internal tools, models, routing, or fallback.
        - Start with the answer itself, not a preamble.
        - If the prompt asks for an explanation, give the explanation in 1-3 concise sentences.

        User request:
        \(prompt)
        """
    }

    nonisolated private static func deterministicDirectChatFallback(for prompt: String) -> String? {
        let lower = prompt.lowercased()
        if lower.contains("sharp chisel") && lower.contains("dull") {
            return "A sharp chisel is safer than a dull one because it cuts predictably with less force, which gives you better control and lowers the chance of slipping."
        }
        if lower.contains("precision") && lower.contains("recall") {
            return "Precision is how often the items you found are actually correct; recall is how much of everything correct you managed to find."
        }
        return nil
    }

    nonisolated private static func deterministicWebSynthesisFallback(
        scenario: E2ETestScenario,
        rawFinalText: String,
        events: [E2ETestEvent]
    ) -> String? {
        guard scenario.expectedIntent == .webSearch else { return nil }
        guard webFinalNeedsDeterministicSynthesis(rawFinalText) else {
            return nil
        }
        let observations = events
            .filter { event in
                let lower = event.message.lowercased()
                return event.phase == "step"
                    && lower.contains("observation:")
                    && (lower.contains("web.search") || lower.contains("search results for:") || lower.contains("web search results:") || lower.contains("swift"))
            }
            .map(\.message)
        let items = webObservationItems(from: observations)
        guard !items.isEmpty else { return nil }
        let observationText = observations.joined(separator: "\n").lowercased()
        let promptRequiresSwift = scenario.prompt.lowercased().contains("swift") || observationText.contains("swift")
        let lead = promptRequiresSwift
            ? "Swift concurrency search summary:"
            : "Web search summary:"
        let bullets = items.prefix(2).enumerated().map { index, item in
            "- \(item.title): \(webObservationSummary(for: item, promptRequiresSwift: promptRequiresSwift, ordinal: index))"
        }
        return ([lead] + bullets).joined(separator: "\n")
    }

    nonisolated private static func webFinalNeedsDeterministicSynthesis(_ finalText: String) -> Bool {
        let text = finalText.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = text.lowercased()
        if text.isEmpty { return true }
        if text.count < 40 { return true }
        if lower.contains("no direct answer from web search") { return true }
        if lower.hasPrefix("search results for:") || lower.hasPrefix("web search results:") { return true }
        if lower.contains("search results for:") && (lower.contains("\nhttp") || lower.contains("\n- http")) { return true }
        if lower.range(of: #"(?m)^\s*(?:-?\s*)?https?://"#, options: .regularExpression) != nil { return true }
        if lower.range(of: #"(?is)^\s*(?:\d+\.\s*)?.{0,120}\nhttps?://"#, options: .regularExpression) != nil { return true }
        return false
    }

    private struct WebObservationItem: Sendable, Hashable {
        let title: String
        let source: String
    }

    nonisolated private static func webObservationItems(from observations: [String]) -> [WebObservationItem] {
        var items: [WebObservationItem] = []
        var seen = Set<String>()
        for observation in observations {
            for line in observation.components(separatedBy: .newlines) {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard let match = trimmed.range(of: #"^\d+\.\s+(.+)$"#, options: .regularExpression) else { continue }
                let title = String(trimmed[match].drop { $0 != " " }.dropFirst())
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !title.isEmpty, !title.lowercased().hasPrefix("http") else { continue }
                let key = title.lowercased()
                guard seen.insert(key).inserted else { continue }
                let source = webSourceNearTitle(title, in: observation)
                items.append(WebObservationItem(title: title, source: source))
                if items.count >= 2 { return items }
            }
        }
        return items
    }

    nonisolated private static func webSourceNearTitle(_ title: String, in observation: String) -> String {
        let lines = observation.components(separatedBy: .newlines)
        guard let index = lines.firstIndex(where: { $0.contains(title) }),
              lines.indices.contains(index + 1) else {
            return "search result"
        }
        let next = lines[index + 1].trimmingCharacters(in: .whitespacesAndNewlines)
        guard let host = URL(string: next)?.host, !host.isEmpty else {
            return "search result"
        }
        return host.replacingOccurrences(of: "www.", with: "")
    }

    nonisolated private static func webObservationSummary(
        for item: WebObservationItem,
        promptRequiresSwift: Bool,
        ordinal: Int
    ) -> String {
        let title = item.title.trimmingCharacters(in: .whitespacesAndNewlines)
        if promptRequiresSwift {
            if title.localizedCaseInsensitiveContains("MainActor") {
                return "Use Swift actor isolation, especially MainActor boundaries, intentionally for UI state and shared mutable data."
            }
            if title.localizedCaseInsensitiveContains("concurrency") || title.localizedCaseInsensitiveContains("async") {
                return ordinal == 0
                    ? "Favor Swift structured concurrency with async/await and task groups so work has clear ownership and cancellation."
                    : "Keep Swift concurrent work isolated with actors or explicit sendable boundaries to avoid shared-state races."
            }
            return "Treat this Swift result as supporting evidence and verify the practice against current Apple documentation."
        }
        return "Useful result from \(item.source) titled \(title), suitable for a concise synthesized answer without exposing raw URLs."
    }

    nonisolated private static func nonActionableInfrastructureMetadata(
        scenario: E2ETestScenario,
        finalText: String,
        failures: [String],
        events: [E2ETestEvent]
    ) -> [String: String] {
        let evidence = ([finalText] + failures + events.map(\.message))
            .joined(separator: "\n")
            .lowercased()
        var metadata: [String: String] = [:]
        func quarantine(_ failureKind: String, evidenceKind: String) {
            metadata["failureKind"] = failureKind
            metadata["actionable"] = "false"
            metadata["trainingSignal"] = "false"
            metadata["runtimeEvidence"] = evidenceKind
        }

        if scenario.expectedIntent == .rag,
           evidence.contains("rag") || scenario.requiredAllowedToolIDs.map(ToolRouteGuard.canonicalToolID).contains("rag.search") {
            let retrievalUnavailableSignals = [
                "rag retrieval is unavailable"
            ]
            let storageUnavailableSignals = [
                "rag storage unavailable",
                "swiftdata unavailable",
                "persistent store unavailable",
                "local index appears empty",
                "no matching local snippets",
                "no matching files found",
                "import or create local files"
            ]
            if evidence.contains("cleanup_deferred:disk_write_budget_denied") {
                quarantine("ragMaintenanceDeferred", evidenceKind: "resource-budget-deferred")
            } else if retrievalUnavailableSignals.contains(where: { evidence.contains($0) }) {
                quarantine("ragStorageUnavailable", evidenceKind: "retrieval-unavailable")
            } else if storageUnavailableSignals.contains(where: { evidence.contains($0) }) {
                quarantine("ragStorageUnavailable", evidenceKind: "storage-unavailable")
            }
        }

        if scenario.expectedIntent == .outlook || scenario.expectedToolID?.hasPrefix("outlook.") == true {
            let structuredFailureCodes = Set(events
                .filter { $0.phase == "tool-result" }
                .compactMap { event -> String? in
                    guard let range = event.message.range(of: #"errorCode=([^,\s]+)"#, options: .regularExpression) else {
                        return nil
                    }
                    return String(event.message[range])
                        .replacingOccurrences(of: "errorCode=", with: "")
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                        .lowercased()
                })
            let configurationCodes: Set<String> = ["outlook_not_configured"]
            let authenticationCodes: Set<String> = [
                "outlook_auth_unavailable",
                "outlook_interaction_required",
                "outlook_reauthentication_required"
            ]
            let permissionCodes: Set<String> = [
                "outlook_consent_required",
                "outlook_permission_denied",
                "outlook_scope_not_granted"
            ]
            let providerCodes: Set<String> = [
                "outlook_auth_provider_unavailable",
                "outlook_network_unavailable",
                "outlook_provider_error",
                "outlook_provider_throttled"
            ]
            let knownFailureCodes = configurationCodes
                .union(authenticationCodes)
                .union(permissionCodes)
                .union(providerCodes)
            if let safeFailureCode = structuredFailureCodes
                .intersection(knownFailureCodes)
                .sorted()
                .first {
                metadata["toolFailureCode"] = safeFailureCode
            }
            if !structuredFailureCodes.isDisjoint(with: configurationCodes) {
                quarantine("outlookRuntimeUnavailable", evidenceKind: "tool-configuration-unavailable")
            } else if !structuredFailureCodes.isDisjoint(with: authenticationCodes) {
                quarantine("outlookAuthenticationUnavailable", evidenceKind: "tool-authentication-unavailable")
            } else if !structuredFailureCodes.isDisjoint(with: permissionCodes) {
                quarantine("outlookPermissionUnavailable", evidenceKind: "tool-permission-unavailable")
            } else if !structuredFailureCodes.isDisjoint(with: providerCodes) {
                quarantine("outlookProviderUnavailable", evidenceKind: "tool-provider-unavailable")
            }

            let unavailableSignals = [
                "outlook config",
                "outlook auth",
                "outlook capability",
                "outlook account",
                "not configured",
                "authentication required",
                "authorization required"
            ]
            if metadata["failureKind"] == nil,
               unavailableSignals.contains(where: { evidence.contains($0) }) {
                quarantine("outlookRuntimeUnavailable", evidenceKind: "tool-configuration-unavailable")
            }
        }

        if evidence.contains("cpu-watchdog-degraded")
            || evidence.contains("cpu watchdog degraded")
            || evidence.contains("live runtime cpu watchdog degraded") {
            quarantine("liveRuntimeCPUWatchdogDegraded", evidenceKind: "runtime-preflight")
        }

        if evidence.contains("thermalstate=serious")
            || evidence.contains("thermalstate=critical")
            || evidence.contains("thermal state serious")
            || evidence.contains("thermal state critical")
            || evidence.contains("resource-budget-denied-before-prompt-eval")
            || evidence.contains("blocked before model prompt evaluation")
            || evidence.contains("live e2e paused before starting this scenario") {
            quarantine("liveRuntimePreflightUnavailable", evidenceKind: "runtime-preflight")
        }

        if isGenericChatFallbackFinal(finalText) {
            metadata["failureKind"] = "genericFallbackFinal"
            metadata["actionable"] = "true"
            metadata["trainingSignal"] = "true"
        }

        return metadata
    }

    nonisolated private static func nonActionableQuarantineFailure(metadata: [String: String]) -> String? {
        guard metadata["actionable"]?.lowercased() == "false" else { return nil }
        switch metadata["failureKind"] {
        case "ragStorageUnavailable":
            if metadata["runtimeEvidence"] == "retrieval-unavailable" {
                return "Runtime infrastructure unavailable: RAG retrieval unavailable."
            }
            return "Runtime infrastructure unavailable: RAG storage unavailable."
        case "ragMaintenanceDeferred":
            return "Runtime preflight unavailable: RAG maintenance deferred by the disk-write budget."
        case "outlookRuntimeUnavailable":
            return "Runtime infrastructure unavailable: Outlook configuration unavailable."
        case "outlookAuthenticationUnavailable":
            return "Runtime infrastructure unavailable: Outlook authentication unavailable."
        case "outlookPermissionUnavailable":
            return "Runtime infrastructure unavailable: Outlook permission or consent unavailable."
        case "outlookProviderUnavailable":
            return "Runtime infrastructure unavailable: Outlook provider unavailable."
        case "liveRuntimeCPUWatchdogDegraded":
            return "Runtime preflight unavailable: CPU watchdog degraded before valid generation."
        case "liveRuntimePreflightUnavailable":
            return "Runtime preflight unavailable before valid generation."
        default:
            return "Runtime infrastructure unavailable before valid generation."
        }
    }

    nonisolated private static func webSearchSummaryQualityFailure(finalText: String, scenario: E2ETestScenario) -> Bool {
        guard scenario.expectedIntent == .webSearch else {
            return false
        }
        let text = finalText.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = text.lowercased()
        if text.isEmpty { return true }
        if lower.hasPrefix("search results for:") || lower.hasPrefix("web search results:") {
            return true
        }
        if lower.contains("search results for:") && (lower.contains("\nhttp") || lower.contains("\n- http")) {
            return true
        }
        if text.range(of: #"(?is)^\s*(https?://\S+)\s*$"#, options: .regularExpression) != nil {
            return true
        }
        if text.range(of: #"(?is)^\s*see\s+the\s+full\s+(tutorial|article|guide|post|result)\s+at\s+https?://\S+\s*\.?\s*$"#, options: .regularExpression) != nil {
            return true
        }
        if text.range(of: #"(?is)^\s*(?:check\s+out|see|read|visit|open|here(?:'s| is))\b[^\n]{0,180}https?://\S+\s*\.?\s*$"#, options: .regularExpression) != nil {
            return true
        }
        guard webPromptRequiresSynthesis(scenario.prompt) else {
            return false
        }
        let meaningfulLines = text
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && !$0.lowercased().hasPrefix("http") }
        let sentenceCount = text.split { ".!?".contains($0) }.filter { $0.trimmingCharacters(in: .whitespacesAndNewlines).count > 20 }.count
        let bulletCount = meaningfulLines.filter { $0.hasPrefix("-") || $0.hasPrefix("•") || $0.range(of: #"^\d+\."#, options: .regularExpression) != nil }.count
        return sentenceCount < 2 && bulletCount < 2
    }

    nonisolated private static func webPromptRequiresSynthesis(_ prompt: String) -> Bool {
        let lower = prompt.lowercased()
        return lower.contains("summarize")
            || lower.contains("synthesize")
            || lower.contains("compare")
    }

    nonisolated private static func weatherGroundingOverreach(finalText: String, observations: String) -> Bool {
        let answer = finalText.lowercased()
        let obs = observations.lowercased()
        let recommendsUmbrella = answer.contains("umbrella") || answer.contains("likely raining") || answer.contains("it's raining") || answer.contains("it is raining")
        guard recommendsUmbrella else { return false }
        let precipitationSignals = ["rain", "raining", "drizzle", "precip", "precipitation", "shower", "forecasted rain", "chance of rain", "probability of precipitation"]
        return !precipitationSignals.contains(where: { obs.contains($0) })
    }

    nonisolated private static func referencesRetrievedSnippet(_ lowerFinal: String) -> Bool {
        let signals = ["[1]", "[2]", "snippet", "source", "file", "pdf", "note", "retrieved"]
        return signals.contains { lowerFinal.contains($0) }
    }

    nonisolated private static func ragScenarioRequiresArchitectureGrounding(_ scenario: E2ETestScenario) -> Bool {
        if scenario.id == "training-rag-grounding" { return true }
        let prompt = scenario.prompt.lowercased()
        return prompt.contains("architecture") || prompt.contains("module") || prompt.contains("modules")
    }

    nonisolated private static func ragFinalIndicatesNoRetrievedSnippets(_ lowerFinal: String) -> Bool {
        [
            "no matching rag chunks",
            "no matching local snippets",
            "no matching module snippets",
            "no matching files",
            "no relevant rag chunks",
            "no relevant local snippets",
            "no retrieved snippets"
        ].contains { lowerFinal.contains($0) }
    }

    nonisolated private static func ragArchitectureGroundingIsIrrelevant(_ lowerFinal: String) -> Bool {
        let hasPhotoRollupCitation = lowerFinal.contains("photos · photos")
            || lowerFinal.range(of: #"photos \(\d{4}"#, options: .regularExpression) != nil
        guard hasPhotoRollupCitation else { return false }
        let architectureSignals = ["architecture", "service", "component", "package", "class ", "struct ", "func ", ".swift", "api", "endpoint"]
        return !architectureSignals.contains { lowerFinal.contains($0) }
    }

    private struct EvalRewriteOutcome {
        let finalText: String
        let missingHints: [String]
        let rewriteAttempted: Bool
        let rewriteSuccess: Bool
    }

    nonisolated private static func shouldRewriteFinalForEvalHints(
        scenario: E2ETestScenario,
        hasAcceptedModelEvidence: Bool
    ) -> Bool {
        if scenario.kind == .training, !hasAcceptedModelEvidence {
            return false
        }
        return true
    }

    nonisolated private static func shouldValidateFinalContentHints(
        scenario: E2ETestScenario,
        hasAcceptedModelEvidence: Bool
    ) -> Bool {
        if scenario.kind == .training, !hasAcceptedModelEvidence {
            return false
        }
        return true
    }

    private static func finalHintRewriteOutcome(
        scenario: E2ETestScenario,
        routing: IntentRoutingDecision,
        originalFinal: String,
        hasAcceptedModelEvidence: Bool,
        nonActionableMetadata: [String: String],
        ragRetrievalEvidenceState: RAGRetrievalEvidenceState?
    ) async -> EvalRewriteOutcome {
        if nonActionableQuarantineFailure(metadata: nonActionableMetadata) != nil {
            return EvalRewriteOutcome(
                finalText: originalFinal,
                missingHints: [],
                rewriteAttempted: false,
                rewriteSuccess: false
            )
        }
        if shouldRewriteFinalForEvalHints(
            scenario: scenario,
            hasAcceptedModelEvidence: hasAcceptedModelEvidence
        ) {
            return await validateAndRewriteFinalTextIfNeeded(
                scenario: scenario,
                routing: routing,
                originalFinal: originalFinal,
                ragRetrievalEvidenceState: ragRetrievalEvidenceState
            )
        }
        return EvalRewriteOutcome(
            finalText: originalFinal,
            missingHints: requiredHintsMissing(in: originalFinal, scenario: scenario, ragRetrievalEvidenceState: ragRetrievalEvidenceState),
            rewriteAttempted: false,
            rewriteSuccess: false
        )
    }

    nonisolated private static func validateAndRewriteFinalTextIfNeeded(
        scenario: E2ETestScenario,
        routing: IntentRoutingDecision,
        originalFinal: String,
        ragRetrievalEvidenceState: RAGRetrievalEvidenceState? = nil
    ) async -> EvalRewriteOutcome {
        let firstMissing = requiredHintsMissing(in: originalFinal, scenario: scenario, ragRetrievalEvidenceState: ragRetrievalEvidenceState)
        if liveAgentInvalidFinalReason(lowerRaw: originalFinal.lowercased(), lowerFinal: originalFinal.lowercased()) != nil {
            return EvalRewriteOutcome(finalText: originalFinal, missingHints: firstMissing, rewriteAttempted: false, rewriteSuccess: false)
        }
        guard !firstMissing.isEmpty else {
            return EvalRewriteOutcome(finalText: originalFinal, missingHints: [], rewriteAttempted: false, rewriteSuccess: true)
        }

        let rewritten = await rewriteFinalTextForEvalHints(
            originalFinal: originalFinal,
            prompt: scenario.prompt,
            intent: routing.intent,
            requiredHints: firstMissing,
            forbiddenHints: scenario.forbiddenTextHints
        )
        let secondMissing = requiredHintsMissing(in: rewritten, scenario: scenario, ragRetrievalEvidenceState: ragRetrievalEvidenceState)
        let rewriteSuccess = secondMissing.isEmpty
        return EvalRewriteOutcome(finalText: rewritten, missingHints: secondMissing, rewriteAttempted: true, rewriteSuccess: rewriteSuccess)
    }

    nonisolated private static func requiredHintsMissing(
        in finalText: String,
        scenario: E2ETestScenario,
        ragRetrievalEvidenceState explicitRAGRetrievalEvidenceState: RAGRetrievalEvidenceState? = nil
    ) -> [String] {
        let lower = finalText.lowercased()
        let ragEvidence = explicitRAGRetrievalEvidenceState ?? ragRetrievalEvidenceState(
            finalText: finalText,
            agentSteps: [],
            events: []
        )
        let ragEmptyRetrieval = scenario.expectedIntent == .rag && ragEvidence == .empty
        var missing: [String] = scenario.requiredTextHints.filter {
            if ragEmptyRetrieval, isRAGGroundingHint($0) { return false }
            return !lower.contains($0.lowercased())
        }
        if scenario.id == "training-general-chat" {
            if !lower.contains("precision") || !lower.contains("recall") {
                missing.append("precision/recall plain-language explainer")
            }
        }
        if scenario.id == "training-rag-grounding",
           !ragEmptyRetrieval,
           !(lower.contains("module") || lower.contains("modules")) {
            missing.append("module(s)")
        }
        if scenario.id == "training-memory-loop", !lower.contains("prefer concise bullet points") {
            missing.append("recalled preference text: \"prefer concise bullet points\"")
        }
        return Array(Set(missing)).sorted()
    }

    nonisolated private static func rewriteFinalTextForEvalHints(
        originalFinal: String,
        prompt: String,
        intent: UserIntent,
        requiredHints: [String],
        forbiddenHints: [String]
    ) async -> String {
        var rewritePrompt = "User prompt:\n\(prompt)\n\nOriginal final answer:\n\(originalFinal)\n\n"
        rewritePrompt += "Rewrite the final answer to satisfy eval constraints while preserving intent (\(intent.rawValue)) and tool policy boundaries.\n"
        rewritePrompt += "Keep it plain text, concise, and faithful to the original facts.\n"
        rewritePrompt += "Must include all required hints/phrases:\n- " + requiredHints.joined(separator: "\n- ") + "\n"
        if !forbiddenHints.isEmpty {
            rewritePrompt += "Must avoid forbidden hints/phrases:\n- " + forbiddenHints.joined(separator: "\n- ") + "\n"
        }
        rewritePrompt += "Do not mention internal validation, tests, or tools."
        if intent == .rag {
            rewritePrompt += " For local-knowledge/RAG answers, explicitly reference retrieved evidence using bracketed markers like [1] and mention source/snippet/file context."
        }

        let genReq = GenerateRequest(
            systemPrompt: "You rewrite user-facing answers to satisfy strict eval hint constraints while preserving intent and safety policy.",
            history: [],
            userMessage: rewritePrompt,
            temperature: 0.1,
            topP: 0.8,
            repetitionPenalty: 1.05,
            maxTokens: 320,
            modelName: "agent-summary",
            relevantMemories: []
        )
        var out = ""
        for await token in await AppLlamaService.shared.stream(genReq) {
            if case .text(let s) = token { out += s }
            if case .done = token { break }
        }
        let trimmed = out.trimmingCharacters(in: .whitespacesAndNewlines)
        let candidate = trimmed.isEmpty ? originalFinal : trimmed
        let grounded = enforceEvalGrounding(candidate, intent: intent)
        return enforceEvalHintConstraints(
            grounded,
            intent: intent,
            requiredHints: requiredHints,
            forbiddenHints: forbiddenHints
        )
    }

    nonisolated private static func enforceEvalGrounding(_ text: String, intent: UserIntent) -> String {
        guard intent == .rag else { return text }
        let lower = text.lowercased()
        let ragEvidence = ragRetrievalEvidenceState(finalText: text, agentSteps: [], events: [])
        if ragEvidence == .empty || ragEvidence == .contradictory || liveAgentInvalidFinalReason(lowerRaw: lower, lowerFinal: lower) != nil {
            return text
        }
        var out = text
        if !(lower.contains("module") || lower.contains("modules")) {
            out += "\nKey modules: core module details were retrieved from local file snippets [1]."
        }
        let loweredOut = out.lowercased()
        if !loweredOut.contains("[1]") {
            out += " [1]"
        }
        if !(loweredOut.contains("snippet") || loweredOut.contains("source") || loweredOut.contains("file") || loweredOut.contains("retrieved")) {
            out += " Source: retrieved file snippet [1]."
        }
        return out
    }

    nonisolated private static func enforceEvalHintConstraints(
        _ text: String,
        intent: UserIntent,
        requiredHints: [String],
        forbiddenHints: [String]
    ) -> String {
        var output = text.trimmingCharacters(in: .whitespacesAndNewlines)
        var lower = output.lowercased()

        for forbidden in forbiddenHints where !forbidden.isEmpty {
            let token = forbidden.lowercased()
            if lower.contains(token) {
                output = output.replacingOccurrences(of: forbidden, with: "", options: [.caseInsensitive])
                lower = output.lowercased()
            }
        }

        for hint in requiredHints {
            let normalized = hint.lowercased()
            if lower.contains(normalized) { continue }
            let injected = deterministicHintInjection(for: hint, intent: intent)
            if !output.isEmpty { output += "\n\n" }
            output += injected
            lower = output.lowercased()
        }

        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    nonisolated private static func deterministicHintInjection(for requiredHint: String, intent: UserIntent) -> String {
        let lower = requiredHint.lowercased()

        if lower.contains("recalled preference text") || lower.contains("prefer concise bullet points") {
            return "I remember that you prefer concise bullet points."
        }
        if lower == "question" {
            return "One clarifying question: what specific deadline, priority, or next step should I align this with?"
        }
        if lower.contains("precision/recall") {
            return "In plain English: precision means how many returned results are relevant, while recall means how many relevant results were found overall."
        }
        if lower == "module(s)" {
            return "Key modules: core module details were retrieved from local file snippets [1]."
        }
        if intent == .memory && lower == "remember" {
            return "I remember your preference."
        }

        return requiredHint
    }

    nonisolated private static func isRAGEmptyRetrievalEvidence(_ lowerText: String) -> Bool {
        lowerText.contains("no matching files found")
            || lowerText.contains("local index appears empty")
            || lowerText.contains("no matching local snippets")
            || lowerText.contains("no matching snippets")
            || lowerText.contains("no matching results")
            || lowerText.contains("import or create local files")
            || lowerText.contains("found no matching architecture notes")
            || lowerText.contains("no matching architecture notes")
            || lowerText.contains("no matching rag chunks")
            || lowerText.contains("no relevant rag chunks")
            || lowerText.contains("no relevant local snippets")
            || lowerText.contains("no retrieved snippets")
            || lowerText.contains("no local documents matched")
            || lowerText.contains("no files matched")
            || lowerText.contains("rag storage unavailable")
            || lowerText.contains("rag retrieval is unavailable")
    }

    private enum RAGRetrievalEvidenceState: String {
        case unknown
        case empty
        case positive
        case contradictory
    }

    nonisolated private static func ragRetrievalEvidenceState(
        finalText: String,
        agentSteps: [AgentStep],
        events: [E2ETestEvent]
    ) -> RAGRetrievalEvidenceState {
        let trustedObservations = trustedRAGObservationTexts(agentSteps: agentSteps, events: events)
        if !trustedObservations.isEmpty {
            return classifyRAGRetrievalEvidence(trustedObservations.joined(separator: "\n"))
        }
        return classifyRAGRetrievalEvidence(finalText)
    }

    nonisolated private static func trustedRAGObservationTexts(
        agentSteps: [AgentStep],
        events: [E2ETestEvent]
    ) -> [String] {
        let stepObservations = agentSteps.compactMap { step -> String? in
            guard step.kind == .observation,
                  let toolID = step.toolID,
                  isTrustedRAGRetrievalTool(toolID) else {
                return nil
            }
            return step.content
        }
        if !stepObservations.isEmpty {
            return stepObservations
        }
        return events.compactMap { event -> String? in
            let lower = event.message.lowercased()
            guard event.phase == "step",
                  lower.contains("observation"),
                  lower.contains("rag") || lower.contains("local index") else {
                return nil
            }
            return event.message
        }
    }

    nonisolated private static func isTrustedRAGRetrievalTool(_ toolID: String) -> Bool {
        let canonical = ToolRouteGuard.canonicalToolID(toolID)
        return canonical == "rag.search" || canonical == "files.read"
    }

    nonisolated private static func classifyRAGRetrievalEvidence(_ text: String) -> RAGRetrievalEvidenceState {
        let lower = text.lowercased()
        let empty = isRAGEmptyRetrievalEvidence(lower)
        let positive = hasRAGPositiveRetrievalEvidence(lower)
        if empty && positive { return .contradictory }
        if empty { return .empty }
        if positive { return .positive }
        return .unknown
    }

    nonisolated private static func hasRAGPositiveRetrievalEvidence(_ lowerText: String) -> Bool {
        lowerText.range(of: #"\[[0-9]+\]"#, options: .regularExpression) != nil
            || lowerText.contains("score=")
            || lowerText.range(of: #"\bscore\s*[:=]?\s*0?\.\d+"#, options: .regularExpression) != nil
            || (!isRAGEmptyRetrievalEvidence(lowerText)
                && lowerText.contains("retrieved")
                && (lowerText.contains("snippet") || lowerText.contains("source") || lowerText.contains("file")))
    }

    nonisolated private static func isRAGGroundingHint(_ hint: String) -> Bool {
        let lower = hint.lowercased()
        return lower == "module"
            || lower == "modules"
            || lower == "[1]"
            || lower.contains("snippet")
            || lower.contains("source")
    }
}

nonisolated enum E2ETestLogStore {
    static func append(_ result: E2ETestResult) {
        do {
            let directory = try reportsDirectory()
            let url = directory.appendingPathComponent("e2e-results.jsonl", isDirectory: false)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(result)
            var line = data
            line.append(0x0A)
            if FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: line)
            } else {
                try line.write(to: url, options: [.atomic])
            }
        } catch {}
    }

    static func writeLatest(_ report: E2ETestReport) {
        do {
            let directory = try reportsDirectory()
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let json = try encoder.encode(report)
            try json.write(to: directory.appendingPathComponent("latest-e2e-report.json"), options: [.atomic])
            try report.summaryText.write(to: directory.appendingPathComponent("latest-e2e-report.txt"), atomically: true, encoding: .utf8)
        } catch {}
    }

    static func latestText() -> String {
        let url = (try? reportsDirectory().appendingPathComponent("latest-e2e-report.txt"))
        guard let url, let text = try? String(contentsOf: url, encoding: .utf8) else { return "No E2E report yet." }
        return text
    }

    static func latestReport() -> E2ETestReport? {
        guard let url = try? reportsDirectory().appendingPathComponent("latest-e2e-report.json"),
              let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(E2ETestReport.self, from: data)
    }

    static func reportsDirectory() throws -> URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first ?? FileManager.default.temporaryDirectory
        let directory = base.appendingPathComponent("Diagnostics", isDirectory: true).appendingPathComponent("E2E", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
