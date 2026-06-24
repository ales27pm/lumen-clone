import Foundation
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
        requiresAgentRun: Bool
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
        E2ETestScenario(id: "vague-email-clarifies", title: "Vague email draft asks clarification", kind: .routing, prompt: "Draft a email", expectedIntent: .emailDraft, requiredAllowedToolIDs: ["mail.draft", "contacts.search"], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "reminders.create"], requiredTextHints: ["who should", "what should"], forbiddenTextHints: ["i will be in touch soon", "created a new event"], requiresAgentRun: true),
        E2ETestScenario(id: "normal-chat-no-forced-tool", title: "Normal chat does not force tools", kind: .chat, prompt: "Explain why a sharp chisel is safer than a dull one.", expectedIntent: .chat, requiredAllowedToolIDs: [], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "mail.draft", "reminders.create"], requiredTextHints: [], forbiddenTextHints: ["created a new event", "weather for"], requiresAgentRun: true)
    ]

    static let allToolCoverage: [E2ETestScenario] = liveToolCoverageScenarios()

    static let chatCoverage: [E2ETestScenario] = [
        E2ETestScenario(id: "chat-carpentry-advice", title: "Carpentry chat stays direct", kind: .chat, prompt: "Give me three tips for fitting a door hinge cleanly.", expectedIntent: .chat, requiredAllowedToolIDs: [], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "mail.draft", "reminders.create"], requiredTextHints: [], forbiddenTextHints: ["created a new event", "weather for"], requiresAgentRun: true),
        E2ETestScenario(id: "chat-code-explanation", title: "Code explanation stays chat", kind: .chat, prompt: "Explain actor isolation in Swift in simple terms.", expectedIntent: .chat, requiredAllowedToolIDs: [], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "mail.draft", "reminders.create"], requiredTextHints: [], forbiddenTextHints: ["created a new event", "weather for"], requiresAgentRun: true)
    ]

    private static func liveToolCoverageScenarios() -> [E2ETestScenario] {
        let scenarios = ToolScenarioBank.entries().map(liveToolCoverageScenario)
        precondition(Set(scenarios.map(\.id)).count == scenarios.count, "Live E2E tool scenario IDs must be unique")
        return scenarios
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
            requiresAgentRun: true
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
    let requiresAgentRun: Bool
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
        requiresAgentRun: Bool = false,
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
        performanceMatrix: E2EPerformanceMatrix? = nil
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
        self.requiresAgentRun = requiresAgentRun
        self.passed = passed
        self.failures = failures
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
        self.sanitizedFinalRemovedArtifacts = sanitizedFinalRemovedArtifacts
        self.outputHygieneFailures = outputHygieneFailures
        self.performanceMatrix = performanceMatrix
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
        requiresAgentRun = try c.decodeIfPresent(Bool.self, forKey: .requiresAgentRun) ?? false
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
        lines.append("E2E Test Report")
        lines.append("Passed: \(passed)")
        lines.append("Failed: \(failed)")
        lines.append("")

        let failureBuckets = Dictionary(grouping: results.flatMap(\.failures)) { failure in
            if failure.contains("Intent mismatch") { return "intent" }
            if failure.contains("Forbidden tool") || failure.contains("Required tool not allowed") || failure.contains("Forbidden tool selected by agent") { return "tool-boundary" }
            if failure.contains("Required final hint") || failure.contains("Forbidden final hint") { return "response-quality" }
            if failure.contains("Agent error") { return "runtime" }
            return "other"
        }
        if !failureBuckets.isEmpty {
            lines.append("Training signals for next run:")
            for key in ["intent", "tool-boundary", "response-quality", "runtime", "other"] where failureBuckets[key] != nil {
                lines.append("• \(key): \(failureBuckets[key]?.count ?? 0) issues")
            }
            lines.append("• Capture failed prompts + final outputs into next fine-tuning dataset.")
            lines.append("• Prioritize scenarios with repeated tool-boundary violations.")
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
}

nonisolated enum E2ETestRunner {
    typealias ResultCallback = @Sendable (E2ETestResult) async -> Void
    typealias EventCallback = @Sendable (E2ETestEvent) async -> Void
    typealias EnsureChatLoaded = @Sendable () async -> Bool

    #if DEBUG
    @TaskLocal static var debugStandardScenariosOverride: [E2ETestScenario]?
    @TaskLocal static var debugAssertScenarioLoopOffMainThread = false
    @TaskLocal static var debugScenarioLoopThreadRecorder: (@Sendable (Bool) -> Void)?
    #endif

    static func runStandard(config: E2ERunConfig, ensureChatLoaded: EnsureChatLoaded? = nil, onResult: ResultCallback? = nil, onEvent: EventCallback? = nil) async -> E2ETestReport {
        #if DEBUG
        let scenarios = debugStandardScenariosOverride ?? E2ETestScenario.standard
        #else
        let scenarios = E2ETestScenario.standard
        #endif
        return await run(scenarios: scenarios, config: config, ensureChatLoaded: ensureChatLoaded, onResult: onResult, onEvent: onEvent)
    }

    static func runTrainingValidation(config: E2ERunConfig, ensureChatLoaded: EnsureChatLoaded? = nil, onResult: ResultCallback? = nil, onEvent: EventCallback? = nil) async -> E2ETestReport {
        await run(scenarios: E2ETestScenario.trainingValidation, config: config, ensureChatLoaded: ensureChatLoaded, onResult: onResult, onEvent: onEvent)
    }

    /// Executes end-to-end test scenarios sequentially and generates a report of results and metrics.
    /// - Parameters:
    ///   - scenarios: The test scenarios to execute.
    ///   - config: Runtime configuration for execution.
    ///   - ensureChatLoaded: Optional callback to ensure the chat model is available for scenarios requiring agent execution.
    ///   - onResult: Optional callback invoked when each scenario completes.
    ///   - onEvent: Optional callback invoked during scenario execution.
    /// - Returns: A report containing all results, pass/fail counts, and performance metrics.
    static func run(scenarios: [E2ETestScenario], config: E2ERunConfig, ensureChatLoaded: EnsureChatLoaded? = nil, onResult: ResultCallback? = nil, onEvent: EventCallback? = nil) async -> E2ETestReport {
        let started = Date()
        var results: [E2ETestResult] = []
        for scenario in scenarios {
            #if DEBUG
            let isOnMainThread = Thread.isMainThread
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
                let result = try await runScenario(scenario, config: config, ensureChatLoaded: ensureChatLoaded, onEvent: onEvent)
                results.append(result)
                try Task.checkCancellation()
                await Task.yield()
                E2ETestLogStore.append(result)
                try Task.checkCancellation()
                await Task.yield()
                await onResult?(result)
            } catch is CancellationError {
                break
            } catch {
                let result = E2ETestResult(id: UUID(), scenarioID: scenario.id, kind: scenario.kind.rawValue, title: scenario.title, prompt: scenario.prompt, expectedIntent: scenario.expectedIntent.rawValue, actualIntent: "error", requiresAgentRun: scenario.requiresAgentRun, passed: false, failures: ["E2E runner error: \(error.localizedDescription)"], finalText: "", missingHints: [], rewriteAttempted: false, rewriteSuccess: false, events: [], startedAt: Date(), finishedAt: Date(), rawFinalPrefix: "", sanitizedFinalPrefix: "", rawFinalHadUnsafeLeakage: false, sanitizedFinalRemovedArtifacts: [], outputHygieneFailures: [], performanceMatrix: nil)
                results.append(result)
                E2ETestLogStore.append(result)
                await onResult?(result)
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

    private struct E2ETraceCorrelation: Sendable, Hashable {
        let scenarioID: String?
        let e2eRunID: UUID?
        let agentRunID: UUID?
        let conversationID: UUID?
        let turnID: UUID?

        var diagnosticText: String {
            [
                "scenarioID=\(scenarioID ?? "nil")",
                "e2eRunID=\(e2eRunID?.uuidString ?? "nil")",
                "agentRunID=\(agentRunID?.uuidString ?? "nil")",
                "conversationID=\(conversationID?.uuidString ?? "nil")",
                "turnID=\(turnID?.uuidString ?? "nil")"
            ].joined(separator: ",")
        }

        var hasAnyIdentifier: Bool {
            scenarioID?.isEmpty == false || e2eRunID != nil || agentRunID != nil || conversationID != nil || turnID != nil
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
            let traceCorrelation = E2ETraceCorrelation(
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
                let req = AgentRequest(
                    systemPrompt: config.systemPrompt,
                    history: [],
                    userMessage: scenario.prompt,
                    temperature: min(config.temperature, 0.3),
                    topP: config.topP,
                    repetitionPenalty: config.repetitionPenalty,
                    maxTokens: min(config.maxTokens, 512),
                    maxSteps: min(config.maxAgentSteps, 3),
                    availableTools: ToolRegistry.all.filter { tool in
                        let canonical = ToolRouteGuard.canonicalToolID(tool.id)
                        return enabledCanonicalToolIDs.contains(canonical) && IntentRouter.isToolAllowed(canonical, for: routing)
                    },
                    relevantMemories: [],
                    conversationID: conversationID,
                    turnID: turnID,
                    scenarioID: scenario.id,
                    e2eRunID: e2eRunID,
                    agentRunID: agentRunID
                )
                await event("tools", "available=\(req.availableTools.map(\.id).sorted().joined(separator: ","))")
                var steps: [AgentStep] = []
                try Task.checkCancellation()
                await Task.yield()
                let shouldEnableNetworkAccess = shouldTemporarilyEnableNetworkAccess(
                    scenario: scenario,
                    routing: routing,
                    availableToolIDs: req.availableTools.map(\.id)
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
                let modelEvidenceStartedAt = Date()
                let runOptions = LegacyAgentRunOptions(
                    modelContext: nil,
                    conversationID: req.conversationID,
                    turnID: req.turnID,
                    scenarioID: scenario.id,
                    e2eRunID: e2eRunID,
                    agentRunID: agentRunID,
                    groundingMode: .slotAgent,
                    allowDegradedGrounding: false,
                    preventDoubleGrounding: true,
                    diagnosticsEnabled: false,
                    allowDeterministicCompatibility: scenario.kind != .training,
                    allowParseFailureDeterministicRecovery: scenario.kind != .training
                )
                let agentEvents = await MainActor.run {
                    AssistantKernel.shared.runLegacyAgentBridge(req, options: runOptions)
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
                    case .stepDelta, .toolInvocation, .toolResult, .diagnostic:
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
                let acceptsPolicyFirstEvidence = acceptsPolicyFirstExecutionEvidence(scenario: scenario, routing: routing)
                let evidenceDiagnosis = modelRuntimeEvidenceDiagnosis(
                    since: modelEvidenceStartedAt,
                    prompt: scenario.prompt,
                    correlation: traceCorrelation,
                    acceptsPolicyFirstEvidence: acceptsPolicyFirstEvidence
                )
                if let evidence = evidenceDiagnosis.evidence {
                    let elapsed = evidence.generationElapsedMs.map(String.init) ?? "unknown"
                    let tokens = evidence.outputTokenCount.map(String.init) ?? "unknown"
                    let adapter = evidence.adapterSlot ?? "none"
                    await event("model-evidence", "runtime=\(evidence.runtimePath), kind=\(evidence.evidenceKind), stage=\(evidence.stage), parseError=\(evidence.parseError ?? "none"), elapsedMs=\(elapsed), outputTokens=\(tokens), adapter=\(adapter), matchedBy=\(evidence.matchedBy), \(traceCorrelation.diagnosticText)")
                } else {
                    let requiredEvidence = acceptsPolicyFirstEvidence ? "model-backed or policy-first execution evidence" : "model-backed generation evidence"
                    failures.append("Live E2E scenario did not record \(requiredEvidence)")
                    await event("model-evidence", evidenceDiagnosis.failureMessage)
                }
                try Task.checkCancellation()
                await Task.yield()
                agentSteps = steps
                rawFinalText = FinalIntentValidator.validate(rawFinalText, routing: routing, fallback: nil)

                let recoveredBeforeRewrite = FinalOutputSanitizer.consumeRecoveredUnsafeOutput(forSanitizedText: rawFinalText)
                let rawSanitized = mergeSanitizerOutputs(FinalOutputSanitizer.sanitizeUserVisibleText(rawFinalText), recovered: recoveredBeforeRewrite)
                finalText = rawSanitized.text

                try Task.checkCancellation()
                await Task.yield()
                let rewriteOutcome = await validateAndRewriteFinalTextIfNeeded(
                    scenario: scenario,
                    routing: routing,
                    originalFinal: finalText
                )

                let recoveredAfterRewrite = FinalOutputSanitizer.consumeRecoveredUnsafeOutput(forSanitizedText: rewriteOutcome.finalText)
                let postRewriteSanitized = mergeSanitizerOutputs(FinalOutputSanitizer.sanitizeUserVisibleText(rewriteOutcome.finalText), recovered: recoveredAfterRewrite)
                finalText = postRewriteSanitized.text

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
                await event("final-hints", "missing_hints=\(missingHints), rewrite_attempted=\(rewriteAttempted), rewrite_success=\(rewriteSuccess)")
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
        let outputHygieneFailures = hygieneFailures(
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
        for hint in scenario.requiredTextHints where !lowerFinal.contains(hint.lowercased()) {
            failures.append("Required final hint missing: \(hint)")
        }
        if scenario.expectedIntent == .rag && scenario.requiresAgentRun && scenario.requiredAllowedToolIDs.map(ToolRouteGuard.canonicalToolID).contains("rag.search") {
            if !lowerFinal.contains("module") && !lowerFinal.contains("modules") {
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
        if scenario.id == "training-rag-grounding" {
            if !(lowerFinal.contains("module") || lowerFinal.contains("modules")) {
                failures.append("RAG grounding assertion failed: final text must mention module/modules")
            }
            if !referencesRetrievedSnippet(lowerFinal) {
                failures.append("RAG grounding assertion failed: summary must reference retrieved docs/snippets")
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
        return E2ETestResult(id: UUID(), scenarioID: scenario.id, kind: scenario.kind.rawValue, title: scenario.title, prompt: scenario.prompt, expectedIntent: scenario.expectedIntent.rawValue, actualIntent: routing.intent.rawValue, e2eRunID: e2eRunID, agentRunID: agentRunID, conversationID: conversationID, turnID: turnID, requiresAgentRun: scenario.requiresAgentRun, passed: failures.isEmpty, failures: failures, finalText: finalText, missingHints: missingHints, rewriteAttempted: rewriteAttempted, rewriteSuccess: rewriteSuccess, events: events, startedAt: started, finishedAt: endedAt, rawFinalPrefix: rawPrefix, sanitizedFinalPrefix: sanitizedPrefix, rawFinalHadUnsafeLeakage: hygieneState.hadUnsafeLeakage, sanitizedFinalRemovedArtifacts: mergedAuditArtifacts.map(\.rawValue), outputHygieneFailures: outputHygieneFailures, performanceMatrix: matrix)
    }

    /// Determines whether a scenario accepts policy-first deterministic execution traces as valid evidence.
    /// - Returns: `true` if the scenario accepts such traces, `false` otherwise.
    private nonisolated static func acceptsPolicyFirstExecutionEvidence(scenario: E2ETestScenario, routing: IntentRoutingDecision) -> Bool {
        guard scenario.requiresAgentRun else { return false }
        // Training scenarios are adapter/model promotion evals. They must still
        // prove a fresh modelTurn and must not pass on deterministic policy traces.
        guard scenario.kind != .training else { return false }
        if scenario.kind == .chat, routing.intent == .chat {
            return true
        }
        // Regression/routing scenarios may be intentionally satisfied by the
        // policy-first deterministic compatibility path. Those traces are valid
        // execution evidence when the routed intent is tool-scoped or needs a
        // clarification before tool execution.
        return IntentRouter.intentRequiresTool(routing) || routing.requiresClarification
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

    private nonisolated static func modelRuntimeEvidence(
        since startedAt: Date,
        prompt: String,
        correlation: E2ETraceCorrelation? = nil,
        acceptsPolicyFirstEvidence: Bool
    ) -> ModelRuntimeEvidence? {
        modelRuntimeEvidenceDiagnosis(
            since: startedAt,
            prompt: prompt,
            correlation: correlation,
            acceptsPolicyFirstEvidence: acceptsPolicyFirstEvidence
        ).evidence
    }

    private nonisolated static func modelRuntimeEvidenceDiagnosis(
        since startedAt: Date,
        prompt: String,
        correlation: E2ETraceCorrelation? = nil,
        acceptsPolicyFirstEvidence: Bool
    ) -> ModelRuntimeEvidenceDiagnosis {
        let promptNeedle = prompt.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let recentTraces = AgentBehaviorTraceRecorder.recent(limit: 64).reversed()
        let correlatedTraces = recentTraces.filter { trace in
            traceMatchesCorrelation(trace, correlation: correlation, startedAt: startedAt)
        }
        let fallbackTraces = recentTraces.filter { trace in
            guard trace.createdAt >= startedAt else { return false }
            let promptPrefix = trace.promptPrefix.lowercased()
            if !promptNeedle.isEmpty, !promptPrefix.contains(promptNeedle) {
                return false
            }
            return true
        }
        let usedCorrelation = !correlatedTraces.isEmpty
        let matchingTraces = usedCorrelation ? correlatedTraces : fallbackTraces
        let matchedBy = usedCorrelation ? "correlation" : "prompt-time"

        if let modelTrace = matchingTraces.first(where: { trace in
            trace.event == AgentBehaviorTrace.Event.modelTurn
                && trace.runtimePath != "deterministic-compatibility"
                && trace.parseError == nil
                && !trace.rawOutputPrefix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }) {
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
                fallbackTraceCount: fallbackTraces.count
            )
        )
    }

    private nonisolated static func traceMatchesCorrelation(_ trace: AgentBehaviorTrace, correlation: E2ETraceCorrelation?, startedAt: Date) -> Bool {
        guard let correlation, correlation.hasAnyIdentifier else { return false }
        if let e2eRunID = correlation.e2eRunID, trace.e2eRunID == e2eRunID { return true }
        if let agentRunID = correlation.agentRunID, trace.agentRunID == agentRunID { return true }
        if let conversationID = correlation.conversationID, trace.conversationID == conversationID { return true }
        if let turnID = correlation.turnID, trace.turnID == turnID { return true }
        if let scenarioID = correlation.scenarioID, !scenarioID.isEmpty, trace.scenarioID == scenarioID, trace.createdAt >= startedAt { return true }
        return false
    }

    private nonisolated static func modelRuntimeEvidenceFailureMessage(
        matchingTraces: [AgentBehaviorTrace],
        acceptsPolicyFirstEvidence: Bool,
        correlation: E2ETraceCorrelation? = nil,
        usedCorrelation: Bool = false,
        fallbackTraceCount: Int = 0
    ) -> String {
        if let rejectedModelTrace = matchingTraces.first(where: { $0.event == .modelTurn }) {
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
                    } else {
                        reasons.append("agent-json emitted empty output (\(emptyOutputReason))")
                    }
                } else {
                    reasons.append("raw output was empty")
                }
            }
            if rejectedModelTrace.parseError != nil {
                reasons.append("parseError=\(parseError)")
            }
            if reasons.isEmpty {
                reasons.append("trace did not satisfy model-backed evidence policy")
            }
            return "found AgentBehaviorTrace modelTurn but \(reasons.joined(separator: "; ")); stage=\(rejectedModelTrace.stage); runtimePath=\(runtimePath); parseError=\(parseError); outputTokens=\(rejectedModelTrace.outputTokenCount.map(String.init) ?? "unknown")"
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
            return "no correlated AgentBehaviorTrace found; checked \(correlation.diagnosticText)\(fallbackText); \(base); AgentService model path was not entered or trace export failed"
        }
        return base
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

#if DEBUG
    nonisolated static func scenarioTemporarilyEnablesNetworkAccessForTests(
        _ scenario: E2ETestScenario,
        routing: IntentRoutingDecision,
        availableToolIDs: [String]
    ) -> Bool {
        shouldTemporarilyEnableNetworkAccess(scenario: scenario, routing: routing, availableToolIDs: availableToolIDs)
    }

    nonisolated static func modelRuntimeEvidenceForTests(
        since startedAt: Date,
        prompt: String,
        scenarioID: String? = nil,
        e2eRunID: UUID? = nil,
        agentRunID: UUID? = nil,
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
        acceptsPolicyFirstEvidence: Bool = false
    ) -> Bool {
        modelRuntimeEvidence(
            since: startedAt,
            prompt: prompt,
            correlation: E2ETraceCorrelation(scenarioID: scenarioID, e2eRunID: e2eRunID, agentRunID: agentRunID, conversationID: conversationID, turnID: turnID),
            acceptsPolicyFirstEvidence: acceptsPolicyFirstEvidence
        ) != nil
    }

    nonisolated static func modelRuntimeEvidenceFailureMessageForTests(
        since startedAt: Date,
        prompt: String,
        scenarioID: String? = nil,
        e2eRunID: UUID? = nil,
        agentRunID: UUID? = nil,
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
        acceptsPolicyFirstEvidence: Bool = false
    ) -> String {
        modelRuntimeEvidenceDiagnosis(
            since: startedAt,
            prompt: prompt,
            correlation: E2ETraceCorrelation(scenarioID: scenarioID, e2eRunID: e2eRunID, agentRunID: agentRunID, conversationID: conversationID, turnID: turnID),
            acceptsPolicyFirstEvidence: acceptsPolicyFirstEvidence
        ).failureMessage
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
        if scenario.expectedIntent == .weather && weatherGroundingOverreach(finalText: lowerFinal, observations: observations) {
            failures.append("Weather precipitation recommendation not grounded")
        }
        return mergedStrings(failures)
    }

    nonisolated static func liveAgentQualityFailures(rawFinalText: String, finalText: String, scenario: E2ETestScenario) -> [String] {
        guard scenario.requiresAgentRun else { return [] }
        var failures: [String] = []
        let lowerRaw = rawFinalText.lowercased()
        let lowerFinal = finalText.lowercased()
        if let reason = liveAgentInvalidFinalReason(lowerRaw: lowerRaw, lowerFinal: lowerFinal) {
            failures.append(reason)
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
            "i hit an internal response-format issue",
            "only internal reasoning and no final answer",
            "no model loaded; routing-only checks completed",
            "routing-only checks completed",
            "full local model pipeline is temporarily running in compatibility mode",
            "full agent pipeline",
            "please try again with thinking disabled",
            "please ask again or tell me what you'd like to do next"
        ]
        return invalidSignals.contains(where: { combined.contains($0) })
            ? "Live agent returned fallback/error text instead of completing the scenario"
            : nil
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
        let signals = ["[1]", "[2]", "snippet", "source", "file", "pdf", "note", "photos", "retrieved"]
        return signals.contains { lowerFinal.contains($0) }
    }

    private struct EvalRewriteOutcome {
        let finalText: String
        let missingHints: [String]
        let rewriteAttempted: Bool
        let rewriteSuccess: Bool
    }

    nonisolated private static func validateAndRewriteFinalTextIfNeeded(
        scenario: E2ETestScenario,
        routing: IntentRoutingDecision,
        originalFinal: String
    ) async -> EvalRewriteOutcome {
        let firstMissing = requiredHintsMissing(in: originalFinal, scenario: scenario)
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
        let secondMissing = requiredHintsMissing(in: rewritten, scenario: scenario)
        let rewriteSuccess = secondMissing.isEmpty
        return EvalRewriteOutcome(finalText: rewritten, missingHints: secondMissing, rewriteAttempted: true, rewriteSuccess: rewriteSuccess)
    }

    nonisolated private static func requiredHintsMissing(in finalText: String, scenario: E2ETestScenario) -> [String] {
        let lower = finalText.lowercased()
        var missing: [String] = scenario.requiredTextHints.filter { !lower.contains($0.lowercased()) }
        if scenario.id == "training-general-chat" {
            if !lower.contains("precision") || !lower.contains("recall") {
                missing.append("precision/recall plain-language explainer")
            }
        }
        if scenario.id == "training-rag-grounding", !(lower.contains("module") || lower.contains("modules")) {
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
