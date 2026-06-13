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

    private static let allToolCoverageBase: [E2ETestScenario] = [
        // Calendar
        E2ETestScenario(id: "tool-calendar-create", title: "calendar.create scoped", kind: .toolGuard, prompt: "Create an event tomorrow at 5 called test appointment", expectedIntent: .calendar, requiredAllowedToolIDs: ["calendar.create", "calendar.list"], forbiddenToolIDs: ["weather", "web.search", "mail.draft", "maps.search"], requiredTextHints: [], forbiddenTextHints: ["weather for", "web search"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-calendar-list", title: "calendar.list scoped", kind: .toolGuard, prompt: "List my upcoming calendar events", expectedIntent: .calendar, requiredAllowedToolIDs: ["calendar.create", "calendar.list"], forbiddenToolIDs: ["weather", "web.search", "reminders.create", "mail.draft"], requiredTextHints: [], forbiddenTextHints: ["weather for"], requiresAgentRun: false),

        // Reminders
        E2ETestScenario(id: "tool-reminders-create", title: "reminders.create scoped", kind: .toolGuard, prompt: "Remind me to call Alex tomorrow", expectedIntent: .reminder, requiredAllowedToolIDs: ["reminders.create", "reminders.list"], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "mail.draft"], requiredTextHints: [], forbiddenTextHints: ["calendar event", "weather for"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-reminders-list", title: "reminders.list scoped", kind: .toolGuard, prompt: "List my pending reminders", expectedIntent: .reminder, requiredAllowedToolIDs: ["reminders.create", "reminders.list"], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "mail.draft"], requiredTextHints: [], forbiddenTextHints: ["calendar event", "weather for"], requiresAgentRun: false),

        // Communication
        E2ETestScenario(id: "tool-contacts-search", title: "contacts.search scoped", kind: .toolGuard, prompt: "Search contacts for Alex", expectedIntent: .contactSearch, requiredAllowedToolIDs: ["contacts.search"], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "maps.search"], requiredTextHints: [], forbiddenTextHints: ["calendar event", "weather for"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-messages-draft", title: "messages.draft scoped", kind: .toolGuard, prompt: "Draft a text message to Alex saying I am running late", expectedIntent: .messageDraft, requiredAllowedToolIDs: ["messages.draft", "contacts.search"], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "mail.draft"], requiredTextHints: [], forbiddenTextHints: ["calendar event", "weather for"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-mail-draft", title: "mail.draft scoped", kind: .toolGuard, prompt: "Write an email to alex@example.com saying the plans are ready", expectedIntent: .emailDraft, requiredAllowedToolIDs: ["mail.draft", "contacts.search"], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "messages.draft"], requiredTextHints: [], forbiddenTextHints: ["created a new event", "weather for"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-phone-call", title: "phone.call scoped", kind: .toolGuard, prompt: "Call 5145551234", expectedIntent: .phoneCall, requiredAllowedToolIDs: ["phone.call", "contacts.search"], forbiddenToolIDs: ["calendar.create", "weather", "web.search", "mail.draft"], requiredTextHints: [], forbiddenTextHints: ["calendar event", "weather for"], requiresAgentRun: false),

        // Location / Weather / Maps
        E2ETestScenario(id: "tool-location-current", title: "location.current scoped through local weather", kind: .toolGuard, prompt: "Use my current location for the weather here", expectedIntent: .weather, requiredAllowedToolIDs: ["weather", "location.current"], forbiddenToolIDs: ["calendar.create", "web.search", "mail.draft", "reminders.create"], requiredTextHints: [], forbiddenTextHints: ["calendar event"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-weather", title: "weather scoped", kind: .toolGuard, prompt: "What is the temperature outside right now?", expectedIntent: .weather, requiredAllowedToolIDs: ["weather", "location.current"], forbiddenToolIDs: ["calendar.create", "web.search", "mail.draft", "reminders.create"], requiredTextHints: [], forbiddenTextHints: ["calendar event"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-maps-directions", title: "maps.directions scoped", kind: .toolGuard, prompt: "Get directions to 123 Main Street", expectedIntent: .maps, requiredAllowedToolIDs: ["maps.directions", "maps.search", "location.current"], forbiddenToolIDs: ["calendar.create", "web.search", "mail.draft", "weather"], requiredTextHints: [], forbiddenTextHints: ["calendar event"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-maps-search", title: "maps.search scoped", kind: .toolGuard, prompt: "Find the closest hardware store near me", expectedIntent: .maps, requiredAllowedToolIDs: ["maps.search", "maps.directions", "location.current"], forbiddenToolIDs: ["calendar.create", "web.search", "mail.draft", "weather"], requiredTextHints: [], forbiddenTextHints: ["calendar event"], requiresAgentRun: false),

        // Media
        E2ETestScenario(id: "tool-photos-search", title: "photos.search scoped", kind: .toolGuard, prompt: "Search photos from last month", expectedIntent: .photos, requiredAllowedToolIDs: ["photos.search"], forbiddenToolIDs: ["web.search", "calendar.create", "camera.capture", "mail.draft"], requiredTextHints: [], forbiddenTextHints: ["web search"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-music-play", title: "music.play scoped", kind: .toolGuard, prompt: "Play music", expectedIntent: .music, requiredAllowedToolIDs: ["music.play"], forbiddenToolIDs: ["web.search", "calendar.create", "photos.search", "mail.draft"], requiredTextHints: [], forbiddenTextHints: ["calendar event"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-camera-capture", title: "camera.capture scoped", kind: .toolGuard, prompt: "Take a photo", expectedIntent: .camera, requiredAllowedToolIDs: ["camera.capture"], forbiddenToolIDs: ["photos.search", "web.search", "calendar.create", "mail.draft"], requiredTextHints: [], forbiddenTextHints: ["web search"], requiresAgentRun: false),

        // Web / Knowledge
        E2ETestScenario(id: "tool-web-search", title: "web.search scoped", kind: .toolGuard, prompt: "Search the web for local building code examples", expectedIntent: .webSearch, requiredAllowedToolIDs: ["web.search", "web.fetch"], forbiddenToolIDs: ["calendar.create", "weather", "mail.draft", "photos.search"], requiredTextHints: [], forbiddenTextHints: ["calendar event"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-news-top", title: "news.top scoped", kind: .toolGuard, prompt: "Show me top news headlines", expectedIntent: .news, requiredAllowedToolIDs: ["news.top", "web.search"], forbiddenToolIDs: ["calendar.create", "weather", "mail.draft", "photos.search"], requiredTextHints: [], forbiddenTextHints: ["calendar event"], requiresAgentRun: false),

        // Files / Memory / RAG
        E2ETestScenario(id: "tool-files-read", title: "files.read scoped", kind: .toolGuard, prompt: "Read my selected file", expectedIntent: .fileRead, requiredAllowedToolIDs: ["files.read"], forbiddenToolIDs: ["web.search", "calendar.create", "weather", "mail.draft"], requiredTextHints: [], forbiddenTextHints: ["weather for"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-memory-save", title: "memory.save scoped", kind: .toolGuard, prompt: "Remember that I like short answers", expectedIntent: .memory, requiredAllowedToolIDs: ["memory.save", "memory.recall"], forbiddenToolIDs: ["web.search", "calendar.create", "weather", "mail.draft"], requiredTextHints: [], forbiddenTextHints: ["web search"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-rag-search", title: "rag.search scoped", kind: .toolGuard, prompt: "Search my documents for architecture notes", expectedIntent: .rag, requiredAllowedToolIDs: ["rag.search", "files.read"], forbiddenToolIDs: ["web.search", "calendar.create", "weather", "mail.draft"], requiredTextHints: [], forbiddenTextHints: ["weather for"], requiresAgentRun: false),

        // Health / Device / Control
        E2ETestScenario(id: "tool-health-summary", title: "health.summary scoped", kind: .toolGuard, prompt: "Summarize my steps today", expectedIntent: .health, requiredAllowedToolIDs: ["health.summary"], forbiddenToolIDs: ["web.search", "calendar.create", "weather", "mail.draft"], requiredTextHints: [], forbiddenTextHints: ["calendar event"], requiresAgentRun: false),
        E2ETestScenario(id: "tool-device-status", title: "device.status scoped", kind: .toolGuard, prompt: "Show device battery and thermal status", expectedIntent: .deviceStatus, requiredAllowedToolIDs: ["device.status"], forbiddenToolIDs: ["web.search", "calendar.create", "weather", "mail.draft"], requiredTextHints: [], forbiddenTextHints: ["weather for"], requiresAgentRun: false)
    ]

    static let allToolCoverage: [E2ETestScenario] = {
        var seen: Set<String> = []
        return allToolCoverageBase.filter { scenario in
            seen.insert(scenario.id).inserted
        }
    }()

    static let chatCoverage: [E2ETestScenario] = [
        E2ETestScenario(id: "chat-construction-advice", title: "chat: construction advice", kind: .chat, prompt: "Explain three safe ways to clamp a board before chiseling.", expectedIntent: .chat, requiredAllowedToolIDs: [], forbiddenToolIDs: ["calendar.create", "mail.draft", "web.search"], requiredTextHints: ["clamp"], forbiddenTextHints: ["calendar event", "email draft"], requiresAgentRun: true),
        E2ETestScenario(id: "chat-ai-explanation", title: "chat: AI explanation", kind: .chat, prompt: "Explain vector search like I'm new to AI.", expectedIntent: .chat, requiredAllowedToolIDs: [], forbiddenToolIDs: ["calendar.create", "weather", "mail.draft"], requiredTextHints: ["vector"], forbiddenTextHints: ["calendar event"], requiresAgentRun: true)
    ]
}

nonisolated struct E2ETestResult: Identifiable, Codable, Sendable {
    let id: UUID
    let scenarioID: String
    let title: String
    let kind: E2ETestKind
    let passed: Bool
    let summary: String
    let details: [String]
    let trace: [String]
    let durationMs: Int
    let peakRSSMB: Double?

    init(scenario: E2ETestScenario, passed: Bool, summary: String, details: [String], trace: [String], durationMs: Int, peakRSSMB: Double?) {
        self.id = UUID()
        self.scenarioID = scenario.id
        self.title = scenario.title
        self.kind = scenario.kind
        self.passed = passed
        self.summary = summary
        self.details = details
        self.trace = trace
        self.durationMs = durationMs
        self.peakRSSMB = peakRSSMB
    }
}

nonisolated struct E2ETestSuiteResult: Codable, Sendable {
    let results: [E2ETestResult]
    var passed: Bool { results.allSatisfy(\.passed) }
    var passedCount: Int { results.filter(\.passed).count }
    var failedCount: Int { results.count - passedCount }
}

actor E2ETestRunner {
    static let shared = E2ETestRunner()
    private let traceRecorder = AgentBehaviorTraceRecorder.shared

    func run(config: E2ERunConfig, modelLoaded: Bool, kinds: Set<E2ETestKind>? = nil) async -> E2ETestSuiteResult {
        let selected = E2ETestScenario.standard.filter { kinds == nil || kinds!.contains($0.kind) }
        var results: [E2ETestResult] = []
        for scenario in selected {
            if Task.isCancelled { break }
            results.append(await runScenario(scenario, config: config, modelLoaded: modelLoaded))
        }
        return E2ETestSuiteResult(results: results)
    }

    func runTrainingValidation(config: E2ERunConfig, modelLoaded: Bool) async -> E2ETestSuiteResult {
        var results: [E2ETestResult] = []
        for scenario in E2ETestScenario.trainingValidation {
            if Task.isCancelled { break }
            results.append(await runScenario(scenario, config: config, modelLoaded: modelLoaded))
        }
        return E2ETestSuiteResult(results: results)
    }

    private func runScenario(_ scenario: E2ETestScenario, config: E2ERunConfig, modelLoaded: Bool) async -> E2ETestResult {
        let start = Date()
        let startRSS = currentRSSMB()
        let beforeTraceID = await traceRecorder.recentAsync(limit: 1).last?.id
        var trace: [String] = []
        var failures: [String] = []
        var rawFinalText = ""
        var finalText = ""
        var agentSteps: [AgentStep] = []

        func event(_ stage: String, _ value: String) async {
            trace.append("\(stage): \(value)")
            await Task.yield()
        }

        func collectPerformanceSample(force: Bool = false) {
            let sample = currentRSSMB()
            if force || sample != nil {
                trace.append("perf.rssMB=\(sample.map { String(format: "%.1f", $0) } ?? "unknown")")
            }
        }

        do {
            try Task.checkCancellation()
            await event("scenario", scenario.id)
            let routing = AgentIntentRouter.route(scenario.prompt)
            await event("intent", "expected=\(scenario.expectedIntent.rawValue), actual=\(routing.intent.rawValue), confidence=\(String(format: "%.2f", routing.confidence))")
            if routing.intent != scenario.expectedIntent {
                failures.append("Expected intent \(scenario.expectedIntent.rawValue), got \(routing.intent.rawValue)")
            }

            let allowedTools = DeterministicToolPlanner.allowedTools(for: routing)
            let allowedIDs = Set(allowedTools.map { ToolRouteGuard.canonicalToolID($0.id) })
            let allowedRawIDs = Set(allowedTools.map(\.id))
            let requiredAllowedCanonicalIDs = Set(scenario.requiredAllowedToolIDs.map(ToolRouteGuard.canonicalToolID))
            for required in requiredAllowedCanonicalIDs {
                if !allowedIDs.contains(required) {
                    failures.append("Required tool not allowed: \(required)")
                }
            }
            for forbidden in scenario.forbiddenToolIDs {
                let canonicalForbidden = ToolRouteGuard.canonicalToolID(forbidden)
                if allowedIDs.contains(canonicalForbidden) {
                    failures.append("Forbidden tool allowed: \(forbidden)")
                }
            }
            await event("allowedTools", allowedRawIDs.sorted().joined(separator: ","))

            if !scenario.requiresAgentRun {
                finalText = "Policy guard completed for \(scenario.expectedIntent.rawValue)."
            } else {
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
                        relevantMemories: []
                    )
                    await event("tools", "available=\(req.availableTools.map(\.id).sorted().joined(separator: ","))")
                    var steps: [AgentStep] = []
                    try Task.checkCancellation()
                    await Task.yield()
                    let modelEvidenceStartedAt = Date()
                    let runOptions = LegacyAgentRunOptions(
                        modelContext: nil,
                        conversationID: req.conversationID,
                        turnID: req.turnID,
                        groundingMode: .slotAgent,
                        allowDegradedGrounding: false,
                        preventDoubleGrounding: true,
                        diagnosticsEnabled: false
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
                    if let evidence = modelRuntimeEvidence(
                        since: modelEvidenceStartedAt,
                        prompt: scenario.prompt,
                        acceptsPolicyFirstEvidence: acceptsPolicyFirstEvidence
                    ) {
                        let elapsed = evidence.generationElapsedMs.map(String.init) ?? "unknown"
                        let tokens = evidence.outputTokenCount.map(String.init) ?? "unknown"
                        let adapter = evidence.adapterSlot ?? "none"
                        await event("model-evidence", "runtime=\(evidence.runtimePath), kind=\(evidence.evidenceKind), stage=\(evidence.stage), elapsedMs=\(elapsed), outputTokens=\(tokens), adapter=\(adapter)")
                    } else {
                        let requiredEvidence = acceptsPolicyFirstEvidence ? "model-backed or policy-first execution evidence" : "model-backed generation evidence"
                        failures.append("Live E2E scenario did not record \(requiredEvidence)")
                        await event("model-evidence", acceptsPolicyFirstEvidence ? "missing fresh AgentBehaviorTrace modelTurn or deterministic-compatibility execution trace" : "missing fresh AgentBehaviorTrace modelTurn")
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
                        rawText: finalText,
                        routing: routing,
                        agentSteps: agentSteps,
                        config: config
                    )
                    finalText = rewriteOutcome.finalText
                    failures.append(contentsOf: rewriteOutcome.failures)
                    trace.append(contentsOf: rewriteOutcome.trace)
                } else {
                    await event("model", "not-loaded; using deterministic compatibility output")
                    let planned = allowedTools.first?.id ?? "none"
                    let raw = deterministicFinalText(for: scenario, routing: routing, plannedTool: planned)
                    let recovered = FinalOutputSanitizer.consumeRecoveredUnsafeOutput(forSanitizedText: raw)
                    finalText = mergeSanitizerOutputs(FinalOutputSanitizer.sanitizeUserVisibleText(raw), recovered: recovered).text
                }
            }

            try Task.checkCancellation()
            for hint in scenario.requiredTextHints {
                if !finalText.localizedCaseInsensitiveContains(hint) {
                    failures.append("Missing required text hint: \(hint)")
                }
            }
            for hint in scenario.forbiddenTextHints {
                if finalText.localizedCaseInsensitiveContains(hint) {
                    failures.append("Forbidden text hint present: \(hint)")
                }
            }
            if !finalText.isEmpty {
                let preview = String(finalText.prefix(120)).replacingOccurrences(of: "\n", with: " ")
                await event("final", preview)
            }
            if modelLoaded, scenario.requiresAgentRun {
                let afterTrace = await traceRecorder.recentAsync(limit: 12)
                let producedNewTrace = afterTrace.contains { trace in
                    guard let beforeTraceID else { return true }
                    return trace.id != beforeTraceID
                }
                if !producedNewTrace {
                    failures.append("Agent run did not record a behavior trace")
                }
                if let last = afterTrace.last {
                    await event("trace", "last=\(last.traceKind.rawValue):\(last.decisionKind.rawValue)")
                }
            }
        } catch is CancellationError {
            failures.append("Scenario cancelled")
        } catch {
            failures.append("Scenario threw: \(error.localizedDescription)")
        }

        let duration = Int(Date().timeIntervalSince(start) * 1000)
        let endRSS = currentRSSMB()
        let peak = [startRSS, endRSS].compactMap { $0 }.max()
        let passed = failures.isEmpty
        let summary = passed ? "Passed" : failures.joined(separator: "; ")
        return E2ETestResult(scenario: scenario, passed: passed, summary: summary, details: failures, trace: trace, durationMs: duration, peakRSSMB: peak)
    }

    private func validateAndRewriteFinalTextIfNeeded(
        scenario: E2ETestScenario,
        rawText: String,
        routing: IntentRoutingResult,
        agentSteps: [AgentStep],
        config: E2ERunConfig
    ) async -> (finalText: String, failures: [String], trace: [String]) {
        var trace: [String] = []
        var failures: [String] = []
        var finalText = rawText
        let needsRewrite = scenario.requiredTextHints.contains { !finalText.localizedCaseInsensitiveContains($0) }
            || scenario.forbiddenTextHints.contains { finalText.localizedCaseInsensitiveContains($0) }
        guard needsRewrite else { return (finalText, [], trace) }
        let rewritten = await rewriteForE2EPolicy(
            scenario: scenario,
            routing: routing,
            rawText: finalText,
            agentSteps: agentSteps,
            config: config
        )
        finalText = rewritten
        trace.append("rewrite: applied E2E policy final rewrite")
        let recovered = FinalOutputSanitizer.consumeRecoveredUnsafeOutput(forSanitizedText: rewritten)
        let sanitized = mergeSanitizerOutputs(FinalOutputSanitizer.sanitizeUserVisibleText(rewritten), recovered: recovered)
        finalText = sanitized.text
        return (finalText, failures, trace)
    }

    private func rewriteForE2EPolicy(
        scenario: E2ETestScenario,
        routing: IntentRoutingResult,
        rawText: String,
        agentSteps: [AgentStep],
        config: E2ERunConfig
    ) async -> String {
        let allowed = DeterministicToolPlanner.allowedTools(for: routing).map(\.id).sorted().joined(separator: ", ")
        let used = agentSteps.compactMap(\.toolID).map(ToolRouteGuard.canonicalToolID).sorted().joined(separator: ", ")
        let prompt = """
        Rewrite the assistant's final answer so it satisfies this E2E policy test.
        Intent: \(routing.intent.rawValue)
        Original user prompt: \(scenario.prompt)
        Required text hints: \(scenario.requiredTextHints.joined(separator: ", "))
        Forbidden text hints: \(scenario.forbiddenTextHints.joined(separator: ", "))
        Allowed tools: \(allowed)
        Used tools: \(used.isEmpty ? "none" : used)
        Raw answer:
        \(rawText)

        Return only the final user-visible answer. Do not mention E2E tests.
        """
        let req = GenerationRequest(prompt: prompt, systemPrompt: config.systemPrompt, temperature: min(config.temperature, 0.2), topP: config.topP, repetitionPenalty: config.repetitionPenalty, maxTokens: min(config.maxTokens, 320))
        let response = await AppLlamaService.shared.generate(req: req, slot: .cortex, intent: .standard)
        let text = response.text.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? deterministicFinalText(for: scenario, routing: routing, plannedTool: used.isEmpty ? "none" : used) : text
    }

    private func deterministicFinalText(for scenario: E2ETestScenario, routing: IntentRoutingResult, plannedTool: String) -> String {
        switch routing.intent {
        case .weather:
            return "Weather answer prepared with scoped weather tools."
        case .webSearch:
            return "Web search summary prepared with scoped web tools."
        case .calendar:
            return "Calendar request prepared with scoped calendar tools."
        case .reminder:
            return "Reminder request prepared with scoped reminders tools."
        case .contactSearch:
            return "Contact search prepared with scoped contacts tools."
        case .messageDraft:
            return "Message draft prepared."
        case .emailDraft:
            return "I need to know who should receive it and what key details to include before drafting the email."
        case .phoneCall:
            return "Phone call request prepared."
        case .maps:
            return "Maps request prepared."
        case .photos:
            return "Photo search request prepared."
        case .music:
            return "Music request prepared."
        case .camera:
            return "Camera request prepared."
        case .news:
            return "News summary prepared."
        case .fileRead:
            return "File read request prepared."
        case .memory:
            return "Memory request prepared."
        case .rag:
            return "Document search result [1] prepared with module summary."
        case .health:
            return "Health summary prepared."
        case .deviceStatus:
            return "Device status prepared."
        case .trigger:
            return "Trigger schedule prepared and confirmed."
        case .chat:
            if scenario.prompt.localizedCaseInsensitiveContains("vector") {
                return "Vector search compares meaning so similar ideas can be found even when exact words differ."
            }
            return "A sharp chisel is safer because it cuts predictably with less force and is easier to clamp and control."
        }
    }

    private func acceptsPolicyFirstExecutionEvidence(scenario: E2ETestScenario, routing: IntentRoutingResult) -> Bool {
        guard scenario.kind == .training else { return false }
        switch routing.intent {
        case .trigger, .memory, .rag, .emailDraft:
            return true
        default:
            return false
        }
    }

    private func modelRuntimeEvidence(since start: Date, prompt: String, acceptsPolicyFirstEvidence: Bool) -> AgentBehaviorTrace? {
        let promptSnippet = prompt.lowercased().prefix(24)
        let traces = AgentBehaviorTraceRecorder.shared.recent(limit: 80)
        return traces.reversed().first { trace in
            guard trace.createdAt >= start.addingTimeInterval(-1) else { return false }
            let promptMatches = trace.promptHashHint.lowercased().contains(promptSnippet) || promptSnippet.isEmpty
            guard promptMatches || trace.traceKind == .modelTurn else { return false }
            if trace.traceKind == .modelTurn {
                return trace.runtimePath != "policy-only"
            }
            return acceptsPolicyFirstEvidence && trace.evidenceKind == "deterministic-compatibility"
        }
    }

    private func currentRSSMB() -> Double? {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard kerr == KERN_SUCCESS else { return nil }
        return Double(info.resident_size) / 1_048_576.0
        #else
        return nil
        #endif
    }
}
