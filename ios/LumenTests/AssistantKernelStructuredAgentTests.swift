import Foundation
import XCTest
@testable import Lumen

@MainActor
final class AssistantKernelStructuredAgentTests: XCTestCase {
    private actor ScriptedLlamaService: LlamaRuntimeStreamingService {
        private var chatLoaded: Bool
        var isChatLoaded: Bool { chatLoaded }
        let isEmbedLoaded = false
        private let scripts: [[GenerationToken]]
        private let readiness: ExecutorRuntimePreflightResult
        private let loadsChatOnReadiness: Bool
        private(set) var readinessCallCount = 0
        private(set) var preflightCallCount = 0
        private(set) var streamCallCount = 0
        private(set) var callOrder: [String] = []

        init(
            isChatLoaded: Bool,
            scripts: [[GenerationToken]],
            readiness: ExecutorRuntimePreflightResult? = nil,
            loadsChatOnReadiness: Bool = false
        ) {
            self.chatLoaded = isChatLoaded
            self.scripts = scripts
            self.readiness = readiness ?? ExecutorRuntimePreflightResult(
                passed: true,
                reason: "scripted runtime ready",
                baseModelExists: true,
                resourceGateAllowed: true,
                ensureReadySucceeded: true
            )
            self.loadsChatOnReadiness = loadsChatOnReadiness
        }

        func prepareStructuredRuntime(
            slot: LumenModelSlot,
            allowsLoadedMemoryPressureContinuation: Bool
        ) -> ExecutorRuntimePreflightResult {
            readinessCallCount += 1
            callOrder.append("readiness")
            if readiness.passed, loadsChatOnReadiness {
                chatLoaded = true
            }
            return readiness
        }

        func structuredPromptPreflight(_ request: GenerateRequest, slot: LumenModelSlot) -> LlamaStructuredPromptPreflightSnapshot {
            preflightCallCount += 1
            callOrder.append("preflight")
            let chars = request.systemPrompt.count + request.userMessage.count
            return .init(contextSize: 4_096, finalPromptChars: chars, estimatedPromptTokens: max(1, chars / 4))
        }

        func stream(_ req: GenerateRequest, slot: LumenModelSlot) -> AsyncStream<GenerationToken> {
            let index = streamCallCount
            streamCallCount += 1
            callOrder.append("stream")
            let tokens = scripts.indices.contains(index) ? scripts[index] : []
            return AsyncStream { continuation in
                for token in tokens {
                    continuation.yield(token)
                }
                continuation.finish()
            }
        }

        func embed(_ text: String) async throws -> [Double] { [] }

        func embedWithIdentity(_ text: String) async throws -> EmbeddingRuntimeResult {
            EmbeddingRuntimeResult(vector: [], modelIdentifier: "test:structured")
        }
    }

    private struct StubWeatherTool: LocalTool {
        let observation: String

        init(observation: String = "It is 21°C and clear.") {
            self.observation = observation
        }

        let definition = SecureToolDefinition(
            id: "weather",
            displayName: "Weather",
            description: "Returns a grounded weather observation.",
            category: .readOnly,
            requiredPermissions: [],
            supportsBackgroundExecution: true,
            requiresUserApproval: false,
            argumentSchemaDescription: "{}",
            resultPrivacyLevel: .low,
            maxOutputCharacters: 2_000
        )

        func validateArguments(_ arguments: [String: String]) throws {}

        func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
            ToolResult(
                invocationID: invocation.id,
                status: .success,
                displayText: observation,
                modelText: observation,
                structuredPayload: ["temperature": "21", "conditions": "clear"],
                privacyLevel: .low,
                metricsSummary: "weather_success",
                errorCode: nil
            )
        }
    }

    func testToolRequiredFirstTurnUsesActionOnlySchema() throws {
        #if DEBUG
        let request = structuredRequest(
            "What's the weather here?",
            allowedToolIDs: ["weather"]
        )
        let schema = StructuredAgentKernelExecutor.structuredAgentResponseSchemaForTests(
            request: request,
            availableTools: [try tool("weather")],
            stepIndex: 0,
            hasObservations: false
        )

        XCTAssertEqual(schema, StructuredAgentKernelExecutor.structuredAgentActionResponseSchema)
        XCTAssertTrue(schema.contains(#""required":["action"]"#))
        XCTAssertFalse(schema.contains(#""required":["final"]"#))
        #endif
    }

    func testNoToolTurnUsesFinalOnlySchema() {
        #if DEBUG
        let request = structuredRequest("Say hello.", allowedToolIDs: [])
        let schema = StructuredAgentKernelExecutor.structuredAgentResponseSchemaForTests(
            request: request,
            availableTools: [],
            stepIndex: 0,
            hasObservations: false
        )

        XCTAssertEqual(schema, StructuredAgentKernelExecutor.structuredAgentFinalResponseSchema)
        XCTAssertTrue(schema.contains(#""required":["final"]"#))
        XCTAssertFalse(schema.contains(#""required":["action"]"#))
        #endif
    }

    func testWebSearchSynthesizesSwiftSummaryAfterFirstObservation() throws {
        #if DEBUG
        let request = structuredRequest(
            "Search the web for Swift concurrency best practices.",
            allowedToolIDs: ["web.search"],
            traceCorrelation: AgentTraceCorrelation(
                scenarioID: "training-web-research",
                e2eRunID: UUID(),
                agentRunID: UUID(),
                conversationID: UUID(),
                turnID: UUID()
            )
        )
        let observation = """
        <lumen_web_payload>{"results":[{"title":"Swift.org Concurrency","snippet":"Swift concurrency uses async/await, tasks, actors, and structured concurrency to keep code safe."},{"title":"Apple Developer Documentation","snippet":"Swift tasks and actors help isolate mutable state while preserving responsive apps."}]}</lumen_web_payload>
        """

        XCTAssertTrue(StructuredAgentKernelExecutor.shouldStopAfterFirstWebObservationForTests(
            request: request,
            actionTool: "web.search",
            observations: [("web.search", observation)]
        ))

        let final = try XCTUnwrap(StructuredAgentKernelExecutor.deterministicObservationFallbackForTests(
            observations: [("web.search", observation)],
            intent: .webSearch
        ))
        XCTAssertTrue(final.contains("Swift"))
        XCTAssertTrue(final.contains("Summary:"))
        XCTAssertFalse(final.localizedCaseInsensitiveContains("No direct answer from web search"))
        #endif
    }

    func testPlaceholderWeatherFinalIsRejectedBeforeTrustedObservation() throws {
        #if DEBUG
        let request = structuredRequest(
            "What's the weather here?",
            allowedToolIDs: ["weather"]
        )
        let tools = [try tool("weather")]

        XCTAssertTrue(StructuredAgentKernelExecutor.toolRequiredFinalNeedsActionForTests(
            "[insert local weather information]",
            request: request,
            availableTools: tools,
            observations: []
        ))
        XCTAssertTrue(StructuredAgentKernelExecutor.toolRequiredFinalNeedsActionForTests(
            "It is sunny and 21°C.",
            request: request,
            availableTools: tools,
            observations: []
        ))
        XCTAssertFalse(StructuredAgentKernelExecutor.toolRequiredFinalNeedsActionForTests(
            "[insert local weather information]",
            request: request,
            availableTools: tools,
            observations: [("weather", "The weather is clear with a temperature of 21°C.")]
        ))
        #endif
    }

    func testDisjointStructuredAllowlistDoesNotBroadenToAllSecureTools() {
        #if DEBUG
        let sourceIDs = StructuredAgentKernelExecutor.structuredToolSourceIDsForTests(
            secureIDs: ["weather", "web.search", "maps.search"],
            optionIDs: ["web.search"],
            routingIDs: ["weather"]
        )

        XCTAssertEqual(sourceIDs, Set(["web.search"]))
        XCTAssertFalse(sourceIDs.contains("weather"))
        XCTAssertFalse(sourceIDs.contains("maps.search"))
        #endif
    }

    func testStructuredToolScopeIsEmptyWhenSecureRegistryFiltersEveryRequestedTool() {
        #if DEBUG
        let sourceIDs = StructuredAgentKernelExecutor.structuredToolSourceIDsForTests(
            secureIDs: ["weather"],
            optionIDs: ["web.search"],
            routingIDs: ["web.search"]
        )

        XCTAssertTrue(sourceIDs.isEmpty)
        #endif
    }

    func testToolRequiredEmptySecureScopeStopsBeforeModelGeneration() async {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        defer { AgentBehaviorTraceRecorder.clear() }
        let service = ScriptedLlamaService(
            isChatLoaded: true,
            scripts: [[.text(#"{"final":"unsupported weather claim"}"#), .done]]
        )
        let kernel = AssistantKernel(
            router: AssistantRuntimeRouter(llamaService: service, allowDiagnosticFallbackSelection: false),
            toolRegistry: SecureToolRegistry(tools: [])
        )
        let request = structuredRequest(
            "What's the weather here?",
            allowedToolIDs: ["weather"],
            traceCorrelation: AgentTraceCorrelation(
                scenarioID: "secure-empty-weather",
                e2eRunID: UUID(),
                agentRunID: UUID(),
                conversationID: UUID(),
                turnID: UUID()
            )
        )

        var final = ""
        for await event in kernel.run(request) {
            if case .final(let text) = event { final = text }
        }

        XCTAssertTrue(final.localizedCaseInsensitiveContains("unavailable"))
        let preflightCallCount = await service.preflightCallCount
        let streamCallCount = await service.streamCallCount
        XCTAssertEqual(preflightCallCount, 0)
        XCTAssertEqual(streamCallCount, 0)
        let readinessCallCount = await service.readinessCallCount
        XCTAssertEqual(readinessCallCount, 0)
        XCTAssertFalse(AgentBehaviorTraceRecorder.recent(limit: 10).contains { $0.stage.hasPrefix("agent-json-step-") })
        #endif
    }

    func testInjectedRuntimeExecutesPreflightActionToolAndFinal() async throws {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        defer { AgentBehaviorTraceRecorder.clear() }
        let service = ScriptedLlamaService(
            isChatLoaded: true,
            scripts: [
                [.text(#"{"action":{"tool":"weather","args":{}}}"#), .done],
                [.text(#"{"final":"It is 21°C and clear."}"#), .done]
            ]
        )
        let kernel = AssistantKernel(
            router: AssistantRuntimeRouter(llamaService: service, allowDiagnosticFallbackSelection: false),
            toolRegistry: SecureToolRegistry(tools: [StubWeatherTool()])
        )
        let correlation = AgentTraceCorrelation(
            scenarioID: "scripted-weather",
            e2eRunID: UUID(),
            agentRunID: UUID(),
            conversationID: UUID(),
            turnID: UUID()
        )
        let request = structuredRequest("What's the weather?", allowedToolIDs: ["weather"], traceCorrelation: correlation)
        var sawAction = false
        var sawToolSuccess = false
        var final = ""
        for await event in kernel.run(request) {
            switch event {
            case .step(let step) where step.kind == .action && step.toolID == "weather":
                sawAction = true
            case .toolResult(let result) where result.status == .success:
                sawToolSuccess = true
            case .final(let text):
                final = text
            default:
                break
            }
        }

        XCTAssertTrue(sawAction)
        XCTAssertTrue(sawToolSuccess)
        XCTAssertEqual(final, "It is 21°C and clear.")
        let preflightCallCount = await service.preflightCallCount
        let streamCallCount = await service.streamCallCount
        XCTAssertEqual(preflightCallCount, 2)
        XCTAssertEqual(streamCallCount, 2)
        let readinessCallCount = await service.readinessCallCount
        let callOrder = await service.callOrder
        XCTAssertEqual(readinessCallCount, 2)
        XCTAssertEqual(callOrder, ["readiness", "preflight", "stream", "readiness", "preflight", "stream"])
        let traces = AgentBehaviorTraceRecorder.recent(limit: 10).filter { $0.scenarioID == correlation.scenarioID }
        let modelTraces = traces.filter { $0.event == .modelTurn }
        XCTAssertEqual(modelTraces.count, 2)
        XCTAssertTrue(modelTraces.allSatisfy { $0.runtimePath == "agent-model" })
        let actionTrace = modelTraces.first { $0.selectedToolID == "weather" }
        let finalTrace = modelTraces.first { $0.emittedFinalInActionTurn }
        XCTAssertEqual(actionTrace?.successfulObservationCount, 0)
        XCTAssertNil(actionTrace?.finalizerAccepted)
        XCTAssertEqual(finalTrace?.successfulObservationCount, 1)
        XCTAssertEqual(finalTrace?.finalizerAccepted, true)
        #endif
    }

    func testStructuredFinalTraceRecordsActualObservationFinalizerRejection() async {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        defer { AgentBehaviorTraceRecorder.clear() }
        let service = ScriptedLlamaService(
            isChatLoaded: true,
            scripts: [
                [.text(#"{"action":{"tool":"weather","args":{}}}"#), .done],
                [.text(#"{"final":"It is 21°C and clear."}"#), .done]
            ]
        )
        let kernel = AssistantKernel(
            router: AssistantRuntimeRouter(llamaService: service, allowDiagnosticFallbackSelection: false),
            toolRegistry: SecureToolRegistry(tools: [StubWeatherTool(observation: #"{"kind":"unsafe"}"#)])
        )
        let correlation = AgentTraceCorrelation(
            scenarioID: "scripted-unsafe-weather-observation",
            e2eRunID: UUID(),
            agentRunID: UUID(),
            conversationID: UUID(),
            turnID: UUID()
        )

        for await _ in kernel.run(structuredRequest(
            "What's the weather?",
            allowedToolIDs: ["weather"],
            traceCorrelation: correlation
        )) {}

        let finalTrace = AgentBehaviorTraceRecorder.recent(limit: 10).first {
            $0.scenarioID == correlation.scenarioID && $0.emittedFinalInActionTurn
        }
        XCTAssertEqual(finalTrace?.successfulObservationCount, 1)
        XCTAssertEqual(finalTrace?.finalizerAccepted, false)
        XCTAssertEqual(finalTrace?.finalizerRejectionReason, "unsafe-observation")
        #endif
    }

    func testStreamAcquisitionFailureRecordsNoStartedOrLoadedModel() async {
        #if DEBUG
        AgentBehaviorTraceRecorder.clear()
        defer { AgentBehaviorTraceRecorder.clear() }
        let service = ScriptedLlamaService(isChatLoaded: false, scripts: [])
        let kernel = AssistantKernel(
            router: AssistantRuntimeRouter(llamaService: service, allowDiagnosticFallbackSelection: false),
            toolRegistry: SecureToolRegistry(tools: [])
        )
        let correlation = AgentTraceCorrelation(
            scenarioID: "scripted-stream-acquisition-failure",
            e2eRunID: UUID(),
            agentRunID: UUID(),
            conversationID: UUID(),
            turnID: UUID()
        )
        let request = structuredRequest("Say hello.", allowedToolIDs: [], traceCorrelation: correlation)
        for await _ in kernel.run(request) {}

        let trace = AgentBehaviorTraceRecorder.recent(limit: 10).last { $0.scenarioID == correlation.scenarioID }
        XCTAssertEqual(trace?.streamStarted, false)
        XCTAssertEqual(trace?.modelLoaded, false)
        XCTAssertEqual(trace?.firstChunkReceived, false)
        XCTAssertEqual(trace?.textChunkCount, 0)
        XCTAssertTrue(trace?.emptyOutputReason?.hasPrefix("structured-runtime-unavailable:") == true)
        let preflightCallCount = await service.preflightCallCount
        let streamCallCount = await service.streamCallCount
        XCTAssertEqual(preflightCallCount, 1)
        XCTAssertEqual(streamCallCount, 0)
        let readinessCallCount = await service.readinessCallCount
        let callOrder = await service.callOrder
        XCTAssertEqual(readinessCallCount, 1)
        XCTAssertEqual(callOrder, ["readiness", "preflight"])
        #endif
    }

    func testColdInjectedRuntimeReadinessPrecedesPromptPreflight() async {
        #if DEBUG
        let service = ScriptedLlamaService(
            isChatLoaded: false,
            scripts: [[.text(#"{"final":"Hello."}"#), .done]],
            loadsChatOnReadiness: true
        )
        let kernel = AssistantKernel(
            router: AssistantRuntimeRouter(llamaService: service, allowDiagnosticFallbackSelection: false),
            toolRegistry: SecureToolRegistry(tools: [])
        )

        var final = ""
        for await event in kernel.run(structuredRequest("Say hello.", allowedToolIDs: [])) {
            if case .final(let text) = event { final = text }
        }

        XCTAssertEqual(final, "Hello.")
        let callOrder = await service.callOrder
        XCTAssertEqual(callOrder, ["readiness", "preflight", "stream"])
        #endif
    }

    func testPlausibleToolRequiredFinalRetriesActionBeforeAnswering() async {
        #if DEBUG
        let service = ScriptedLlamaService(
            isChatLoaded: true,
            scripts: [
                [.text(#"{"final":"It is sunny and 30°C."}"#), .done],
                [.text(#"{"action":{"tool":"weather","args":{}}}"#), .done],
                [.text(#"{"final":"It is 21°C and clear."}"#), .done]
            ]
        )
        let kernel = AssistantKernel(
            router: AssistantRuntimeRouter(llamaService: service, allowDiagnosticFallbackSelection: false),
            toolRegistry: SecureToolRegistry(tools: [StubWeatherTool()])
        )

        var sawToolSuccess = false
        var final = ""
        for await event in kernel.run(structuredRequest("What's the weather?", allowedToolIDs: ["weather"])) {
            if case .toolResult(let result) = event, result.status == .success {
                sawToolSuccess = true
            }
            if case .final(let text) = event { final = text }
        }

        XCTAssertTrue(sawToolSuccess)
        XCTAssertEqual(final, "It is 21°C and clear.")
        let readinessCallCount = await service.readinessCallCount
        let preflightCallCount = await service.preflightCallCount
        let streamCallCount = await service.streamCallCount
        XCTAssertEqual(readinessCallCount, 3)
        XCTAssertEqual(preflightCallCount, 3)
        XCTAssertEqual(streamCallCount, 3)
        #endif
    }

    func testNonSuccessToolResultStopsBeforeTrustedObservationReuse() {
        #if DEBUG
        XCTAssertTrue(StructuredAgentKernelExecutor.shouldStopAfterToolResultForTests(.failed))
        XCTAssertTrue(StructuredAgentKernelExecutor.shouldStopAfterToolResultForTests(.denied))
        XCTAssertTrue(StructuredAgentKernelExecutor.shouldStopAfterToolResultForTests(.unavailable))
        XCTAssertFalse(StructuredAgentKernelExecutor.shouldStopAfterToolResultForTests(.success))
        #endif
    }

    func testPhoneCallContinuationAfterContactSearchEmitsApprovalBoundary() throws {
        #if DEBUG
        let routing = IntentRoutingDecision(
            intent: .phoneCall,
            allowedToolIDs: ["contacts.search", "phone.call"],
            requiresClarification: false,
            clarificationPrompt: nil
        )
        let continuation = try XCTUnwrap(StructuredAgentKernelExecutor.phoneCallContinuationAfterContactObservationForTests(
            routing: routing,
            actionTool: "contacts.search",
            observation: "Sarah — +1 (555) 123-4567",
            availableToolIDs: ["contacts.search", "phone.call"]
        ))

        XCTAssertEqual(continuation.step.kind, .approvalBoundary)
        XCTAssertEqual(continuation.step.toolID, "phone.call")
        XCTAssertEqual(continuation.step.toolArgs?["number"], "+15551234567")
        XCTAssertTrue(continuation.text.contains("Approval required for phone.call"))
        #endif
    }

    func testRAGEmptyObservationPreservesEmptyStateInsteadOfSummary() throws {
        #if DEBUG
        let noMatches = try XCTUnwrap(StructuredAgentKernelExecutor.deterministicObservationFallbackForTests(
            observations: [("rag.search", "No matching snippets found in the local index.")],
            intent: .rag
        ))
        XCTAssertEqual(noMatches, "No matching snippets were found in the local index.")
        XCTAssertFalse(noMatches.contains("Summary\n"))

        let emptyIndex = try XCTUnwrap(StructuredAgentKernelExecutor.ragOrFilesEmptyObservationFinalForTests(
            observations: [("rag.search", "RAG storage unavailable: local index appears empty.")]
        ))
        XCTAssertTrue(emptyIndex.contains("RAG retrieval is unavailable"))
        XCTAssertFalse(emptyIndex.contains("Key modules"))

        let unavailable = try XCTUnwrap(StructuredAgentKernelExecutor.deterministicObservationFallbackForTests(
            observations: [("rag.search", "RAG storage unavailable.")],
            intent: .rag
        ))
        XCTAssertEqual(unavailable, "RAG retrieval is unavailable right now. RAG storage unavailable.")
        XCTAssertFalse(unavailable.contains("Summary\n"))
        XCTAssertFalse(unavailable.contains("Key modules"))
        #endif
    }

    func testSelectedActionIsSchemaValidatedBeforeExecution() throws {
        let web = try tool("web.search")
        let missingRequiredArg = AgentAction(tool: "web.search", args: [:])
        switch StructuredToolCallValidator.validate(action: missingRequiredArg, availableTools: [web]) {
        case .success:
            XCTFail("web.search without query must not validate")
        case .failure(let error):
            XCTAssertTrue(error.diagnostic.contains("missing_required_argument:web.search.query"))
        }

        let valid = AgentAction(tool: "web.search", args: ["query": .string("Swift concurrency")])
        switch StructuredToolCallValidator.validate(action: valid, availableTools: [web]) {
        case .success(let call):
            XCTAssertEqual(call.canonicalToolID, "web.search")
            XCTAssertEqual(call.arguments["query"], "Swift concurrency")
        case .failure(let error):
            XCTFail("Expected valid web.search action, got \(error.diagnostic)")
        }
    }

    func testApprovalToolsEmitBoundaryFinalWithoutExecutionCopy() {
        #if DEBUG
        let action = AgentAction(
            tool: "mail.draft",
            args: [
                "to": .string("sarah@example.com"),
                "body": .string("Hello")
            ]
        )
        let final = StructuredAgentKernelExecutor.approvalBoundaryFinalForTests(
            toolID: "mail.draft",
            action: action
        )

        XCTAssertTrue(final.contains("Approval required for mail.draft"))
        XCTAssertTrue(final.contains("I did not prepare or send the email yet."))
        #endif
    }

    func testTriggerCreateInvalidScheduleEnumRepairsBeforeApprovalBoundary() throws {
        #if DEBUG
        let prompt = "Schedule a trigger to summarize reminders tonight and confirm what will run."
        let routing = IntentRoutingDecision(
            intent: .trigger,
            allowedToolIDs: ["trigger.create", "trigger.list", "trigger.cancel"],
            requiresClarification: false,
            clarificationPrompt: nil
        )
        let badAction = AgentAction(tool: "trigger.create", args: [
            "title": .string("Reminder summary tonight"),
            "prompt": .string("Summarize reminders"),
            "schedule": .string("tonight")
        ])
        let error = StructuredToolCallValidationError.invalidEnumValue(
            tool: "trigger.create",
            argument: "schedule",
            allowed: ["absolute", "interval", "relative"]
        )

        let repair = try XCTUnwrap(StructuredAgentKernelExecutor.repairedTriggerCreateActionIfNeededForTests(
            modelAction: badAction,
            validationError: error,
            routing: routing,
            prompt: prompt,
            availableToolIDs: ["trigger.create", "trigger.list", "trigger.cancel"]
        ))
        XCTAssertEqual(repair.action.args["schedule"]?.stringValue, "relative")
        XCTAssertEqual(repair.action.args["inMinutes"]?.intValue, 120)
        XCTAssertEqual(repair.reflection.kind, .reflection)

        switch StructuredToolCallValidator.validate(action: repair.action, availableTools: ToolRegistry.all) {
        case .success(let call):
            XCTAssertEqual(call.canonicalToolID, "trigger.create")
            XCTAssertEqual(call.arguments["schedule"], "relative")
        case .failure(let error):
            XCTFail("Expected repaired trigger.create to validate, got \(error.diagnostic)")
        }

        let final = StructuredAgentKernelExecutor.approvalBoundaryFinalForTests(
            toolID: "trigger.create",
            action: repair.action
        )
        XCTAssertTrue(final.contains("Approval required for trigger.create"))
        XCTAssertTrue(final.contains("I did not schedule an agent run yet."))
        #endif
    }

    func testTriggerCreateInvalidDailyScheduleRepairsToAbsoluteTiming() throws {
        #if DEBUG
        let prompt = "Schedule a trigger to summarize reminders every day at 9am."
        let routing = IntentRoutingDecision(
            intent: .trigger,
            allowedToolIDs: ["trigger.create", "trigger.list", "trigger.cancel"],
            requiresClarification: false,
            clarificationPrompt: nil
        )
        let badAction = AgentAction(tool: "trigger.create", args: [
            "title": .string("Reminder summary"),
            "prompt": .string("Summarize reminders"),
            "schedule": .string("daily")
        ])
        let error = StructuredToolCallValidationError.invalidEnumValue(
            tool: "trigger.create",
            argument: "schedule",
            allowed: ["absolute", "interval", "relative"]
        )

        let repair = try XCTUnwrap(StructuredAgentKernelExecutor.repairedTriggerCreateActionIfNeededForTests(
            modelAction: badAction,
            validationError: error,
            routing: routing,
            prompt: prompt,
            availableToolIDs: ["trigger.create", "trigger.list", "trigger.cancel"]
        ))
        XCTAssertEqual(repair.action.args["schedule"]?.stringValue, "absolute")
        XCTAssertEqual(repair.action.args["atTime"]?.stringValue, "09:00")
        XCTAssertNil(repair.action.args["inMinutes"])
        #endif
    }

    func testTriggerCreateInvalidScheduleDoesNotInventRelativeFallbackWithoutPlan() {
        #if DEBUG
        let routing = IntentRoutingDecision(
            intent: .trigger,
            allowedToolIDs: ["trigger.create", "trigger.list"],
            requiresClarification: false,
            clarificationPrompt: nil
        )
        let badAction = AgentAction(tool: "trigger.create", args: [
            "title": .string("Reminder summary"),
            "prompt": .string("Summarize reminders"),
            "schedule": .string("yearly")
        ])
        let error = StructuredToolCallValidationError.invalidEnumValue(
            tool: "trigger.create",
            argument: "schedule",
            allowed: ["absolute", "interval", "relative"]
        )

        XCTAssertNil(StructuredAgentKernelExecutor.repairedTriggerCreateActionIfNeededForTests(
            modelAction: badAction,
            validationError: error,
            routing: routing,
            prompt: "Schedule a trigger sometime.",
            availableToolIDs: ["trigger.create", "trigger.list"]
        ))
        #endif
    }

    func testMemorySaveThenRecallInvariantRepairsPrematureRecall() throws {
        #if DEBUG
        let prompt = "Remember that I prefer concise bullet points, then tell me what you remembered."
        let plan = try XCTUnwrap(MemoryCommandPlan.saveThenRecall(from: prompt))
        let availableToolIDs: Set<String> = ["memory.save", "memory.recall"]
        let prematureRecall = AgentAction(tool: "memory.recall", args: ["query": .string("concise bullet points")])
        let saveRepair = StructuredAgentKernelExecutor.repairedMemoryActionIfNeededForTests(
            modelAction: prematureRecall,
            memoryPlan: plan,
            steps: [],
            availableToolIDs: availableToolIDs
        )

        XCTAssertEqual(saveRepair.action.tool, "memory.save")
        XCTAssertEqual(saveRepair.action.args["content"]?.stringValue, "I prefer concise bullet points")
        XCTAssertNotNil(saveRepair.reflection)

        let savedSteps = [
            AgentStep(
                kind: .action,
                content: "memory.save(content=I prefer concise bullet points, kind=fact)",
                toolID: "memory.save",
                toolArgs: ["content": "I prefer concise bullet points", "kind": "fact"]
            )
        ]
        let recallRepair = StructuredAgentKernelExecutor.repairedMemoryActionIfNeededForTests(
            modelAction: prematureRecall,
            memoryPlan: plan,
            steps: savedSteps,
            availableToolIDs: availableToolIDs
        )
        XCTAssertEqual(recallRepair.action.tool, "memory.recall")
        #endif
    }

    func testMalformedSecondMemoryTurnRepairsIntoRecall() throws {
        #if DEBUG
        let prompt = "Remember that I prefer concise bullet points, then tell me what you remembered."
        let plan = try XCTUnwrap(MemoryCommandPlan.saveThenRecall(from: prompt))
        let savedSteps = [
            AgentStep(
                kind: .action,
                content: "memory.save(content=I prefer concise bullet points, kind=fact)",
                toolID: "memory.save",
                toolArgs: ["content": "I prefer concise bullet points", "kind": "fact"]
            ),
            AgentStep(
                kind: .observation,
                content: "Memory saved.",
                toolID: "memory.save"
            )
        ]

        let required = StructuredAgentKernelExecutor.nextRequiredMemoryActionForTests(
            memoryPlan: plan,
            steps: savedSteps,
            availableToolIDs: ["memory.save", "memory.recall"]
        )

        XCTAssertEqual(required?.tool, "memory.recall")
        XCTAssertEqual(required?.args["query"]?.stringValue, "prefer concise bullet points")
        #endif
    }

    func testMapsSearchRepairsDuplicateLocationContinuation() throws {
        #if DEBUG
        let routing = IntentRoutingDecision(
            intent: .maps,
            allowedToolIDs: ["location.current", "maps.search"],
            requiresClarification: false,
            clarificationPrompt: nil
        )
        let steps = [
            AgentStep(kind: .action, content: "location.current", toolID: "location.current"),
            AgentStep(kind: .observation, content: "Position snapshot is disabled in this build.", toolID: "location.current")
        ]
        let duplicateLocation = AgentAction(tool: "location.current", args: [:])

        let repair = StructuredAgentKernelExecutor.repairedMapsSearchActionIfNeededForTests(
            modelAction: duplicateLocation,
            routing: routing,
            prompt: "Find coffee near me.",
            steps: steps,
            availableToolIDs: ["location.current", "maps.search"]
        )

        XCTAssertEqual(repair?.action.tool, "maps.search")
        XCTAssertEqual(repair?.action.args["query"]?.stringValue, "coffee")
        XCTAssertNotNil(repair?.reflection)
        #endif
    }

    func testMapsSearchContinuesAfterNonSuccessLocationResult() throws {
        #if DEBUG
        let routing = IntentRoutingDecision(
            intent: .maps,
            allowedToolIDs: ["location.current", "maps.search"],
            requiresClarification: false,
            clarificationPrompt: nil
        )
        let steps = [
            AgentStep(kind: .action, content: "location.current", toolID: "location.current"),
            AgentStep(kind: .observation, content: "Position snapshot is disabled in this build.", toolID: "location.current")
        ]

        XCTAssertTrue(StructuredAgentKernelExecutor.shouldContinueAfterNonSuccessToolResultForTests(
            .unavailable,
            routing: routing,
            actionTool: "location.current",
            prompt: "Find coffee near me.",
            steps: steps,
            availableToolIDs: ["location.current", "maps.search"]
        ))
        XCTAssertFalse(StructuredAgentKernelExecutor.shouldContinueAfterNonSuccessToolResultForTests(
            .unavailable,
            routing: routing,
            actionTool: "rag.search",
            prompt: "Find coffee near me.",
            steps: steps,
            availableToolIDs: ["location.current", "maps.search"]
        ))
        XCTAssertFalse(StructuredAgentKernelExecutor.shouldContinueAfterNonSuccessToolResultForTests(
            .unavailable,
            routing: routing,
            actionTool: "location.current",
            prompt: "Find coffee near me.",
            steps: steps,
            availableToolIDs: ["location.current", "maps.search"],
            stepIndex: 1,
            maxSteps: 2
        ))
        #endif
    }

    func testMemoryFinalRequiresSuccessfulSaveAndRecallObservations() throws {
        let request = structuredRequest(
            "Remember that I prefer concise bullet points, then tell me what you remembered.",
            allowedToolIDs: ["memory.save", "memory.recall"]
        )
        let steps = [
            AgentStep(kind: .action, content: "memory.save", toolID: "memory.save", toolArgs: ["content": "I prefer concise bullet points"]),
            AgentStep(kind: .observation, content: "Memory saved.", toolID: "memory.save"),
            AgentStep(kind: .action, content: "memory.recall", toolID: "memory.recall", toolArgs: ["query": "concise bullet points"]),
            AgentStep(kind: .observation, content: "Memory recall failed.", toolID: "memory.recall")
        ]

        let failedRecallFinal = StructuredAgentKernelExecutor.postprocessStructuredFinalAnswerForTests(
            "Recall failed.",
            request: request,
            availableTools: [try tool("memory.save"), try tool("memory.recall")],
            observations: [("memory.save", "Memory saved.")],
            steps: steps
        )
        XCTAssertEqual(failedRecallFinal, "Recall failed.")

        let successfulFinal = StructuredAgentKernelExecutor.postprocessStructuredFinalAnswerForTests(
            "fallback",
            request: request,
            availableTools: [try tool("memory.save"), try tool("memory.recall")],
            observations: [("memory.save", "Memory saved."), ("memory.recall", "I prefer concise bullet points")],
            steps: steps
        )
        XCTAssertEqual(successfulFinal, "I remember that you prefer concise bullet points.")
    }

    func testStructuredExecutorSourceContainsStrictEvidenceAndLoopGuards() throws {
        let source = try String(
            contentsOf: repoRoot().appendingPathComponent("ios/Lumen/Assistant/StructuredAgentKernelExecutor.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(source.contains("AgentBehaviorTraceEmitter.recordModelTurn"))
        XCTAssertTrue(source.contains("correlation: request.traceCorrelation"))
        XCTAssertTrue(source.contains("stage: \"agent-json-step-\\(stepIndex)\""))
        XCTAssertTrue(source.contains("runtimePath: \"agent-model\""))
        XCTAssertTrue(source.contains("selectedToolID: turn.action.map"))
        XCTAssertTrue(source.contains("allowedToolIDs: availableTools.map"))
        XCTAssertTrue(source.contains("parseError: turn.parseError?.rawValue"))
        XCTAssertTrue(source.contains("modelLoaded: diagnostics.modelLoaded"))
        XCTAssertTrue(source.contains("streamStarted: diagnostics.streamStarted"))
        XCTAssertTrue(source.contains("firstChunkReceived: diagnostics.firstChunkReceived"))
        XCTAssertTrue(source.contains("finalChunkReceived: diagnostics.finalChunkReceived"))
        XCTAssertTrue(source.contains("streamTerminationReason: diagnostics.streamTerminationReason"))
        XCTAssertTrue(source.contains("Duplicate tool call blocked"))
        XCTAssertTrue(source.contains("deterministic web synthesis fallback used after observations"))
        XCTAssertTrue(source.contains("prompt: AgentDiagnosticFileRedactor.summary(label: \"prompt\", text: request.userMessage)"))
        XCTAssertTrue(source.contains("toolArguments: AgentDiagnosticFileRedactor.redactedMap"))
        XCTAssertTrue(source.contains("Structured agent-json model turn completed."))

        let postprocessIndex = try XCTUnwrap(source.range(of: "postprocessStructuredFinalAnswer(finalAnswer")?.lowerBound)
        let finalDeltaIndex = try XCTUnwrap(source.range(of: "continuation.yield(.finalDelta(finalAnswer))", range: postprocessIndex..<source.endIndex)?.lowerBound)
        let finalIndex = try XCTUnwrap(source.range(of: "continuation.yield(.final(finalAnswer))", range: finalDeltaIndex..<source.endIndex)?.lowerBound)
        XCTAssertLessThan(postprocessIndex, finalDeltaIndex)
        XCTAssertLessThan(finalDeltaIndex, finalIndex)

        let validateIndex = try XCTUnwrap(source.range(of: "StructuredToolCallValidator.validate")?.lowerBound)
        let executeIndex = try XCTUnwrap(source.range(of: "toolRegistry.execute")?.lowerBound)
        XCTAssertLessThan(validateIndex, executeIndex)
    }

    private func structuredRequest(
        _ prompt: String,
        allowedToolIDs: [String],
        traceCorrelation: AgentTraceCorrelation? = nil
    ) -> AgentKernelRequest {
        AgentKernelRequest(
            userMessage: prompt,
            options: AgentKernelOptions(
                allowHeavyRuntime: true,
                allowDegradedMode: true,
                requireUserVisibleFinal: true,
                diagnosticsEnabled: true,
                maxSteps: 4,
                prefersFoundationModels: false,
                temperature: 0.05,
                topP: 0.6,
                maxTokens: 384,
                forceModelBackedToolPlanning: true,
                structuredMode: .requiredAgentJSON,
                structuredAllowedToolIDs: allowedToolIDs
            ),
            traceCorrelation: traceCorrelation
        )
    }

    private func tool(_ id: String) throws -> ToolDefinition {
        try XCTUnwrap(ToolRegistry.find(id: id))
    }

    private func repoRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
