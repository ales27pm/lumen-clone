import Foundation
import XCTest
@testable import Lumen

@MainActor
final class AssistantKernelStructuredAgentTests: XCTestCase {
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
        #endif
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
