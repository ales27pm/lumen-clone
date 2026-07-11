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

    func testSelectedActionIsSchemaValidatedBeforeExecution() throws {
        let web = try tool("web.search")
        let missingRequiredArg = AgentAction(tool: "web.search", args: [:])
        switch StructuredToolCallValidator.validate(action: missingRequiredArg, availableTools: [web]) {
        case .success:
            XCTFail("web.search without query must not validate")
        case .failure(let error):
            XCTAssertTrue(error.diagnostic.contains("missing_required_argument:web.search:query"))
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
