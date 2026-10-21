import Foundation
import XCTest
@testable import Lumen

@MainActor
final class CoreAIGemmaRuntimeTests: XCTestCase {
    private enum Failure: Error, Equatable { case load, preflight, inference }

    private actor ScriptedRuntime {
        let outputs: [String]
        let failure: Failure?
        let promptTokens: Int
        private(set) var order: [String] = []
        private(set) var requests: [GenerateRequest] = []

        init(outputs: [String] = [], failure: Failure? = nil, promptTokens: Int = 200) {
            self.outputs = outputs
            self.failure = failure
            self.promptTokens = promptTokens
        }

        func prepare() throws {
            order.append("prepare")
            if failure == .load { throw Failure.load }
        }

        func preflight(_ request: GenerateRequest) throws -> LlamaStructuredPromptPreflightSnapshot {
            order.append("preflight")
            if failure == .preflight { throw Failure.preflight }
            return .init(contextSize: 32_768, finalPromptChars: request.systemPrompt.count + request.userMessage.count, estimatedPromptTokens: promptTokens)
        }

        func generate(_ request: GenerateRequest) throws -> String {
            order.append("generate")
            let index = requests.count
            requests.append(request)
            if failure == .inference { throw Failure.inference }
            guard outputs.indices.contains(index) else { throw Failure.inference }
            return outputs[index]
        }
    }

    private actor ToolCalls {
        private(set) var ids: [String] = []
        func record(_ id: String) { ids.append(id) }
    }

    private struct CountingTool: LocalTool {
        let definition: SecureToolDefinition
        let calls: ToolCalls

        init(id: String, calls: ToolCalls) {
            self.calls = calls
            definition = SecureToolDefinition(
                id: id,
                displayName: id,
                description: "Returns synthetic observations for kernel tests.",
                category: .readOnly,
                requiredPermissions: [],
                supportsBackgroundExecution: true,
                requiresUserApproval: false,
                argumentSchemaDescription: "{}",
                resultPrivacyLevel: .low,
                maxOutputCharacters: 2_000
            )
        }

        func validateArguments(_ arguments: [String: String]) throws {}

        func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
            await calls.record(definition.id)
            return ToolResult(
                invocationID: invocation.id,
                status: .success,
                displayText: "It is 21°C and clear.",
                modelText: "It is 21°C and clear.",
                structuredPayload: [:],
                privacyLevel: .low,
                metricsSummary: "synthetic_observation",
                errorCode: nil
            )
        }
    }

    func testSelectedGemmaTakesPriorityOverAvailableLlama() {
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { _ in "llama" }),
            coreAI: .init(isAvailable: true),
            allowDiagnosticFallbackSelection: false
        )
        XCTAssertEqual(router.runtime(for: foregroundContext()), .coreAI)
    }

    func testMissingPreferredGemmaFailsClosedEvenWithAvailableLlama() {
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { _ in "llama" }),
            coreAI: .init(isAvailable: false, isPreferred: true),
            allowDiagnosticFallbackSelection: false
        )
        let selection = router.selection(for: foregroundContext())
        XCTAssertEqual(selection.runtime, .unavailable)
        XCTAssertFalse(selection.reason.isEmpty)
    }

    func testDisabledGemmaLeavesExplicitLlamaRouteAvailable() {
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { _ in "llama" }),
            coreAI: .disabled,
            allowDiagnosticFallbackSelection: false
        )
        XCTAssertEqual(router.runtime(for: foregroundContext()), .llama)
    }

    func testBackgroundMaintenanceNeverSelectsGemma() {
        let router = AssistantRuntimeRouter(
            llama: .init(isAvailable: false),
            coreAI: .init(isAvailable: true),
            allowDiagnosticFallbackSelection: false
        )
        let context = AssistantTurnContext(task: .backgroundTrigger, input: "maintenance", isForeground: false, lowPowerMode: false, thermalState: .nominal)
        XCTAssertEqual(router.runtime(for: context), .unavailable)
    }

    func testThermalAndComputeDenialCannotSelectGemma() {
        let router = AssistantRuntimeRouter(coreAI: .init(isAvailable: true), allowDiagnosticFallbackSelection: false)
        let thermal = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .serious)
        let denied = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal, allowHeavyRuntime: false)
        XCTAssertEqual(router.runtime(for: thermal), .unavailable)
        XCTAssertEqual(router.runtime(for: denied), .unavailable)
    }

    func testGemmaDoesNotReplaceSemanticEmbeddingRuntime() {
        let router = AssistantRuntimeRouter(
            llama: .init(embedHandler: { _ in [Float(1)] }),
            coreAI: .init(isAvailable: true),
            allowDiagnosticFallbackSelection: false
        )
        let context = AssistantTurnContext(task: .embedding, input: "retrieval", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        XCTAssertEqual(router.runtime(for: context), .llama)
    }

    func testPlainTextKernelUsesCoreAIGenerationAdapter() async throws {
        let coreAI = CoreAIGemmaRuntimeAdapter(isAvailable: true, generateHandler: { request in
            "Core AI received: \(request.prompt)"
        })
        let kernel = AssistantKernel(router: AssistantRuntimeRouter(coreAI: coreAI, allowDiagnosticFallbackSelection: false))
        let output = try await kernel.runTextTurn(foregroundContext())
        XCTAssertEqual(output, "Core AI received: hello")
    }

    func testGemmaMessagesFoldSystemAndSchemaWithoutChangingAssistantHistoryIntoInstructions() {
        let request = GenerateRequest(
            systemPrompt: "Ask for approval before side effects.",
            history: [(.user, "Earlier question"), (.assistant, "Earlier answer"), (.tool, "Trusted observation")],
            userMessage: "Current question",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            modelName: "agent-json",
            relevantMemories: [],
            responseFormat: .constrainedJSON(schema: #"{"required":["final"]}"#)
        )
        let messages = CoreAIGemmaRuntimeAdapter.messages(for: request)
        XCTAssertEqual(messages.map(\.role), ["user", "assistant", "user"])
        XCTAssertTrue(messages.first?.content.contains("Ask for approval before side effects.") == true)
        XCTAssertTrue(messages.first?.content.contains(#"{"required":["final"]}"#) == true)
        XCTAssertTrue(messages.first?.content.contains("Earlier question") == true)
        XCTAssertEqual(messages[1].content, "Earlier answer")
        XCTAssertTrue(messages.last?.content.contains("Trusted observation") == true)
        XCTAssertTrue(messages.last?.content.contains("Current question") == true)
    }

    func testStructuredGemmaUsesExistingSecureToolLoopAfterPrepareAndTokenPreflight() async {
        let runtime = ScriptedRuntime(outputs: [
            #"{"action":{"tool":"weather","args":{}}}"#,
            #"{"final":"It is 21°C and clear."}"#
        ])
        let calls = ToolCalls()
        let kernel = makeKernel(runtime: runtime, tools: [CountingTool(id: "weather", calls: calls)])
        let events = await collect(kernel, request: structuredRequest("What's the weather?", tools: ["weather"]))
        let executed = await calls.ids
        let order = await runtime.order
        let requests = await runtime.requests
        XCTAssertEqual(executed, ["weather"])
        XCTAssertEqual(order, ["prepare", "preflight", "generate", "prepare", "preflight", "generate"])
        XCTAssertTrue(requests.allSatisfy {
            if case .constrainedJSON = $0.responseFormat { return true }
            return false
        })
        XCTAssertTrue(requests.last?.userMessage.contains("21°C") == true)
        XCTAssertTrue(events.contains { if case .final(let text) = $0 { return text.contains("21°C") }; return false })
    }

    func testPrepareAndTokenizerFailuresCannotRepairIntoMemorySideEffects() async {
        for failure in [Failure.load, .preflight] {
            let runtime = ScriptedRuntime(failure: failure)
            let calls = ToolCalls()
            let kernel = makeKernel(runtime: runtime, tools: [CountingTool(id: "memory.save", calls: calls), CountingTool(id: "memory.recall", calls: calls)])
            let events = await collect(kernel, request: structuredRequest("Remember that I prefer concise bullet points, then tell me what you remembered.", tools: ["memory.save", "memory.recall"]))
            let executed = await calls.ids
            let requests = await runtime.requests
            XCTAssertTrue(executed.isEmpty)
            XCTAssertTrue(requests.isEmpty)
            XCTAssertTrue(events.contains { if case .error = $0 { return true }; return false })
            XCTAssertFalse(events.contains { if case .toolInvocation = $0 { return true }; return false })
        }
    }

    func testCoreAIActionWithExtraArgumentsDoesNotReachNativeTool() async {
        let runtime = ScriptedRuntime(outputs: [#"{"action":{"tool":"weather","args":{"unexpected":true}}}"#])
        let calls = ToolCalls()
        let kernel = makeKernel(runtime: runtime, tools: [CountingTool(id: "weather", calls: calls)])
        let events = await collect(kernel, request: structuredRequest("What's the weather?", tools: ["weather"]))
        let executed = await calls.ids
        XCTAssertTrue(executed.isEmpty)
        XCTAssertFalse(events.contains { if case .toolInvocation = $0 { return true }; return false })
    }

    func testInferenceFailureCannotRepairIntoMemorySideEffects() async {
        let runtime = ScriptedRuntime(failure: .inference)
        let calls = ToolCalls()
        let kernel = makeKernel(runtime: runtime, tools: [CountingTool(id: "memory.save", calls: calls), CountingTool(id: "memory.recall", calls: calls)])
        let events = await collect(kernel, request: structuredRequest("Remember that I prefer concise bullet points, then tell me what you remembered.", tools: ["memory.save", "memory.recall"]))
        let executed = await calls.ids
        let order = await runtime.order
        XCTAssertTrue(executed.isEmpty)
        XCTAssertEqual(order, ["prepare", "preflight", "generate"])
        XCTAssertTrue(events.contains { if case .error = $0 { return true }; return false })
        XCTAssertFalse(events.contains { if case .toolInvocation = $0 { return true }; return false })
    }

    func testTokenizerCountOverCapacityPreventsGenerationAndTools() async {
        let runtime = ScriptedRuntime(promptTokens: 32_700)
        let calls = ToolCalls()
        let kernel = makeKernel(runtime: runtime, tools: [CountingTool(id: "weather", calls: calls)])
        let events = await collect(kernel, request: structuredRequest("What's the weather?", tools: ["weather"]))
        let order = await runtime.order
        let executed = await calls.ids
        XCTAssertEqual(order, ["prepare", "preflight", "preflight"])
        XCTAssertTrue(executed.isEmpty)
        XCTAssertTrue(events.contains { if case .error = $0 { return true }; return false })
    }

    func testAdapterCancellationDoesNotBecomeACompletedDecision() async {
        let coreAI = CoreAIGemmaRuntimeAdapter(isAvailable: true, structuredHandler: { _ in throw CancellationError() })
        do {
            _ = try await coreAI.streamStructured(generationRequest())
            XCTFail("A cancelled generation must throw before any completed decision stream exists.")
        } catch is CancellationError {
            // The kernel receives typed cancellation, not JSON or a done token.
        } catch {
            XCTFail("Unexpected cancellation category: \(type(of: error))")
        }
    }

    func testExplicitE2ECoreAIPreflightChecksTokenizerWithoutClaimingInference() async {
        let runtime = ScriptedRuntime()
        let adapter = CoreAIGemmaRuntimeAdapter(
            isAvailable: true,
            preflightHandler: { try await runtime.preflight($0) },
            prepareHandler: { try await runtime.prepare() }
        )
        let result = await E2ETestRunner.coreAIExecutorRuntimePreflight(adapter: adapter)
        let order = await runtime.order
        XCTAssertTrue(result.passed)
        XCTAssertEqual(result.runtimeKind, AssistantRuntimeKind.coreAI.rawValue)
        XCTAssertTrue(result.ensureReadySucceeded)
        XCTAssertFalse(result.smokeProbeSucceeded)
        XCTAssertNil(result.activeAdapterSlot)
        XCTAssertEqual(order, ["prepare", "preflight"])
    }

    func testE2ECoreAIPreflightFailsClosedOnTokenizerFailureOrFullContext() async {
        for runtime in [ScriptedRuntime(failure: .preflight), ScriptedRuntime(promptTokens: 32_700)] {
            let adapter = CoreAIGemmaRuntimeAdapter(
                isAvailable: true,
                preflightHandler: { try await runtime.preflight($0) },
                prepareHandler: { try await runtime.prepare() }
            )
            let result = await E2ETestRunner.coreAIExecutorRuntimePreflight(adapter: adapter)
            let requests = await runtime.requests
            XCTAssertFalse(result.passed)
            XCTAssertFalse(result.ensureReadySucceeded)
            XCTAssertFalse(result.smokeProbeSucceeded)
            XCTAssertNotNil(result.failureKind)
            XCTAssertTrue(requests.isEmpty)
        }
    }

    func testE2ERequiresActualCompletedCoreAIStructuredEvidence() async throws {
        #if DEBUG
        let scenarioID = "coreai-evidence-\(UUID().uuidString)"
        let runtime = ScriptedRuntime(outputs: [#"{"final":"Hello."}"#])
        let kernel = makeKernel(runtime: runtime, tools: [])
        _ = await collect(kernel, request: structuredRequest("Say hello.", tools: [], correlation: .init(scenarioID: scenarioID)))
        let trace = try XCTUnwrap(AgentBehaviorTraceRecorder.recent(limit: 20).last { $0.scenarioID == scenarioID })
        XCTAssertEqual(trace.runtimePath, AssistantRuntimeKind.coreAI.rawValue)
        XCTAssertEqual(trace.selectedRuntime, AssistantRuntimeKind.coreAI.rawValue)
        XCTAssertNil(trace.activeAdapterSlot)
        XCTAssertNil(trace.adapterSlot)
        XCTAssertTrue(E2ETestRunner.isValidModelBackedEvidenceTraceForTests(trace, requiresPrimaryAgentJSON: true))
        #endif
    }

    private func makeKernel(runtime: ScriptedRuntime, tools: [any LocalTool]) -> AssistantKernel {
        let coreAI = CoreAIGemmaRuntimeAdapter(
            isAvailable: true,
            structuredHandler: { try await runtime.generate($0) },
            preflightHandler: { try await runtime.preflight($0) },
            prepareHandler: { try await runtime.prepare() }
        )
        return AssistantKernel(
            router: AssistantRuntimeRouter(coreAI: coreAI, allowDiagnosticFallbackSelection: false),
            toolRegistry: SecureToolRegistry(tools: tools)
        )
    }

    private func collect(_ kernel: AssistantKernel, request: AgentKernelRequest) async -> [AgentKernelEvent] {
        var events: [AgentKernelEvent] = []
        for await event in kernel.run(request) { events.append(event) }
        return events
    }

    private func foregroundContext() -> AssistantTurnContext {
        AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal)
    }

    private func structuredRequest(_ prompt: String, tools: [String], correlation: AgentTraceCorrelation? = nil) -> AgentKernelRequest {
        AgentKernelRequest(userMessage: prompt, options: AgentKernelOptions(
            allowHeavyRuntime: true,
            allowDegradedMode: true,
            requireUserVisibleFinal: true,
            diagnosticsEnabled: true,
            maxSteps: 4,
            prefersFoundationModels: false,
            temperature: 0,
            maxTokens: 384,
            forceModelBackedToolPlanning: true,
            structuredMode: .requiredAgentJSON,
            structuredAllowedToolIDs: tools
        ), traceCorrelation: correlation)
    }

    private func generationRequest() -> GenerateRequest {
        GenerateRequest(systemPrompt: "Return JSON", history: [], userMessage: "hello", temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 128, modelName: "agent-json", relevantMemories: [])
    }
}
