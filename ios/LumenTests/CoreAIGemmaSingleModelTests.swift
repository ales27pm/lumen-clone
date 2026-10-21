import Foundation
import XCTest
@testable import Lumen

@MainActor
final class CoreAIGemmaSingleModelTests: XCTestCase {
    private actor Calls {
        private(set) var gemma: [String] = []
        private(set) var llama: [String] = []
        private(set) var structured: [GenerateRequest] = []
        private let replies: [String]

        init(replies: [String] = []) { self.replies = replies }
        func generate(_ request: TextGenerationRequest) -> String {
            gemma.append(request.prompt)
            return "Lumen response"
        }
        func legacy(_ request: TextGenerationRequest) -> String {
            llama.append(request.prompt)
            return "Legacy response"
        }
        func next(_ request: GenerateRequest) throws -> String {
            let index = structured.count
            structured.append(request)
            guard replies.indices.contains(index) else { throw CoreAIGemmaError.inferenceFailed }
            return replies[index]
        }
    }

    func testEveryForegroundGenerationTaskUsesTheSameGemmaAdapter() async throws {
        let calls = Calls()
        let kernel = AssistantKernel(router: AssistantRuntimeRouter(
            llama: .init(generateHandler: { await calls.legacy($0) }),
            coreAI: .init(isAvailable: true, generateHandler: { await calls.generate($0) }),
            allowDiagnosticFallbackSelection: false
        ))
        let tasks: [AssistantTaskKind] = [.chat, .agentPlan, .toolDecision, .summarization, .memoryExtraction, .speechCommandParsing]
        for task in tasks {
            let response = try await kernel.runTextTurn(AssistantTurnContext(
                task: task,
                input: "request-\(String(describing: task))",
                isForeground: true,
                lowPowerMode: false,
                thermalState: .nominal
            ))
            XCTAssertEqual(response, "Lumen response")
        }
        let gemma = await calls.gemma
        let llama = await calls.llama
        XCTAssertEqual(gemma.count, tasks.count)
        XCTAssertTrue(llama.isEmpty)
    }

    func testGemmaBackgroundTasksCannotSelectLegacyOrDiagnosticModels() {
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { _ in "must not run" }),
            coreAI: .init(isAvailable: true),
            allowDiagnosticFallbackSelection: true
        )
        for task in [AssistantTaskKind.backgroundTrigger, .remConsolidation, .chat, .summarization] {
            let selection = router.selection(for: AssistantTurnContext(
                task: task,
                input: "background",
                isForeground: false,
                lowPowerMode: false,
                thermalState: .nominal
            ))
            XCTAssertEqual(selection.runtime, .unavailable)
            XCTAssertNotEqual(selection.runtime, .llama)
            XCTAssertNotEqual(selection.runtime, .deterministicFallback)
        }
    }

    func testExplicitLlamaAlternativeRetainsItsForegroundGenerationPath() async throws {
        let calls = Calls()
        let kernel = AssistantKernel(router: AssistantRuntimeRouter(
            llama: .init(generateHandler: { await calls.legacy($0) }),
            coreAI: .disabled,
            allowDiagnosticFallbackSelection: false
        ))
        let response = try await kernel.runTextTurn(AssistantTurnContext(
            task: .summarization, input: "summarize", isForeground: true, lowPowerMode: false, thermalState: .nominal
        ))
        let llama = await calls.llama
        let gemma = await calls.gemma
        XCTAssertEqual(response, "Legacy response")
        XCTAssertEqual(llama, ["summarize"])
        XCTAssertTrue(gemma.isEmpty)
    }

    func testGemmaIsTheOnlySelectableGeneratorWhileEmbeddingsRemainIndependent() {
        let matrix = AssistantRuntimeCapabilityMatrix.current(
            llama: .init(generateHandler: { _ in "legacy" }, embedHandler: { _ in [Float(1)] }),
            coreAI: .init(isAvailable: true)
        )
        XCTAssertEqual(matrix.selectableGenerationRuntimes, [.coreAI])
        XCTAssertEqual(matrix.selectableEmbeddingRuntimes, [.llama])
        XCTAssertEqual(matrix.row(for: .llama)?.generationSelectable, false)
        XCTAssertEqual(matrix.row(for: .llama)?.embeddingSelectable, true)
        XCTAssertTrue(matrix.row(for: .llama)?.status.contains("Generation reserved for Gemma") == true)
        XCTAssertEqual(matrix.row(for: .deterministicFallback)?.generationSelectable, false)
    }

    func testDisablingToolsStillGeneratesThroughGemmaWithoutNativeActions() async {
        let calls = Calls()
        let kernel = AssistantKernel(router: AssistantRuntimeRouter(
            llama: .init(generateHandler: { await calls.legacy($0) }),
            coreAI: .init(isAvailable: true, generateHandler: { await calls.generate($0) })
        ))
        var finalText = ""
        var invokedTool = false
        let request = AgentKernelRequest(userMessage: "Remember that I prefer concise answers.", options: options(structured: false, allowsTools: false))
        for await event in kernel.run(request) {
            if case .final(let text) = event { finalText = text }
            if case .toolInvocation = event { invokedTool = true }
        }
        let gemma = await calls.gemma
        let llama = await calls.llama
        XCTAssertEqual(finalText, "Lumen response")
        XCTAssertEqual(gemma.count, 1)
        XCTAssertTrue(llama.isEmpty)
        XCTAssertFalse(invokedTool)
    }

    func testToolDisabledStructuredOutputCannotExecuteEvenWhenModelProposesAction() async {
        let calls = Calls(replies: [
            #"{"action":{"tool":"memory.save","args":{"content":"synthetic preference"}}}"#,
            #"{"final":"I can discuss that preference, but tools are disabled."}"#
        ])
        let kernel = scriptedKernel(calls)
        var invokedTool = false
        var finalText = ""
        for await event in kernel.run(AgentKernelRequest(
            userMessage: "Remember that I prefer concise answers.",
            options: options(structured: true, allowsTools: false)
        )) {
            if case .toolInvocation = event { invokedTool = true }
            if case .final(let text) = event { finalText = text }
        }
        let requests = await calls.structured
        XCTAssertFalse(invokedTool)
        XCTAssertEqual(requests.count, 2)
        XCTAssertTrue(finalText.contains("tools are disabled"))
        XCTAssertTrue(requests.allSatisfy { $0.systemPrompt.contains("No tools are available") })
    }

    func testLegacyRequestAdapterUsesOneUnifiedKernelTurnAndPreservesContext() async {
        let calls = Calls(replies: [#"{"final":"Unified answer."}"#])
        let kernel = scriptedKernel(calls)
        let conversationID = UUID()
        let legacy = AgentRequest(
            systemPrompt: "Use concise language.",
            history: [(.user, "Earlier request"), (.assistant, "Earlier answer")],
            userMessage: "Say hello.",
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1,
            maxTokens: 128,
            maxSteps: 2,
            availableTools: [],
            relevantMemories: [],
            conversationID: conversationID
        )
        var finalText = ""
        for await event in kernel.runAdaptedLegacyRequest(legacy, options: .default) {
            if case .done(let text, _) = event { finalText = text }
        }
        let requests = await calls.structured
        XCTAssertEqual(finalText, "Unified answer.")
        XCTAssertEqual(requests.count, 1)
        XCTAssertTrue(requests[0].systemPrompt.hasPrefix("You are Lumen, one unified local assistant"))
        XCTAssertFalse(requests[0].systemPrompt.contains("You are Executor"))
        XCTAssertTrue(requests[0].systemPrompt.contains("Use concise language."))
        XCTAssertTrue(requests[0].userMessage.contains("Earlier answer"))
        let adapted = AssistantKernel.adaptLegacyRequest(legacy, options: .default)
        XCTAssertEqual(adapted.conversationID, conversationID)
        XCTAssertFalse(adapted.options.allowsToolExecution)
        XCTAssertEqual(adapted.options.structuredMode, .requiredAgentJSON)
    }

    func testLegacyHeadlessAdaptationCannotStartAHeavyModel() {
        let legacy = AgentRequest(systemPrompt: "", history: [], userMessage: "Summarize", temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 128, maxSteps: 2, availableTools: [], relevantMemories: [])
        var legacyOptions = LegacyAgentRunOptions.default
        legacyOptions.groundingMode = .headlessTrigger
        let adapted = AssistantKernel.adaptLegacyRequest(legacy, options: legacyOptions)
        XCTAssertEqual(adapted.source, .trigger)
        XCTAssertEqual(adapted.task, .backgroundTrigger)
        XCTAssertFalse(adapted.options.allowHeavyRuntime)
    }

    func testHeadlessGemmaPromptContainsNoSeparateModelAssignments() {
        let prompt = HeadlessAgentKernelRunner.unifiedSystemPrompt(
            basePrompt: "Keep the answer concise.",
            style: MimicryProfile(state: "neutral", style: "calm-direct", constraints: ["Report completed actions only."])
        )
        XCTAssertTrue(prompt.contains("one unified local assistant"))
        XCTAssertTrue(prompt.contains("Keep the answer concise."))
        XCTAssertTrue(prompt.contains("Report completed actions only."))
        for legacy in ["Role contracts:", "Current v1 slot assignments:", "Missing slots:", "You are Executor", "Mimicry v1"] {
            XCTAssertFalse(prompt.contains(legacy))
        }
    }

    func testLoadedContextCapacityControlsGemmaPromptBudget() {
        let content = "BEGIN " + String(repeating: "retrieved context ", count: 2_000) + " END"
        let request = GenerateRequest(systemPrompt: "Answer from context.", history: [], userMessage: content, temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 128, modelName: "chat", relevantMemories: [])
        let short = CoreAIGemmaRuntimeAdapter.messages(for: request, contextSize: 8_192)
        let long = CoreAIGemmaRuntimeAdapter.messages(for: request, contextSize: 32_768)
        let shortCount = short.reduce(0) { $0 + $1.content.count }
        let longCount = long.reduce(0) { $0 + $1.content.count }
        XCTAssertGreaterThan(longCount, shortCount)
        XCTAssertTrue(long.first?.content.contains("one unified local assistant") == true)
        XCTAssertTrue(long.last?.content.hasSuffix(" END") == true)
        XCTAssertEqual(long.last?.role, "user")
    }

    func testToolPermissionOptionRoundTripsAndOlderRequestsKeepTheirContract() throws {
        let disabled = options(structured: false, allowsTools: false)
        let encoded = try JSONEncoder().encode(disabled)
        XCTAssertFalse(try JSONDecoder().decode(AgentKernelOptions.self, from: encoded).allowsToolExecution)
        var legacy = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        legacy.removeValue(forKey: "allowsToolExecution")
        let oldData = try JSONSerialization.data(withJSONObject: legacy)
        XCTAssertTrue(try JSONDecoder().decode(AgentKernelOptions.self, from: oldData).allowsToolExecution)
    }

    private func scriptedKernel(_ calls: Calls) -> AssistantKernel {
        let coreAI = CoreAIGemmaRuntimeAdapter(
            isAvailable: true,
            structuredHandler: { try await calls.next($0) },
            preflightHandler: { _ in .init(contextSize: 32_768, finalPromptChars: 2_000, estimatedPromptTokens: 500) },
            prepareHandler: { }
        )
        return AssistantKernel(router: AssistantRuntimeRouter(coreAI: coreAI, allowDiagnosticFallbackSelection: false))
    }

    private func options(structured: Bool, allowsTools: Bool) -> AgentKernelOptions {
        AgentKernelOptions(
            allowHeavyRuntime: true,
            allowDegradedMode: true,
            requireUserVisibleFinal: true,
            diagnosticsEnabled: false,
            maxSteps: 2,
            prefersFoundationModels: false,
            temperature: 0,
            maxTokens: 128,
            forceModelBackedToolPlanning: structured,
            structuredMode: structured ? .requiredAgentJSON : .automatic,
            allowsToolExecution: allowsTools
        )
    }
}
