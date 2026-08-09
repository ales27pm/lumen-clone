import XCTest
@testable import Lumen

@MainActor
final class AssistantKernelRunContractTests: XCTestCase {
    private actor RequestCapture {
        private var request: TextGenerationRequest?

        func record(_ request: TextGenerationRequest) {
            self.request = request
        }

        func snapshot() -> TextGenerationRequest? {
            request
        }
    }

    func testKernelRunEmitsFinalAndDoneEvents() async {
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { request in
                "kernel final: \(request.prompt)"
            })
        )
        let kernel = AssistantKernel(router: router)
        let request = AgentKernelRequest(userMessage: "hello", task: .chat, source: .chat, options: .chat)

        var finalText: String?
        var doneText: String?
        var sawRuntimeDiagnostic = false

        for await event in kernel.run(request, modelContext: nil) {
            switch event {
            case .final(let text):
                finalText = text
            case .done(let text, _):
                doneText = text
            case .diagnostic(let diagnostic) where diagnostic.stage == "runtime-selection":
                sawRuntimeDiagnostic = true
            default:
                break
            }
        }

        XCTAssertTrue(finalText?.contains("hello") == true)
        XCTAssertTrue(doneText?.contains("hello") == true)
        XCTAssertTrue(sawRuntimeDiagnostic)
    }

    func testKernelRunHonorsHeavyRuntimeOption() async {
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { _ in
                "llama should not run"
            })
        )
        let kernel = AssistantKernel(router: router)
        let options = AgentKernelOptions(
            allowHeavyRuntime: false,
            allowDegradedMode: true,
            requireUserVisibleFinal: true,
            diagnosticsEnabled: true,
            maxSteps: 2,
            prefersFoundationModels: true
        )
        let request = AgentKernelRequest(userMessage: "hello", task: .chat, source: .chat, options: options)

        var finalText: String?
        var selectedRuntime: String?
        var runtimeMetadata: [String: String] = [:]

        for await event in kernel.run(request, modelContext: nil) {
            switch event {
            case .final(let text):
                finalText = text
            case .diagnostic(let diagnostic) where diagnostic.stage == "runtime-selection":
                selectedRuntime = diagnostic.metadata["runtime"]
                runtimeMetadata = diagnostic.metadata
            default:
                break
            }
        }

        XCTAssertEqual(finalText, "Diagnostic deterministic runtime response.")
        XCTAssertEqual(selectedRuntime, AssistantRuntimeKind.deterministicFallback.rawValue)
        XCTAssertEqual(runtimeMetadata["budgetPolicy"], LumenSlotBudgetPolicy.foregroundInteractive.rawValue)
        XCTAssertEqual(runtimeMetadata["budgetDenialReason"], "foregroundInteractive: heavyRuntime=false")
        XCTAssertEqual(runtimeMetadata["selectionReason"], "foregroundInteractive: heavyRuntime=false")
    }

    func testForceModelBackedToolPlanningBypassesNativeToolShortcut() async {
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { request in
                "model planned: \(request.prompt)"
            })
        )
        let kernel = AssistantKernel(router: router)
        let options = AgentKernelOptions(
            allowHeavyRuntime: true,
            allowDegradedMode: false,
            requireUserVisibleFinal: true,
            diagnosticsEnabled: true,
            maxSteps: 2,
            prefersFoundationModels: false,
            forceModelBackedToolPlanning: true
        )
        let request = AgentKernelRequest(
            userMessage: "What is the weather here?",
            task: .chat,
            source: .diagnostics,
            options: options
        )

        var finalText = ""
        var sawNativeToolRouting = false
        for await event in kernel.run(request, modelContext: nil) {
            switch event {
            case .final(let text):
                finalText = text
            case .diagnostic(let diagnostic) where diagnostic.stage == "native-tool-routing":
                sawNativeToolRouting = true
            default:
                break
            }
        }

        XCTAssertEqual(finalText, "model planned: What is the weather here?")
        XCTAssertFalse(sawNativeToolRouting)
    }

    func testKernelRunPreservesGroundingContextAndSamplingOptions() async {
        let capture = RequestCapture()
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { request in
                await capture.record(request)
                return "grounded"
            })
        )
        let kernel = AssistantKernel(router: router)
        let memory = MemoryContextItem(
            content: "User prefers concise answers.",
            scope: .userPreference,
            authority: .preferenceOnly,
            createdAt: nil,
            expiresAt: nil,
            source: "test",
            topic: "style"
        )
        let attachment = ChatAttachment(
            name: "notes.txt",
            kind: .text,
            path: "/tmp/notes.txt",
            byteSize: 42
        )
        let options = AgentKernelOptions(
            allowHeavyRuntime: true,
            allowDegradedMode: true,
            requireUserVisibleFinal: true,
            diagnosticsEnabled: false,
            maxSteps: 3,
            prefersFoundationModels: false,
            temperature: 0.21,
            topP: 0.82,
            repetitionPenalty: 1.18,
            maxTokens: 384
        )
        let request = AgentKernelRequest(
            userMessage: "Draft a concise paragraph about alpine weather patterns.",
            history: [
                AgentKernelMessage(messageRole: .user, content: "Earlier user turn"),
                AgentKernelMessage(messageRole: .assistant, content: "Earlier assistant answer")
            ],
            systemPrompt: "Answer in bullets.",
            relevantMemories: [memory],
            attachments: [attachment],
            task: .summarization,
            source: .chat,
            options: options
        )

        for await _ in kernel.run(request, modelContext: nil) {}

        guard let generated = await capture.snapshot() else {
            XCTFail("Expected kernel to call the text generation runtime")
            return
        }
        XCTAssertEqual(generated.prompt, request.userMessage)
        XCTAssertEqual(generated.systemPrompt, "Answer in bullets.")
        XCTAssertEqual(generated.history.count, 2)
        XCTAssertEqual(generated.history.first?.role, .user)
        XCTAssertEqual(generated.history.last?.content, "Earlier assistant answer")
        XCTAssertEqual(generated.relevantMemories, [memory])
        XCTAssertEqual(generated.attachments, [attachment])
        XCTAssertEqual(generated.temperature, 0.21)
        XCTAssertEqual(generated.topP, 0.82)
        XCTAssertEqual(generated.repetitionPenalty, 1.18)
        XCTAssertLessThanOrEqual(generated.maxTokens, 384)
    }

    func testKernelRunSuppressesDiagnosticsWhenDisabled() async {
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { _ in
                "done"
            })
        )
        let kernel = AssistantKernel(router: router)
        let options = AgentKernelOptions(
            allowHeavyRuntime: true,
            allowDegradedMode: true,
            requireUserVisibleFinal: true,
            diagnosticsEnabled: false,
            maxSteps: 2,
            prefersFoundationModels: true
        )
        let request = AgentKernelRequest(userMessage: "hello", task: .chat, source: .chat, options: options)

        var sawDiagnostic = false
        for await event in kernel.run(request, modelContext: nil) {
            if case .diagnostic = event {
                sawDiagnostic = true
            }
        }

        XCTAssertFalse(sawDiagnostic)
    }

    func testBackgroundTriggerToolIntentUsesNativeBackgroundSafeExecution() async {
        let capture = RequestCapture()
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { request in
                await capture.record(request)
                return "background trigger should use native tool execution"
            })
        )
        let kernel = AssistantKernel(router: router)
        let options = AgentKernelOptions(
            allowHeavyRuntime: false,
            allowDegradedMode: true,
            requireUserVisibleFinal: true,
            diagnosticsEnabled: true,
            maxSteps: 2,
            prefersFoundationModels: false,
            maxTokens: 128
        )
        let request = AgentKernelRequest(
            userMessage: "what do you remember about my workshop preferences",
            task: .backgroundTrigger,
            source: .trigger,
            options: options
        )

        var routingMetadata: [String: String]?
        var doneSteps: [AgentStep] = []
        var invocation: ToolInvocation?
        var result: ToolResult?
        for await event in kernel.run(request, modelContext: nil) {
            switch event {
            case .diagnostic(let diagnostic) where diagnostic.stage == "native-tool-routing":
                routingMetadata = diagnostic.metadata
            case .toolInvocation(let emitted):
                invocation = emitted
            case .toolResult(let emitted):
                result = emitted
            case .done(_, let steps):
                doneSteps = steps
            default:
                break
            }
        }

        XCTAssertEqual(routingMetadata?["mode"], "background-safe")
        XCTAssertTrue(routingMetadata?["availableToolIDs"]?.contains("memory.recall") == true)
        XCTAssertEqual(invocation?.toolID, "memory.recall")
        XCTAssertEqual(invocation?.source, .backgroundTrigger)
        XCTAssertEqual(result?.status, .unavailable)
        XCTAssertTrue(doneSteps.contains { $0.kind == .action && $0.toolID == "memory.recall" })
        let generatedRequest = await capture.snapshot()
        XCTAssertNil(generatedRequest)
    }

    func testBackgroundTriggerUnavailableToolIntentSkipsBeforeTextRuntime() async {
        let capture = RequestCapture()
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { request in
                await capture.record(request)
                return "background trigger should not use text runtime"
            })
        )
        let kernel = AssistantKernel(router: router)
        let options = AgentKernelOptions(
            allowHeavyRuntime: true,
            allowDegradedMode: true,
            requireUserVisibleFinal: true,
            diagnosticsEnabled: true,
            maxSteps: 2,
            prefersFoundationModels: false,
            maxTokens: 128
        )
        let request = AgentKernelRequest(
            userMessage: "search the web for the latest iOS background task docs",
            task: .backgroundTrigger,
            source: .trigger,
            options: options
        )

        var skipDiagnostic: [String: String]?
        var errorText: String?
        var sawFinal = false
        var sawDone = false
        var sawObservation = false
        for await event in kernel.run(request, modelContext: nil) {
            switch event {
            case .diagnostic(let diagnostic) where diagnostic.stage == "background-tool-execution":
                skipDiagnostic = diagnostic.metadata
            case .error(let text):
                errorText = text
            case .final:
                sawFinal = true
            case .done:
                sawDone = true
            case .step(let step) where step.kind == .observation:
                sawObservation = true
            default:
                break
            }
        }

        XCTAssertEqual(skipDiagnostic?["status"], BackgroundToolExecutionAssessment.Status.noBackgroundSafeRoutedTools.rawValue)
        XCTAssertTrue(errorText?.contains("no routed tool is allowed to run in background") == true)
        XCTAssertFalse(sawFinal)
        XCTAssertFalse(sawDone)
        XCTAssertFalse(sawObservation)
        let generatedRequest = await capture.snapshot()
        XCTAssertNil(generatedRequest)
    }

    func testForegroundToolBridgeResolvesRecentContactPronounBeforePlanning() async {
        let capture = RequestCapture()
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { request in
                await capture.record(request)
                return "text runtime should not run"
            })
        )
        let kernel = AssistantKernel(router: router)
        let request = AgentKernelRequest(
            userMessage: "Call him",
            history: [
                AgentKernelMessage(messageRole: .assistant, content: "Contact search results:\n• Alexis Boulet — 14504943059\n• Alexis Boulet — no phone")
            ],
            task: .chat,
            source: .chat,
            options: AgentKernelOptions(
                allowHeavyRuntime: true,
                allowDegradedMode: true,
                requireUserVisibleFinal: true,
                diagnosticsEnabled: true,
                maxSteps: 2,
                prefersFoundationModels: true
            )
        )

        var doneSteps: [AgentStep] = []
        var routingMetadata: [String: String]?
        var invocation: ToolInvocation?
        for await event in kernel.run(request, modelContext: nil) {
            switch event {
            case .diagnostic(let diagnostic) where diagnostic.stage == "native-tool-routing":
                routingMetadata = diagnostic.metadata
            case .toolInvocation(let emitted):
                invocation = emitted
            case .done(_, let steps):
                doneSteps = steps
            default:
                break
            }
        }

        let contactAction = doneSteps.first { $0.kind == .action && $0.toolID == "contacts.search" }
        XCTAssertEqual(routingMetadata?["referenceRewrite"], "true")
        XCTAssertEqual(contactAction?.toolArgs?["query"], "Alexis Boulet")
        XCTAssertEqual(invocation?.toolID, "contacts.search")
        XCTAssertEqual(invocation?.source, .modelProposed)
        let generatedRequest = await capture.snapshot()
        XCTAssertNil(generatedRequest)
    }

    func testPolicyFirstNativeToolTurnEmitsCorrelatedActionAndFinalTraces() async {
        AgentBehaviorTraceRecorder.clear()
        let e2eRunID = UUID()
        let agentRunID = UUID()
        let conversationID = UUID()
        let turnID = UUID()
        let correlation = AgentTraceCorrelation(
            scenarioID: "live-weather-direct",
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID
        )
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { _ in
                "text runtime should not run"
            })
        )
        let kernel = AssistantKernel(router: router)
        let request = AgentKernelRequest(
            conversationID: conversationID,
            turnID: turnID,
            userMessage: "What is the weather here?",
            task: .chat,
            source: .diagnostics,
            options: AgentKernelOptions(
                allowHeavyRuntime: true,
                allowDegradedMode: true,
                requireUserVisibleFinal: true,
                diagnosticsEnabled: true,
                maxSteps: 2,
                prefersFoundationModels: false
            ),
            traceCorrelation: correlation
        )

        for await _ in kernel.run(request, modelContext: nil) {}

        let traces = AgentBehaviorTraceRecorder.recent(limit: 10)
            .filter { $0.scenarioID == "live-weather-direct" }
        let actionTrace = traces.first { $0.event == .toolAction }
        let finalTrace = traces.last { $0.event == .finalAnswer }
        XCTAssertEqual(actionTrace?.runtimePath, "deterministic-compatibility")
        XCTAssertEqual(actionTrace?.selectedToolID, "weather")
        XCTAssertEqual(actionTrace?.e2eRunID, e2eRunID)
        XCTAssertEqual(actionTrace?.agentRunID, agentRunID)
        XCTAssertEqual(actionTrace?.conversationID, conversationID)
        XCTAssertEqual(actionTrace?.turnID, turnID)
        XCTAssertEqual(actionTrace?.modelLoaded, false)
        XCTAssertEqual(finalTrace?.runtimePath, "deterministic-compatibility")
        XCTAssertEqual(finalTrace?.event, .finalAnswer)
        XCTAssertEqual(finalTrace?.e2eRunID, e2eRunID)
        XCTAssertFalse(finalTrace?.rawOutputPrefix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
        AgentBehaviorTraceRecorder.clear()
    }

    func testNativeToolTurnApprovalRequiredActionsStopBeforeExecution() async {
        let cases: [(prompt: String, expectedToolID: String)] = [
            ("Text 5551234567 that I am late.", "messages.draft"),
            ("Email alex@example.com with subject Update and body Hello.", "mail.draft"),
            ("Set an appointment called Dentist in 30 minutes in my calendar.", "calendar.create"),
            ("Schedule a trigger to summarize reminders in 10 minutes.", "trigger.create"),
            ("Call 5551234567.", "phone.call"),
            ("Schedule an alarm called Test in 10 minutes.", "alarm.schedule"),
            ("Reindex my imported files.", "rag.index_files"),
            ("Reindex photo metadata for the last 3 months.", "rag.index_photos"),
            ("Mark Outlook message about q3 as read.", "outlook.message.mark_read")
        ]

        for testCase in cases {
            let capture = RequestCapture()
            let router = AssistantRuntimeRouter(
                llama: .init(generateHandler: { request in
                    await capture.record(request)
                    return "text runtime should not run"
                })
            )
            let kernel = AssistantKernel(router: router)
            let request = AgentKernelRequest(
                userMessage: testCase.prompt,
                task: .chat,
                source: .chat,
                options: AgentKernelOptions(
                    allowHeavyRuntime: true,
                    allowDegradedMode: true,
                    requireUserVisibleFinal: true,
                    diagnosticsEnabled: true,
                    maxSteps: 3,
                    prefersFoundationModels: false
                )
            )

            var finalText: String?
            var doneSteps: [AgentStep] = []
            var invocations: [ToolInvocation] = []
            var results: [ToolResult] = []
            for await event in kernel.run(request, modelContext: nil) {
                switch event {
                case .final(let text):
                    finalText = text
                case .toolInvocation(let invocation):
                    invocations.append(invocation)
                case .toolResult(let result):
                    results.append(result)
                case .done(_, let steps):
                    doneSteps = steps
                default:
                    break
                }
            }

            let approvalStep = doneSteps.first { $0.kind == .approvalBoundary && $0.toolID == testCase.expectedToolID }
            XCTAssertNotNil(approvalStep, "Expected approval boundary for \(testCase.expectedToolID) from prompt: \(testCase.prompt)")
            let declaredArguments = ToolRegistry.find(id: testCase.expectedToolID)?.capabilityContract.arguments ?? []
            if declaredArguments.isEmpty {
                XCTAssertEqual(approvalStep?.toolArgs, [:])
            } else {
                XCTAssertEqual(approvalStep?.toolArgs?.isEmpty, false)
            }
            XCTAssertTrue(finalText?.contains("Approval required for \(testCase.expectedToolID)") == true)
            XCTAssertFalse(doneSteps.contains { $0.kind == .action && $0.toolID == testCase.expectedToolID })
            XCTAssertTrue(invocations.isEmpty, "Approval boundary must stop before any tool invocation for \(testCase.expectedToolID)")
            XCTAssertTrue(results.isEmpty, "Approval boundary must stop before any tool result for \(testCase.expectedToolID)")
            let generatedRequest = await capture.snapshot()
            XCTAssertNil(generatedRequest)
        }
    }

    func testAgentKernelOptionsClampMaxSteps() {
        let options = AgentKernelOptions(
            allowHeavyRuntime: true,
            allowDegradedMode: true,
            requireUserVisibleFinal: true,
            diagnosticsEnabled: true,
            maxSteps: 0,
            prefersFoundationModels: true,
            temperature: -1,
            topP: 2,
            repetitionPenalty: 0
        )

        XCTAssertEqual(options.maxSteps, 1)
        XCTAssertEqual(options.temperature, 0)
        XCTAssertEqual(options.topP, 1)
        XCTAssertEqual(options.repetitionPenalty, 0.1)
    }

    func testKernelRunMapsToLegacyEventsDuringMigration() async {
        let kernelEvent = AgentKernelEvent.finalDelta("partial")
        guard case .finalDelta(let text) = kernelEvent.legacyAgentEvent else {
            XCTFail("Expected finalDelta legacy event")
            return
        }
        XCTAssertEqual(text, "partial")
        XCTAssertNil(AgentKernelEvent.final("partial").legacyAgentEvent)
    }
}
