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

        for await event in kernel.run(request, modelContext: nil) {
            switch event {
            case .final(let text):
                finalText = text
            case .diagnostic(let diagnostic) where diagnostic.stage == "runtime-selection":
                selectedRuntime = diagnostic.metadata["runtime"]
            default:
                break
            }
        }

        XCTAssertEqual(finalText, "Lumen is running in limited local mode.")
        XCTAssertEqual(selectedRuntime, AssistantRuntimeKind.deterministicFallback.rawValue)
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
            prefersFoundationModels: true,
            temperature: 0.21,
            topP: 0.82,
            repetitionPenalty: 1.18,
            maxTokens: 384
        )
        let request = AgentKernelRequest(
            userMessage: "Use the attached notes and my preferences.",
            history: [
                AgentKernelMessage(messageRole: .user, content: "Earlier user turn"),
                AgentKernelMessage(messageRole: .assistant, content: "Earlier assistant answer")
            ],
            systemPrompt: "Answer in bullets.",
            relevantMemories: [memory],
            attachments: [attachment],
            task: .chat,
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
