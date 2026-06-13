import XCTest
@testable import Lumen

@MainActor
final class AssistantKernelRunContractTests: XCTestCase {
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

    func testAgentKernelOptionsClampMaxSteps() {
        let options = AgentKernelOptions(
            allowHeavyRuntime: true,
            allowDegradedMode: true,
            requireUserVisibleFinal: true,
            diagnosticsEnabled: true,
            maxSteps: 0,
            prefersFoundationModels: true
        )

        XCTAssertEqual(options.maxSteps, 1)
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
