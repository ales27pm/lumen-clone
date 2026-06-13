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

        XCTAssertEqual(finalText, "kernel final: hello")
        XCTAssertEqual(doneText, "kernel final: hello")
        XCTAssertTrue(sawRuntimeDiagnostic)
    }

    func testKernelRunMapsToLegacyEventsDuringMigration() async {
        let kernelEvent = AgentKernelEvent.finalDelta("partial")
        guard case .finalDelta(let text) = kernelEvent.legacyAgentEvent else {
            XCTFail("Expected finalDelta legacy event")
            return
        }
        XCTAssertEqual(text, "partial")
    }
}
