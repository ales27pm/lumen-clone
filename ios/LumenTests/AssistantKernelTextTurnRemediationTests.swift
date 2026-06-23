import XCTest
@testable import Lumen

@MainActor
final class AssistantKernelTextTurnRemediationTests: XCTestCase {
    func testRunTextTurnRejectsEmbedding() async {
        let kernel = AssistantKernel()
        let context = AssistantTurnContext(task: .embedding, input: "x", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        do {
            _ = try await kernel.runTextTurn(context)
            XCTFail("Embedding should not run through text turn")
        } catch AssistantKernel.KernelError.unsupportedTaskForTextTurn(let task) {
            XCTAssertEqual(task, .embedding)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testRunTextTurnRejectsSafetyClassification() async {
        let kernel = AssistantKernel()
        let context = AssistantTurnContext(task: .safetyClassification, input: "x", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        do {
            _ = try await kernel.runTextTurn(context)
            XCTFail("Safety classification should not run through text turn")
        } catch AssistantKernel.KernelError.unsupportedTaskForTextTurn(let task) {
            XCTAssertEqual(task, .safetyClassification)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testRunTextTurnDoesNotSelectUnwiredLlamaMarkedAvailable() async throws {
        let router = AssistantRuntimeRouter(llama: .init(isAvailable: true, unavailableReason: nil))
        let kernel = AssistantKernel(router: router)
        let context = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal)

        let output = try await kernel.runTextTurn(context)

        XCTAssertEqual(output, "Lumen is running in limited local mode.")
        XCTAssertEqual(kernel.selectRuntime(for: context), .deterministicFallback)
    }

    func testRunTextTurnUsesInjectedLlamaGenerationAdapter() async throws {
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { request in
                "wired: \(request.prompt)"
            })
        )
        let kernel = AssistantKernel(router: router)
        let context = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal)

        let output = try await kernel.runTextTurn(context)

        XCTAssertEqual(output, "wired: hello")
        XCTAssertEqual(kernel.selectRuntime(for: context), .llama)
    }

}
