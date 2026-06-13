import XCTest
@testable import Lumen

@MainActor
final class AssistantKernelLlamaRuntimeAdapterTests: XCTestCase {
    func testDefaultRouterSelectsLiveLlamaWhenHeavyRuntimeAllowed() {
        let router = AssistantRuntimeRouter()
        let context = AssistantTurnContext(
            task: .chat,
            input: "hello",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal
        )

        XCTAssertEqual(router.runtime(for: context), .llama)
    }

    func testRunTextTurnFallsBackWhenLiveLlamaHasNoLoadedModel() async throws {
        let kernel = AssistantKernel()
        let context = AssistantTurnContext(
            task: .chat,
            input: "hello",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal
        )

        let output = try await kernel.runTextTurn(context)

        XCTAssertEqual(output, "Lumen is running in limited local mode.")
    }

    func testLiveLlamaAdapterReportsAvailableToRouterButThrowsUntilModelLoaded() async {
        let adapter = LlamaRuntimeAdapter.live()
        XCTAssertTrue(adapter.isAvailable)

        do {
            _ = try await adapter.generate(request: TextGenerationRequest(prompt: "hello", systemPrompt: "", maxTokens: 8))
        } catch LocalRuntimeError.unavailable(let reason) {
            XCTAssertTrue(reason.contains("no loaded chat model") || reason.contains("no visible output"))
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
}
