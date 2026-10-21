import XCTest
@testable import Lumen

final class LlamaGenerationCancellationTests: XCTestCase {
    func testZeroBudgetStreamFinishesAndReleasesRuntimeHandoff() async {
        let service = AppLlamaService.shared
        let request = GenerateRequest(
            systemPrompt: "Synthetic test", history: [], userMessage: "Hello",
            temperature: 0, topP: 1, repetitionPenalty: 1,
            maxTokens: 0, modelName: "synthetic", relevantMemories: []
        )
        let stream = await service.stream(request, slot: .mouth)
        let finished = expectation(description: "zero-budget stream drained")
        let consumer = Task {
            var sawDone = false
            for await event in stream {
                if case .done = event { sawDone = true }
            }
            XCTAssertTrue(sawDone)
            finished.fulfill()
        }
        defer { consumer.cancel() }
        await fulfillment(of: [finished], timeout: 3)
        let active = await service.hasActiveGenerationForTesting(requestID: request.id)
        XCTAssertFalse(active)
    }

    func testCancellationTokenPreservesReasonAndThrowsCancellationError() {
        let token = LlamaGenerationCancellationToken()

        XCTAssertFalse(token.isCancelled)
        token.cancel(reason: "scene-transition-test")

        XCTAssertTrue(token.isCancelled)
        XCTAssertEqual(token.reason, "scene-transition-test")
        XCTAssertThrowsError(try token.checkCancellation()) { error in
            XCTAssertTrue(error is CancellationError)
        }
    }
}
