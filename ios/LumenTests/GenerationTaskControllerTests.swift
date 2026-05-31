import XCTest
@testable import Lumen

@MainActor
final class GenerationTaskControllerTests: XCTestCase {
    func testStartupGuardRunsOnlyOncePerKey() {
        let controller = GenerationTaskController<String>()
        var count = 0

        controller.startupIfNeeded(for: "chat") { count += 1 }
        controller.startupIfNeeded(for: "chat") { count += 1 }

        XCTAssertEqual(count, 1)
    }

    func testBeginCancelsPreviousAndPreventsStaleRequest() async {
        let controller = GenerationTaskController<String>()
        let firstTask = Task<Void, Never> { await Task.yield() }
        let first = controller.begin(for: "chat", task: firstTask)
        let secondTask = Task<Void, Never> { await Task.yield() }
        let second = controller.begin(for: "chat", task: secondTask)

        XCTAssertFalse(controller.isCurrent(first, for: "chat"))
        XCTAssertTrue(controller.isCurrent(second, for: "chat"))
        XCTAssertTrue(firstTask.isCancelled)
    }

    func testCancelClearsActiveRequest() {
        let controller = GenerationTaskController<String>()
        let task = Task<Void, Never> { await Task.yield() }
        let requestID = controller.begin(for: "voice", task: task)

        controller.cancel(for: "voice")

        XCTAssertFalse(controller.isCurrent(requestID, for: "voice"))
        XCTAssertTrue(task.isCancelled)
    }

    func testSingleActiveGenerationPerConversationInvariant() {
        let controller = GenerationTaskController<String>()
        let firstTask = Task<Void, Never> { await Task.yield() }
        _ = controller.begin(for: "conversation-1", task: firstTask)
        XCTAssertTrue(controller.hasActiveGeneration(for: "conversation-1"))

        let secondTask = Task<Void, Never> { await Task.yield() }
        _ = controller.begin(for: "conversation-1", task: secondTask)
        XCTAssertTrue(firstTask.isCancelled)
        XCTAssertTrue(controller.hasActiveGeneration(for: "conversation-1"))
        controller.assertSingleActiveGeneration(for: "conversation-1")
    }

    func testNoDetachedStateWriteAudit() async {
        let controller = GenerationTaskController<String>()
        await Task(priority: .utility) {
            let task = Task<Void, Never> { await Task.yield() }
            _ = await MainActor.run {
                controller.begin(for: "audit", task: task)
            }
        }.value

        XCTAssertTrue(controller.hasActiveGeneration(for: "audit"))
        controller.assertSingleActiveGeneration(for: "audit")
    }
}
