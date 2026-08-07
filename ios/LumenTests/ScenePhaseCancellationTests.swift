import XCTest
import SwiftUI
@testable import Lumen

private actor PendingModelLoadGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var isReleased = false
    private(set) var isWaiting = false

    func wait() async {
        guard !isReleased else { return }
        isWaiting = true
        await withCheckedContinuation { continuation in
            self.continuation = continuation
        }
        isWaiting = false
    }

    func release() {
        isReleased = true
        continuation?.resume()
        continuation = nil
    }
}

@MainActor
final class ScenePhaseCancellationTests: XCTestCase {
    override func setUp() async throws {
        try await super.setUp()
        SceneTransitionCoordinator.shared.resetForTesting()
        ModelLoader.resetLoadTasksForTesting()
        AppCancellationBus.shared.resetForTesting()
    }

    override func tearDown() async throws {
        ResourceBudgetGate.testSnapshotOverride = nil
        ModelLoader.resetLoadTasksForTesting()
        SceneTransitionCoordinator.shared.resetForTesting()
        AppCancellationBus.shared.resetForTesting()
        try await super.tearDown()
    }

    func testBackgroundScenePhaseDoesNotCancelRuntimeWork() async {
        XCTAssertFalse(ResourceBudgetGate.shouldCancelForScenePhase(.background))
    }

    func testBackgroundScenePhaseCancelsPendingChatAndEmbeddingLoads() async {
        let chatCancelled = expectation(description: "chat load observed cancellation")
        let embeddingCancelled = expectation(description: "embedding load observed cancellation")
        let chatTask = cancellableLoadTask(cancelled: chatCancelled)
        let embeddingTask = cancellableLoadTask(cancelled: embeddingCancelled)
        ModelLoader.installChatLoadTaskForTesting(chatTask)
        ModelLoader.installEmbeddingLoadTaskForTesting(embeddingTask)

        XCTAssertTrue(ModelLoader.hasActiveChatLoadTaskForTesting)
        XCTAssertTrue(ModelLoader.hasActiveEmbeddingLoadTaskForTesting)

        SceneTransitionCoordinator.shared.handleScenePhaseChange(.background)

        await fulfillment(of: [chatCancelled, embeddingCancelled], timeout: 3.0)
        await waitForPendingLoadsToDrain()
        XCTAssertFalse(ModelLoader.hasActiveChatLoadTaskForTesting)
        XCTAssertFalse(ModelLoader.hasActiveEmbeddingLoadTaskForTesting)
        XCTAssertNil(AppCancellationBus.shared.lastCancellationReason)
    }

    func testCancellationRetainsPendingOwnershipUntilUnderlyingLoadFinishes() async {
        let gate = PendingModelLoadGate()
        let task = Task {
            await gate.wait()
            return false
        }
        ModelLoader.installChatLoadTaskForTesting(task)
        for _ in 0..<100 {
            if await gate.isWaiting { break }
            await Task.yield()
        }
        let reachedWait = await gate.isWaiting
        XCTAssertTrue(reachedWait)

        ModelLoader.cancelActiveLoads()

        XCTAssertTrue(ModelLoader.hasActiveChatLoadTaskForTesting)
        await gate.release()
        await waitForPendingLoadsToDrain()
        XCTAssertFalse(ModelLoader.hasActiveChatLoadTaskForTesting)
    }

    func testExplicitLifecycleCancellationStillCancelsRuntimeWork() async {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .active,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )

        let appState = AppState()
        let stored = StoredModel(
            name: "Cancellation Test Chat",
            repoId: "local/test",
            fileName: "cancellation-test.gguf",
            sizeBytes: 1,
            quantization: "test",
            parameters: "test",
            role: .chat,
            localPath: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).path
        )
        appState.activeChatModelID = stored.id.uuidString
        let started = expectation(description: "test load task started")
        let task = Task { @MainActor in
            started.fulfill()
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 10_000_000)
            }
            return false
        }
        ModelLoader.installChatLoadTaskForTesting(task)
        await fulfillment(of: [started], timeout: 3.0)

        RuntimeLifecycleCanceller.cancelForSceneTransition(reason: "test")

        let loaded = await ModelLoader.ensureChatLoaded(appState: appState, stored: [stored], intent: .userChat)
        XCTAssertFalse(loaded)
        XCTAssertFalse(ModelLoader.hasActiveChatLoadTaskForTesting)
    }

    private func cancellableLoadTask(cancelled: XCTestExpectation) -> Task<Bool, Never> {
        Task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 10_000_000)
            }
            cancelled.fulfill()
            return false
        }
    }

    private func waitForPendingLoadsToDrain() async {
        for _ in 0..<100 {
            if !ModelLoader.hasActiveChatLoadTaskForTesting,
               !ModelLoader.hasActiveEmbeddingLoadTaskForTesting {
                return
            }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
    }
}
