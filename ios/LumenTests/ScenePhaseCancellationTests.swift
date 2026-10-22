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

    func testCancelActiveLoadsImmediatelyCancelsRegisteredCoreAILoadsWithoutLegacyPendingTasks() async {
        let gemmaGate = PendingModelLoadGate()
        let embeddingGate = PendingModelLoadGate()
        let gemma = Task { await gemmaGate.wait() }
        let embeddings = Task { await embeddingGate.wait() }
        // CoreAI registers its child task directly; it does not populate the
        // legacy ModelLoader pending-task fields.
        AppCancellationBus.shared.registerCancellation({ gemma.cancel() }, category: .modelLoad)
        AppCancellationBus.shared.registerCancellation({ embeddings.cancel() }, category: .modelLoad)
        XCTAssertFalse(ModelLoader.hasActiveChatLoadTaskForTesting)
        XCTAssertFalse(ModelLoader.hasActiveEmbeddingLoadTaskForTesting)
        XCTAssertFalse(gemma.isCancelled)
        XCTAssertFalse(embeddings.isCancelled)

        ModelLoader.cancelActiveLoads()

        // This assertion must hold before yielding to any queued actor method.
        XCTAssertTrue(gemma.isCancelled)
        XCTAssertTrue(embeddings.isCancelled)
        XCTAssertEqual(AppCancellationBus.shared.activeRegistrationCount(category: .modelLoad), 0)
        await gemmaGate.release()
        await embeddingGate.release()
        await gemma.value
        await embeddings.value
    }

    func testCoalescedMemoryWarningImmediatelyCancelsNewLoadAndMaintenanceRegistrations() async {
        let monitor = MemoryPressureMonitor(notificationCenter: NotificationCenter())
        let firstWarningAt = Date()
        monitor.recordWarningForTesting(count: 1, at: firstWarningAt)
        let loadGate = PendingModelLoadGate()
        let maintenanceGate = PendingModelLoadGate()
        let load = Task { await loadGate.wait() }
        let maintenance = Task { await maintenanceGate.wait() }
        AppCancellationBus.shared.registerCancellation({ load.cancel() }, category: .modelLoad)
        AppCancellationBus.shared.registerCancellation({ maintenance.cancel() }, category: .modelMaintenance)

        // The coalesced path returns before scheduling any runtime unload. New
        // registrations must still observe this second warning immediately.
        await monitor.handleWarning(now: firstWarningAt.addingTimeInterval(0.1))

        XCTAssertTrue(load.isCancelled)
        XCTAssertTrue(maintenance.isCancelled)
        XCTAssertEqual(monitor.warningCount, 1)
        XCTAssertEqual(AppCancellationBus.shared.activeRegistrationCount(category: .modelLoad), 0)
        XCTAssertEqual(AppCancellationBus.shared.activeRegistrationCount(category: .modelMaintenance), 0)
        await loadGate.release()
        await maintenanceGate.release()
        await load.value
        await maintenance.value
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

        let loaded = await ModelLoader.ensureChatLoaded(appState: appState, stored: [stored], intent: .userChat, coreAISelected: false)
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
