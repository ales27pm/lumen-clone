import XCTest
import SwiftUI
@testable import Lumen

@MainActor
final class ScenePhaseCancellationTests: XCTestCase {
    override func tearDown() async throws {
        ResourceBudgetGate.testSnapshotOverride = nil
        ModelLoader.resetLoadTasksForTesting()
        try await super.tearDown()
    }

    func testBackgroundScenePhaseCancelsRuntimeWork() async {
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
        await fulfillment(of: [started], timeout: 1.0)

        XCTAssertTrue(ResourceBudgetGate.shouldCancelForScenePhase(.background))
        RuntimeLifecycleCanceller.cancelForSceneTransition(reason: "test")
        XCTAssertTrue(ModelLoader.hasActiveChatLoadTaskForTesting)

        let loaded = await ModelLoader.ensureChatLoaded(appState: appState, stored: [stored], intent: .userChat)
        XCTAssertFalse(loaded)
        XCTAssertFalse(ModelLoader.hasActiveChatLoadTaskForTesting)
    }
}
