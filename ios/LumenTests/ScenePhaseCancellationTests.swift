import XCTest
import SwiftUI
@testable import Lumen

@MainActor
final class ScenePhaseCancellationTests: XCTestCase {
    func testBackgroundScenePhaseCancelsRuntimeWork() {
        XCTAssertTrue(ResourceBudgetGate.shouldCancelForScenePhase(.background))
        RuntimeLifecycleCanceller.cancelForSceneTransition(reason: "test")
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .background))
    }
}
