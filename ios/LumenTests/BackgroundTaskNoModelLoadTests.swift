import XCTest
import SwiftUI
@testable import Lumen

@MainActor
final class BackgroundTaskNoModelLoadTests: XCTestCase {
    override func tearDown() async throws {
        ResourceBudgetGate.testSnapshotOverride = nil
        try await super.tearDown()
    }

    func testBackgroundTaskCannotStartModelLoad() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0)
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .background))
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.background.rawValue))
    }
}
