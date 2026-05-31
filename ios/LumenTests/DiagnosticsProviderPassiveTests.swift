import XCTest
import SwiftUI
@testable import Lumen

@MainActor
final class DiagnosticsProviderPassiveTests: XCTestCase {
    override func tearDown() async throws {
        ResourceBudgetGate.testSnapshotOverride = nil
        try await super.tearDown()
    }

    func testDiagnosticsIntentCannotTriggerModelLoad() async {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .diagnostics))
        _ = await DiagnosticsProvider().collect()
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .diagnostics))
    }
}
