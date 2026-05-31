import XCTest
import SwiftUI
@testable import Lumen

@MainActor
final class ModelLoaderPolicyTests: XCTestCase {
    override func tearDown() async throws {
        ResourceBudgetGate.testSnapshotOverride = nil
        ModelLoader.cancelActiveLoads()
        try await super.tearDown()
    }

    func testOnlyExplicitUserChatAndVoiceCanStartModelLoadWhenGateAllows() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0)
        XCTAssertTrue(ModelLoader.canStartModelLoad(intent: .userChat))
        XCTAssertTrue(ModelLoader.canStartModelLoad(intent: .userVoice))
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .appStartup))
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .diagnostics))
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .background))
    }

    func testUserChatAndVoiceDeniedWhenGateDenies() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0)
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .userChat))
        XCTAssertFalse(ModelLoader.canStartModelLoad(intent: .userVoice))
    }
}
