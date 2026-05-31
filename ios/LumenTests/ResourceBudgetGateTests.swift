import XCTest
import SwiftUI
@testable import Lumen

@MainActor
final class ResourceBudgetGateTests: XCTestCase {
    override func tearDown() async throws {
        ResourceBudgetGate.testSnapshotOverride = nil
        try await super.tearDown()
    }

    func testInactiveAndBackgroundCancel() {
        XCTAssertTrue(ResourceBudgetGate.shouldCancelForScenePhase(.inactive))
        XCTAssertTrue(ResourceBudgetGate.shouldCancelForScenePhase(.background))
        XCTAssertFalse(ResourceBudgetGate.shouldCancelForScenePhase(.active))
    }

    func testLowPowerDeniesNonExplicitHeavyWork() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: true, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.diagnostics.rawValue))
        XCTAssertTrue(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
    }

    func testSeriousAndCriticalThermalDenyHeavyWork() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .serious, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .critical, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userVoice.rawValue))
    }

    func testStaleMemoryWarningAllowsLaterExplicitUserWork() {
        let staleWarning = Date().addingTimeInterval(-(MemoryPressureMonitor.modelLoadSuppressionInterval + 1))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 1, lastMemoryWarningAt: staleWarning)
        XCTAssertTrue(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
    }

    func testUnknownAndMemoryWarningDenyHeavyWork() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: nil, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .unknown, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 1, lastMemoryWarningAt: Date())
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
    }
}
