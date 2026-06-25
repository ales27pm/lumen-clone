import XCTest
@testable import Lumen

final class ComputePolicyTests: XCTestCase {
    func testBackgroundConstrained() {
        let decision = ComputePolicy.decide(for: .init(task: .chat, isForeground: false, lowPowerMode: false, thermalState: .nominal))
        XCTAssertEqual(decision.maxTokens, 256)
        XCTAssertFalse(decision.allowHeavyRuntime)
    }

    func testForegroundLowPowerReducesTokenBudgetWithoutDenyingHeavyRuntime() {
        let decision = ComputePolicy.decide(for: .init(task: .chat, isForeground: true, lowPowerMode: true, thermalState: .nominal))
        XCTAssertEqual(decision.maxTokens, 512)
        XCTAssertTrue(decision.allowHeavyRuntime)
        XCTAssertEqual(decision.budgetPolicy, .foregroundInteractive)
        XCTAssertNil(decision.denialReason)
    }

    func testExplicitHeavyRuntimeDisallowCarriesDenialReason() {
        let decision = ComputePolicy.decide(for: .init(task: .chat, isForeground: true, lowPowerMode: false, thermalState: .nominal, allowHeavyRuntime: false))
        XCTAssertEqual(decision.maxTokens, 512)
        XCTAssertFalse(decision.allowHeavyRuntime)
        XCTAssertEqual(decision.budgetPolicy, .foregroundInteractive)
        XCTAssertEqual(decision.denialReason, "foregroundInteractive: heavyRuntime=false")
    }

    func testBackgroundRemUsesMaintenanceBudgetWhenUnconstrained() {
        let decision = ComputePolicy.decide(for: .init(task: .remConsolidation, isForeground: false, lowPowerMode: false, thermalState: .nominal))
        XCTAssertEqual(decision.maxTokens, 512)
        XCTAssertTrue(decision.allowHeavyRuntime)
        XCTAssertEqual(decision.budgetPolicy, .maintenanceIdle)
    }

    func testBackgroundRemDeniesLowPowerMaintenanceWork() {
        let decision = ComputePolicy.decide(for: .init(task: .remConsolidation, isForeground: false, lowPowerMode: true, thermalState: .nominal))
        XCTAssertFalse(decision.allowHeavyRuntime)
        XCTAssertEqual(decision.denialReason, "maintenanceIdle: lowPowerMode=true")
    }
}
