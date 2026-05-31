import XCTest
@testable import Lumen

final class BackgroundTaskPolicyTests: XCTestCase {
    func testCriticalThermalDenied() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .triggerScan, lowPowerMode: false, thermalState: .critical, isForeground: false, backgroundAgentsEnabled: true, requiresNetwork: false, estimatedCost: 1))
        XCTAssertFalse(d.allow)
    }

    func testSeriousThermalDenied() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .triggerScan, lowPowerMode: false, thermalState: .serious, isForeground: false, backgroundAgentsEnabled: true, requiresNetwork: false, estimatedCost: 1))
        XCTAssertFalse(d.allow)
    }

    func testNetworkRequiredDeniedWhenNetworkCannotBeAllowed() {
        let d = BackgroundTaskPolicy.decide(.init(taskKind: .ragMaintenance, lowPowerMode: true, thermalState: .nominal, isForeground: false, backgroundAgentsEnabled: true, requiresNetwork: true, estimatedCost: 1))
        XCTAssertFalse(d.allow)
        XCTAssertFalse(d.allowNetwork)
    }
}
