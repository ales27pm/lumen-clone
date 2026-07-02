import XCTest
@testable import Lumen

final class PermissionsCenterTests: XCTestCase {
    func testAlarmRequestRequiresRuntimeUsageDescription() {
        XCTAssertFalse(PermissionsCenter.alarmUsageDescriptionPresent(infoDictionary: [:]))
        XCTAssertFalse(PermissionsCenter.alarmUsageDescriptionPresent(infoDictionary: ["NSAlarmKitUsageDescription": "   "]))
        XCTAssertTrue(PermissionsCenter.alarmUsageDescriptionPresent(infoDictionary: ["NSAlarmKitUsageDescription": "Alarm scheduling"]))
    }
}
