import XCTest
@testable import Lumen

final class DeviceCapabilityProfilerTests: XCTestCase {
    func testThermalMapping() {
        XCTAssertEqual(DeviceThermalState.from(processThermalState: .nominal), .nominal)
        XCTAssertEqual(DeviceThermalState.from(processThermalState: .fair), .fair)
        XCTAssertEqual(DeviceThermalState.from(processThermalState: .serious), .serious)
        XCTAssertEqual(DeviceThermalState.from(processThermalState: .critical), .critical)
    }
}
