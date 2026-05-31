import XCTest
@testable import Lumen

final class AssistantPermissionStateTests: XCTestCase {
    func testStateRawValuesStable() { XCTAssertEqual(AssistantPermissionState.granted.rawValue, "granted") }
}
