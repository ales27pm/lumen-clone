import XCTest
@testable import Lumen

final class PermissionGateTests: XCTestCase {
    func testBackgroundNotDeterminedDenied() {
        let d = PermissionGate.evaluate(domain: .camera, state: .notDetermined, isForeground: false)
        XCTAssertFalse(d.allowed)
    }
}
