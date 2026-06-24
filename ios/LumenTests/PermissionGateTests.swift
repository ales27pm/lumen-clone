import XCTest
@testable import Lumen

final class PermissionGateTests: XCTestCase {
    func testBackgroundNotDeterminedDenied() {
        let d = PermissionGate.evaluate(domain: .camera, state: .notDetermined, isForeground: false)
        XCTAssertFalse(d.allowed)
    }

    func testRouteGuardDoesNotRequestPermissionFromBackground() {
        let decision = ToolRouteGuard.permissionGateDecision(
            for: .location,
            state: .notDetermined,
            isForeground: false
        )

        if case .denied(let message) = decision {
            XCTAssertTrue(message.contains("location access"))
        } else {
            XCTFail("Background permission gate should deny instead of requesting permission")
        }
    }

    func testRouteGuardAllowsForegroundPermissionRequest() {
        let decision = ToolRouteGuard.permissionGateDecision(
            for: .location,
            state: .notDetermined,
            isForeground: true
        )

        XCTAssertEqual(decision, .request)
    }
}
