import XCTest
@testable import Lumen

final class LumenIntentPolicyTests: XCTestCase {
    func testSensitiveActionRequiresOpenApp() {
        XCTAssertTrue(LumenIntentPolicy.requiresOpenAppForSensitiveAction("calendar.read"))
    }
}
