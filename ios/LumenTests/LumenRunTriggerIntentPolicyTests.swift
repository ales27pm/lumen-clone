import XCTest
@testable import Lumen

final class LumenRunTriggerIntentPolicyTests: XCTestCase {
    func testSensitivePromptRequiresOpenApp() {
        XCTAssertTrue(LumenIntentPolicy.requiresOpenAppForSensitiveAction("calendar sync"))
    }
}
