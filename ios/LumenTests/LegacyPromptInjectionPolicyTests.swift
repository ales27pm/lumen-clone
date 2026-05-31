import XCTest
@testable import Lumen

final class LegacyPromptInjectionPolicyTests: XCTestCase {
    func testProfiles() {
        XCTAssertTrue(LegacyPromptInjectionPolicy.headlessTrigger.backgroundSafeToolsOnly)
        XCTAssertFalse(LegacyPromptInjectionPolicy.foregroundChat.backgroundSafeToolsOnly)
    }
}
