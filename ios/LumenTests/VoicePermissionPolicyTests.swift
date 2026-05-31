import XCTest
@testable import Lumen

final class VoicePermissionPolicyTests: XCTestCase {
    func testInterruptionPolicyStopsForegroundSession() {
        XCTAssertTrue(VoiceInterruptionHandler.shouldInterruptOnBackground())
    }
}
