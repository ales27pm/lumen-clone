import XCTest
@testable import Lumen

final class VoiceModePermissionDeniedTests: XCTestCase {
    func testDeniedStateValueExists() {
        let state: VoiceSessionState = .denied("x")
        if case .denied = state { XCTAssertTrue(true) } else { XCTFail() }
    }
}
