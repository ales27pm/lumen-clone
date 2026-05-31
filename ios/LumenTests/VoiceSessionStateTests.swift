import XCTest
@testable import Lumen

final class VoiceSessionStateTests: XCTestCase {
    func testIdleDefault() {
        let s: VoiceSessionState = .idle
        XCTAssertEqual(s, .idle)
    }
}
