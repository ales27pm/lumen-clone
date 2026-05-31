import XCTest
@testable import Lumen

@MainActor
final class VoiceModeLifecycleTests: XCTestCase {
    func testBackgroundInterruptTransitionsState() {
        let c = VoiceSessionController()
        c.state = .listening
        c.handleAppDidEnterBackground()
        XCTAssertEqual(c.state, .interrupted)
    }
}
