import XCTest
@testable import Lumen

@MainActor
final class VoiceServiceCoexistenceTests: XCTestCase {
    func testCancelStopsControllerState() {
        let c = VoiceSessionController()
        c.state = .speaking
        c.cancel()
        XCTAssertEqual(c.state, .idle)
    }
}
