import XCTest
@testable import Lumen

@MainActor
final class VoiceModeControllerIntegrationTests: XCTestCase {
    func testControllerStartsOnlyByExplicitCall() {
        let c = VoiceSessionController()
        XCTAssertEqual(c.state, .idle)
    }
}
