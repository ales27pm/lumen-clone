import XCTest
@testable import Lumen

@MainActor
final class VoiceModeSceneTransitionTests: XCTestCase {
    func testBackgroundSceneStopsVoiceRecognitionState() {
        let controller = VoiceSessionController()
        controller.state = .listening
        controller.handleAppDidEnterBackground()
        XCTAssertEqual(controller.state, .interrupted)
    }
}
