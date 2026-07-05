import XCTest
@testable import Lumen

@MainActor
final class VoiceModeLifecycleTests: XCTestCase {
    override func tearDown() {
        VoiceService.shared.resetAudioStartupForTests()
        super.tearDown()
    }

    func testBackgroundInterruptTransitionsState() {
        let c = VoiceSessionController()
        c.state = .listening
        c.handleAppDidEnterBackground()
        XCTAssertEqual(c.state, .interrupted)
    }

    func testStartPushToTalkDoesNotEnterListeningWhenAudioStartupFails() async {
        let snapshot = VoiceInputReadinessSnapshot(
            isInputAvailable: false,
            availableInputCount: 0,
            currentRouteInputCount: 0,
            currentRouteOutputCount: 1,
            availableInputsSummary: "none",
            routeInputsSummary: "none",
            routeOutputsSummary: "CarAudio",
            sampleRate: nil,
            channelCount: nil,
            exceptionError: nil
        )
        VoiceService.shared.configureAudioStartupForTests(Self.fakeStartup(snapshot: snapshot))
        let controller = VoiceSessionController(
            recognition: SpeechRecognitionService(requestPermissionsHandler: { true })
        )

        await controller.startPushToTalk { _ in
            XCTFail("Startup failure must not produce a final transcript")
        }

        if case .failed(let reason) = controller.state {
            XCTAssertTrue(reason.contains("Voice input is unavailable"))
        } else {
            XCTFail("Expected .failed, got \(controller.state)")
        }
        XCTAssertFalse(VoiceService.shared.isListening)
    }

    private static func fakeStartup(snapshot: VoiceInputReadinessSnapshot) -> VoiceAudioStartup {
        VoiceAudioStartup(
            activateAudioSession: { .success },
            inputReadinessSnapshot: { _ in snapshot },
            installInputTap: { _, _ in .success },
            prepareAndStartEngine: { _ in .success },
            stopEngine: { _ in },
            removeInputTap: { _ in }
        )
    }
}
