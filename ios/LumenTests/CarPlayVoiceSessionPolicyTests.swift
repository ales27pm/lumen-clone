import XCTest
@testable import Lumen

final class CarPlayVoiceSessionPolicyTests: XCTestCase {
    func testAlertCombinesTitleAndMessageSoReasonIsVisible() {
        let title = CarPlayVoiceSessionPolicy.compactAlertTitle(
            title: "Lumen unavailable",
            message: "Open Lumen on iPhone to allow microphone and speech recognition."
        )

        XCTAssertTrue(title.contains("Lumen unavailable"))
        XCTAssertTrue(title.contains("microphone"))
    }

    func testSeriousAndCriticalThermalBlockCarPlayModelRuns() {
        XCTAssertFalse(CarPlayVoiceSessionPolicy.blocksModelRun(thermalState: .nominal))
        XCTAssertFalse(CarPlayVoiceSessionPolicy.blocksModelRun(thermalState: .fair))
        XCTAssertTrue(CarPlayVoiceSessionPolicy.blocksModelRun(thermalState: .serious))
        XCTAssertTrue(CarPlayVoiceSessionPolicy.blocksModelRun(thermalState: .critical))
    }

    func testAskOnlyAcceptedWhenIdle() {
        XCTAssertTrue(CarPlayVoiceSessionPolicy.acceptsAsk(in: .idle))
        XCTAssertFalse(CarPlayVoiceSessionPolicy.acceptsAsk(in: .requestingPermission))
        XCTAssertFalse(CarPlayVoiceSessionPolicy.acceptsAsk(in: .listening))
        XCTAssertFalse(CarPlayVoiceSessionPolicy.acceptsAsk(in: .thinking))
        XCTAssertFalse(CarPlayVoiceSessionPolicy.acceptsAsk(in: .speaking))
        XCTAssertFalse(CarPlayVoiceSessionPolicy.acceptsAsk(in: .unavailable))
    }

    func testSpokenAnswerIsSanitizedAndBoundedForCarPlay() {
        let raw = "<think>private</think>" + String(repeating: "A", count: 500)
        let spoken = CarPlayVoiceSessionPolicy.spokenAnswer(from: raw, maxCharacters: 40)

        XCTAssertFalse(spoken.contains("<think>"))
        XCTAssertLessThanOrEqual(spoken.count, 40)
    }
}
