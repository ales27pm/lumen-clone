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

    func testVoiceStateActivationPolicySkipsDuplicateState() {
        let decision = CarPlayVoiceStateActivationPolicy.decision(
            requestedStateID: "listening",
            previousStateID: "listening",
            lastActivationUptime: 100,
            nowUptime: 101
        )

        XCTAssertEqual(decision, .skipDuplicate)
    }

    func testVoiceStateActivationPolicyDelaysRapidDifferentState() {
        let decision = CarPlayVoiceStateActivationPolicy.decision(
            requestedStateID: "thinking",
            previousStateID: "listening",
            lastActivationUptime: 100,
            nowUptime: 100.1,
            minimumInterval: 0.35
        )

        if case .delay(let delay) = decision {
            XCTAssertEqual(delay, 0.25, accuracy: 0.001)
        } else {
            XCTFail("Expected delayed CarPlay voice state activation, got \(decision)")
        }
    }

    func testVoiceStateActivationPolicyAllowsStateAfterInterval() {
        let decision = CarPlayVoiceStateActivationPolicy.decision(
            requestedStateID: "speaking",
            previousStateID: "thinking",
            lastActivationUptime: 100,
            nowUptime: 100.5,
            minimumInterval: 0.35
        )

        XCTAssertEqual(decision, .activateNow)
    }

    func testVoiceStateActivationPolicyClampsBackwardClockMovement() {
        let decision = CarPlayVoiceStateActivationPolicy.decision(
            requestedStateID: "speaking",
            previousStateID: "thinking",
            lastActivationUptime: 100,
            nowUptime: 99,
            minimumInterval: 0.35
        )

        XCTAssertEqual(decision, .delay(0.35))
    }
}
