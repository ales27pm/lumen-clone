import XCTest
@testable import Lumen

@MainActor
final class VoiceModeControllerIntegrationTests: XCTestCase {
    func testControllerStartsOnlyByExplicitCall() {
        let c = VoiceSessionController()
        XCTAssertEqual(c.state, .idle)
    }

    func testStreamingChunkerWaitsForNaturalBoundaryDuringStreaming() {
        let chunk = VoiceStreamingChunker.nextChunk(
            in: "This is still mid phrase",
            startingAt: 0,
            finishedStreaming: false,
            minimumStreamingCharacters: 12
        )

        XCTAssertNil(chunk)
    }

    func testStreamingChunkerUsesSoftBoundaryAfterEnoughText() {
        let chunk = VoiceStreamingChunker.nextChunk(
            in: "Here is the first useful clause, with more text still arriving",
            startingAt: 0,
            finishedStreaming: false,
            minimumStreamingCharacters: 24
        )

        XCTAssertEqual(chunk?.text, "Here is the first useful clause,")
        XCTAssertEqual(chunk?.nextOffset, 32)
    }

    func testStreamingChunkerFlushesRemainderWhenStreamCompletes() {
        let text = "First sentence. Final phrase"
        let first = VoiceStreamingChunker.nextChunk(in: text, startingAt: 0, finishedStreaming: false)
        let final = VoiceStreamingChunker.nextChunk(
            in: text,
            startingAt: first?.nextOffset ?? 0,
            finishedStreaming: true
        )

        XCTAssertEqual(first?.text, "First sentence.")
        XCTAssertEqual(final?.text, "Final phrase")
    }

    func testStreamingChunkerAdvancesPastWhitespaceOnlyRemainder() {
        let chunk = VoiceStreamingChunker.nextChunk(
            in: "   ",
            startingAt: 0,
            finishedStreaming: true
        )

        XCTAssertEqual(chunk?.text, "")
        XCTAssertEqual(chunk?.nextOffset, 3)
    }

    func testTurnCompletionRejectsStaleSpeechTurn() {
        let active = UUID()
        let stale = UUID()

        XCTAssertFalse(VoiceTurnCompletionPolicy.acceptsSpeechCompletion(
            turnID: stale,
            activeSpeechTurnID: active
        ))
        XCTAssertFalse(VoiceTurnCompletionPolicy.shouldResumeHandsFree(
            handsFree: true,
            turnID: stale,
            activeSpeechTurnID: active
        ))
    }

    func testTurnCompletionAllowsHandsFreeResumeForCurrentSpeechTurn() {
        let active = UUID()

        XCTAssertTrue(VoiceTurnCompletionPolicy.acceptsSpeechCompletion(
            turnID: active,
            activeSpeechTurnID: active
        ))
        XCTAssertTrue(VoiceTurnCompletionPolicy.shouldResumeHandsFree(
            handsFree: true,
            turnID: active,
            activeSpeechTurnID: active
        ))
        XCTAssertFalse(VoiceTurnCompletionPolicy.shouldResumeHandsFree(
            handsFree: false,
            turnID: active,
            activeSpeechTurnID: active
        ))
    }
}
