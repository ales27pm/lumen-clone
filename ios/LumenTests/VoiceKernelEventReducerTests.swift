import XCTest
@testable import Lumen

final class VoiceKernelEventReducerTests: XCTestCase {
    func testTokenStreamingUpdatesResponseTextForSpeech() {
        var state = VoiceKernelEventState()

        let first = VoiceKernelEventReducer.reduce(.token("Hel"), state: &state, lastUserMessage: "Say hello", routing: Self.chatRouting)
        let second = VoiceKernelEventReducer.reduce(.finalDelta("lo."), state: &state, lastUserMessage: "Say hello", routing: Self.chatRouting)

        XCTAssertTrue(first.textChanged)
        XCTAssertTrue(second.textChanged)
        XCTAssertTrue(second.shouldSpeakPending)
        XCTAssertEqual(state.finalText, "Hello.")
        XCTAssertEqual(state.responseText, "Hello.")
    }

    func testFinalResponseIsReadyForTTS() {
        var state = VoiceKernelEventState()

        _ = VoiceKernelEventReducer.reduce(.token("draft"), state: &state, lastUserMessage: "Prompt", routing: Self.chatRouting)
        let mutation = VoiceKernelEventReducer.reduce(.final("Final answer."), state: &state, lastUserMessage: "Prompt", routing: Self.chatRouting)

        XCTAssertTrue(mutation.textChanged)
        XCTAssertTrue(mutation.shouldStartSpeaking)
        XCTAssertTrue(mutation.shouldSpeakPending)
        XCTAssertEqual(state.finalText, "Final answer.")
        XCTAssertEqual(state.responseText, "Final answer.")
    }

    func testStreamingResponseDoesNotIntentValidatePartialText() {
        var state = VoiceKernelEventState()

        _ = VoiceKernelEventReducer.reduce(.token("The temp"), state: &state, lastUserMessage: "Weather?", routing: Self.weatherRouting)
        let streaming = VoiceKernelEventReducer.streamingResponseText(from: state.finalText, lastUserMessage: "Weather?")

        XCTAssertEqual(state.responseText, "The temp")
        XCTAssertEqual(streaming, "The temp")
    }

    func testToolInvocationAndResultBecomeVoiceStatusSteps() {
        let invocationID = UUID()
        let invocation = ToolInvocation(
            id: invocationID,
            toolID: "calendar.read",
            arguments: ["date": "today"],
            source: .modelProposed,
            conversationID: nil,
            turnID: nil,
            createdAt: Date()
        )
        let result = ToolResult(
            invocationID: invocationID,
            status: .success,
            displayText: "Found two events.",
            modelText: "2 events",
            structuredPayload: nil,
            privacyLevel: .low,
            metricsSummary: "2 rows",
            errorCode: nil
        )
        var state = VoiceKernelEventState()

        let invocationMutation = VoiceKernelEventReducer.reduce(.toolInvocation(invocation), state: &state, lastUserMessage: "Calendar", routing: Self.chatRouting)
        let resultMutation = VoiceKernelEventReducer.reduce(.toolResult(result), state: &state, lastUserMessage: "Calendar", routing: Self.chatRouting)

        XCTAssertTrue(invocationMutation.stepsChanged)
        XCTAssertTrue(resultMutation.stepsChanged)
        XCTAssertEqual(state.steps.count, 2)
        XCTAssertEqual(state.steps[0].id, invocationID)
        XCTAssertEqual(state.steps[0].kind, .action)
        XCTAssertEqual(state.steps[0].toolID, "calendar.read")
        XCTAssertEqual(state.steps[0].toolArgs?["date"], "today")
        XCTAssertEqual(state.steps[1].kind, .observation)
        XCTAssertEqual(state.steps[1].content, "Found two events.")
        XCTAssertEqual(state.steps[1].toolArgs?["status"], ToolResultStatus.success.rawValue)
    }

    func testDegradedDiagnosticCreatesVoiceReflectionStep() {
        var state = VoiceKernelEventState()
        let diagnostic = AgentKernelDiagnosticEvent(
            stage: "runtime-selection",
            message: "Using deterministic fallback.",
            metadata: ["runtime": AssistantRuntimeKind.deterministicFallback.rawValue]
        )

        let mutation = VoiceKernelEventReducer.reduce(.diagnostic(diagnostic), state: &state, lastUserMessage: "Prompt", routing: Self.chatRouting)

        XCTAssertTrue(mutation.diagnosticsChanged)
        XCTAssertTrue(mutation.stepsChanged)
        XCTAssertTrue(state.degraded)
        XCTAssertEqual(state.diagnostics.count, 1)
        XCTAssertEqual(state.steps.first?.kind, .reflection)
        XCTAssertEqual(state.steps.first?.content, "Using deterministic fallback.")
    }

    func testErrorResponseIsReadyForTTS() {
        var state = VoiceKernelEventState()

        let mutation = VoiceKernelEventReducer.reduce(.error("Model failed."), state: &state, lastUserMessage: "Prompt", routing: Self.chatRouting)

        XCTAssertTrue(mutation.textChanged)
        XCTAssertTrue(mutation.shouldSpeakPending)
        XCTAssertEqual(state.errorMessage, "Model failed.")
        XCTAssertEqual(state.finalText, "Model failed.")
        XCTAssertEqual(state.responseText, "Model failed.")
    }

    func testDoneMarksCompletionAndAppliesFinalSteps() {
        var state = VoiceKernelEventState()
        let step = AgentStep(kind: .thought, content: "Done thinking")

        let mutation = VoiceKernelEventReducer.reduce(
            .done(finalText: "Complete.", steps: [step]),
            state: &state,
            lastUserMessage: "Prompt",
            routing: Self.chatRouting
        )

        XCTAssertTrue(mutation.textChanged)
        XCTAssertTrue(mutation.stepsChanged)
        XCTAssertTrue(state.isDone)
        XCTAssertEqual(state.finalText, "Complete.")
        XCTAssertEqual(state.responseText, "Complete.")
        XCTAssertEqual(state.steps, [step])
    }

    func testCancellationMarksVoiceStateWithoutChangingSpeechText() {
        var state = VoiceKernelEventState()
        _ = VoiceKernelEventReducer.reduce(.token("Partial."), state: &state, lastUserMessage: "Prompt", routing: Self.chatRouting)

        let mutation = VoiceKernelEventReducer.cancel(state: &state, reason: "scene-background")

        XCTAssertFalse(mutation.textChanged)
        XCTAssertTrue(state.isCancelled)
        XCTAssertEqual(state.cancellationReason, "scene-background")
        XCTAssertEqual(state.responseText, "Partial.")
    }

    private static var chatRouting: IntentRoutingDecision {
        IntentRoutingDecision(intent: .chat, allowedToolIDs: [], requiresClarification: false, clarificationPrompt: nil)
    }

    private static var weatherRouting: IntentRoutingDecision {
        IntentRoutingDecision(intent: .weather, allowedToolIDs: ["weather"], requiresClarification: false, clarificationPrompt: nil)
    }
}
