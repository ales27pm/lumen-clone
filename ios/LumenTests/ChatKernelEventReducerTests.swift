import XCTest
@testable import Lumen

final class ChatKernelEventReducerTests: XCTestCase {
    func testTokenAndFinalDeltaAppendVisibleStreamingText() {
        var state = ChatKernelEventState()

        let tokenMutation = ChatKernelEventReducer.reduce(
            .token("Hel"),
            state: &state,
            lastUserMessage: "Say hello"
        )
        let deltaMutation = ChatKernelEventReducer.reduce(
            .finalDelta("lo"),
            state: &state,
            lastUserMessage: "Say hello"
        )

        XCTAssertTrue(tokenMutation.textChanged)
        XCTAssertTrue(deltaMutation.textChanged)
        XCTAssertEqual(state.finalText, "Hello")
        XCTAssertEqual(state.streamingText, "Hello")
    }

    func testFinalEventReplacesStreamedTextWithoutDuplicating() {
        var state = ChatKernelEventState()

        _ = ChatKernelEventReducer.reduce(.finalDelta("draft"), state: &state, lastUserMessage: "Prompt")
        let mutation = ChatKernelEventReducer.reduce(.final("final answer"), state: &state, lastUserMessage: "Prompt")

        XCTAssertTrue(mutation.textChanged)
        XCTAssertEqual(state.finalText, "final answer")
        XCTAssertEqual(state.streamingText, "final answer")
    }

    func testToolInvocationAndResultBecomeVisibleSteps() {
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
        var state = ChatKernelEventState()

        let invocationMutation = ChatKernelEventReducer.reduce(.toolInvocation(invocation), state: &state, lastUserMessage: "Calendar")
        let resultMutation = ChatKernelEventReducer.reduce(.toolResult(result), state: &state, lastUserMessage: "Calendar")

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

    func testDegradedDiagnosticCreatesReflectionStep() {
        var state = ChatKernelEventState()
        let diagnostic = AgentKernelDiagnosticEvent(
            stage: "runtime-selection",
            message: "Using deterministic fallback.",
            metadata: ["runtime": AssistantRuntimeKind.deterministicFallback.rawValue]
        )

        let mutation = ChatKernelEventReducer.reduce(.diagnostic(diagnostic), state: &state, lastUserMessage: "Prompt")

        XCTAssertTrue(mutation.diagnosticsChanged)
        XCTAssertTrue(mutation.stepsChanged)
        XCTAssertTrue(state.degraded)
        XCTAssertEqual(state.diagnostics.count, 1)
        XCTAssertEqual(state.steps.first?.kind, .reflection)
        XCTAssertEqual(state.steps.first?.content, "Using deterministic fallback.")
    }

    func testErrorBecomesVisibleFinalText() {
        var state = ChatKernelEventState()

        let mutation = ChatKernelEventReducer.reduce(.error("Model failed"), state: &state, lastUserMessage: "Prompt")

        XCTAssertTrue(mutation.textChanged)
        XCTAssertEqual(state.errorMessage, "Model failed")
        XCTAssertEqual(state.finalText, "Model failed")
        XCTAssertEqual(state.streamingText, "Model failed")
    }

    func testDoneMarksCompletionAndAppliesFinalSteps() {
        var state = ChatKernelEventState()
        let step = AgentStep(kind: .thought, content: "Done thinking")

        let mutation = ChatKernelEventReducer.reduce(
            .done(finalText: "Complete", steps: [step]),
            state: &state,
            lastUserMessage: "Prompt"
        )

        XCTAssertTrue(mutation.textChanged)
        XCTAssertTrue(mutation.stepsChanged)
        XCTAssertTrue(state.isDone)
        XCTAssertEqual(state.finalText, "Complete")
        XCTAssertEqual(state.streamingText, "Complete")
        XCTAssertEqual(state.steps, [step])
    }
}
