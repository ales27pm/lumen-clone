import Testing
@testable import Lumen

struct ReasoningAwareStreamParserTests {
    @Test func plainAnswerNoThinkTags() {
        let result = parse(["Hello there."])
        #expect(result.visibleAnswer == "Hello there.")
        #expect(result.reasoningText == nil)
        #expect(result.rawModelOutput == "Hello there.")
        #expect(result.parserWarnings.isEmpty)
    }

    @Test func completeThinkBlockRoutesReasoningAndFinal() {
        let result = parse(["<think>reasoning</think>final"], captureReasoning: true)
        #expect(result.visibleAnswer == "final")
        #expect(result.reasoningText == "reasoning")
        #expect(result.rawModelOutput == "<think>reasoning</think>final")
    }

    @Test func partialStreamChunksWhereTagsAreSplit() {
        let result = parse(["<thi", "nk>abc</th", "ink>final"], captureReasoning: true)
        #expect(result.visibleAnswer == "final")
        #expect(result.reasoningText == "abc")
        #expect(!result.visibleAnswer.contains("abc"))
    }

    @Test func missingClosingTagMarksUnterminatedAndFallsBackWhenNoFinal() {
        let result = parse(["<think>only reasoning"], captureReasoning: true)
        #expect(result.visibleAnswer == ReasoningAwareStreamParserConfig.onlyReasoningFallback)
        #expect(result.reasoningText == "only reasoning")
        #expect(result.unterminatedReasoningBlock)
        #expect(result.parserWarnings.contains("Unterminated <think> block at end of stream."))
    }

    @Test func missingOpeningTagButClosingTagAppears() {
        let result = parse(["visible</think>final"], captureReasoning: true)
        #expect(result.visibleAnswer == "visiblefinal")
        #expect(result.reasoningText == nil)
        #expect(result.parserWarnings.contains("Closing </think> appeared without a matching opening tag."))
    }

    @Test func nestedThinkTagsAreDefensive() {
        let result = parse(["<think>a<think>b</think>c</think>final"], captureReasoning: true)
        #expect(result.visibleAnswer == "final")
        #expect(result.reasoningText == "abc")
        #expect(result.parserWarnings.contains("Nested <think> block encountered."))
    }

    @Test func multipleThinkBlocksAreCaptured() {
        let result = parse(["<think>a</think>one<think>b</think>two"], captureReasoning: true)
        #expect(result.visibleAnswer == "onetwo")
        #expect(result.reasoningText == "ab")
    }

    @Test func hugeReasoningBlockIsTruncated() {
        let result = parse(
            ["<think>", String(repeating: "x", count: 32), "</think>final"],
            captureReasoning: true,
            budget: 8
        )
        #expect(result.visibleAnswer == "final")
        #expect(result.reasoningText == "xxxxxxxx")
        #expect(result.reasoningWasTruncated)
        #expect(result.parserWarnings.contains("Reasoning exceeded developer trace budget and was truncated."))
    }

    @Test func reasoningFollowedByFinalAnswer() {
        let result = parse(["<think>abc</think>\n\nFinal answer."], captureReasoning: true)
        #expect(result.visibleAnswer == "\n\nFinal answer.")
        #expect(result.reasoningText == "abc")
    }

    @Test func finalAnswerBeforeThinkBlockThenAnotherFinalSegment() {
        let result = parse(["First. <think>hidden</think> Second."], captureReasoning: true)
        #expect(result.visibleAnswer == "First.  Second.")
        #expect(result.reasoningText == "hidden")
    }

    @Test func normalModeNeverExposesReasoningText() {
        let result = parse(["<think>Okay, the user wants internal notes.</think>Final."], captureReasoning: false)
        #expect(result.visibleAnswer == "Final.")
        #expect(result.reasoningText == nil)
        #expect(!result.visibleAnswer.contains("Okay, the user wants"))
    }

    @Test func uppercaseThinkTagsAreAccepted() {
        let result = parse(["<THINK>LOUD</THINK>quiet"], captureReasoning: true)
        #expect(result.visibleAnswer == "quiet")
        #expect(result.reasoningText == "LOUD")
    }

    @Test func chatMessageRenderContentUsesVisibleContentBeforeRawContent() {
        let message = ChatMessage(
            role: .assistant,
            content: "<think>private</think>Final",
            visibleContent: "Final",
            reasoningTrace: "private",
            rawModelOutput: "<think>private</think>Final"
        )
        #expect(message.assistantRenderContent == "Final")
        #expect(!message.assistantRenderContent.contains("private"))
    }

    @Test func incompleteSplitOpeningTagIsDiscardedAtFinish() {
        let result = parse(["final <thi"], captureReasoning: true)
        #expect(result.visibleAnswer == "final ")
        #expect(result.parserWarnings.contains("Incomplete reasoning tag at end of stream was discarded."))
    }

    private func parse(_ chunks: [String], captureReasoning: Bool = false, budget: Int = 16_384) -> ReasoningAwareStreamParserResult {
        var parser = ReasoningAwareStreamParser(
            config: ReasoningAwareStreamParserConfig(
                captureReasoning: captureReasoning,
                reasoningTraceBudgetCharacters: budget
            )
        )
        for chunk in chunks {
            _ = parser.ingest(chunk)
        }
        _ = parser.finish()
        return parser.result
    }
}
