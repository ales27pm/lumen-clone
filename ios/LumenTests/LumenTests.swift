//
//  LumenTests.swift
//  LumenTests
//
//  Created by Rork on April 20, 2026.
//

import Foundation
import Testing
import XCTest
@testable import Lumen

struct LumenTests {

    @Test func parserAcceptsNestedAction() async throws {
        let raw = #"{"thought":"look up data","action":{"tool":"weather.current","args":{"city":"Boston"}}}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.action?.tool == "weather.current")
        #expect(turn.action?.args["city"]?.stringValue == "Boston")
        #expect(turn.final == nil)
    }

    @Test func parserAcceptsFlatAction() async throws {
        let raw = #"{"tool":"weather.current","args":{"city":"Boston"}}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.action?.tool == "weather.current")
        #expect(turn.action?.args["city"]?.stringValue == "Boston")
    }

    @Test func parserRejectsMixedTurn() async throws {
        let raw = #"{"action":{"tool":"weather.current","args":{"city":"Boston"}},"final":"sunny"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == .mixedTurn)
        #expect(turn.action == nil)
        #expect(turn.final == nil)
    }

    @Test func parserAcceptsMultipleObjectsSelectingLastValidTurn() async throws {
        let raw = #"{"final":"first"}{"final":"second"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.final == "second")
        #expect(turn.action == nil)
    }

    @Test func parserAcceptsMultipleObjectsWithFirstMalformedSecondValid() async throws {
        let raw = #"{"final":bad}{"final":"second"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.final == "second")
    }

    @Test func parserAcceptsMultipleObjectsWithNoiseOutsideJSONRanges() async throws {
        let raw = #"note {"final":"first"} {"final":"second"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.final == "second")
        #expect(turn.hadNoise)
    }

    @Test func parserRejectsMultipleObjectsWhenAllAreInvalid() async throws {
        let raw = #"{"final":bad}{"action":42}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == .multipleJSONObjects)
    }

    @Test func parserAcceptsTypedNumberArgs() async throws {
        let raw = #"{"action":{"tool":"weather.current","args":{"days":3}}}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.action?.tool == "weather.current")
        #expect(turn.action?.args["days"]?.intValue == 3)
        #expect(turn.action?.args["days"]?.stringValue == "3")
    }

    @Test func parserRejectsMalformedEscapes() async throws {
        let raw = #"{"final":"bad \q escape"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == .malformedEscapeSequence)
    }

    @Test func parserAcceptsValidTrailingJSONWhenPrefixContainsMalformedEscapedProse() async throws {
        let raw = #"prefix prose with malformed escape \q and quote "still prose" {"final":"ok"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.final == "ok")
        #expect(turn.hadNoise)
    }

    @Test func parserAcceptsValidTrailingJSONWhenPrefixContainsMalformedEscapedQuotedProse() async throws {
        let raw = #"prefix "prose with malformed escape \q" {"final":"ok"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.final == "ok")
        #expect(turn.hadNoise)
    }

    @Test func parserRejectsMissingActionAndFinal() async throws {
        let raw = #"{"thought":"I should think first"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == .missingActionOrFinal)
    }

    @Test func parserRejectsIncompleteJSON() async throws {
        let raw = #"{"final":"unterminated""#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == .incompleteJSON)
    }

    @Test func parserRejectsInvalidThoughtType() async throws {
        let raw = #"{"thought":123,"final":"x"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == .invalidThoughtType)
    }

    @Test func parserRejectsInvalidFinalType() async throws {
        let raw = #"{"thought":"ok","final":42}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == .invalidFinalType)
    }

    @Test func parserAcceptsReasoningAndFinalAnswerAliases() async throws {
        let raw = #"{"reasoning":"Need no tools","final_answer":"All good"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.thought == "Need no tools")
        #expect(turn.final == "All good")
        #expect(turn.action == nil)
    }

    @Test func parserAcceptsSentinelLiteralFinal() async throws {
        let raw = #"{"thought":"<PRIVATE_REASONING>","final":"<USER_FINAL_TEXT>"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.final == "<USER_FINAL_TEXT>")
    }

    @Test func parserAcceptsNoisyOutputOutsideJSON() async throws {
        let raw = #"prefix {"final":"ok"} suffix"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.final == "ok")
        #expect(turn.hadNoise)
    }

    @Test func parserAcceptsCodeFenceWrappedJSON() async throws {
        let raw = """
        ```json
        {"final":"ok"}
        ```
        """
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.final == "ok")
        #expect(!turn.hadNoise)
    }

    @Test func parserSelectsDeterministicCandidateWithMultipleObjectsAndStrayText() async throws {
        let raw = #"notice {"final":"fallback"} stray {"tool":"web.search","args":{"query":"swift"}} trailing"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.action?.tool == "web.search")
        #expect(turn.action?.args["query"]?.stringValue == "swift")
        #expect(turn.hadNoise)
    }

    @Test func parserAndDiagnosticsSelectSameCandidateWhenScoresDiffer() async throws {
        let raw = #"prefix {"final":"fallback"} middle {"tool":"web.search","args":{"query":"swift"}} suffix"#
        let turn = AgentTurnParser.parse(raw)
        let snapshot = AgentNoiseInspector.inspect(raw)
        #expect(turn.parseError == nil)
        #expect(turn.action?.tool == "web.search")
        #expect(snapshot.selectedJSON == #"{"tool":"web.search","args":{"query":"swift"}}"#)
    }

    @Test func parserAndDiagnosticsSelectSameCandidateWhenScoresTieAndRecencyWins() async throws {
        let raw = #"{"final":"first"}{"final":"second"}"#
        let turn = AgentTurnParser.parse(raw)
        let snapshot = AgentNoiseInspector.inspect(raw)
        #expect(turn.parseError == nil)
        #expect(turn.final == "second")
        #expect(snapshot.selectedJSON == #"{"final":"second"}"#)
    }

    @Test func parserRejectsMixedActionShapes() async throws {
        let raw = #"{"action":{"tool":"web.search","args":{"query":"llama"}},"tool":"web.search","args":{"query":"llama"}}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == .mixedActionShapes)
    }

    @Test func parserAcceptsNestedActionWithNameAliasAndArgumentsAlias() async throws {
        let raw = #"{"action":{"name":"web.search","arguments":{"query":"swift testing"}}}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.action?.tool == "web.search")
        #expect(turn.action?.args["query"]?.stringValue == "swift testing")
    }

    @Test func parserAcceptsFlatActionUsingInputAlias() async throws {
        let raw = #"{"tool":"memory.save","input":{"content":"remember this","kind":"fact"}}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.action?.tool == "memory.save")
        #expect(turn.action?.args["content"]?.stringValue == "remember this")
        #expect(turn.action?.args["kind"]?.stringValue == "fact")
    }

    @Test func parserAcceptsInputAliasWithTypedArgs() async throws {
        let raw = #"{"tool":"memory.save","input":{"content":"remember this","priority":1,"pinned":true}}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.action?.tool == "memory.save")
        #expect(turn.action?.args["content"]?.stringValue == "remember this")
        #expect(turn.action?.args["priority"]?.intValue == 1)
        #expect(turn.action?.args["pinned"]?.boolValue == true)
    }

    @Test func parserAcceptsRawInputStringContainingJSONObject() async throws {
        let raw = #"{"tool":"web.search","input":"{\"query\":\"swift testing\"}"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.action?.tool == "web.search")
        #expect(turn.action?.args["query"]?.stringValue == "swift testing")
    }

    @Test func parserAcceptsRawInputStringContainingJSONObjectWithTypedArgs() async throws {
        let raw = #"{"tool":"web.search","input":"{\"query\":\"swift testing\",\"limit\":3,\"fresh\":true}"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.action?.tool == "web.search")
        #expect(turn.action?.args["query"]?.stringValue == "swift testing")
        #expect(turn.action?.args["limit"]?.intValue == 3)
        #expect(turn.action?.args["fresh"]?.boolValue == true)
    }

    @Test func parserAcceptsRawInputStringAsFreeTextQueryFallback() async throws {
        let raw = #"{"tool":"web.search","input":"swift testing"}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == nil)
        #expect(turn.action?.tool == "web.search")
        #expect(turn.action?.args["query"]?.stringValue == "swift testing")
    }

    @Test func parserRejectsNestedActionMissingToolName() async throws {
        let raw = #"{"action":{"args":{"query":"x"}}}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == .missingActionTool)
    }

    @Test func parserRejectsFlatActionMissingToolName() async throws {
        let raw = #"{"args":{"query":"x"}}"#
        let turn = AgentTurnParser.parse(raw)
        #expect(turn.parseError == .missingActionTool)
    }

    @Test func streamingScannerEmitsThoughtThenFinalAcrossChunks() async throws {
        let scanner = StreamingJSONScanner()
        let events1 = scanner.feed(#"{"thought":"Plan"#)
        #expect(events1.count == 1)
        #expect(scanner.thought == "Plan")

        let events2 = scanner.feed(#" tool use","final":"Done"#)
        #expect(events2.count >= 1)
        #expect(scanner.thought == "Plan tool use")
        #expect(scanner.final == "Done")

        let events3 = scanner.feed(#""}"#)
        #expect(events3.isEmpty || scanner.final == "Done")
    }

    @Test func streamingScannerDecodesEscapes() async throws {
        let scanner = StreamingJSONScanner()
        _ = scanner.feed(#"{"thought":"line1\nline2","final":"tab\tok"}"#)
        #expect(scanner.thought == "line1\nline2")
        #expect(scanner.final == "tab\tok")
    }

    @Test func streamingScannerDecodesUnicodeEscapesAcrossChunks() async throws {
        let scanner = StreamingJSONScanner()
        _ = scanner.feed(#"{"thought":"Hello \u004"#)
        _ = scanner.feed(#"1","final":"\u263A"}"#)
        #expect(scanner.thought == "Hello A")
        #expect(scanner.final == "☺")
    }

    @Test func streamingScannerReturnsNoEventsWhenTrackedKeysAbsent() async throws {
        let scanner = StreamingJSONScanner()
        let events = scanner.feed(#"{"action":{"tool":"web.search","args":{"query":"x"}}}"#)
        #expect(events.isEmpty)
        #expect(scanner.thought.isEmpty)
        #expect(scanner.final.isEmpty)
    }

    @Test func streamingScannerIgnoresFinalWordInsideThoughtText() async throws {
        let scanner = StreamingJSONScanner()
        let events1 = scanner.feed(#"{"thought":"I might write \"final\" later"#)
        #expect(events1.count == 1)
        #expect(scanner.thought == #"I might write "final" later"#)
        #expect(scanner.final.isEmpty)

        let events2 = scanner.feed(#"","final":"Actual final"}"#)
        #expect(events2.count >= 1)
        #expect(scanner.final == "Actual final")
    }

    @Test func agentActionDedupeKeyAndDisplayAreDeterministic() async throws {
        let action1 = AgentAction(tool: "web.search", args: ["q": .string("swift"), "city": .string("Boston")])
        let action2 = AgentAction(tool: "web.search", args: ["city": .string("Boston"), "q": .string("swift")])
        #expect(action1.dedupeKey == action2.dedupeKey)
        #expect(action1.displayContent == "web.search(city=Boston, q=swift)")
    }

    @Test func placeholderDetectorFlagsSentinelVariantsAndPartialCopies() async throws {
        #expect(SchemaPlaceholderDetector.isPlaceholderFinal("<USER_FINAL_TEXT>"))
        #expect(SchemaPlaceholderDetector.isPlaceholderFinal("<PRIVATE_REASONING>"))
        #expect(SchemaPlaceholderDetector.isPlaceholderFinal("<USER_FI"))
        #expect(SchemaPlaceholderDetector.isPlaceholderFinal("answer shown to the user"))
    }

    @Test func placeholderDetectorPrefixMatchesStreamingSentinelCopy() async throws {
        #expect(SchemaPlaceholderDetector.isPlaceholderPrefix("<USE"))
        #expect(SchemaPlaceholderDetector.isPlaceholderPrefix("<PRIVATE_REAS"))
        #expect(SchemaPlaceholderDetector.isSchemaPlaceholderPrefix("<USER_FIN"))
        #expect(!SchemaPlaceholderDetector.isPlaceholderPrefix("Here is the answer"))
    }

    @Test func placeholderRepairFallsBackForSentinelCopies() async throws {
        let fallback = "I couldn't produce a valid answer. Try rephrasing, or switch off Agent Mode for this prompt."
        #expect(SchemaPlaceholderDetector.repairOrFallback("<USER_FINAL_TEXT>") == fallback)
        #expect(SchemaPlaceholderDetector.repairOrFallback("<PRIVATE_REASONING>") == fallback)
        #expect(SchemaPlaceholderDetector.repairOrFallback("   ") == fallback)
        #expect(SchemaPlaceholderDetector.repairOrFallback("Use two eggs and whisk.") == "Use two eggs and whisk.")
    }

    @Test @MainActor func toolExecutorReturnsUnknownToolMessage() async throws {
        let result = await ToolExecutor.shared.execute("tool.that.does.not.exist", arguments: [String: String]())
        #expect(result == "Unknown tool: tool.that.does.not.exist")
    }

    @Test func agentStepCodecRoundTripPreservesValues() async throws {
        let original = [
            AgentStep(kind: .thought, content: "Need weather", toolID: nil, toolArgs: nil),
            AgentStep(kind: .action, content: "weather.current(city=Boston)", toolID: "weather.current", toolArgs: ["city": "Boston"]),
            AgentStep(kind: .observation, content: "72°F and sunny", toolID: "weather.current", toolArgs: nil),
            AgentStep(kind: .reflection, content: "Ready to answer", toolID: nil, toolArgs: nil)
        ]
        let encoded = AgentStepCodec.encode(original)
        #expect(encoded != nil)
        let decoded = AgentStepCodec.decode(encoded)
        #expect(decoded.count == original.count)
        #expect(decoded.map(\.kind) == original.map(\.kind))
        #expect(decoded.map(\.content) == original.map(\.content))
        #expect(decoded[1].toolID == "weather.current")
        #expect(decoded[1].toolArgs?["city"] == "Boston")
    }

    @Test @MainActor func sanitizeHistoryContentStripsMarkdownXmlAndToolWrappers() async throws {
        let raw = """
        <tool_call>
        ```json
        {"name":"web.search","arguments":{"query":"Swift"}}
        ```
        </tool_call>
        <response>Sure — let's continue with your question.</response>
        """
        let sanitized = AgentService.shared.sanitizeHistoryContentForTests(raw)
        #expect(sanitized == "Sure — let's continue with your question.")
    }

    @Test @MainActor func sanitizeHistoryContentCollapsesRepeatedStructuralPunctuation() async throws {
        let raw = #"Answer:::: {{{{done}}}} [[ok]] <final>Great!!!</final> ##"#
        let sanitized = AgentService.shared.sanitizeHistoryContentForTests(raw)
        #expect(sanitized == "Answer:::: {done} [ok] Great!!! ##")
    }

    @Test @MainActor func sanitizeHistoryContentPreservesURLPathAndMarkupTokens() async throws {
        let raw = #"Noise {{{{ https://example.com/a//b?x=1::2#frag /Users/me/Hybrid Coder/Models/model.gguf ./Sources//AgentService.swift **bold** __under__ ~~strike~~ }}"#
        let sanitized = AgentService.shared.sanitizeHistoryContentForTests(raw)
        #expect(sanitized == "Noise { https://example.com/a//b?x=1::2#frag /Users/me/Hybrid Coder/Models/model.gguf ./Sources//AgentService.swift **bold** __under__ ~~strike~~ }")
    }

    @Test @MainActor func sanitizeSystemPromptForStructuredOutputRemovesCoderFormattingPressure() async throws {
        let sanitized = AgentService.shared.sanitizeSystemPromptForStructuredOutputForTests(Presets.coder.prompt)
        #expect(!sanitized.lowercased().contains("fenced code"))
        #expect(!sanitized.lowercased().contains("markdown"))
        #expect(sanitized.contains("You are Lumen in coder mode."))
        #expect(sanitized.contains("Prefer Swift/SwiftUI for iOS."))
    }

    @Test @MainActor func sanitizeSystemPromptForStructuredOutputRemovesResearcherFormattingPressure() async throws {
        let sanitized = AgentService.shared.sanitizeSystemPromptForStructuredOutputForTests(Presets.researcher.prompt)
        #expect(!sanitized.lowercased().contains("headings"))
        #expect(!sanitized.lowercased().contains("step-by-step"))
        #expect(!sanitized.lowercased().contains("step by step"))
        #expect(sanitized.contains("You are Lumen in researcher mode."))
        #expect(sanitized.contains("Be thorough, cite reasoning"))
    }

    @Test @MainActor func structuredAgentTurnMaxTokensUsesDedicatedCap() async throws {
        let low = AgentService.shared.structuredTurnMaxTokensForTests(from: 32)
        let mid = AgentService.shared.structuredTurnMaxTokensForTests(from: 256)
        let high = AgentService.shared.structuredTurnMaxTokensForTests(from: 4_096)

        #expect(low == 128)
        #expect(mid == 256)
        #expect(high == 384)
        #expect(low <= 384)
        #expect(mid <= 384)
        #expect(high <= 384)
    }

    @Test func parseFailureSummaryAggregatesByErrorAndNoiseSignatures() async throws {
        let lines = [
            try makeParseFailureTraceLine(parseError: "invalidJSONObject", prefixNoise: "Prefix NOISE alpha", suffixNoise: "Suffix one"),
            try makeParseFailureTraceLine(parseError: "invalidJSONObject", prefixNoise: "prefix noise alpha", suffixNoise: "suffix one"),
            try makeParseFailureTraceLine(parseError: "invalidJSONObject", prefixNoise: "prefix noise beta", suffixNoise: "suffix one"),
            try makeParseFailureTraceLine(parseError: "missingActionOrFinal", prefixNoise: nil, suffixNoise: nil),
        ]
        let jsonl = lines.joined(separator: "\n")
        let summary = AgentParseFailureSummaryLoader.load(fromJSONLText: jsonl, topN: 5)

        #expect(summary.totalLines == 4)
        #expect(summary.decodedLines == 4)
        #expect(summary.skippedLines == 0)
        #expect(summary.topEntries.count == 3)
        #expect(summary.topEntries[0].count == 2)
        #expect(summary.topEntries[0].parseError == "invalidJSONObject")
        #expect(summary.topEntries[0].suffixSignature.hasPrefix("suffix one#"))
        #expect(summary.topEntries[0].prefixSignature.hasPrefix("prefix noise alpha#"))
    }

    @Test func parseFailureSummarySkipsCorruptLinesAndRespectsTopN() async throws {
        let validA = try makeParseFailureTraceLine(parseError: "invalidFinalType", prefixNoise: "pre a", suffixNoise: "suf a")
        let validB = try makeParseFailureTraceLine(parseError: "invalidThoughtType", prefixNoise: "pre b", suffixNoise: "suf b")
        let jsonl = [validA, "{not-json", validA, validB].joined(separator: "\n")
        let summary = AgentParseFailureSummaryLoader.load(fromJSONLText: jsonl, topN: 1)

        #expect(summary.totalLines == 4)
        #expect(summary.decodedLines == 3)
        #expect(summary.skippedLines == 1)
        #expect(summary.topEntries.count == 1)
        #expect(summary.topEntries[0].parseError == "invalidFinalType")
        #expect(summary.topEntries[0].count == 2)
    }

    @Test func parseNoiseSummarySkipsCorruptLinesAndRespectsTopN() async throws {
        let validA = try makeParseNoiseTraceLine(modelName: "agent-json", stepIndex: 0, prefixNoise: "pre a", suffixNoise: "suf a")
        let validB = try makeParseNoiseTraceLine(modelName: "agent-json", stepIndex: 1, prefixNoise: "pre b", suffixNoise: "suf b")
        let jsonl = [validA, "{not-json", validA, validB].joined(separator: "\n")
        let summary = AgentParseNoiseSummaryLoader.load(fromJSONLText: jsonl, topN: 1)

        #expect(summary.totalLines == 4)
        #expect(summary.decodedLines == 3)
        #expect(summary.skippedLines == 1)
        #expect(summary.topEntries.count == 1)
        #expect(summary.topEntries[0].modelName == "agent-json")
        #expect(summary.topEntries[0].stepIndex == 0)
        #expect(summary.topEntries[0].count == 2)
    }

    @Test func parseNoiseSummaryGroupsByNormalizedSignaturesModelAndStep() async throws {
        let lines = [
            try makeParseNoiseTraceLine(modelName: "agent-json", stepIndex: 1, prefixNoise: "Prefix Noise Alpha", suffixNoise: "Suffix One"),
            try makeParseNoiseTraceLine(modelName: "agent-json", stepIndex: 1, prefixNoise: "prefix   noise alpha", suffixNoise: "suffix one"),
            try makeParseNoiseTraceLine(modelName: "agent-json", stepIndex: 2, prefixNoise: "prefix noise alpha", suffixNoise: "suffix one"),
            try makeParseNoiseTraceLine(modelName: "agent-thought", stepIndex: 1, prefixNoise: "prefix noise alpha", suffixNoise: "suffix one"),
        ]
        let summary = AgentParseNoiseSummaryLoader.load(fromJSONLText: lines.joined(separator: "\n"), topN: 5)

        #expect(summary.totalLines == 4)
        #expect(summary.decodedLines == 4)
        #expect(summary.skippedLines == 0)
        #expect(summary.topEntries.count == 3)
        #expect(summary.topEntries[0].count == 2)
        #expect(summary.topEntries[0].modelName == "agent-json")
        #expect(summary.topEntries[0].stepIndex == 1)
        #expect(summary.topEntries[0].prefixSignature.hasPrefix("prefix noise alpha#"))
        #expect(summary.topEntries[0].suffixSignature.hasPrefix("suffix one#"))
    }

    @Test func parseFailureSummaryComputesRecentTrendWindowsAndRegression() async throws {
        let now = Date(timeIntervalSince1970: 2_000_000)
        let lines = [
            try makeParseFailureTraceLine(parseError: "invalidJSONObject", prefixNoise: "alpha", suffixNoise: "one", createdAt: now.addingTimeInterval(-172_800)),
            try makeParseFailureTraceLine(parseError: "invalidJSONObject", prefixNoise: "alpha", suffixNoise: "one", createdAt: now.addingTimeInterval(-36_000)),
            try makeParseFailureTraceLine(parseError: "invalidJSONObject", prefixNoise: "alpha", suffixNoise: "one", createdAt: now.addingTimeInterval(-3_600)),
            try makeParseFailureTraceLine(parseError: "invalidJSONObject", prefixNoise: "alpha", suffixNoise: "one", createdAt: now.addingTimeInterval(-600)),
            try makeParseFailureTraceLine(parseError: "missingActionOrFinal", prefixNoise: "beta", suffixNoise: "two", createdAt: now.addingTimeInterval(-300)),
        ]
        let summary = AgentParseFailureSummaryLoader.load(fromJSONLText: lines.joined(separator: "\n"), topN: 5)

        #expect(summary.decodedLines == 5)
        #expect(summary.recentLineWindowSize == 5)
        #expect(summary.recent24hCount == 4)
        #expect(summary.recent24hTopEntries.count == 2)
        #expect(summary.recent24hTopEntries[0].parseError == "invalidJSONObject")
        #expect(summary.recent24hTopEntries[0].recentCount == 3)
        #expect(summary.recent24hTopEntries[0].isRegression)
        #expect(summary.recent24hTopEntries[0].recentShare > summary.recent24hTopEntries[0].baselineShare)
    }

    @Test func parseNoiseSummaryComputesRecentTrendWindowsAndRegression() async throws {
        let now = Date(timeIntervalSince1970: 3_000_000)
        let lines = [
            try makeParseNoiseTraceLine(modelName: "agent-json", stepIndex: 1, prefixNoise: "alpha", suffixNoise: "one", createdAt: now.addingTimeInterval(-200_000)),
            try makeParseNoiseTraceLine(modelName: "agent-json", stepIndex: 1, prefixNoise: "alpha", suffixNoise: "one", createdAt: now.addingTimeInterval(-180_000)),
            try makeParseNoiseTraceLine(modelName: "agent-json", stepIndex: 1, prefixNoise: "alpha", suffixNoise: "one", createdAt: now.addingTimeInterval(-1_000)),
            try makeParseNoiseTraceLine(modelName: "agent-json", stepIndex: 1, prefixNoise: "alpha", suffixNoise: "one", createdAt: now.addingTimeInterval(-500)),
            try makeParseNoiseTraceLine(modelName: "agent-json", stepIndex: 1, prefixNoise: "alpha", suffixNoise: "one", createdAt: now.addingTimeInterval(-100)),
            try makeParseNoiseTraceLine(modelName: "agent-thought", stepIndex: 1, prefixNoise: "beta", suffixNoise: "two", createdAt: now.addingTimeInterval(-50)),
        ]
        let summary = AgentParseNoiseSummaryLoader.load(fromJSONLText: lines.joined(separator: "\n"), topN: 5)

        #expect(summary.decodedLines == 6)
        #expect(summary.recentLineWindowSize == 6)
        #expect(summary.recent24hCount == 4)
        #expect(summary.recent24hTopEntries.count == 2)
        #expect(summary.recent24hTopEntries[0].modelName == "agent-json")
        #expect(summary.recent24hTopEntries[0].stepIndex == 1)
        #expect(summary.recent24hTopEntries[0].recentCount == 3)
        #expect(summary.recent24hTopEntries[0].isRegression)
        #expect(summary.recent24hTopEntries[0].recentShare > summary.recent24hTopEntries[0].baselineShare)
    }

    @Test func agentRoutingAttachmentNormalizationReducesStructuralNoiseForCodeHeavyContent() async throws {
        let content = """
        ```json
        {{{{{{{{{{{{{{{{{{{{{{{{{{{{{{
        "tool":"web.search","args":{"query":"swift parser"}}
        ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
        /////////////////////////////////////////////////
        ###############################################
        ```
        """
        let attachment = try makeTestAttachment(name: "code-heavy.txt", kind: .text, content: content)
        let budget = PromptBudget(totalChars: 12_000, attachmentsShare: 4_000, memoriesShare: 0, historyShare: 2_000)

        let rawAssembly = PromptAssembler.assemble(
            systemPrompt: "sys",
            history: [],
            userMessage: "route this",
            memories: [],
            attachments: [attachment],
            budget: budget,
            attachmentNormalization: .preserveRaw
        )
        let normalizedAssembly = PromptAssembler.assemble(
            systemPrompt: "sys",
            history: [],
            userMessage: "route this",
            memories: [],
            attachments: [attachment],
            budget: budget,
            attachmentNormalization: .agentRouting
        )

        let rawNoise = structuralNoiseScore(rawAssembly.systemPrompt)
        let normalizedNoise = structuralNoiseScore(normalizedAssembly.systemPrompt)

        #expect(normalizedNoise < rawNoise)
        #expect(normalizedAssembly.systemPrompt.contains("\"tool\":\"web.search\""))
        #expect(!normalizedAssembly.systemPrompt.contains("```json"))
    }

    @Test func agentRoutingAttachmentNormalizationKeepsPdfExtractedRelevantContent() async throws {
        let content = """
        ```text
        ----- PAGE 1 -----
        ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        Invoice Number: INV-9381
        Due Date: 2026-05-01
        Total: $4,920.55
        [][()][()][()][()][()][()][()][()][()][()][()][()][()][()]
        ----- END PAGE -----
        ```
        """
        let attachment = try makeTestAttachment(name: "scan-extract.txt", kind: .text, content: content)
        let budget = PromptBudget(totalChars: 12_000, attachmentsShare: 4_000, memoriesShare: 0, historyShare: 2_000)

        let rawAssembly = PromptAssembler.assemble(
            systemPrompt: "sys",
            history: [],
            userMessage: "extract invoice total",
            memories: [],
            attachments: [attachment],
            budget: budget,
            attachmentNormalization: .preserveRaw
        )
        let normalizedAssembly = PromptAssembler.assemble(
            systemPrompt: "sys",
            history: [],
            userMessage: "extract invoice total",
            memories: [],
            attachments: [attachment],
            budget: budget,
            attachmentNormalization: .agentRouting
        )

        #expect(structuralNoiseScore(normalizedAssembly.systemPrompt) < structuralNoiseScore(rawAssembly.systemPrompt))
        #expect(normalizedAssembly.systemPrompt.contains("Invoice Number: INV-9381"))
        #expect(normalizedAssembly.systemPrompt.contains("Total: $4,920.55"))
        #expect(!normalizedAssembly.systemPrompt.contains("```text"))
    }

    @Test @MainActor func sanitizeInternalErrorNoiseRemovesRepairPrefixArtifacts() async throws {
        let raw = """
        I hit an internal formatting issue and repaired it into a plain answer. Prefix noise: Generation error: The operation couldn't be completed.
        (SwiftLlama.LlamaError error 0.) No valid JSON object found in raw model output.
        Here's the real answer: hi there.
        """
        let cleaned = AgentService().sanitizeInternalErrorNoiseForTests(raw)
        #expect(cleaned == "Here's the real answer: hi there.")
    }

    @Test @MainActor func sanitizeInternalErrorNoiseRemovesStandaloneGenerationError() async throws {
        let raw = """
        Generation error: The operation couldn't be completed.
        (SwiftLlama.LlamaError error 0.)
        """
        let cleaned = AgentService().sanitizeInternalErrorNoiseForTests(raw)
        #expect(cleaned.isEmpty)
    }

}

private func makeParseFailureTraceLine(
    parseError: String,
    prefixNoise: String?,
    suffixNoise: String?,
    createdAt: Date = Date(timeIntervalSince1970: 1_000)
) throws -> String {
    let trace = AgentParseFailureTrace(
        id: UUID(),
        createdAt: createdAt,
        parseError: parseError,
        modelName: "agent-json",
        temperature: 0.1,
        topP: 0.8,
        maxTokens: 512,
        stepIndex: 0,
        systemPromptPrefix: "system",
        userTurnPrefix: "user",
        rawOutputPrefix: "raw",
        streamedThoughtPrefix: "",
        streamedFinalPrefix: "",
        selectedJSONPrefix: nil,
        prefixNoise: prefixNoise,
        suffixNoise: suffixNoise
    )
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    let data = try encoder.encode(trace)
    return String(decoding: data, as: UTF8.self)
}

private func makeParseNoiseTraceLine(
    modelName: String,
    stepIndex: Int,
    prefixNoise: String?,
    suffixNoise: String?,
    createdAt: Date = Date(timeIntervalSince1970: 1_000)
) throws -> String {
    let trace = AgentParseNoiseTrace(
        id: UUID(),
        createdAt: createdAt,
        modelName: modelName,
        temperature: 0.1,
        topP: 0.8,
        maxTokens: 512,
        stepIndex: stepIndex,
        systemPromptPrefix: "system",
        userTurnPrefix: "user",
        rawOutputPrefix: "raw",
        selectedJSONPrefix: nil,
        prefixNoise: prefixNoise,
        suffixNoise: suffixNoise
    )
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    let data = try encoder.encode(trace)
    return String(decoding: data, as: UTF8.self)
}

private func makeTestAttachment(name: String, kind: ChatAttachment.Kind, content: String) throws -> ChatAttachment {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    XCTAssertNoThrow(try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true), "Creating a temporary attachment directory should succeed before writing test fixture content.")
    let fileURL = dir.appendingPathComponent(name)
    XCTAssertNoThrow(try content.write(to: fileURL, atomically: true, encoding: .utf8), "Writing attachment fixture content to disk should succeed so prompt assembly tests can load the file.")
    return ChatAttachment(
        name: name,
        kind: kind,
        path: fileURL.path,
        byteSize: content.utf8.count
    )
}

private func structuralNoiseScore(_ text: String) -> Int {
    let pattern = #"[`|#=_\-\*<>\[\]\(\)\{\}/\\:;,+]{4,}"#
    guard let regex = try? NSRegularExpression(pattern: pattern) else { return 0 }
    let range = NSRange(location: 0, length: (text as NSString).length)
    let matches = regex.matches(in: text, options: [], range: range)
    return matches.reduce(0) { $0 + $1.range.length }
}
