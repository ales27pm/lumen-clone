import Testing
@testable import Lumen

struct FinalOutputSanitizerTests {
    @Test func removesCompleteThinkBlock() {
        let out = FinalOutputSanitizer.sanitizeUserVisibleText("<think>secret</think>Answer")
        #expect(out.text == "Answer")
        #expect(out.removedArtifacts.contains(.thinkBlock))
    }

    @Test func removesMalformedThinkPrefix() {
        let out = FinalOutputSanitizer.sanitizeUserVisibleText("<think>secret no close")
        #expect(!out.text.contains("<think"))
        #expect(out.removedArtifacts.contains(.malformedThinkPrefix))
    }

    @Test func preservesAnswerAfterThinkBlock() {
        let out = FinalOutputSanitizer.sanitizeUserVisibleText("<think>a</think>\n\nHi there")
        #expect(out.text == "Hi there")
    }

    @Test func removesLumenWebPayload() {
        let out = FinalOutputSanitizer.sanitizeUserVisibleText("Before <lumen_web_payload>{\"kind\":\"searchResults\"}</lumen_web_payload> after")
        #expect(out.text == "Before after")
        #expect(out.removedArtifacts.contains(.lumenWebPayload))
    }

    @Test func removesRawSearchResultsJSONObjectOutsidePayloadTags() {
        let raw = "Before {\"kind\":\"searchResults\",\"results\":[{\"mediaKind\":\"page\",\"title\":\"Doc\"}]} after"
        let out = FinalOutputSanitizer.sanitizeUserVisibleText(raw)
        #expect(out.text == "Before after")
        #expect(out.removedArtifacts.contains(.rawToolPayload))
        #expect(!out.text.contains("searchResults"))
        #expect(!out.text.contains("mediaKind"))
    }

    @Test func removesRawSearchResultsMarkerLineWhenJSONIsMalformed() {
        let raw = "Answer line\n{\"kind\":\"searchResults\",\"sourcePageURL\":\"https://example.com\"\nFollow-up"
        let out = FinalOutputSanitizer.sanitizeUserVisibleText(raw)
        #expect(out.text == "Answer line\nFollow-up")
        #expect(out.removedArtifacts.contains(.rawToolPayload))
    }

    @Test func removesBareSearchResultsJSONObject() {
        let raw = "{\"kind\":\"searchResults\",\"results\":[{\"mediaKind\":\"page\"}]}"
        let out = FinalOutputSanitizer.sanitizeUserVisibleText(raw)
        #expect(out.removedArtifacts.contains(.rawToolPayload))
        #expect(!out.text.lowercased().contains("searchresults"))
    }

    @Test func defaultSanitizerPreservesLegitimateFallbackPrefixedModelText() {
        let raw = "\(FinalOutputSanitizer.fallback) This is the actual model answer about error handling."
        let out = FinalOutputSanitizer.sanitizeUserVisibleText(raw)
        #expect(out.text == raw)
        #expect(!out.removedArtifacts.contains(.injectedFallbackPrefix))
        #expect(!out.hadUnsafeLeakage)
    }

    @Test func explicitInjectedProvenanceStripsFallbackPrefix() {
        let raw = "\(FinalOutputSanitizer.fallback) This is the actual model answer about error handling."
        let out = FinalOutputSanitizer.sanitizeUserVisibleText(raw, isInjectedProvenance: true)
        #expect(out.text == "This is the actual model answer about error handling.")
        #expect(out.removedArtifacts.contains(.injectedFallbackPrefix))
        #expect(out.hadUnsafeLeakage)
    }

    @Test func recoveredUnsafeOutputCanBeConsumedWithRawOrSanitizedText() {
        let raw = "<think>x</think>safe"
        let sanitized = FinalOutputSanitizer.sanitizeUserVisibleText(raw)
        let recoveredByRaw = FinalOutputSanitizer.consumeRecoveredUnsafeOutput(forSanitizedText: raw)
        #expect(recoveredByRaw?.text == sanitized.text)
        #expect(recoveredByRaw?.removedArtifacts.contains(.thinkBlock) == true)

        _ = FinalOutputSanitizer.sanitizeUserVisibleText(raw)
        let recoveredBySanitized = FinalOutputSanitizer.consumeRecoveredUnsafeOutput(forSanitizedText: sanitized.text)
        #expect(recoveredBySanitized?.text == sanitized.text)
        #expect(recoveredBySanitized?.removedArtifacts.contains(.thinkBlock) == true)
    }

    @Test func handlesOutputThatBecomesEmpty() {
        let out = FinalOutputSanitizer.sanitizeUserVisibleText("<think>only</think>")
        #expect(out.removedArtifacts.contains(.emptyAfterSanitization))
        #expect(!out.text.isEmpty)
    }

    @Test func preservesMarkdownAndLinks() {
        let out = FinalOutputSanitizer.sanitizeUserVisibleText("Use **bold** and [link](https://example.com)")
        #expect(out.text.contains("**bold**"))
        #expect(out.text.contains("https://example.com"))
    }

    @Test func deterministicRepeatedCalls() {
        let raw = "<think>x</think> Answer"
        #expect(FinalOutputSanitizer.sanitizeUserVisibleText(raw) == FinalOutputSanitizer.sanitizeUserVisibleText(raw))
    }

    @Test func removesMalformedHiddenReasoningTags() {
        let thinkResponse = FinalOutputSanitizer.sanitizeUserVisibleText("<thinkresponse>secret</thinkresponse> visible")
        #expect(thinkResponse.text == "visible")
        #expect(thinkResponse.removedArtifacts.contains(.thinkBlock))

        let thinking = FinalOutputSanitizer.sanitizeUserVisibleText("<thinking>secret</thinking>\nAnswer")
        #expect(thinking.text == "Answer")

        let analysisOnly = FinalOutputSanitizer.sanitizeUserVisibleText("<analysis>secret")
        #expect(analysisOnly.text == FinalOutputSanitizer.fallback)
        #expect(analysisOnly.removedArtifacts.contains(.emptyAfterSanitization))

        let trailingReasoning = FinalOutputSanitizer.sanitizeUserVisibleText("Answer\n<reasoning>secret</reasoning>")
        #expect(trailingReasoning.text == "Answer")

        let thinkResponseUnderscore = FinalOutputSanitizer.sanitizeUserVisibleText("<think_response>secret</think_response>\nClean")
        #expect(thinkResponseUnderscore.text == "Clean")
    }

    @Test func sanitizedHiddenMarkerVariantsDoNotLeak() {
        let samples = [
            "<thinkresponse>secret</thinkresponse> visible",
            "<thinking>secret</thinking>\nAnswer",
            "<analysis>secret",
            "Answer\n<reasoning>secret</reasoning>",
            "<think_response>secret</think_response>\nClean",
            "<chain_of_thought>secret</chain_of_thought> Safe"
        ]

        for sample in samples {
            let lowered = FinalOutputSanitizer.sanitizeUserVisibleText(sample).text.lowercased()
            #expect(!lowered.contains("<think"))
            #expect(!lowered.contains("<analysis"))
            #expect(!lowered.contains("<reasoning"))
            #expect(!lowered.contains("<thinking"))
            #expect(!lowered.contains("<chain_of_thought"))
        }
    }

    @Test func modelOutputSanitizerRemovesHiddenReasoningVariants() {
        #expect(ModelOutputSanitizer.stripHiddenBlocks("<analysis>secret</analysis>\nAnswer") == "Answer")
        #expect(ModelOutputSanitizer.stripHiddenBlocks("<think_response>secret</think_response>\nClean") == "Clean")
        #expect(ModelOutputSanitizer.stripHiddenBlocks("<reasoning>secret") == "")
    }
}

extension FinalOutputSanitizerTests {
    @Test func streamingSanitizerWithholdsFallbackWhileHoldbackWindowIsEmpty() {
        var sanitizer = StreamingFinalOutputSanitizer()
        let delta = sanitizer.ingest("Short valid answer")
        #expect(delta.isEmpty)
        #expect(!delta.contains(FinalOutputSanitizer.fallback))

        let finalization = sanitizer.finish()
        switch finalization {
        case let .append(final, remainingDelta):
            #expect(final.text == "Short valid answer")
            #expect(remainingDelta == "Short valid answer")
            #expect(!remainingDelta.contains(FinalOutputSanitizer.fallback))
        case .replace:
            Issue.record("Expected append finalization when no partial text was emitted")
        }
    }

    @Test func streamingSanitizerNeverPrependsFallbackToValidDelayedOutput() {
        var sanitizer = StreamingFinalOutputSanitizer()
        let first = sanitizer.ingest("edge allows for more precise cutting and controlled application of force.")
        #expect(!first.contains(FinalOutputSanitizer.fallback))

        let finalization = sanitizer.finish()
        switch finalization {
        case let .append(final, remainingDelta):
            #expect(!final.text.contains(FinalOutputSanitizer.fallback))
            #expect(!remainingDelta.contains(FinalOutputSanitizer.fallback))
            #expect(final.text.hasPrefix("edge allows"))
        case let .replace(final):
            #expect(!final.text.contains(FinalOutputSanitizer.fallback))
            #expect(final.text.hasPrefix("edge allows"))
        }
    }

    @Test func streamingFinalizationUsesProvenanceToStripSanitizerFallbackPrefix() {
        var sanitizer = StreamingFinalOutputSanitizer()
        let unsafeOnlyPrefix = "<think>secret</think>" + String(repeating: " ", count: 220)
        let first = sanitizer.ingest(unsafeOnlyPrefix)
        let second = sanitizer.ingest("\(FinalOutputSanitizer.fallback) Real answer.")
        #expect(first.isEmpty)
        #expect(second.isEmpty)

        let finalization = sanitizer.finish()
        switch finalization {
        case let .append(final, remainingDelta):
            #expect(final.text == "Real answer.")
            #expect(remainingDelta == "Real answer.")
            #expect(final.removedArtifacts.contains(.injectedFallbackPrefix))
            #expect(final.removedArtifacts.contains(.thinkBlock))
        case let .replace(final):
            #expect(final.text == "Real answer.")
            #expect(final.removedArtifacts.contains(.injectedFallbackPrefix))
            #expect(final.removedArtifacts.contains(.thinkBlock))
        }
    }

    @Test func streamingFinalizationPreservesLegitimateFallbackPrefixedModelText() {
        var sanitizer = StreamingFinalOutputSanitizer()
        _ = sanitizer.ingest("\(FinalOutputSanitizer.fallback) Real answer.")

        let finalization = sanitizer.finish()
        switch finalization {
        case let .append(final, remainingDelta):
            #expect(final.text == "\(FinalOutputSanitizer.fallback) Real answer.")
            #expect(remainingDelta == final.text)
            #expect(!final.removedArtifacts.contains(.injectedFallbackPrefix))
        case let .replace(final):
            #expect(final.text == "\(FinalOutputSanitizer.fallback) Real answer.")
            #expect(!final.removedArtifacts.contains(.injectedFallbackPrefix))
        }
    }

    @Test func streamingSanitizerWithholdsSplitThinkMarker() {
        var sanitizer = StreamingFinalOutputSanitizer()
        let first = sanitizer.ingest("Hello <thi")
        let second = sanitizer.ingest("nk>secret</think> world")
        let finalization = sanitizer.finish()
        let final: SanitizedFinalOutput
        let remainingDelta: String
        switch finalization {
        case let .append(output, delta):
            final = output
            remainingDelta = delta
        case let .replace(output):
            final = output
            remainingDelta = ""
        }

        #expect(!first.lowercased().contains("<thi"))
        #expect(!second.lowercased().contains("think"))
        #expect(final.text == "Hello world")
        #expect(remainingDelta.isEmpty)
    }

    @Test func streamingSanitizerWithholdsSplitPayloadMarker() {
        var sanitizer = StreamingFinalOutputSanitizer()
        _ = sanitizer.ingest("Before <lumen_")
        let delta = sanitizer.ingest("web_payload>{\"kind\":\"searchResults\"}</lumen_web_payload> after")
        let finalization = sanitizer.finish()
        let final: SanitizedFinalOutput
        switch finalization {
        case let .append(output, _), let .replace(output):
            final = output
        }

        #expect(!delta.lowercased().contains("lumen_web_payload"))
        #expect(final.text == "Before after")
    }

    @Test func streamingSanitizerWithholdsSplitRawJSONPayload() {
        var sanitizer = StreamingFinalOutputSanitizer()
        let one = sanitizer.ingest("Result: {\"kind\":\"search")
        let two = sanitizer.ingest("Results\",\"results\":[{\"mediaKind\":\"page\"}]}")
        let finalization = sanitizer.finish()
        let final: SanitizedFinalOutput
        switch finalization {
        case let .append(output, _), let .replace(output):
            final = output
        }

        #expect(!one.lowercased().contains("searchresults"))
        #expect(!two.lowercased().contains("searchresults"))
        #expect(final.text == "Result:")
    }

    @Test func streamingFinalizationProvidesRemainingDeltaForWhitespaceNormalization() {
        var sanitizer = StreamingFinalOutputSanitizer()
        let streamed = sanitizer.ingest("Hello  <think>x</think>\n\nworld")
        let finalization = sanitizer.finish()
        switch finalization {
        case let .append(final, remainingDelta):
            #expect(streamed == "Hello")
            #expect(remainingDelta == " world")
            #expect(streamed + remainingDelta == final.text)
            #expect(final.text == "Hello world")
        case .replace:
            Issue.record("Expected append finalization for whitespace normalization case")
        }
    }
}
