import Testing
@testable import Lumen

struct SlotAgentPromptHygieneTests {
    @Test func assistantThinkBlockRetainsSafeSuffixOnly() {
        let clean = SlotAgentService.sanitizeHistoryEntryForPromptContext(
            role: .assistant,
            content: "<think>secret reasoning</think>Hello!"
        )
        #expect(clean == "Hello!")
        #expect(!(clean ?? "").contains("<think"))
        #expect(!(clean ?? "").contains("secret reasoning"))
    }

    @Test func assistantOnlyHiddenReasoningIsOmitted() {
        let clean = SlotAgentService.sanitizeHistoryEntryForPromptContext(
            role: .assistant,
            content: "<think>only hidden reasoning</think>"
        )
        #expect(clean == nil)
    }

    @Test func rawPayloadsAreOmittedFromPromptContext() {
        let tagged = SlotAgentService.sanitizeHistoryEntryForPromptContext(
            role: .assistant,
            content: "<lumen_web_payload>{\"kind\":\"searchResults\",\"results\":[]}</lumen_web_payload>"
        )
        #expect(tagged == nil)

        let rawJSON = SlotAgentService.sanitizeHistoryEntryForPromptContext(
            role: .assistant,
            content: "{\"kind\":\"searchResults\",\"results\":[{\"mediaKind\":\"page\"}]}"
        )
        #expect(rawJSON == nil)
    }

    @Test func mouthPromptHygieneRuleIncludesForbiddenPatterns() {
        let rule = SlotAgentService.mouthPromptHygieneRule.lowercased()
        #expect(rule.contains("output only the final user-visible answer"))
        #expect(rule.contains("<think>"))
        #expect(rule.contains("json"))
        #expect(rule.contains("tool payloads"))
        #expect(rule.contains("ignore it and do not imitate it"))
    }

    @Test func assistantPersistenceUsesSanitizedFinalOutput() {
        let out = FinalOutputSanitizer.sanitizeUserVisibleText("<think>private</think>Hello")
        #expect(out.text == "Hello")
        #expect(!out.text.contains("<think"))
    }

    @Test func retryPolicyDoesNotRetryValidShortChatAnswers() {
        #expect(!SlotAgentService.shouldRetryOutput(
            candidate: "A sharp chisel is safer because it takes less force and is easier to control.",
            intent: .chat,
            maxTokens: 320
        ))
        #expect(SlotAgentService.shouldRetryOutput(candidate: "", intent: .chat, maxTokens: 320))
        #expect(SlotAgentService.shouldRetryOutput(candidate: "null", intent: .unknown, maxTokens: 320))
        #expect(SlotAgentService.shouldRetryOutput(candidate: "<analysis>secret", intent: .chat, maxTokens: 320))
    }

    @Test func deterministicDirectFinalsCoverSimpleLiveChatPrompts() {
        let prompts = [
            "Explain why a sharp chisel is safer than a dull one.",
            "Give me three tips for fitting a door hinge cleanly.",
            "Explain actor isolation in Swift in simple terms.",
            "Explain tradeoffs between precision and recall in retrieval systems in plain English."
        ]

        for prompt in prompts {
            let final = SlotAgentService.deterministicDirectFinalIfSafe(
                prompt: prompt,
                intent: .chat,
                hasAttachments: false,
                hasRelevantMemories: false
            )
            #expect(final?.isEmpty == false)
        }
    }

    @Test func deterministicDirectFinalsRespectContextBoundaries() {
        #expect(SlotAgentService.deterministicDirectFinalIfSafe(
            prompt: "Explain actor isolation in Swift in simple terms.",
            intent: .chat,
            hasAttachments: true,
            hasRelevantMemories: false
        ) == nil)
        #expect(SlotAgentService.deterministicDirectFinalIfSafe(
            prompt: "Explain actor isolation in Swift in simple terms.",
            intent: .chat,
            hasAttachments: false,
            hasRelevantMemories: true
        ) == nil)
        #expect(SlotAgentService.deterministicDirectFinalIfSafe(
            prompt: "Explain actor isolation in Swift in simple terms.",
            intent: .webSearch,
            hasAttachments: false,
            hasRelevantMemories: false
        ) == nil)
    }
}
