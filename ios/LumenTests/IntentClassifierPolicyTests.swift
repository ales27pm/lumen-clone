import Testing
@testable import Lumen

struct IntentClassifierPolicyTests {
    @Test func bundledUnavailableFallsBackToDeterministic() {
        let fallback = DeterministicIntentFallback.classify("explain this Swift code")
        let result = IntentClassifierPolicy.resolve(modelResult: nil, deterministic: fallback)
        #expect(result.intent == fallback.intent)
        #expect(result.source == .deterministicFallback)
    }

    @Test func highConfidenceWeatherBeatsWeakDeterministicChat() {
        let fallback = IntentClassificationResult(intent: .chat, confidence: 0.75, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .deterministicFallback, diagnostics: nil)
        let model = IntentClassificationResult(intent: .weather, confidence: 0.9, alternatives: [IntentAlternative(intent: .weather, confidence: 0.9)], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        let result = IntentClassifierPolicy.resolve(modelResult: model, deterministic: fallback)
        #expect(result.intent == .weather)
    }

    @Test func lowConfidenceFallsBackToDeterministic() {
        let fallback = DeterministicIntentFallback.classify("set alarm for 6")
        let model = IntentClassificationResult(intent: .weather, confidence: 0.2, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        let result = IntentClassifierPolicy.resolve(modelResult: model, deterministic: fallback)
        #expect(result.intent == fallback.intent)
    }

    @Test func mediumConfidenceSameIntentMerges() {
        let fallback = DeterministicIntentFallback.classify("weather here")
        let model = IntentClassificationResult(intent: .weather, confidence: 0.55, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        let result = IntentClassifierPolicy.resolve(modelResult: model, deterministic: fallback)
        #expect(result.source == .policyMerged)
        #expect(result.confidence > model.confidence)
    }

    @Test func writeIntentDisagreementPrefersDeterministic() {
        let fallback = DeterministicIntentFallback.classify("create an event tomorrow")
        let model = IntentClassificationResult(intent: .maps, confidence: 0.68, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        let result = IntentClassifierPolicy.resolve(modelResult: model, deterministic: fallback)
        #expect(result.intent == .calendar)
    }

    @Test func chatIntentHasNoAllowedTools() {
        let chat = IntentClassificationResult(intent: .chat, confidence: 0.9, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        #expect(chat.asRoutingDecision().allowedToolIDs.isEmpty)
    }

    @Test func selectedIntentNeverExposesToolsOutsideMapping() {
        let model = IntentClassificationResult(intent: .weather, confidence: 0.9, alternatives: [IntentAlternative(intent: .calendar, confidence: 0.2)], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        let resolved = IntentClassifierPolicy.resolve(modelResult: model, deterministic: DeterministicIntentFallback.classify("hello"))
        let routing = resolved.asRoutingDecision()
        #expect(routing.allowedToolIDs == IntentToolMapping.allowedToolIDs(for: .weather))
        #expect(!routing.allowedToolIDs.contains("calendar.create"))
    }

    @Test func policySanitizesConfidenceAndDeduplicatesAlternatives() {
        let model = IntentClassificationResult(
            intent: .weather,
            confidence: 1.7,
            alternatives: [
                IntentAlternative(intent: .weather, confidence: 0.8),
                IntentAlternative(intent: .weather, confidence: 0.6),
                IntentAlternative(intent: .maps, confidence: 0.2)
            ],
            requiresClarification: false,
            clarificationPrompt: nil,
            source: .bundledModel,
            diagnostics: nil
        )
        let resolved = IntentClassifierPolicy.resolve(modelResult: model, deterministic: DeterministicIntentFallback.classify("hi"))
        #expect(resolved.confidence == 1.0)
        let weatherCount = resolved.alternatives.filter { $0.intent == .weather }.count
        #expect(weatherCount == 1)
    }

    @Test func umbrellaPromptCanBeWeatherFromModel() {
        let model = IntentClassificationResult(intent: .weather, confidence: 0.82, alternatives: [], requiresClarification: false, clarificationPrompt: nil, source: .bundledModel, diagnostics: nil)
        let fallback = DeterministicIntentFallback.classify("should I bring an umbrella")
        let result = IntentClassifierPolicy.resolve(modelResult: model, deterministic: fallback)
        #expect(result.intent == .weather)
    }

    @Test func explainSwiftCodeStaysChatWhenModelUnavailable() {
        let fallback = DeterministicIntentFallback.classify("explain this Swift code")
        let result = IntentClassifierPolicy.resolve(modelResult: nil, deterministic: fallback)
        #expect(result.intent == .chat)
    }

    @MainActor
    @Test func priorityOverridesRunBeforeBundledModel() async {
        let result = await IntentClassifierService.shared.classify("Help me message Jordan with a complete ETA and apology.")
        #expect(result.intent == .messageDraft)
        #expect(result.diagnostics == "deterministic_priority_override")
    }

    @MainActor
    @Test func liveE2ERoutingRegressionsUsePriorityOverrides() async {
        let cases: [(String, UserIntent)] = [
            ("Draft a quick email update to Taylor about the delay and ask one question.", .emailDraft),
            ("Place a call to Alex from contacts.", .phoneCall),
            ("Remind me to call Alex tomorrow.", .reminder),
            ("Remind me to text Alex tomorrow.", .reminder),
            ("Remind me to email Sarah next week.", .reminder),
            ("Create a reminder to call the supplier.", .reminder),
            ("Open the camera and prepare to take a photo.", .camera),
            ("Show whether I was walking or driving recently.", .motion),
            ("Read this web URL: https://example.com.", .webSearch),
            ("Open and read architecture-notes.md.", .files),
            ("Save this note: prioritize bullet points.", .memory),
            ("Reindex local files for retrieval.", .rag),
            ("Refresh the file retrieval index.", .rag),
            ("Refresh the photo retrieval index.", .rag)
        ]

        for (prompt, expectedIntent) in cases {
            let result = await IntentClassifierService.shared.classify(prompt)
            #expect(result.intent == expectedIntent)
            #expect(result.diagnostics == "deterministic_priority_override")
        }
    }
}
