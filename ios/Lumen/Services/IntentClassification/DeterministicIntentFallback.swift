import Foundation

nonisolated enum DeterministicIntentFallback {
    static func classify(_ text: String) -> IntentClassificationResult {
        if let clarification = ambiguityClarification(text) {
            return clarification
        }

        let decision = IntentRouter.classify(text)
        let confidence: Double
        switch decision.intent {
        case .chat: confidence = 0.75
        case .unknown: confidence = 0.50
        default: confidence = decision.allowedToolIDs.isEmpty ? 0.50 : 0.90
        }
        return IntentClassificationResult(
            intent: decision.intent,
            confidence: confidence,
            alternatives: [IntentAlternative(intent: decision.intent, confidence: confidence)],
            requiresClarification: decision.requiresClarification,
            clarificationPrompt: decision.clarificationPrompt,
            source: .deterministicFallback,
            diagnostics: "deterministic"
        )
    }

    static func ambiguityClarification(_ text: String) -> IntentClassificationResult? {
        let value = normalized(text)
        guard !value.isEmpty else { return nil }

        if isAmbiguousMeetingLookup(value) {
            return clarification(
                prompt: "Do you mean a calendar event or a nearby meeting location?",
                alternatives: [.calendar, .maps]
            )
        }

        if isAmbiguousReferenceAction(value) {
            return clarification(
                prompt: "What would you like me to act on?",
                alternatives: [.unknown]
            )
        }

        return nil
    }

    private static func clarification(prompt: String, alternatives: [UserIntent]) -> IntentClassificationResult {
        let primary = alternatives.first ?? .unknown
        return IntentClassificationResult(
            intent: primary,
            confidence: 0.45,
            alternatives: alternatives.map { IntentAlternative(intent: $0, confidence: $0 == primary ? 0.45 : 0.40) },
            requiresClarification: true,
            clarificationPrompt: prompt,
            source: .deterministicFallback,
            diagnostics: "deterministic_ambiguity_clarification"
        )
    }

    private static func isAmbiguousMeetingLookup(_ text: String) -> Bool {
        let hasLookupVerb = containsAny(text, ["find", "show", "search", "check", "open", "where is", "where are"])
        let hasPersonalMeeting = containsAny(text, ["my meeting", "my appointment", "my event"])
        let lacksCalendarScope = !containsAny(text, ["calendar", "schedule", "event details"])
        let lacksLocalScope = !containsAny(text, ["near me", "nearby", "closest", "nearest", "directions", "address"])
        return hasLookupVerb && hasPersonalMeeting && lacksCalendarScope && lacksLocalScope
    }

    private static func isAmbiguousReferenceAction(_ text: String) -> Bool {
        let exact = [
            "book that", "book it", "schedule that", "schedule it",
            "cancel that", "cancel it", "send it", "send that",
            "do it", "do that", "that one", "this one"
        ]
        if exact.contains(text) { return true }

        let referenceTargets = [" it", " that", " this", " them"]
        let actionVerbs = ["book", "schedule", "cancel", "send", "move", "delete", "open", "call"]
        return actionVerbs.contains { verb in text.hasPrefix("\(verb) ") }
            && referenceTargets.contains { target in text.contains(target) }
            && !text.contains("http")
    }

    private static func normalized(_ text: String) -> String {
        text.lowercased()
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func containsAny(_ value: String, _ needles: [String]) -> Bool {
        needles.contains { value.contains($0) }
    }
}
