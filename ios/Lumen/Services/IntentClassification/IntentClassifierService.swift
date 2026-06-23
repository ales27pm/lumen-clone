import Foundation

final class IntentClassifierService: Sendable {
    static let shared = IntentClassifierService()
    private init() {}

    /// Determines the routing decision for the given text.
    ///
    /// If the classified intent is a web search and is appropriate for dynamic public lookup, the location access tool is automatically enabled.
    ///
    /// - Returns: The routing decision with the classified intent and allowed tools.
    func route(_ text: String) async -> IntentRoutingDecision {
        let result = await classify(text)
        let routing = result.asRoutingDecision()
        if result.intent == .webSearch, ToolRouteGuard.shouldUseWebSearchForDynamicPublicLookup(text) {
            return IntentRoutingDecision(
                intent: routing.intent,
                allowedToolIDs: routing.allowedToolIDs.union(["location.current"]),
                requiresClarification: routing.requiresClarification,
                clarificationPrompt: routing.clarificationPrompt
            )
        }
        return routing
    }

    /// Classifies the intent of the provided text.
    /// - Returns: An IntentClassificationResult with the determined intent and classification details.
    func classify(_ text: String) async -> IntentClassificationResult {
        if let ambiguity = DeterministicIntentFallback.ambiguityClarification(text) {
            return IntentClarificationPolicy.apply(ambiguity, to: text)
        }

        if let override = IntentRouter.priorityOverride(text) {
            let result = IntentClassificationResult(
                intent: override.intent,
                confidence: 0.99,
                alternatives: [IntentAlternative(intent: override.intent, confidence: 0.99)],
                requiresClarification: override.requiresClarification,
                clarificationPrompt: override.clarificationPrompt,
                source: .deterministicFallback,
                diagnostics: "deterministic_priority_override"
            )
            return IntentClarificationPolicy.apply(result, to: text)
        }
        let deterministic = DeterministicIntentFallback.classify(text)
        let modelResult = await BundledIntentClassifier.shared.classify(text)
        let resolved = IntentClassifierPolicy.resolve(modelResult: modelResult, deterministic: deterministic)
        return IntentClarificationPolicy.apply(resolved, to: text)
    }
}
