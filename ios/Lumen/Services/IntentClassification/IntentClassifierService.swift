import Foundation

@MainActor
final class IntentClassifierService {
    static let shared = IntentClassifierService()
    private init() {}

    func route(_ text: String) async -> IntentRoutingDecision {
        await classify(text).asRoutingDecision()
    }

    func classify(_ text: String) async -> IntentClassificationResult {
        if let override = IntentRouter.priorityOverride(text) {
            return IntentClassificationResult(
                intent: override.intent,
                confidence: 0.99,
                alternatives: [IntentAlternative(intent: override.intent, confidence: 0.99)],
                requiresClarification: override.requiresClarification,
                clarificationPrompt: override.clarificationPrompt,
                source: .deterministicFallback,
                diagnostics: "deterministic_priority_override"
            )
        }
        let deterministic = DeterministicIntentFallback.classify(text)
        let modelResult = await BundledIntentClassifier.shared.classify(text)
        return IntentClassifierPolicy.resolve(modelResult: modelResult, deterministic: deterministic)
    }
}
