import Foundation

nonisolated enum DeterministicIntentFallback {
    static func classify(_ text: String) -> IntentClassificationResult {
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
}
