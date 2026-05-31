import Foundation

nonisolated enum IntentClassifierPolicy {
    static func resolve(modelResult: IntentClassificationResult?, deterministic fallback: IntentClassificationResult) -> IntentClassificationResult {
        guard let modelResult else { return sanitized(fallback, source: .deterministicFallback) }
        if modelResult.confidence >= 0.72 {
            return sanitized(modelResult, source: .bundledModel)
        }
        if modelResult.confidence >= 0.50 {
            return resolveMediumConfidence(modelResult: modelResult, fallback: fallback)
        }
        return sanitized(fallback, source: .deterministicFallback)
    }

    private static func resolveMediumConfidence(modelResult: IntentClassificationResult, fallback: IntentClassificationResult) -> IntentClassificationResult {
        if modelResult.intent == fallback.intent {
            return sanitized(IntentClassificationResult(intent: modelResult.intent, confidence: min(0.99, modelResult.confidence + 0.05), alternatives: modelResult.alternatives, requiresClarification: modelResult.requiresClarification || fallback.requiresClarification, clarificationPrompt: modelResult.clarificationPrompt ?? fallback.clarificationPrompt, source: .policyMerged, diagnostics: "merged:agree"), source: .policyMerged)
        }
        if isApprovalSensitive(fallback.intent) {
            return sanitized(fallback, source: .policyMerged)
        }
        if isSemanticNonDestructive(modelResult.intent), modelResult.confidence >= 0.65 {
            return sanitized(IntentClassificationResult(intent: modelResult.intent, confidence: modelResult.confidence, alternatives: modelResult.alternatives, requiresClarification: modelResult.requiresClarification, clarificationPrompt: modelResult.clarificationPrompt, source: .policyMerged, diagnostics: "merged:model_preferred"), source: .policyMerged)
        }
        return sanitized(fallback, source: .policyMerged)
    }

    private static func sanitized(_ result: IntentClassificationResult, source: IntentClassificationResult.Source) -> IntentClassificationResult {
        let boundedConfidence = min(max(result.confidence, 0.0), 1.0)
        let primary = IntentAlternative(intent: result.intent, confidence: boundedConfidence)
        let candidates = result.withAllowedAlternatives().alternatives + [primary]
        let alternatives = candidates
            .filter { $0.confidence.isFinite }
            .sorted { $0.confidence > $1.confidence }
            .reduce(into: [IntentAlternative]()) { acc, item in
                if !acc.contains(where: { $0.intent == item.intent }) {
                    acc.append(item)
                }
            }
        return IntentClassificationResult(
            intent: result.intent,
            confidence: boundedConfidence,
            alternatives: Array(alternatives.prefix(5)),
            requiresClarification: result.requiresClarification,
            clarificationPrompt: result.clarificationPrompt,
            source: source,
            diagnostics: result.diagnostics
        )
    }

    private static func isApprovalSensitive(_ intent: UserIntent) -> Bool {
        switch intent {
        case .calendar, .emailDraft, .messageDraft, .phoneCall, .alarm, .trigger, .outlook:
            return true
        default:
            return false
        }
    }

    private static func isSemanticNonDestructive(_ intent: UserIntent) -> Bool {
        switch intent {
        case .weather, .maps, .webSearch, .rag, .memory, .files, .photos, .health, .motion, .camera:
            return true
        default:
            return false
        }
    }
}
