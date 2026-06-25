import Foundation

struct RAGContextResult: Sendable {
    let selected: [RAGRetrievalResult]
    let totalChars: Int
    let totalTokens: Int
    let candidateCount: Int
    let confidence: Double

    init(
        selected: [RAGRetrievalResult],
        totalChars: Int,
        totalTokens: Int? = nil,
        candidateCount: Int? = nil,
        confidence: Double? = nil
    ) {
        self.selected = selected
        self.totalChars = max(0, totalChars)
        self.totalTokens = totalTokens ?? ContextBudgetAllocator.estimateTokens(forCharacterCount: max(0, totalChars))
        self.candidateCount = candidateCount ?? selected.count
        self.confidence = confidence ?? Self.confidence(for: selected)
    }

    private static func confidence(for selected: [RAGRetrievalResult]) -> Double {
        guard let topScore = selected.map(\.score).max() else { return 0 }
        return min(max(topScore, 0), 1)
    }
}

enum RAGContextBuilder {
    static func build(results: [RAGRetrievalResult], budgetChars: Int) -> RAGContextResult {
        var picked:[RAGRetrievalResult] = []; var chars = 0; var seen = Set<String>()
        for r in results.sorted(by: { $0.score > $1.score }) {
            let key = "\(r.source.id)#\(r.chunkID.uuidString)"
            guard !seen.contains(key) else { continue }
            let c = r.excerpt.count
            if chars + c > budgetChars { continue }
            seen.insert(key); picked.append(r); chars += c
        }
        return .init(
            selected: picked,
            totalChars: chars,
            candidateCount: results.count,
            confidence: confidence(for: picked)
        )
    }

    static func build(results: [RAGRetrievalResult], budgetTokens: Int) -> RAGContextResult {
        build(results: results, budgetChars: max(0, budgetTokens) * ContextBudgetAllocator.defaultCharsPerToken)
    }

    private static func confidence(for selected: [RAGRetrievalResult]) -> Double {
        guard let topScore = selected.map(\.score).max() else { return 0 }
        let topScoreSignal = min(max(topScore, 0), 1)
        let coverageSignal = min(Double(selected.count) / 4.0, 1.0)
        return (topScoreSignal * 0.7) + (coverageSignal * 0.3)
    }
}
