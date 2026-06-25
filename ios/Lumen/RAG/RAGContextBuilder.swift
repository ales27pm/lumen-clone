import Foundation

struct RAGContextResult: Sendable {
    let selected: [RAGRetrievalResult]
    let totalChars: Int
    let totalTokens: Int
    let candidateCount: Int
    let selectedSourceCount: Int
    let diversityPassApplied: Bool
    let confidence: Double

    init(
        selected: [RAGRetrievalResult],
        totalChars: Int,
        totalTokens: Int? = nil,
        candidateCount: Int? = nil,
        selectedSourceCount: Int? = nil,
        diversityPassApplied: Bool = false,
        confidence: Double? = nil
    ) {
        self.selected = selected
        self.totalChars = max(0, totalChars)
        self.totalTokens = totalTokens ?? ContextBudgetAllocator.estimateTokens(forCharacterCount: max(0, totalChars))
        self.candidateCount = candidateCount ?? selected.count
        self.selectedSourceCount = selectedSourceCount ?? Set(selected.map(\.source.id)).count
        self.diversityPassApplied = diversityPassApplied
        self.confidence = confidence ?? Self.confidence(for: selected)
    }

    private static func confidence(for selected: [RAGRetrievalResult]) -> Double {
        guard let topScore = selected.map(\.score).max() else { return 0 }
        return min(max(topScore, 0), 1)
    }
}

enum RAGContextBuilder {
    private static let primaryChunksPerSource = 2

    static func build(results: [RAGRetrievalResult], budgetChars: Int) -> RAGContextResult {
        let budgetChars = max(0, budgetChars)
        var chars = 0
        var seen = Set<String>()
        var sourceCounts: [String: Int] = [:]
        var picked: [RAGRetrievalResult] = []
        let sorted = results.sorted(by: sortByScoreThenExcerpt)
        let deduped = sorted.filter { r in
            let key = "\(r.source.id)#\(r.chunkID.uuidString)"
            guard !seen.contains(key) else { return false }
            seen.insert(key)
            return true
        }

        func canFit(_ r: RAGRetrievalResult) -> Bool {
            let c = r.excerpt.count
            return chars + c <= budgetChars
        }

        func pick(_ r: RAGRetrievalResult) {
            picked.append(r)
            chars += r.excerpt.count
            sourceCounts[r.source.id, default: 0] += 1
        }

        for r in deduped {
            guard canFit(r) else { continue }
            guard sourceCounts[r.source.id, default: 0] < primaryChunksPerSource else { continue }
            pick(r)
        }

        if picked.count < deduped.count {
            let pickedIDs = Set(picked.map(\.chunkID))
            for r in deduped where !pickedIDs.contains(r.chunkID) {
                guard canFit(r) else { continue }
                pick(r)
            }
        }

        let sourceCount = Set(picked.map(\.source.id)).count
        return .init(
            selected: picked,
            totalChars: chars,
            candidateCount: results.count,
            selectedSourceCount: sourceCount,
            diversityPassApplied: sourceCount > 1,
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
        let sourceSignal = min(Double(Set(selected.map(\.source.id)).count) / 3.0, 1.0)
        return (topScoreSignal * 0.64) + (coverageSignal * 0.24) + (sourceSignal * 0.12)
    }

    private static func sortByScoreThenExcerpt(_ lhs: RAGRetrievalResult, _ rhs: RAGRetrievalResult) -> Bool {
        if lhs.score == rhs.score {
            return lhs.excerpt.count > rhs.excerpt.count
        }
        return lhs.score > rhs.score
    }
}
