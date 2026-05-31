import Foundation

struct RAGContextResult: Sendable {
    let selected: [RAGRetrievalResult]
    let totalChars: Int
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
        return .init(selected: picked, totalChars: chars)
    }
}
