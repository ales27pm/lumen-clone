import Foundation

struct ContextQueryRewriteResult: Sendable, Equatable {
    let query: String
    let addedTerms: [String]

    var expansionApplied: Bool { !addedTerms.isEmpty }
    var estimatedTokens: Int { ContextBudgetAllocator.estimateTokens(for: query) }
}

enum ContextQueryRewriter {
    private static let maxAddedTerms = 12
    private static let maxQueryChars = 320
    private static let minTermLength = 3

    static func rewrite(
        userInput: String,
        history: [(role: MessageRole, content: String)] = [],
        relevantMemories: [MemoryContextItem] = []
    ) -> ContextQueryRewriteResult {
        let base = normalized(userInput)
        var seen = Set(salientTerms(in: base))
        var added: [String] = []

        func addTerms(from text: String) {
            guard added.count < maxAddedTerms else { return }
            for term in salientTerms(in: text) {
                guard added.count < maxAddedTerms else { break }
                guard seen.insert(term).inserted else { continue }
                added.append(term)
            }
        }

        for item in relevantMemories.prefix(3) {
            if let topic = item.topic { addTerms(from: topic) }
            addTerms(from: item.content)
        }

        for turn in history.suffix(4).reversed() where turn.role != .tool {
            addTerms(from: turn.content)
        }

        let expanded = ([base] + added).filter { !$0.isEmpty }.joined(separator: " ")
        return .init(query: bounded(expanded), addedTerms: added)
    }

    private static func normalized(_ text: String) -> String {
        text
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func salientTerms(in text: String) -> [String] {
        let stopwords: Set<String> = [
            "about", "after", "again", "avec", "been", "dans", "does", "from", "have", "into",
            "pour", "that", "this", "what", "when", "where", "with", "your",
            "alors", "avoir", "comme", "elle", "fait", "mais", "nous", "plus", "quoi",
            "sans", "sont", "tout", "vous"
        ]
        var seen = Set<String>()
        return text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= minTermLength && !stopwords.contains($0) }
            .filter { seen.insert($0).inserted }
    }

    private static func bounded(_ text: String) -> String {
        guard text.count > maxQueryChars else { return text }
        let end = text.index(text.startIndex, offsetBy: maxQueryChars)
        let prefix = text[..<end]
        if let lastSpace = prefix.lastIndex(where: { $0.isWhitespace }) {
            return String(prefix[..<lastSpace]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return String(prefix).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
