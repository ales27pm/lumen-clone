import Foundation
import SwiftData

nonisolated enum RAGSourceScope: String, CaseIterable, Sendable {
    case all
    case documents
    case notes
    case photos

    var sourceTypes: Set<RAGSourceType>? {
        switch self {
        case .all:
            nil
        case .documents:
            [.file, .pdf, .note]
        case .notes:
            [.note]
        case .photos:
            [.photo]
        }
    }

    static func inferred(fromUserPrompt prompt: String) -> RAGSourceScope? {
        let lower = prompt.lowercased()
        if ["my files", "local files"].contains(where: lower.contains) {
            return .documents
        }
        if ["my notes", "local notes"].contains(where: lower.contains) {
            return .notes
        }
        if ["my photos", "photo library", "photo metadata", "pictures"].contains(where: lower.contains) {
            return .photos
        }
        if ["documents", "document", "report", "pdf"].contains(where: lower.contains) {
            return .documents
        }
        if lower.contains("architecture notes") {
            return .notes
        }
        return nil
    }
}

struct RAGMaintenanceResult: Sendable, Equatable {
    let success: Bool
    let metricSummary: String
}

@MainActor
final class RAGEngine {
    private let indexer = RAGIndexer()

    struct RetrieveResult {
        let results: [RAGRetrievalResult]
        let mode: String
        let diagnostic: String?
    }

    func retrieve(query: String, limit: Int, context: ModelContext) async -> [RAGRetrievalResult] {
        await retrieveWithDiagnostics(query: query, limit: limit, context: context).results
    }

    func retrieveWithDiagnostics(
        query: String,
        relevanceQuery: String? = nil,
        limit: Int,
        sourceTypes: Set<RAGSourceType>? = nil,
        context: ModelContext
    ) async -> RetrieveResult {
        let boundedLimit = max(0, limit)
        guard boundedLimit > 0 else {
            return RetrieveResult(results: [], mode: "empty_limit", diagnostic: "empty_limit")
        }

        let candidateLimit = min(max(boundedLimit * 3, boundedLimit + 8), 60)
        let search = await RAGStore.searchWithDiagnostics(
            query: query,
            context: context,
            limit: candidateLimit,
            sourceTypes: sourceTypes
        )
        let relevanceQuery = relevanceQuery ?? query
        let mapped = search.matches.compactMap { item -> RAGRetrievalResult? in
            let ref = item.chunk.sourceRef ?? item.chunk.id.uuidString
            let rerankedScore = Self.rerankedScore(query: query, chunk: item.chunk, baseScore: item.score)
            guard Self.isPostRerankRelevant(
                query: relevanceQuery,
                content: item.chunk.content,
                title: item.chunk.sourceName,
                rerankedScore: rerankedScore
            ) else {
                return nil
            }
            let excerpt = Self.focusedExcerpt(query: relevanceQuery, content: item.chunk.content, maxLength: 260)
            return RAGRetrievalResult(
                chunkID: item.chunk.id,
                source: .init(id: ref, type: item.chunk.sourceType, title: item.chunk.sourceName, ref: item.chunk.sourceRef),
                excerpt: excerpt.text,
                score: rerankedScore,
                retrievalMode: "\(search.mode)+local_rerank",
                offsetStart: excerpt.offsetStart,
                offsetEnd: excerpt.offsetEnd
            )
        }
        var seen = Set<String>()
        let results = mapped
            .sorted { lhs, rhs in
                if lhs.score == rhs.score {
                    return lhs.excerpt.count > rhs.excerpt.count
                }
                return lhs.score > rhs.score
            }
            .filter { result in
                let key = "\(result.source.id)#\(result.chunkID.uuidString)"
                if seen.contains(key) { return false }
                seen.insert(key)
                return true
            }
            .prefix(boundedLimit)
            .map { $0 }
        let mode = results.first?.retrievalMode ?? search.mode
        return RetrieveResult(results: results, mode: mode, diagnostic: search.diagnostic)
    }

    func buildContext(query: String, budget: Int, context: ModelContext) async -> RAGContextResult {
        let r = await retrieve(query: query, limit: 20, context: context)
        return RAGContextBuilder.build(results: r, budgetChars: budget)
    }

    func index(source: RAGSource, title: String, text: String, metadata: [String:String], context: ModelContext) async throws -> Int {
        try await indexer.indexText(source: source, title: title, text: text, metadata: metadata, context: context)
    }

    func maintenance(context: ModelContext) async -> RAGMaintenanceResult {
        do {
            let stagingPrefix = RAGChunk.replacementStagingSourceTypePrefix
            var descriptor = FetchDescriptor<RAGChunk>(
                predicate: #Predicate<RAGChunk> { !$0.sourceType.starts(with: stagingPrefix) }
            )
            descriptor.fetchLimit = 1
            let hasChunks = try !context.fetch(descriptor).isEmpty
            return .init(success: true, metricSummary: hasChunks ? "maintenance_success_work_done" : "maintenance_success_empty")
        } catch {
            return .init(success: false, metricSummary: "maintenance_failed")
        }
    }

    nonisolated private static func rerankedScore(query: String, chunk: RAGChunk, baseScore: Double, now: Date = Date()) -> Double {
        let terms = queryTerms(query)
        guard !terms.isEmpty else { return clamped(baseScore) }

        let content = chunk.content.lowercased()
        let title = chunk.sourceName.lowercased()
        let matchedTerms = terms.filter { content.contains($0) || title.contains($0) }
        let lexicalCoverage = Double(matchedTerms.count) / Double(terms.count)
        let titleCoverage = Double(terms.filter { title.contains($0) }.count) / Double(terms.count)
        let recency = max(0, 0.08 - now.timeIntervalSince(chunk.createdAt) / (60 * 60 * 24 * 365 * 4))

        return clamped((baseScore * 0.70) + (lexicalCoverage * 0.22) + (titleCoverage * 0.08) + recency)
    }

    /// Applies the final relevance gate after local reranking.
    /// Low-confidence candidates must share a meaningful lexical anchor with the query.
    nonisolated static func isPostRerankRelevant(
        query: String,
        content: String,
        title: String,
        rerankedScore: Double
    ) -> Bool {
        guard rerankedScore.isFinite, rerankedScore >= 0 else { return false }
        let terms = queryTerms(query)
        let highConfidenceSemanticThreshold = 0.50

        // Strong semantic evidence may not share literal words with the query.
        if rerankedScore >= highConfidenceSemanticThreshold { return true }

        // Preserve acronyms embedded in ordinary requests without treating a
        // substring inside a longer word (for example, "ui" in "build") as evidence.
        if hasDirectShortQueryMatch(query: query, content: content, title: title) {
            return true
        }
        guard !terms.isEmpty else { return false }

        // Lower-confidence candidates need a meaningful lexical anchor to fail closed.
        let genericRetrievalTerms: Set<String> = [
            "and", "document", "documents", "file", "files", "find", "for", "key", "latest",
            "local", "look", "note", "notes", "please", "report", "search", "show", "summarize",
            "summary", "tell", "the", "use", "using"
        ]
        let anchors = terms.filter { !genericRetrievalTerms.contains($0) }
        guard !anchors.isEmpty else { return false }

        let candidateTerms = Set(
            "\(title) \(content)".lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { $0.count >= 3 }
        )
        return anchors.contains { anchor in
            candidateTerms.contains { candidate in
                tokensShareLexicalAnchor(anchor, candidate)
            }
        }
    }

    /// Expands the retrieval query while allowing callers to keep the untouched
    /// input as the relevance anchor.
    nonisolated static func expandedSearchQuery(_ query: String) -> String {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()
        let expansionTerms = ["architecture", "module", "service", "component", "package"]
        guard expansionTerms.contains(where: { lower.contains($0) }) || lower.contains("design") || lower.contains("system") else {
            return trimmed
        }

        var expanded = trimmed
        for term in expansionTerms where !lower.contains(term) {
            expanded += " \(term)"
        }
        return expanded
    }

    nonisolated private static func focusedExcerpt(query: String, content: String, maxLength: Int) -> (text: String, offsetStart: Int?, offsetEnd: Int?) {
        let boundedMax = max(64, maxLength)
        guard content.count > boundedMax else {
            return (content.trimmingCharacters(in: .whitespacesAndNewlines), 0, content.count)
        }

        let lower = content.lowercased()
        let terms = queryTerms(query)
        let firstMatch = terms
            .compactMap { term -> String.Index? in lower.range(of: term)?.lowerBound }
            .min()

        guard let matchIndex = firstMatch else {
            let end = content.index(content.startIndex, offsetBy: boundedMax, limitedBy: content.endIndex) ?? content.endIndex
            return (String(content[..<end]).trimmingCharacters(in: .whitespacesAndNewlines), 0, content.distance(from: content.startIndex, to: end))
        }

        let halfWindow = boundedMax / 2
        let matchOffset = content.distance(from: content.startIndex, to: matchIndex)
        let startOffset = max(0, matchOffset - halfWindow)
        let endOffset = min(content.count, startOffset + boundedMax)
        let start = content.index(content.startIndex, offsetBy: startOffset)
        let end = content.index(content.startIndex, offsetBy: endOffset)
        var excerpt = String(content[start..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
        if startOffset > 0 { excerpt = "... " + excerpt }
        if endOffset < content.count { excerpt += " ..." }
        return (excerpt, startOffset, endOffset)
    }

    nonisolated private static func queryTerms(_ query: String) -> [String] {
        let stopwords: Set<String> = [
            "about", "after", "avec", "dans", "from", "have", "into", "pour", "that", "this", "what", "when", "where", "with"
        ]
        var seen = Set<String>()
        return query.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 3 && !stopwords.contains($0) }
            .filter { seen.insert($0).inserted }
    }

    nonisolated private static func hasDirectShortQueryMatch(query: String, content: String, title: String) -> Bool {
        let ignoredTerms: Set<String> = [
            "about", "after", "and", "document", "documents", "file", "files", "find", "for", "from",
            "have", "into", "local", "look", "note", "notes", "please", "pour", "report", "search",
            "show", "summarize", "summary", "tell", "that", "the", "this", "use", "using", "what",
            "when", "where", "with", "avec", "dans"
        ]
        let shortStopwords: Set<String> = [
            "a", "an", "as", "at", "be", "by", "do", "go", "he", "i", "if", "in", "is", "it",
            "la", "le", "me", "my", "of", "on", "or", "so", "to", "up", "us", "we"
        ]
        let rawTerms = lexicalTokens(query).filter {
            $0.count < 3 && !ignoredTerms.contains($0) && !shortStopwords.contains($0)
        }
        guard !rawTerms.isEmpty else { return false }

        let candidateTerms = Set(lexicalTokens(title) + lexicalTokens(content))
        return rawTerms.contains(where: candidateTerms.contains)
    }

    nonisolated private static func lexicalTokens(_ value: String) -> [String] {
        value.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }

    nonisolated private static func tokensShareLexicalAnchor(_ anchor: String, _ candidate: String) -> Bool {
        if anchor == candidate { return true }

        let anchorVariants = inflectionVariants(anchor)
        let candidateVariants = inflectionVariants(candidate)
        if !anchorVariants.isDisjoint(with: candidateVariants) { return true }

        let shorter = anchor.count <= candidate.count ? anchor : candidate
        let longer = anchor.count <= candidate.count ? candidate : anchor
        return shorter.count >= 5 && longer.hasPrefix(shorter)
    }

    nonisolated private static func inflectionVariants(_ token: String) -> Set<String> {
        var variants: Set<String> = [token]
        if token.count > 3, token.hasSuffix("s") {
            variants.insert(String(token.dropLast()))
        }
        if token.count > 4, token.hasSuffix("ies") {
            variants.insert(String(token.dropLast(3)) + "y")
        }
        return variants
    }

    nonisolated private static func clamped(_ value: Double) -> Double {
        min(1, max(0, value))
    }
}
