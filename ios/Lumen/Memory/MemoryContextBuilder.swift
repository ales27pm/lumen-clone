import Foundation
import SwiftData

struct MemoryContextResult {
    let selected: [MemoryItem]
    let totalChars: Int
    let totalTokens: Int
    let candidateCount: Int
    let reasons: [UUID: String]
    let sourceIDs: [UUID]
    let tierCounts: [String: Int]
    let hierarchyPassApplied: Bool
    let diagnostic: String?

    init(
        selected: [MemoryItem],
        totalChars: Int,
        totalTokens: Int? = nil,
        candidateCount: Int? = nil,
        reasons: [UUID: String],
        sourceIDs: [UUID],
        tierCounts: [String: Int] = [:],
        hierarchyPassApplied: Bool = false,
        diagnostic: String? = nil
    ) {
        self.selected = selected
        self.totalChars = max(0, totalChars)
        self.totalTokens = totalTokens ?? ContextBudgetAllocator.estimateTokens(forCharacterCount: max(0, totalChars))
        self.candidateCount = candidateCount ?? selected.count
        self.reasons = reasons
        self.sourceIDs = sourceIDs
        self.tierCounts = tierCounts
        self.hierarchyPassApplied = hierarchyPassApplied
        self.diagnostic = diagnostic
    }
}

enum MemoryContextBuilder {
    enum Tier: String, CaseIterable {
        case pinned
        case working
        case episodic
        case semantic

        var primaryLimit: Int {
            switch self {
            case .pinned: Int.max
            case .working: 2
            case .episodic: 2
            case .semantic: 4
            }
        }
    }

    @MainActor
    static func build(query: String, budgetChars: Int, context: ModelContext) -> MemoryContextResult {
        let budgetChars = max(0, budgetChars)
        let all: [MemoryItem]
        do {
            all = try context.fetch(FetchDescriptor<MemoryItem>())
        } catch {
            return MemoryContextResult(
                selected: [],
                totalChars: 0,
                candidateCount: 0,
                reasons: [:],
                sourceIDs: [],
                diagnostic: "fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            )
        }
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let hasQuery = !q.isEmpty
        let terms = queryTerms(q)
        let now = Date()
        let ranked = all
            .filter { !MemoryStore.isExpired($0, now: now) }
            .map { item in
                RankedMemory(item: item, tier: tier(for: item, now: now), score: score(item, q, terms: terms, hasQuery: hasQuery, now: now))
            }
            .sorted(by: rank)

        var picked: [RankedMemory] = []
        var chars = 0
        var tierCounts: [Tier: Int] = [:]
        var pickedIDs = Set<UUID>()

        func canFit(_ ranked: RankedMemory) -> Bool {
            chars + contextChars(for: ranked.item) <= budgetChars
        }

        func pick(_ ranked: RankedMemory) {
            picked.append(ranked)
            chars += contextChars(for: ranked.item)
            tierCounts[ranked.tier, default: 0] += 1
            pickedIDs.insert(ranked.item.id)
        }

        for m in ranked {
            guard canFit(m) else { continue }
            guard tierCounts[m.tier, default: 0] < m.tier.primaryLimit else { continue }
            pick(m)
        }

        if picked.count < ranked.count {
            for m in ranked where !pickedIDs.contains(m.item.id) {
                guard canFit(m) else { continue }
                pick(m)
            }
        }

        let selected = picked.map(\.item)
        var reasons: [UUID: String] = [:]
        for m in picked {
            let queryMatched = queryMatches(m.item, q: q, terms: terms, hasQuery: hasQuery)
            let base = m.item.isPinned ? "pinned" : (queryMatched ? "query-match" : "recency")
            reasons[m.item.id] = "\(m.tier.rawValue):\(base)"
        }

        return .init(
            selected: selected,
            totalChars: chars,
            candidateCount: all.count,
            reasons: reasons,
            sourceIDs: selected.map(\.id),
            tierCounts: Dictionary(uniqueKeysWithValues: Tier.allCases.map { ($0.rawValue, tierCounts[$0, default: 0]) }),
            hierarchyPassApplied: Set(picked.map(\.tier)).count > 1
        )
    }

    private static func score(_ m: MemoryItem, _ q: String, terms: [String], hasQuery: Bool, now: Date) -> Double {
        var s = 0.0
        if m.isPinned { s += 1.8 }
        if hasQuery && m.content.lowercased().contains(q) { s += 1.0 }
        if hasQuery && m.topic?.lowercased().contains(q) == true { s += 0.7 }
        s += termCoverage(item: m, terms: terms) * 0.9
        switch m.memoryKind {
        case .preference, .person, .project: s += 0.35
        case .conversation: s += 0.15
        case .fact: break
        }
        s += max(0, 0.3 - now.timeIntervalSince(m.createdAt)/(60*60*24*365))
        return s
    }

    private struct RankedMemory {
        let item: MemoryItem
        let tier: Tier
        let score: Double
    }

    private static func rank(_ lhs: RankedMemory, _ rhs: RankedMemory) -> Bool {
        if lhs.score != rhs.score { return lhs.score > rhs.score }
        if lhs.tier != rhs.tier { return tierPriority(lhs.tier) > tierPriority(rhs.tier) }
        return lhs.item.createdAt > rhs.item.createdAt
    }

    private static func tierPriority(_ tier: Tier) -> Int {
        switch tier {
        case .pinned: 4
        case .semantic: 3
        case .episodic: 2
        case .working: 1
        }
    }

    private static func tier(for item: MemoryItem, now: Date) -> Tier {
        if item.isPinned { return .pinned }
        if item.memoryKind == .conversation {
            return now.timeIntervalSince(item.createdAt) < 60 * 60 * 24 * 2 ? .working : .episodic
        }
        if item.freshnessClass == MemoryFreshnessClass.volatile.rawValue || item.freshnessClass == MemoryFreshnessClass.shortLived.rawValue {
            return .episodic
        }
        return .semantic
    }

    private static func contextChars(for item: MemoryItem) -> Int {
        min(220, item.content.count)
    }

    private static func queryMatches(_ item: MemoryItem, q: String, terms: [String], hasQuery: Bool) -> Bool {
        guard hasQuery else { return false }
        if item.content.lowercased().contains(q) || item.topic?.lowercased().contains(q) == true { return true }
        return termCoverage(item: item, terms: terms) > 0
    }

    private static func termCoverage(item: MemoryItem, terms: [String]) -> Double {
        guard !terms.isEmpty else { return 0 }
        let content = item.content.lowercased()
        let topic = item.topic?.lowercased() ?? ""
        let matches = terms.filter { content.contains($0) || topic.contains($0) }
        return Double(matches.count) / Double(terms.count)
    }

    private static func queryTerms(_ query: String) -> [String] {
        let stopwords: Set<String> = [
            "about", "after", "again", "avec", "been", "dans", "does", "from", "have", "into",
            "pour", "that", "this", "what", "when", "where", "with", "your",
            "alors", "avoir", "comme", "elle", "fait", "mais", "nous", "plus", "quoi",
            "sans", "sont", "tout", "vous"
        ]
        var seen = Set<String>()
        return query
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 3 && !stopwords.contains($0) }
            .filter { seen.insert($0).inserted }
    }
}
