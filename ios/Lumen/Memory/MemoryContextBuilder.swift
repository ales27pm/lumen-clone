import Foundation
import SwiftData

struct MemoryContextResult: Sendable {
    let selected: [MemoryItem]
    let totalChars: Int
    let reasons: [UUID: String]
    let sourceIDs: [UUID]
}

enum MemoryContextBuilder {
    @MainActor
    static func build(query: String, budgetChars: Int, context: ModelContext) -> MemoryContextResult {
        let all = (try? context.fetch(FetchDescriptor<MemoryItem>())) ?? []
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let hasQuery = !q.isEmpty
        let now = Date()
        let ranked = all.sorted { a,b in
            score(a, q, hasQuery: hasQuery, now: now) > score(b, q, hasQuery: hasQuery, now: now)
        }
        var picked: [MemoryItem] = []
        var reasons: [UUID:String] = [:]
        var chars = 0
        for m in ranked {
            let c = min(220, m.content.count)
            if chars + c > budgetChars { continue }
            picked.append(m); chars += c
            let queryMatched = hasQuery && (m.content.lowercased().contains(q) || m.topic?.lowercased().contains(q) == true)
            reasons[m.id] = m.isPinned ? "pinned" : (queryMatched ? "query-match" : "recency")
        }
        return .init(selected: picked, totalChars: chars, reasons: reasons, sourceIDs: picked.map(\.id))
    }

    private static func score(_ m: MemoryItem, _ q: String, hasQuery: Bool, now: Date) -> Double {
        var s = 0.0
        if m.isPinned { s += 1.5 }
        if hasQuery && m.content.lowercased().contains(q) { s += 1 }
        if hasQuery && m.topic?.lowercased().contains(q) == true { s += 0.6 }
        s += max(0, 0.3 - now.timeIntervalSince(m.createdAt)/(60*60*24*365))
        return s
    }
}
