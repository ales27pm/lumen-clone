import Foundation
import SwiftData

nonisolated struct MemoryCascadeResult: Sendable {
    let ephemeral: [MemoryContextItem]
    let vectorized: [MemoryContextItem]
    let condensed: [MemoryContextItem]

    var promptFragments: [MemoryContextItem] {
        var fragments: [MemoryContextItem] = []
        if !ephemeral.isEmpty { fragments.append(contentsOf: ephemeral) }
        if !vectorized.isEmpty { fragments.append(contentsOf: vectorized) }
        if !condensed.isEmpty { fragments.append(contentsOf: condensed) }
        return fragments
    }
}

@MainActor
enum MemoryCascade {
    static func recall(
        query: String,
        history: [(role: MessageRole, content: String)],
        context: ModelContext
    ) async -> MemoryCascadeResult {
        let tier1 = history
            .suffix(12)
            .compactMap { item -> MemoryContextItem? in
                let compact = compactAndTrim(item.content, maxLength: 260)
                guard !compact.isEmpty else { return nil }
                return MemoryContextItem(content: compact, scope: .currentTurn, authority: .referenceOnly, createdAt: nil, expiresAt: nil, source: "tier1-ephemeral", topic: nil)
            }

        let tier2 = await MemoryStore.recall(query: query, context: context, limit: 8)
            .filter { $0.source != "rem-condensed" }
            .compactMap { item -> MemoryContextItem? in
                let compact = compactAndTrim(item.content, maxLength: 260)
                guard !compact.isEmpty else { return nil }
                return MemoryContextItem(content: compact, scope: .conversation, authority: .referenceOnly, createdAt: item.createdAt, expiresAt: item.expiresAt, source: item.source, topic: item.topic)
            }

        let queryTokens = tokenSet(query)
        let condensedDescriptor = FetchDescriptor<MemoryItem>(
            predicate: #Predicate<MemoryItem> { $0.source == "rem-condensed" }
        )
        let condensedItems = ((try? context.fetch(condensedDescriptor)) ?? []).filter { item in
            MemoryStore.migrateExpiryIfNeeded(for: item)
            return !MemoryStore.isExpired(item)
        }

        let rankedTier3 = condensedItems
            .map { item in
                (
                    item: item,
                    overlap: overlapScore(tokens: queryTokens, content: item.content)
                )
            }
            .sorted { lhs, rhs in
                if lhs.overlap != rhs.overlap { return lhs.overlap > rhs.overlap }
                return lhs.item.createdAt > rhs.item.createdAt
            }
            .prefix(8)
            .compactMap { pair -> MemoryContextItem? in
                let compact = compactAndTrim(pair.item.content, maxLength: 260)
                guard !compact.isEmpty else { return nil }
                return MemoryContextItem(content: compact, scope: .remCondensed, authority: .backgroundOnly, createdAt: pair.item.createdAt, expiresAt: pair.item.expiresAt, source: pair.item.source, topic: pair.item.topic)
            }

        return MemoryCascadeResult(
            ephemeral: tier1,
            vectorized: tier2,
            condensed: Array(rankedTier3)
        )
    }


    static func condenseIfNeeded(context: ModelContext, minimumCount: Int = 24) async throws {
        let descriptor = FetchDescriptor<MemoryItem>(
            sortBy: [SortDescriptor(\.createdAt, order: .forward)]
        )
        let allItems = try context.fetch(descriptor)

        let candidates = allItems.filter { item in
            MemoryStore.migrateExpiryIfNeeded(for: item)
            guard item.source != "rem-condensed" else { return false }
            guard item.isPinned == false else { return false }
            guard !MemoryStore.isExpired(item) else { return false }
            return !item.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }

        guard candidates.count >= minimumCount else { return }

        let grouped = Dictionary(grouping: candidates) { item -> String in
            let topic = item.topic?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !topic.isEmpty {
                return topic
            }
            return "kind:\(item.kind)"
        }

        for groupKey in grouped.keys.sorted() {
            guard let groupItems = grouped[groupKey], groupItems.count >= 6 else { continue }

            let sortedItems = groupItems.sorted { lhs, rhs in
                if lhs.createdAt != rhs.createdAt { return lhs.createdAt < rhs.createdAt }
                if lhs.content != rhs.content { return lhs.content < rhs.content }
                return lhs.id.uuidString < rhs.id.uuidString
            }
            let newestItems = Array(sortedItems.suffix(12))
            let newestSourceDate = newestItems.last?.createdAt ?? .distantPast
            let cascadeTopic = "cascade:\(groupKey)"

            let existingDescriptor = FetchDescriptor<MemoryItem>(
                predicate: #Predicate<MemoryItem> {
                    $0.source == "rem-condensed" && $0.topic == cascadeTopic
                },
                sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
            )
            if let newestCondensed = try context.fetch(existingDescriptor).first,
               newestCondensed.createdAt > newestSourceDate {
                continue
            }

            var seenLines: Set<String> = []
            let extracts = newestItems.compactMap { item -> String? in
                let compact = compactAndTrim(item.content, maxLength: 140)
                guard !compact.isEmpty else { return nil }
                let dedupeKey = compact.lowercased()
                guard !seenLines.contains(dedupeKey) else { return nil }
                seenLines.insert(dedupeKey)
                return compact
            }

            guard !extracts.isEmpty else { continue }
            let summaryBody = extracts.enumerated().map { index, line in
                "\(index + 1). \(line)"
            }.joined(separator: " ")
            let summary = "Background (low-authority) condensed \(groupKey): \(summaryBody)"

            try await MemoryStore.remember(
                summary,
                kind: .conversation,
                source: "rem-condensed",
                topic: cascadeTopic,
                context: context
            )

            let saved = try context.fetch(existingDescriptor)
            if let latest = saved.first {
                latest.freshnessClass = MemoryFreshnessClass.durable.rawValue
                latest.expiresAt = nil
            }
        }

        try context.save()
    }

    private static func compactAndTrim(_ text: String, maxLength: Int) -> String {
        let normalized = text
            .replacingOccurrences(of: "\n", with: " ")
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !normalized.isEmpty else { return "" }
        if normalized.count <= maxLength { return normalized }
        let end = normalized.index(normalized.startIndex, offsetBy: maxLength)
        return String(normalized[..<end]).trimmingCharacters(in: .whitespacesAndNewlines) + "…"
    }

    private static func tokenSet(_ text: String) -> Set<String> {
        let lower = text.lowercased()
        var tokens: Set<String> = []
        var current = ""

        for scalar in lower.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) {
                current.unicodeScalars.append(scalar)
            } else if !current.isEmpty {
                if current.count >= 2 {
                    tokens.insert(current)
                }
                current.removeAll(keepingCapacity: true)
            }
        }

        if current.count >= 2 {
            tokens.insert(current)
        }

        return tokens
    }

    private static func overlapScore(tokens queryTokens: Set<String>, content: String) -> Int {
        guard !queryTokens.isEmpty else { return 0 }
        let memoryTokens = tokenSet(content)
        return queryTokens.intersection(memoryTokens).count
    }
}
