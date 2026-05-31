import Foundation
import SwiftData

@MainActor
enum MemoryRecall {
    static func recallAndNormalize(
        query: String,
        routing: IntentRoutingDecision,
        context: ModelContext,
        limit: Int = 8
    ) async -> [MemoryContextItem] {
        let rawItems = await MemoryStore.recall(query: query, context: context, limit: limit)
        return rawItems.compactMap { item in
            MemoryStore.migrateExpiryIfNeeded(for: item)
            guard !MemoryStore.isExpired(item) else { return nil }
            let content = item.content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !content.isEmpty else { return nil }
            return MemoryContextItem(
                content: content,
                scope: scope(for: item),
                authority: authority(for: item),
                createdAt: item.createdAt,
                expiresAt: item.expiresAt,
                source: item.source,
                topic: item.topic
            )
        }
    }

    private static func scope(for item: MemoryItem) -> MemoryContextItem.Scope {
        let source = item.source.lowercased()
        if source == "rem-condensed" { return .remCondensed }
        if source.contains("tool") || source.contains("observation") { return .toolObservation }

        switch item.memoryKind {
        case .preference:
            return .userPreference
        case .person:
            return .person
        case .project:
            return .project
        case .conversation:
            return .conversation
        case .fact:
            if item.topic?.lowercased().contains("people") == true || item.topic?.lowercased().contains("contact") == true {
                return .person
            }
            return .conversation
        }
    }

    private static func authority(for item: MemoryItem) -> MemoryContextItem.Authority {
        let source = item.source.lowercased()
        if source == "rem-condensed" { return .backgroundOnly }
        if source.contains("tool") || source.contains("observation") { return .referenceOnly }

        switch item.memoryKind {
        case .preference:
            return .preferenceOnly
        case .person, .project:
            return .referenceOnly
        case .conversation, .fact:
            return .referenceOnly
        }
    }
}
