import Foundation
import SwiftData

@MainActor
enum MemoryRecall {
    struct NormalizedRecallResult {
        let items: [MemoryContextItem]
        let mode: String
        let diagnostic: String?
    }

    static func recallAndNormalize(
        query: String,
        routing: IntentRoutingDecision,
        context: ModelContext,
        limit: Int = 8
    ) async -> [MemoryContextItem] {
        await recallAndNormalizeWithDiagnostics(query: query, routing: routing, context: context, limit: limit).items
    }

    static func recallAndNormalizeWithDiagnostics(
        query: String,
        routing: IntentRoutingDecision,
        context: ModelContext,
        limit: Int = 8
    ) async -> NormalizedRecallResult {
        let result = await MemoryStore.recallWithDiagnostics(query: query, context: context, limit: limit)
        let rawItems = result.items
        let items = rawItems.compactMap { item -> MemoryContextItem? in
            guard !MemoryStore.isExpired(item) else { return nil }
            let content = item.content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !content.isEmpty else { return nil }
            return MemoryContextItem(
                content: content,
                scope: scope(for: item),
                authority: authority(for: item),
                createdAt: item.createdAt,
                expiresAt: MemoryStore.inferredExpiresAt(for: item),
                source: item.source,
                topic: item.topic
            )
        }
        return NormalizedRecallResult(items: items, mode: result.mode, diagnostic: result.diagnostic)
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
