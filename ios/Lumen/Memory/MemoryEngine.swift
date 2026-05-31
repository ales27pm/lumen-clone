import Foundation
import SwiftData

@MainActor
final class MemoryEngine {
    func search(query: String, limit: Int, context: ModelContext) async -> [MemoryItem] {
        await MemoryStore.recall(query: query, context: context, limit: limit)
    }

    func buildContext(query: String, budget: Int, context: ModelContext) -> MemoryContextResult {
        MemoryContextBuilder.build(query: query, budgetChars: budget, context: context)
    }

    func extractCandidates(from messages: [ChatMessage], conversationID: UUID?) -> [MemoryCandidate] {
        messages.compactMap { message in
            guard message.messageRole == .user else { return nil }
            let t = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !t.isEmpty else { return nil }
            let lower = t.lowercased()
            let explicit: MemoryUserExplicitness
            let reason: String
            if lower.contains("remember that") { explicit = .explicitPreference; reason = "remember directive" }
            else if lower.hasPrefix("actually") { explicit = .correction; reason = "user correction" }
            else if lower.contains("i prefer") { explicit = .explicitPreference; reason = "stated preference" }
            else if lower.contains("for lumen") { explicit = .projectFact; reason = "project declaration" }
            else { explicit = .inferred; reason = "pattern match" }
            return MemoryCandidate(text: String(t.prefix(220)), kind: "fact", topics: [], conversationID: conversationID, messageID: message.id, createdAt: message.createdAt, confidence: 0.6, extractionReason: reason, userExplicitness: explicit, sensitivity: .normal)
        }
    }

    func saveCandidateIfAllowed(_ candidate: MemoryCandidate, context: ModelContext) async throws {
        let result = MemoryScorer.score(candidate: candidate)
        guard result.decision == .save else { return }
        try await MemoryStore.remember(candidate.text, kind: .fact, source: "memory-engine", topic: candidate.topics.first, context: context)
    }

    func consolidateDueMemories(context: ModelContext) async { await MemoryConsolidator.consolidate(context: context) }
    func deleteMemory(id: UUID, context: ModelContext) throws {
        var descriptor = FetchDescriptor<MemoryItem>(predicate: #Predicate { $0.id == id })
        descriptor.fetchLimit = 1
        if let item = try context.fetch(descriptor).first {
            context.delete(item)
            try context.save()
        }
    }
    func pinMemory(id: UUID, context: ModelContext) throws {
        var descriptor = FetchDescriptor<MemoryItem>(predicate: #Predicate { $0.id == id })
        descriptor.fetchLimit = 1
        if let item = try context.fetch(descriptor).first {
            item.isPinned = true
            try context.save()
        }
    }
}
