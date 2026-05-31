import Foundation
import SwiftData
import OSLog

@MainActor
enum MemoryStore {
    private static let logger = Logger(subsystem: "ai.lumen.app", category: "persistence")

    private static func persist(_ context: ModelContext, operation: String, scope: String) throws {
        do {
            try context.save()
        } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error=\(String(describing: error), privacy: .public)")
            throw error
        }
    }

    static func auditPersistence(operation: String, scope: String, save: () throws -> Void) -> Bool {
        do {
            try save()
            return true
        } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error=\(String(describing: error), privacy: .public)")
            return false
        }
    }

    nonisolated struct TTLPolicy: Sendable {
        let freshness: MemoryFreshnessClass
        let ttl: TimeInterval?
    }

    static func recall(query: String, context: ModelContext, limit: Int = 5) async -> [MemoryItem] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, limit > 0 else { return [] }
        let queryVec: [Double]
        do {
            queryVec = try await AppLlamaService.shared.embed(trimmed)
        } catch {
            logger.error("memory_embedding_failed op=recall error=\(String(describing: error), privacy: .public)")
            return []
        }
        guard !queryVec.isEmpty else {
            logger.error("memory_embedding_empty op=recall")
            return []
        }

        MemoryVectorIndex.shared.ensureLoaded(context: context)
        var results: [MemoryItem] = []
        results.reserveCapacity(limit)
        var seenIds: Set<PersistentIdentifier> = []
        var topK = max(limit * 3, limit + 8)
        let maxTopK = max(topK, 256)

        while results.count < limit {
            let hits = MemoryVectorIndex.shared.search(query: queryVec, topK: topK, pinBonus: 0.15)
            if hits.isEmpty { break }

            for h in hits {
                if results.count >= limit { break }
                guard seenIds.insert(h.id).inserted else { continue }
                if let item = context.model(for: h.id) as? MemoryItem {
                    migrateExpiryIfNeeded(for: item)
                    guard !isExpired(item) else { continue }
                    results.append(item)
                }
            }

            if results.count >= limit || hits.count < topK || topK >= maxTopK {
                break
            }
            topK = min(topK * 2, maxTopK)
        }
        do { try persist(context, operation: "recall.migrateExpiry", scope: "MemoryItem") } catch {}
        return results
    }

    static func remember(_ content: String, kind: MemoryKind = .fact, source: String = "manual", topic: String? = nil, context: ModelContext) async throws {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if let existing = try? context.fetch(FetchDescriptor<MemoryItem>()) {
            if existing.contains(where: { $0.content.caseInsensitiveCompare(trimmed) == .orderedSame }) {
                return
            }
        }
        let embedding: [Double]
        do {
            embedding = try await AppLlamaService.shared.embed(trimmed)
        } catch {
            logger.error("memory_embedding_failed op=remember error=\(String(describing: error), privacy: .public)")
            throw error
        }
        guard !embedding.isEmpty else {
            logger.error("memory_embedding_empty op=remember")
            throw LlamaError.embeddingFailed("Memory embedding returned empty vector")
        }
        let policy = ttlPolicy(kind: kind, source: source)
        let item = MemoryItem(
            content: trimmed,
            kind: kind,
            source: source,
            embedding: embedding,
            topic: topic,
            expiresAt: policy.ttl.map { Date().addingTimeInterval($0) },
            freshnessClass: policy.freshness
        )
        context.insert(item)
        try persist(context, operation: "remember.insert", scope: "MemoryItem")
        MemoryVectorIndex.shared.ensureLoaded(context: context)
        MemoryVectorIndex.shared.append(id: item.persistentModelID, isPinned: item.isPinned, vector: embedding)
    }

    static func extractAndStore(userText: String, assistantText: String, transientTexts: [String] = [], context: ModelContext) async {
        let durableAssistant = durableAssistantText(assistantText, transientTexts: transientTexts)
        let combined = userText + "\n" + durableAssistant
        for extracted in extractFacts(from: combined) {
            try? await remember(extracted.content, kind: extracted.kind, source: "auto", topic: extracted.topic, context: context)
        }
    }

    static func wipeAll(context: ModelContext) throws {
        guard let all = try? context.fetch(FetchDescriptor<MemoryItem>()) else { return }
        for item in all where !item.isPinned {
            context.delete(item)
        }
        try persist(context, operation: "wipeAll", scope: "MemoryItem")
        MemoryVectorIndex.shared.invalidate()
    }

    static func wipeEverything(context: ModelContext) throws {
        guard let all = try? context.fetch(FetchDescriptor<MemoryItem>()) else { return }
        for item in all { context.delete(item) }
        try persist(context, operation: "wipeEverything", scope: "MemoryItem")
        MemoryVectorIndex.shared.invalidate()
    }

    static func exportJSON(context: ModelContext) -> String {
        guard let all = try? context.fetch(FetchDescriptor<MemoryItem>()) else { return "[]" }
        struct Export: Codable { let content: String; let kind: String; let topic: String?; let pinned: Bool; let createdAt: Date }
        let items = all.map { Export(content: $0.content, kind: $0.kind, topic: $0.topic, pinned: $0.isPinned, createdAt: $0.createdAt) }
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        enc.dateEncodingStrategy = .iso8601
        return (try? String(data: enc.encode(items), encoding: .utf8)) ?? "[]"
    }

    // MARK: - Fact extraction (lightweight, rule-based)

    nonisolated struct Extracted {
        let content: String
        let kind: MemoryKind
        let topic: String?
    }

    nonisolated static func extractFacts(from text: String) -> [Extracted] {
        var results: [Extracted] = []
        let sentences = text
            .replacingOccurrences(of: "\n", with: " ")
            .components(separatedBy: CharacterSet(charactersIn: ".!?"))
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        let prefLove = ["i love", "i like", "i enjoy", "i prefer", "my favorite", "i'm a fan of", "i am a fan of"]
        let durableAllowlist = ["prefer", "favorite", "i am", "i'm", "my name is", "i live", "i work", "working on", "building", "my project", "my app", "my startup"]
        let volatileDenylist = ["weather", "temperature", "forecast", "current location", "located", "search result", "live result", "breaking", "reminder", "alarm", "calendar", "busy", "free", "availability", "tomorrow", "today"]
        let prefHate = ["i hate", "i dislike", "i don't like", "i do not like", "i can't stand"]
        let factSelf = ["i am", "i'm", "i live", "i work", "my name is", "i was born", "i have", "my birthday"]
        let projectMarkers = ["working on", "building", "my project", "my app", "my startup"]
        let personMarkers: [(String, String)] = [
            (#"my (wife|husband|partner|boyfriend|girlfriend|mom|mother|dad|father|brother|sister|son|daughter|friend|boss|manager|teammate|colleague|neighbor|dog|cat) (?:is |named |'s name is |'s )?([A-Z][a-z]+)"#, "relation")
        ]

        var seen: Set<String> = []
        func push(_ e: Extracted) {
            let key = e.content.lowercased()
            if seen.contains(key) { return }
            seen.insert(key)
            results.append(e)
        }
        for s in sentences {
            let lower = s.lowercased()
            guard durableAllowlist.contains(where: { lower.contains($0) }) else { continue }
            if volatileDenylist.contains(where: { lower.contains($0) }) { continue }
            if prefLove.contains(where: { lower.contains($0) }) {
                push(Extracted(content: "User preference: \(cleaned(s))", kind: .preference, topic: nil))
                continue
            }
            if prefHate.contains(where: { lower.contains($0) }) {
                push(Extracted(content: "User dislike: \(cleaned(s))", kind: .preference, topic: nil))
                continue
            }
            if projectMarkers.contains(where: { lower.contains($0) }) {
                push(Extracted(content: "Project: \(cleaned(s))", kind: .project, topic: "projects"))
                continue
            }
            if factSelf.contains(where: { lower.hasPrefix($0) || lower.contains(" \($0) ") }) {
                push(Extracted(content: cleaned(s), kind: .fact, topic: nil))
                continue
            }
            for (pattern, _) in personMarkers {
                if let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]),
                   let match = regex.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)),
                   match.numberOfRanges >= 3,
                   let rRel = Range(match.range(at: 1), in: s),
                   let rName = Range(match.range(at: 2), in: s) {
                    let rel = String(s[rRel])
                    let name = String(s[rName])
                    push(Extracted(content: "\(rel.capitalized): \(name)", kind: .person, topic: "people"))
                    break
                }
            }
        }
        return Array(results.prefix(8))
    }


    nonisolated private static func durableAssistantText(_ assistantText: String, transientTexts: [String]) -> String {
        let base = assistantText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !base.isEmpty else { return "" }
        var filtered = base
        for transient in transientTexts where !transient.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            filtered = filtered.replacingOccurrences(of: transient, with: "", options: [.caseInsensitive])
        }
        let blockedMarkers = ["tool", "observation", "search results", "temporary status"]
        let lines = filtered
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { line in !line.isEmpty && !blockedMarkers.contains(where: { line.lowercased().contains($0) }) }
        return lines.joined(separator: " ")
    }

    nonisolated private static func cleaned(_ s: String) -> String {
        var out = s
        if out.count > 140 { out = String(out.prefix(140)) + "…" }
        return out
    }

    static func migrateExpiryIfNeeded(for item: MemoryItem) {
        guard item.expiresAt == nil || item.freshnessClass == nil else { return }
        let policy = ttlPolicy(kind: item.memoryKind, source: item.source)
        item.freshnessClass = policy.freshness.rawValue
        if item.expiresAt == nil {
            item.expiresAt = policy.ttl.map { item.createdAt.addingTimeInterval($0) }
        }
    }

    static func isExpired(_ item: MemoryItem, now: Date = Date()) -> Bool {
        guard let expiresAt = item.expiresAt else { return false }
        return expiresAt <= now
    }

    nonisolated static func ttlPolicy(kind: MemoryKind, source: String) -> TTLPolicy {
        let lowerSource = source.lowercased()

        if lowerSource.contains("tool") || lowerSource.contains("ephemeral") || lowerSource.contains("observation") {
            return TTLPolicy(freshness: .volatile, ttl: 45 * 60)
        }

        if lowerSource == "rem-condensed" {
            return TTLPolicy(freshness: .durable, ttl: nil)
        }

        if kind == .conversation || lowerSource.contains("crumb") || lowerSource.contains("chat") {
            return TTLPolicy(freshness: .shortLived, ttl: 6 * 60 * 60)
        }

        if kind == .preference || kind == .project || kind == .person {
            return TTLPolicy(freshness: .timeless, ttl: nil)
        }

        return TTLPolicy(freshness: .durable, ttl: 30 * 24 * 60 * 60)
    }
}
