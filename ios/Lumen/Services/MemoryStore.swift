import Foundation
import SwiftData
import OSLog

@MainActor
enum MemoryStore {
    private static let logger = Logger(subsystem: "ai.lumen.app", category: "persistence")

    private static func persist(_ context: ModelContext, operation: String, scope: String) throws {
        guard DiskWriteBudget.shared.canWrite(bytes: 64 * 1024, category: .memory) else { return }
        do {
            try context.save()
            DiskWriteBudget.shared.recordWrite(bytes: 64 * 1024, category: .memory)
        } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            throw error
        }
    }

    static func auditPersistence(operation: String, scope: String, save: () throws -> Void) -> Bool {
        do {
            try save()
            return true
        } catch {
            logger.error("persist_failed op=\(operation, privacy: .public) scope=\(scope, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return false
        }
    }

    nonisolated struct TTLPolicy: Sendable {
        let freshness: MemoryFreshnessClass
        let ttl: TimeInterval?
    }

    struct RecallResult {
        let items: [MemoryItem]
        let mode: String
        let diagnostic: String?
    }

    private struct LexicalRecallResult {
        let items: [MemoryItem]
        let diagnostic: String?
    }

    struct AutoExtractionResult {
        let attempted: Int
        let stored: Int
        let failed: Int
        let skipped: Int
        let diagnostics: [String]
    }

    struct ExportResult {
        let json: String?
        let mode: String
        let diagnostic: String?
    }

    struct RememberResult {
        let mode: String
        let diagnostic: String?
    }

    static func recall(query: String, context: ModelContext, limit: Int = 5) async -> [MemoryItem] {
        await recallWithDiagnostics(query: query, context: context, limit: limit).items
    }

    static func recallWithDiagnostics(query: String, context: ModelContext, limit: Int = 5) async -> RecallResult {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, limit > 0 else {
            return RecallResult(items: [], mode: "empty_query", diagnostic: "empty_query")
        }

        if let budgetDenial = ResourceBudgetGate.budgetDenialReason(policy: .embedding, reason: "memory.recall") {
            logger.error("memory_embedding_budget_denied op=recall reason=\(budgetDenial, privacy: .public)")
            let fallback = lexicalRecallResult(query: trimmed, context: context, limit: limit)
            return RecallResult(
                items: fallback.items,
                mode: "lexical_fallback",
                diagnostic: combinedDiagnostic(primary: budgetDenial, secondary: fallback.diagnostic)
            )
        }

        let queryVec: [Double]
        do {
            queryVec = try await AssistantKernel.runEmbedding(text: SemanticEmbeddingText.memoryQuery(trimmed))
        } catch {
            logger.error("memory_embedding_failed op=recall error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            let fallback = lexicalRecallResult(query: trimmed, context: context, limit: limit)
            return RecallResult(
                items: fallback.items,
                mode: "lexical_fallback",
                diagnostic: combinedDiagnostic(
                    primary: "embedding_failed:\(RuntimeMetricErrorSanitizer.code(for: error))",
                    secondary: fallback.diagnostic
                )
            )
        }
        guard !queryVec.isEmpty else {
            logger.error("memory_embedding_empty op=recall")
            let fallback = lexicalRecallResult(query: trimmed, context: context, limit: limit)
            return RecallResult(
                items: fallback.items,
                mode: "lexical_fallback",
                diagnostic: combinedDiagnostic(primary: "embedding_empty", secondary: fallback.diagnostic)
            )
        }

        let availableItems: [MemoryItem]
        do {
            availableItems = try context.fetch(FetchDescriptor<MemoryItem>())
        } catch {
            let diagnostic = "fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            logger.error("memory_fetch_failed op=recall diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return RecallResult(items: [], mode: "failed", diagnostic: diagnostic)
        }
        guard !availableItems.isEmpty else {
            return RecallResult(items: [], mode: "semantic", diagnostic: "empty_store")
        }
        let itemByID = Dictionary(uniqueKeysWithValues: availableItems.map { ($0.persistentModelID, $0) })

        let vectorLoad = MemoryVectorIndex.shared.ensureLoaded(context: context)
        var diagnostic = vectorLoad.diagnostic
        var results: [MemoryItem] = []
        results.reserveCapacity(limit)
        var seenIds: Set<PersistentIdentifier> = []
        var topK = max(limit * 3, limit + 8)
        let maxTopK = max(topK, 256)
        var sawStaleVectorID = false

        while results.count < limit {
            let hits = MemoryVectorIndex.shared.search(query: queryVec, topK: topK, pinBonus: 0.15)
            if hits.isEmpty { break }

            for h in hits {
                if results.count >= limit { break }
                guard seenIds.insert(h.id).inserted else { continue }
                guard let item = itemByID[h.id] else {
                    sawStaleVectorID = true
                    continue
                }
                guard !isExpired(item) else { continue }
                results.append(item)
            }

            if results.count >= limit || hits.count < topK || topK >= maxTopK {
                break
            }
            topK = min(topK * 2, maxTopK)
        }

        if sawStaleVectorID {
            MemoryVectorIndex.shared.invalidate()
            let reload = MemoryVectorIndex.shared.ensureLoaded(context: context)
            if let reloadDiagnostic = reload.diagnostic {
                diagnostic = diagnostic.map { combinedDiagnostic(primary: $0, secondary: reloadDiagnostic) } ?? reloadDiagnostic
            }
        }
        if results.count < limit {
            let existingIDs = Set(results.map(\.persistentModelID))
            let backfill = lexicalRecallResult(query: trimmed, context: context, limit: limit - results.count, excluding: existingIDs)
            results.append(contentsOf: backfill.items)
            if !backfill.items.isEmpty {
                if let backfillDiagnostic = backfill.diagnostic {
                    diagnostic = diagnostic.map { combinedDiagnostic(primary: $0, secondary: backfillDiagnostic) } ?? backfillDiagnostic
                }
                return RecallResult(items: results, mode: "semantic_with_lexical_backfill", diagnostic: diagnostic)
            }
            if let backfillDiagnostic = backfill.diagnostic {
                diagnostic = diagnostic.map { combinedDiagnostic(primary: $0, secondary: backfillDiagnostic) } ?? backfillDiagnostic
                return RecallResult(items: results, mode: "semantic", diagnostic: diagnostic)
            }
        }
        return RecallResult(items: results, mode: "semantic", diagnostic: diagnostic)
    }

    static func lexicalRecall(
        query: String,
        context: ModelContext,
        limit: Int,
        excluding excludedIDs: Set<PersistentIdentifier> = []
    ) -> [MemoryItem] {
        lexicalRecallResult(query: query, context: context, limit: limit, excluding: excludedIDs).items
    }

    private static func lexicalRecallResult(
        query: String,
        context: ModelContext,
        limit: Int,
        excluding excludedIDs: Set<PersistentIdentifier> = []
    ) -> LexicalRecallResult {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, limit > 0 else {
            return LexicalRecallResult(items: [], diagnostic: "lexical_empty_query")
        }
        let queryLower = trimmed.lowercased()
        let terms = queryLower
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 2 }
        let availableItems: [MemoryItem]
        do {
            availableItems = try context.fetch(FetchDescriptor<MemoryItem>())
        } catch {
            let diagnostic = "lexical_fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            logger.error("memory_fetch_failed op=lexicalRecall diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return LexicalRecallResult(items: [], diagnostic: diagnostic)
        }
        let now = Date()
        let items = availableItems.compactMap { item -> (MemoryItem, Double)? in
            guard !excludedIDs.contains(item.persistentModelID), !isExpired(item, now: now) else { return nil }
            let content = item.content.lowercased()
            let topic = item.topic?.lowercased() ?? ""
            var score = 0.0
            if content == queryLower { score += 5.0 }
            if content.hasPrefix(queryLower) { score += 2.0 }
            if content.contains(queryLower) { score += 1.25 }
            if topic.contains(queryLower) { score += 0.75 }
            let hits = terms.reduce(0) { count, term in
                count + ((content.contains(term) || topic.contains(term)) ? 1 : 0)
            }
            if !terms.isEmpty { score += Double(hits) / Double(terms.count) }
            if item.isPinned { score += 0.5 }
            guard score > 0 else { return nil }
            score += max(0, 0.25 - now.timeIntervalSince(item.createdAt) / (60 * 60 * 24 * 365))
            return (item, score)
        }
        .sorted {
            if $0.1 != $1.1 { return $0.1 > $1.1 }
            return $0.0.createdAt > $1.0.createdAt
        }
        .prefix(limit)
        .map { $0.0 }
        return LexicalRecallResult(items: items, diagnostic: nil)
    }

    private static func combinedDiagnostic(primary: String, secondary: String?) -> String {
        guard let secondary, !secondary.isEmpty else { return primary }
        return "\(primary);\(secondary)"
    }

    static func remember(_ content: String, kind: MemoryKind = .fact, source: String = "manual", topic: String? = nil, context: ModelContext) async throws {
        _ = try await rememberResult(content, kind: kind, source: source, topic: topic, context: context)
    }

    @discardableResult
    static func rememberWithDiagnostics(
        _ content: String,
        kind: MemoryKind = .fact,
        source: String = "manual",
        topic: String? = nil,
        context: ModelContext
    ) async -> RememberResult {
        do {
            return try await rememberResult(content, kind: kind, source: source, topic: topic, context: context)
        } catch {
            let diagnostic = "remember_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            logger.error("memory_store_failed op=remember source=\(source, privacy: .public) kind=\(kind.rawValue, privacy: .public) diagnostic=\(diagnostic, privacy: .public)")
            return RememberResult(mode: "failed", diagnostic: diagnostic)
        }
    }

    private static func rememberResult(_ content: String, kind: MemoryKind, source: String, topic: String?, context: ModelContext) async throws -> RememberResult {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return RememberResult(mode: "skipped", diagnostic: "empty_content")
        }
        do {
            let existing = try context.fetch(FetchDescriptor<MemoryItem>())
            if existing.contains(where: { $0.content.caseInsensitiveCompare(trimmed) == .orderedSame }) {
                return RememberResult(mode: "skipped", diagnostic: "duplicate_memory")
            }
        } catch {
            let diagnostic = "duplicate_fetch_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            logger.error("memory_fetch_failed op=remember.duplicate_check diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            throw error
        }
        let embedding: [Double]
        do {
            let embeddingText = SemanticEmbeddingText.memoryDocument(
                content: trimmed,
                kind: kind,
                source: source,
                topic: topic
            )
            embedding = try await AssistantKernel.runEmbedding(text: embeddingText)
        } catch {
            logger.error("memory_embedding_failed op=remember error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
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
        return RememberResult(mode: "stored", diagnostic: nil)
    }

    @discardableResult
    static func extractAndStore(userText: String, assistantText: String, transientTexts: [String] = [], context: ModelContext) async -> AutoExtractionResult {
        let durableAssistant = durableAssistantText(assistantText, transientTexts: transientTexts)
        let combined = userText + "\n" + durableAssistant
        let cpuToken = CPUWatchdogGuard.shared.begin(category: .memory)
        defer { CPUWatchdogGuard.shared.end(token: cpuToken) }
        var attempted = 0
        var stored = 0
        var failed = 0
        var skipped = 0
        var diagnostics: [String] = []
        for extracted in extractFacts(from: combined) {
            if Task.isCancelled || CPUWatchdogGuard.shared.shouldDegrade(category: .memory) || !ResourceBudgetGate.allowsMaintenance(reason: "memory.extract") {
                skipped += 1
                diagnostics.append("memory_extract_skipped")
                break
            }
            attempted += 1
            do {
                try await remember(extracted.content, kind: extracted.kind, source: "auto", topic: extracted.topic, context: context)
                stored += 1
            } catch {
                failed += 1
                let diagnostic = "remember_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
                diagnostics.append(diagnostic)
                logger.error("memory_auto_store_failed op=extractAndStore diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            }
        }
        return AutoExtractionResult(
            attempted: attempted,
            stored: stored,
            failed: failed,
            skipped: skipped,
            diagnostics: diagnostics
        )
    }

    static func wipeAll(context: ModelContext) throws {
        let all = try context.fetch(FetchDescriptor<MemoryItem>())
        for item in all where !item.isPinned {
            context.delete(item)
        }
        try persist(context, operation: "wipeAll", scope: "MemoryItem")
        MemoryVectorIndex.shared.invalidate()
    }

    static func wipeEverything(context: ModelContext) throws {
        let all = try context.fetch(FetchDescriptor<MemoryItem>())
        for item in all { context.delete(item) }
        try persist(context, operation: "wipeEverything", scope: "MemoryItem")
        MemoryVectorIndex.shared.invalidate()
    }

    static func exportJSON(context: ModelContext) -> String {
        let result = exportJSONWithDiagnostics(context: context)
        if let json = result.json {
            return json
        }
        let diagnostic = result.diagnostic ?? "unknown"
        return #"{"error":"memory_export_failed","diagnostic":"\#(diagnostic)"}"#
    }

    static func exportJSONWithDiagnostics(context: ModelContext) -> ExportResult {
        exportJSONWithDiagnostics(fetch: { try context.fetch(FetchDescriptor<MemoryItem>()) })
    }

    static func exportJSONWithDiagnosticsForTests(fetch: () throws -> [MemoryItem]) -> ExportResult {
        exportJSONWithDiagnostics(fetch: fetch)
    }

    private static func exportJSONWithDiagnostics(fetch: () throws -> [MemoryItem]) -> ExportResult {
        do {
            return ExportResult(json: try exportJSON(items: fetch()), mode: "exported", diagnostic: nil)
        } catch {
            let diagnostic = "export_failed:\(RuntimeMetricErrorSanitizer.code(for: error))"
            logger.error("memory_export_failed op=exportJSON diagnostic=\(diagnostic, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            return ExportResult(json: nil, mode: "failed", diagnostic: diagnostic)
        }
    }

    static func exportJSONThrowing(context: ModelContext) throws -> String {
        let all = try context.fetch(FetchDescriptor<MemoryItem>())
        return try exportJSON(items: all)
    }

    private static func exportJSON(items all: [MemoryItem]) throws -> String {
        struct Export: Codable { let content: String; let kind: String; let topic: String?; let pinned: Bool; let createdAt: Date }
        let items = all.map { Export(content: $0.content, kind: $0.kind, topic: $0.topic, pinned: $0.isPinned, createdAt: $0.createdAt) }
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        enc.dateEncodingStrategy = .iso8601
        let data = try enc.encode(items)
        return String(decoding: data, as: UTF8.self)
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
        // Persisted expiry columns were added after early TestFlight builds. On some upgraded
        // stores, touching those generated SwiftData accessors from the hot chat path can trap
        // before Swift can throw an error. Treat the TTL as computed policy instead of eagerly
        // reading or mutating the optional persisted cache fields during recall.
        _ = item
    }

    static func inferredExpiresAt(for item: MemoryItem) -> Date? {
        ttlPolicy(kind: item.memoryKind, source: item.source).ttl.map { item.createdAt.addingTimeInterval($0) }
    }

    static func isExpired(_ item: MemoryItem, now: Date = Date()) -> Bool {
        guard let expiresAt = inferredExpiresAt(for: item) else { return false }
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
