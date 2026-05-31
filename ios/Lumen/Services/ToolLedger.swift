import Foundation

struct ToolLedgerEntry: Sendable, Hashable {
    let id: UUID
    let conversationID: UUID?
    let turnID: UUID?
    let intent: UserIntent
    let toolID: String
    let query: String
    let result: String
    let createdAt: Date
    let expiresAt: Date

    init(
        id: UUID = UUID(),
        conversationID: UUID?,
        turnID: UUID?,
        intent: UserIntent,
        toolID: String,
        query: String,
        result: String,
        createdAt: Date = Date(),
        ttl: TimeInterval = 60 * 8
    ) {
        self.id = id
        self.conversationID = conversationID
        self.turnID = turnID
        self.intent = intent
        self.toolID = toolID
        self.query = query
        self.result = result
        self.createdAt = createdAt
        self.expiresAt = createdAt.addingTimeInterval(ttl)
    }

    var isExpired: Bool { Date() >= expiresAt }
}

@MainActor
final class ToolLedger {
    static let shared = ToolLedger()
    static let defaultTTL: TimeInterval = 60 * 8

    private var entries: [ToolLedgerEntry] = []
    private init() {}

    func record(conversationID: UUID?, turnID: UUID?, intent: UserIntent, toolID: String, query: String, result: String) {
        pruneExpired()
        let cleanQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanResult = result.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !toolID.isEmpty, !cleanResult.isEmpty else { return }
        entries.append(
            ToolLedgerEntry(
                conversationID: conversationID,
                turnID: turnID,
                intent: intent,
                toolID: toolID,
                query: cleanQuery,
                result: cleanResult
            )
        )
    }

    func currentTurnEntries(conversationID: UUID?, turnID: UUID?) -> [ToolLedgerEntry] {
        pruneExpired()
        return entries.filter { $0.conversationID == conversationID && $0.turnID == turnID }
    }

    func shortTermEntries(conversationID: UUID?, since: Date? = nil) -> [ToolLedgerEntry] {
        let threshold = since ?? Date().addingTimeInterval(-Self.defaultTTL)
        pruneExpired()
        return entries.filter { entry in
            entry.conversationID == conversationID && entry.createdAt >= threshold
        }
    }

    func referencedEntries(conversationID: UUID?, userMessage: String) -> [ToolLedgerEntry] {
        pruneExpired()
        let lower = userMessage.lowercased()
        let requestedPriorResult = ["earlier", "before", "last result", "previous result", "you found", "you said"].contains { lower.contains($0) }
        guard requestedPriorResult else { return [] }
        return shortTermEntries(conversationID: conversationID)
    }

    private func pruneExpired() {
        let now = Date()
        entries.removeAll { $0.expiresAt <= now }
    }
}
