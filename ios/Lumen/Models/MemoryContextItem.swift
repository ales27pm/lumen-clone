import Foundation

nonisolated struct MemoryContextItem: Sendable, Hashable {
    enum Scope: String, Sendable, Hashable, Codable {
        case currentTurn
        case conversation
        case task
        case userPreference
        case person
        case project
        case toolObservation
        case remCondensed
    }

    enum Authority: String, Sendable, Hashable, Codable {
        case sourceOfTruth
        case referenceOnly
        case preferenceOnly
        case backgroundOnly
    }

    let content: String
    let scope: Scope
    let authority: Authority
    let createdAt: Date?
    let expiresAt: Date?
    let source: String?
    let topic: String?

    var id: UUID {
        var hasher = Hasher()
        hasher.combine(content)
        hasher.combine(scope)
        hasher.combine(authority)
        hasher.combine(source)
        hasher.combine(topic)
        let value = UInt64(bitPattern: Int64(hasher.finalize()))
        let hex = String(format: "%016llx%016llx", value, value ^ 0xa5a5a5a5a5a5a5a5)
        return UUID(uuidString: "\(hex.prefix(8))-\(hex.dropFirst(8).prefix(4))-\(hex.dropFirst(12).prefix(4))-\(hex.dropFirst(16).prefix(4))-\(hex.dropFirst(20).prefix(12))") ?? UUID()
    }
}

nonisolated enum MemoryContextAdapter {
    static func fromLegacyStrings(_ values: [String]) -> [MemoryContextItem] {
        values.compactMap { value in
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }

            if let content = trimmed.stripPrefix("Tier 1 Ephemeral: ") {
                let normalized = content.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !normalized.isEmpty else { return nil }
                return MemoryContextItem(content: normalized, scope: .currentTurn, authority: .referenceOnly, createdAt: nil, expiresAt: nil, source: "legacy-tier1", topic: nil)
            }
            if let content = trimmed.stripPrefix("Tier 2 Vectorized: ") {
                let normalized = content.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !normalized.isEmpty else { return nil }
                return MemoryContextItem(content: normalized, scope: .conversation, authority: .referenceOnly, createdAt: nil, expiresAt: nil, source: "legacy-tier2", topic: nil)
            }
            if let content = trimmed.stripPrefix("Tier 3 Condensed: ") {
                let normalized = content.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !normalized.isEmpty else { return nil }
                return MemoryContextItem(content: normalized, scope: .remCondensed, authority: .backgroundOnly, createdAt: nil, expiresAt: nil, source: "legacy-tier3", topic: nil)
            }

            return MemoryContextItem(content: trimmed, scope: .conversation, authority: .referenceOnly, createdAt: nil, expiresAt: nil, source: "legacy", topic: nil)
        }
    }
}

nonisolated private extension String {
    func stripPrefix(_ prefix: String) -> String? {
        guard hasPrefix(prefix) else { return nil }
        return String(dropFirst(prefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
