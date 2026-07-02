import Foundation

nonisolated struct MemoryCommandPlan: Sendable, Hashable {
    enum Kind: Sendable, Hashable {
        case saveThenRecall
    }

    let kind: Kind
    let saveContent: String
    let recallQuery: String

    static func saveThenRecall(from prompt: String) -> MemoryCommandPlan? {
        let text = normalized(prompt)
        guard containsAny(text, ["remember", "save", "note", "keep this in mind", "keep in mind"]),
              containsAny(text, ["tell me what", "what you remembered", "what did you remember", "repeat it back", "then tell", "then recall", "and tell me"]) else {
            return nil
        }
        let fact = extractMemoryFact(from: prompt)
        guard !fact.isEmpty else { return nil }
        return MemoryCommandPlan(kind: .saveThenRecall, saveContent: fact, recallQuery: extractMemoryRecallQuery(fromFact: fact, fallback: prompt))
    }

    static func extractMemoryFact(from prompt: String) -> String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if let name = firstCapture(in: trimmed, pattern: #"(?i)\b(?:can you\s+)?(?:please\s+)?(?:remember|save|note)\s+(?:that\s+)?my name is\s+(.+?)(?:,?\s+then\b|\s+and\s+(?:tell|recall|repeat|confirm)\b|[.!?\n]?$)"#)
            ?? firstCapture(in: trimmed, pattern: #"(?i)\bmy name is\s+([^.!?\n]+)"#) {
            return "User's name is \(cleanCapturedValue(name))"
        }
        if let name = firstCapture(in: trimmed, pattern: #"(?i)\bcall me\s+([^.!?\n]+)"#) {
            return "User prefers to be called \(cleanCapturedValue(name))"
        }
        if let fact = firstCapture(in: trimmed, pattern: #"(?i)\bremember that\s+(.+?)(?:,?\s+then\b|\s+and\s+(?:tell|recall|repeat|confirm)\b|[.!?\n]?$)"#)
            ?? firstCapture(in: trimmed, pattern: #"(?i)\bsave this fact:?\s+(.+?)(?:,?\s+then\b|\s+and\s+(?:tell|recall|repeat|confirm)\b|[.!?\n]?$)"#)
            ?? firstCapture(in: trimmed, pattern: #"(?i)\bkeep this in mind:?\s+(.+?)(?:,?\s+then\b|\s+and\s+(?:tell|recall|repeat|confirm)\b|[.!?\n]?$)"#) {
            return normalizeFactSentence(fact)
        }
        return normalizeFactSentence(trimmed)
    }

    static func extractMemoryRecallQuery(from prompt: String) -> String {
        extractMemoryRecallQuery(fromFact: extractMemoryFact(from: prompt), fallback: prompt)
    }

    private static func extractMemoryRecallQuery(fromFact fact: String, fallback: String) -> String {
        let cleaned = fact
            .replacingOccurrences(of: "User's name is ", with: "", options: [.caseInsensitive])
            .replacingOccurrences(of: "User prefers to be called ", with: "", options: [.caseInsensitive])
            .replacingOccurrences(of: #"(?i)^\s*(?:i\s+am\s+|i'm\s+|i\s+|my\s+)"#, with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !cleaned.isEmpty { return cleaned }
        return fallback.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func normalized(_ text: String) -> String {
        text.lowercased()
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func containsAny(_ text: String, _ needles: [String]) -> Bool {
        needles.contains { text.contains($0) }
    }

    private static func firstCapture(in text: String, pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = text as NSString
        guard let match = regex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)),
              match.numberOfRanges > 1 else { return nil }
        return ns.substring(with: match.range(at: 1))
    }

    private static func cleanCapturedValue(_ value: String) -> String {
        value.trimmingCharacters(in: CharacterSet(charactersIn: " \t\n\r\"'.,!?"))
    }

    private static func normalizeFactSentence(_ value: String) -> String {
        let cleaned = cleanCapturedValue(value)
        guard !cleaned.isEmpty else { return value.trimmingCharacters(in: .whitespacesAndNewlines) }
        return cleaned
    }
}
