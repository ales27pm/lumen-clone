import Foundation

enum PromptGroundingIdempotencyGuard {
    static let marker = "<!-- LUMEN_GROUNDING_V1 -->"
    private static let sectionHeaders = ["[LOCAL MEMORY]", "[LOCAL SOURCES]", "[AVAILABLE LOCAL TOOLS]", "[RUNTIME POLICY]"]

    static func sectionOccurrenceCounts(_ text: String) -> [String: Int] {
        var out: [String: Int] = [:]
        for h in sectionHeaders { out[h] = text.components(separatedBy: h).count - 1 }
        return out
    }

    static func containsGrounding(_ text: String) -> Bool { text.contains(marker) || sectionHeaders.contains(where: { text.contains($0) }) }

    static func stripExistingGrounding(from text: String) -> (text: String, stripped: Bool, ambiguous: Bool) {
        if let range = text.range(of: marker) {
            let prefix = String(text[..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            return (prefix, true, false)
        }
        let counts = sectionOccurrenceCounts(text)
        let total = counts.values.reduce(0,+)
        if total >= 2 {
            let earliest = sectionHeaders.compactMap { text.range(of: $0)?.lowerBound }.min()
            if let idx = earliest {
                return (String(text[..<idx]).trimmingCharacters(in: .whitespacesAndNewlines), true, false)
            }
        }
        return (text, false, total == 1)
    }
}
