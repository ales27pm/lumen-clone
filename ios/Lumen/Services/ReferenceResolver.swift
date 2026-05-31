import Foundation

nonisolated struct ReferenceResolution: Sendable {
    let originalPrompt: String
    let rewrittenPrompt: String
    let resolvedReferences: [String: String]
    let confidence: Double
    let diagnostics: [String]

    var hasRewrite: Bool { originalPrompt != rewrittenPrompt }
}

nonisolated enum ReferenceResolver {
    static func resolve(
        prompt: String,
        history: [(role: MessageRole, content: String)],
        relevantMemories: [MemoryContextItem],
        currentTurnLedger: [ToolLedgerEntry] = []
    ) -> ReferenceResolution {
        let normalizedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedPrompt.isEmpty else {
            return ReferenceResolution(originalPrompt: prompt, rewrittenPrompt: prompt, resolvedReferences: [:], confidence: 0, diagnostics: ["empty_prompt"])
        }

        var rewritten = normalizedPrompt
        var mapping: [String: String] = [:]
        var diagnostics: [String] = []
        var score: Double = 0

        let candidates = recentPersonCandidates(history: history, relevantMemories: relevantMemories)
        if containsPronoun(normalizedPrompt) {
            if let candidate = candidates.first {
                rewritten = rewritePronouns(in: rewritten, with: candidate)
                mapping["pronoun"] = candidate
                score += 0.75
                diagnostics.append("resolved_pronoun_from_recent_context")
            } else {
                diagnostics.append("pronoun_detected_no_safe_referent")
            }
        }

        if containsPreviousReference(normalizedPrompt) {
            if let tool = currentTurnLedger.last?.toolID {
                rewritten = rewritePreviousReference(in: rewritten, replacement: "the previous \(tool) result")
                mapping["previous one"] = "previous \(tool) result"
                score += 0.2
                diagnostics.append("resolved_deictic_from_current_turn_toolledger")
            } else {
                diagnostics.append("deictic_detected_without_current_turn_toolledger")
            }
        }

        let bounded = min(1.0, max(0.0, score))
        return ReferenceResolution(
            originalPrompt: prompt,
            rewrittenPrompt: rewritten,
            resolvedReferences: mapping,
            confidence: bounded,
            diagnostics: diagnostics
        )
    }

    private static func containsPronoun(_ text: String) -> Bool {
        let lower = text.lowercased()
        return [" call her", " text her", " message her", "call him", "text him", "message him", "her ", " him "]
            .contains { lower.contains($0) } || lower.hasPrefix("call her") || lower.hasPrefix("call him") || lower.hasPrefix("text her") || lower.hasPrefix("text him") || lower == "call her" || lower == "call him"
    }

    private static func containsPreviousReference(_ text: String) -> Bool {
        let lower = text.lowercased()
        return ["previous one", "that one", "last one", "use previous"].contains { lower.contains($0) }
    }

    private static func rewritePronouns(in text: String, with referent: String) -> String {
        var out = text
        let patterns = ["\\bher\\b", "\\bhim\\b"]
        for pattern in patterns {
            out = out.replacingOccurrences(of: pattern, with: referent, options: [.regularExpression, .caseInsensitive])
        }
        return out
    }

    private static func rewritePreviousReference(in text: String, replacement: String) -> String {
        var out = text
        ["previous one", "that one", "last one"].forEach { token in
            out = out.replacingOccurrences(of: token, with: replacement, options: [.caseInsensitive])
        }
        return out
    }

    private static func recentPersonCandidates(history: [(role: MessageRole, content: String)], relevantMemories: [MemoryContextItem]) -> [String] {
        var candidates: [String] = []
        var seen: Set<String> = []

        for content in history.suffix(8).reversed().map(\.content) {
            if let name = extractPersonName(content), seen.insert(name.lowercased()).inserted {
                candidates.append(name)
            }
        }

        let memories = relevantMemories
            .filter { $0.scope == .person || ($0.topic?.lowercased().contains("contact") ?? false) || ($0.topic?.lowercased().contains("people") ?? false) }
            .sorted { lhs, rhs in
                (lhs.createdAt ?? .distantPast) > (rhs.createdAt ?? .distantPast)
            }
            .prefix(8)
            .map(\.content)

        for content in memories {
            if let name = extractPersonName(content), seen.insert(name.lowercased()).inserted {
                candidates.append(name)
            }
        }

        return candidates
    }

    private static func extractPersonName(_ text: String) -> String? {
        let bulletPattern = #"•\s*([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){0,3})\s*[—-]"#
        if let match = firstMatch(text, pattern: bulletPattern) { return match }

        let contactPattern = #"\b([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3})\s*[—-]\s*\+?[0-9]"#
        if let match = firstMatch(text, pattern: contactPattern) { return match }

        let pattern = #"\b([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){0,2})\b"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        let blocked: Set<String> = ["Call", "Text", "Message", "Use", "Previous", "Search Contacts", "No", "Contacts", "Assistant", "User"]
        for match in matches.reversed() {
            let candidate = ns.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            if candidate.count >= 3, !blocked.contains(candidate) {
                return candidate
            }
        }
        return nil
    }

    private static func firstMatch(_ text: String, pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = text as NSString
        guard let match = regex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)), match.numberOfRanges > 1 else { return nil }
        return ns.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
