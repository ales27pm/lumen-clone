import Foundation

nonisolated enum AssistantOutputSanitizer {
    static func sanitize(_ text: String, lastUserMessage: String? = nil, debugMode: Bool = false) -> String {
        let withoutWebPayload = WebRichContentPayload.removingMarkers(from: text)
        let withoutToolTraceJSON = debugMode ? withoutWebPayload : removingToolTraceJSONBlobs(from: withoutWebPayload)
        let formattedSources = reformatSourcesSection(in: withoutToolTraceJSON)
        let trimmed = formattedSources.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return trimmed }

        if isLeakedToolJSONArtifact(trimmed) {
            return neutralFallback(for: lastUserMessage)
        }

        if isPrivacyPlaceholderArtifact(trimmed) {
            return neutralFallback(for: lastUserMessage)
        }

        if isFalseToolClaim(trimmed, lastUserMessage: lastUserMessage) {
            return neutralFallback(for: lastUserMessage)
        }

        return trimmed
    }

    static func removingToolTraceJSONBlobs(from text: String) -> String {
        var output = text

        let fencedPattern = #"```json\s*\{[\s\S]*?(?:\"thought\"|\"action\"|\"tool\"|\"tool_trace\"|\"trace\"|\"args\")[\s\S]*?\}\s*```"#
        output = output.replacingOccurrences(of: fencedPattern, with: "", options: .regularExpression)

        let inlinePattern = #"(?mi)^\s*\{\s*\"(?:thought|action|tool|tool_trace|trace|args)\"[\s\S]*?\}\s*$"#
        output = output.replacingOccurrences(of: inlinePattern, with: "", options: .regularExpression)

        return output
            .replacingOccurrences(of: #"\n{3,}"#, with: "\n\n", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func reformatSourcesSection(in text: String) -> String {
        guard let range = text.range(of: #"(?is)sources\s*:\s*(\[[\s\S]*?\])"#, options: .regularExpression) else {
            return text
        }
        let match = String(text[range])
        guard let arrayRange = match.range(of: #"\[[\s\S]*\]"#, options: .regularExpression) else {
            return text
        }
        let jsonArray = String(match[arrayRange])
        guard let data = jsonArray.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(with: data) as? [Any],
              !parsed.isEmpty else {
            return text
        }

        let bullets: [String] = parsed.compactMap { item in
            if let raw = item as? String {
                return "- \(raw)"
            }
            if let object = item as? [String: Any] {
                let title = object["title"] as? String
                let url = object["url"] as? String
                switch (title, url) {
                case let (t?, u?) where !t.isEmpty && !u.isEmpty:
                    return "- [\(t)](\(u))"
                case let (_, u?) where !u.isEmpty:
                    return "- \(u)"
                case let (t?, _) where !t.isEmpty:
                    return "- \(t)"
                default:
                    return nil
                }
            }
            return nil
        }

        guard !bullets.isEmpty else { return text }
        let replacement = "Sources:\n" + bullets.joined(separator: "\n")
        return text.replacingCharacters(in: range, with: replacement)
    }

    static func isFalseToolClaim(_ text: String, lastUserMessage: String? = nil) -> Bool {
        isFalseCalendarCreationClaim(text, lastUserMessage: lastUserMessage)
    }

    static func isFalseCalendarCreationClaim(_ text: String, lastUserMessage: String? = nil) -> Bool {
        let normalized = text.lowercased()
        let claimsCalendarCreation =
            normalized.contains("successfully created a new event") ||
            normalized.contains("created a new event") ||
            normalized.contains("created event") ||
            normalized.contains("calendar event")

        guard claimsCalendarCreation else { return false }

        let mentionsGreetingEvent =
            normalized.contains("hi lumen") ||
            normalized.contains("titled \"hi\"") ||
            normalized.contains("titled 'hi'") ||
            normalized.contains("title: hi")

        return mentionsGreetingEvent || isPureConversation(lastUserMessage)
    }

    static func isLeakedToolJSONArtifact(_ text: String) -> Bool {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = normalized.lowercased()
        let compact = compacted(normalized)

        if lower.hasPrefix("```json") || lower.hasPrefix("```") {
            if lower.contains("\"action\"") || lower.contains("\"tool\"") || lower.contains("\"args\"") || lower.contains("web.search") {
                return true
            }
        }

        if lower.hasPrefix("{") && lower.contains("\"thought\"") && (lower.contains("\"action\"") || lower.contains("\"final\"")) {
            return true
        }

        return compact.contains("thought") && compact.contains("action") && compact.contains("tool") && compact.contains("args")
    }

    static func isPrivacyPlaceholderArtifact(_ text: String) -> Bool {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        guard normalized.hasPrefix("<") && normalized.hasSuffix(">") else { return false }
        return normalized.contains("PRESIDIO_ANONYMIZED_") || normalized.contains("ANONYMIZED_PERSON")
    }

    static func neutralFallback(for lastUserMessage: String?) -> String {
        guard let lastUserMessage else { return "I couldn't complete that cleanly. Try again, or give me one more detail." }
        let compact = compacted(lastUserMessage)
        switch compact {
        case "hi", "hello", "hey", "yo", "sup", "bonjour", "salut", "allo":
            return "Hi."
        case "thanks", "thankyou", "merci":
            return "You're welcome."
        case "ok", "okay":
            return "OK."
        default:
            return "I couldn't complete that cleanly. Try again, or give me one more detail."
        }
    }

    static func isPureConversation(_ text: String?) -> Bool {
        guard let text else { return false }
        let compact = compacted(text)
        return [
            "hi", "hello", "hey", "yo", "sup", "bonjour", "salut", "allo",
            "ok", "okay", "thanks", "thankyou", "merci"
        ].contains(compact)
    }

    private static func compacted(_ text: String) -> String {
        text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: #"[^a-z0-9]+"#, with: "", options: .regularExpression)
    }
}
