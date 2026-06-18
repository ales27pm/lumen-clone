import Foundation

nonisolated enum ToolObservationFinalizer {
    /// Converts a raw tool observation into a sanitized, intent-appropriate final string.
    /// - Returns: A formatted observation string, or `nil` if the observation is empty, unsafe, or does not match the expected intent.
    static func immediateFinalIfSafe(intent: UserIntent, toolID: String, observation: String, originalPrompt: String) -> String? {
        let canonicalTool = ToolRouteGuard.canonicalToolID(toolID)
        let cleanObservation = ModelOutputSanitizer.stripHiddenBlocksPreservingPayloadMarkers(observation)
        guard !cleanObservation.isEmpty else { return nil }
        guard !looksUnsafe(WebRichContentPayload.removingMarkers(from: cleanObservation)) else { return nil }

        let lowerPrompt = originalPrompt.lowercased()
        let payloadMarkers = WebRichContentPayload.decodeAll(from: cleanObservation).map { $0.encodedMarker() }.joined()
        let plainObservation = WebRichContentPayload.removingMarkers(from: cleanObservation).trimmingCharacters(in: .whitespacesAndNewlines)

        switch canonicalTool {
        case "weather":
            guard intent == .weather else { return nil }
            return "Weather update: \(plainObservation)\(payloadMarkers)"
        case "location.current":
            guard intent == .weather || intent == .maps else { return nil }
            return "Current location: \(plainObservation)\(payloadMarkers)"
        case "web.search":
            guard intent == .webSearch else { return nil }
            if asksForDeepSynthesis(lowerPrompt) { return nil }
            return "Web search results:\n\(compactWebResults(from: cleanObservation, fallback: plainObservation))\(payloadMarkers)"
        case "web.fetch":
            guard intent == .webSearch else { return nil }
            if asksForDeepSynthesis(lowerPrompt) { return nil }
            return "Fetched page summary:\n\(plainObservation)\(payloadMarkers)"
        case "outlook.messages.list":
            guard intent == .outlook else { return nil }
            return "Outlook messages:\n\(plainObservation)\(payloadMarkers)"
        case "outlook.message.read":
            guard intent == .outlook else { return nil }
            return "Outlook message:\n\(plainObservation)\(payloadMarkers)"
        case "reminders.list":
            guard intent == .reminder else { return nil }
            return "Reminders:\n\(plainObservation)\(payloadMarkers)"
        case "memory.save":
            guard intent == .memory || intent == .note else { return nil }
            return "Saved to memory: \(plainObservation)\(payloadMarkers)"
        case "memory.recall":
            guard intent == .memory || intent == .note else { return nil }
            return "Memory recall:\n\(plainObservation)\(payloadMarkers)"
        case "maps.search":
            guard intent == .maps else { return nil }
            return "Maps search results:\n\(plainObservation)\(payloadMarkers)"
        case "maps.directions":
            guard intent == .maps else { return nil }
            return "Maps directions:\n\(plainObservation)\(payloadMarkers)"
        case "rag.index_files":
            guard intent == .rag else { return nil }
            return "Local file index updated: \(plainObservation)\(payloadMarkers)"
        case "rag.index_photos":
            guard intent == .rag else { return nil }
            return "Photo index updated: \(plainObservation)\(payloadMarkers)"
        case "rag.search":
            guard intent == .rag else { return nil }
            return "RAG search results:\n\(groundedRAGObservation(plainObservation))\(payloadMarkers)"
        case "alarm.authorization_status":
            guard intent == .alarm else { return nil }
            return "Alarm authorization status: \(plainObservation)\(payloadMarkers)"
        case "alarm.list":
            guard intent == .alarm else { return nil }
            return "Active alarms:\n\(plainObservation)\(payloadMarkers)"
        case "trigger.create":
            guard intent == .trigger else { return nil }
            return "Trigger scheduled: \(plainObservation)\(payloadMarkers)"
        case "trigger.list":
            guard intent == .trigger else { return nil }
            return "Scheduled triggers:\n\(plainObservation)\(payloadMarkers)"
        default:
            return nil
        }
    }

    private static func asksForDeepSynthesis(_ prompt: String) -> Bool {
        ["summarize", "compare", "analyze", "analysis", "deep", "explain", "synthesize", "pros and cons"].contains { prompt.contains($0) }
    }

    /// Determines whether text contains unsafe content markers.
    /// - Returns: `true` if the text contains unsafe markers, `false` otherwise.
    private static func looksUnsafe(_ text: String) -> Bool {
        let lower = text.lowercased()
        return lower.contains("<think") || lower.contains("{\"kind\"") || lower.contains("\"mediakind\"")
    }

    /// Formats and attributes a RAG search observation to the local index.
    ///
    /// - Parameter observation: The raw observation from a RAG search.
    /// - Returns: The observation with local RAG index source attribution and appropriate formatting.
    private static func groundedRAGObservation(_ observation: String) -> String {
        let trimmed = observation.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "No matching local snippets were retrieved. Source: local RAG index." }
        let lower = trimmed.lowercased()
        if trimmed.contains("[") || lower.contains("snippet") || lower.contains("source") {
            return trimmed
        }
        if lower.contains("no matching") || lower.contains("no relevant") {
            return "\(trimmed) Source: local RAG index; no matching module snippets were retrieved."
        }
        return "[1] \(trimmed)\nSource: local RAG index snippet."
    }

    /// Formats search results into a compact list, or displays fallback text when no results are available.
    /// - Parameters:
    ///   - text: A string containing encoded `WebRichContentPayload` items.
    ///   - fallback: Alternative text to display if no search results are found.
    /// - Returns: A formatted string containing up to five search results with titles, optional URLs, and snippets, or the first twelve non-empty lines of fallback text.
    private static func compactWebResults(from text: String, fallback: String) -> String {
        let payloads = WebRichContentPayload.decodeAll(from: text)
        if let payload = payloads.first(where: { $0.kind == .searchResults }), !payload.results.isEmpty {
            return payload.results.prefix(5).enumerated().map { index, result in
                var lines = ["\(index + 1). \(result.title)"]
                if let url = result.url, !url.isEmpty { lines.append(url) }
                if let snippet = result.snippet, !snippet.isEmpty { lines.append(snippet) }
                return lines.joined(separator: "\n")
            }.joined(separator: "\n\n")
        }
        return fallback
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .prefix(12)
            .joined(separator: "\n")
    }
}
