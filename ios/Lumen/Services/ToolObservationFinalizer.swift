import Foundation

nonisolated enum ToolObservationFinalizer {
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
        case "memory.save":
            guard intent == .memory || intent == .note else { return nil }
            return "Saved to memory: \(plainObservation)\(payloadMarkers)"
        case "memory.recall":
            guard intent == .memory || intent == .note else { return nil }
            return "Memory recall:\n\(plainObservation)\(payloadMarkers)"
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

    private static func looksUnsafe(_ text: String) -> Bool {
        let lower = text.lowercased()
        return lower.contains("<think") || lower.contains("{\"kind\"") || lower.contains("\"mediakind\"")
    }

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
