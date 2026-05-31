import Foundation

nonisolated enum FinalIntentValidator {
    static func validate(_ text: String, routing: IntentRoutingDecision, fallback: String?) -> String {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = clean.lowercased()

        if isValid(clean, lower: lower, for: routing) {
            return clean
        }

        if let fallback {
            let fallbackClean = fallback.trimmingCharacters(in: .whitespacesAndNewlines)
            let fallbackLower = fallbackClean.lowercased()
            if isValid(fallbackClean, lower: fallbackLower, for: routing) {
                return fallbackClean
            }
        }

        return safeMessage(for: routing)
    }

    private static func isValid(_ text: String, lower: String, for routing: IntentRoutingDecision) -> Bool {
        guard !text.isEmpty else { return false }
        guard !AssistantOutputSanitizer.isLeakedToolJSONArtifact(text) else { return false }
        guard !looksLikeCalendarLeak(lower, unless: routing.intent == .calendar) else { return false }
        guard !looksLikeWeatherLeak(lower, unless: routing.intent == .weather) else { return false }
        guard !looksLikeEmailLeak(lower, unless: routing.intent == .emailDraft || routing.intent == .outlook) else { return false }
        guard !looksLikeWebSearchLeak(lower, unless: routing.intent == .webSearch) else { return false }

        switch routing.intent {
        case .weather:
            return containsAny(lower, ["weather", "temperature", "humidity", "wind", "feels like", "°c", "rain", "snow", "cloud"])
        case .webSearch:
            return containsAny(lower, ["web", "search", "result", "http", "source", "found", "not available", "no direct answer", "try a different phrasing"])
        case .emailDraft:
            if lower.contains("i will be in touch soon") { return false }
            return !looksLikeCalendarLeak(lower, unless: false) && !looksLikeWeatherLeak(lower, unless: false)
        case .messageDraft:
            return !looksLikeCalendarLeak(lower, unless: false) && !looksLikeWeatherLeak(lower, unless: false) && !looksLikeWebSearchLeak(lower, unless: false)
        case .phoneCall:
            return containsAny(lower, ["call", "phone", "contact", "requires", "unavailable", "couldn’t", "couldn't"])
        case .contactSearch:
            return containsAny(lower, ["contact", "phone", "email", "found", "unavailable", "couldn’t", "couldn't"])
        case .calendar:
            return containsAny(lower, ["calendar", "event", "schedule", "meeting", "appointment", "requires explicit user approval", "did not create"])
        case .reminder:
            return containsAny(lower, ["reminder", "todo", "to-do", "requires explicit user approval", "did not create"])
        case .maps:
            return containsAny(lower, ["map", "maps", "direction", "route", "near", "nearby", "location", "current location", "coordinates", "latitude", "longitude", "place", "unavailable", "couldn’t", "couldn't"])
        case .photos:
            return containsAny(lower, ["photo", "photos", "picture", "image", "library", "unavailable", "couldn’t", "couldn't"])
        case .camera:
            return containsAny(lower, ["camera", "photo", "picture", "capture", "unavailable", "couldn’t", "couldn't"])
        case .health:
            return containsAny(lower, ["health", "steps", "sleep", "heart", "energy", "walking", "unavailable", "couldn’t", "couldn't"])
        case .motion:
            return containsAny(lower, ["motion", "activity", "walking", "running", "stationary", "unavailable", "couldn’t", "couldn't"])
        case .files:
            return containsAny(lower, ["file", "document", "read", "local", "unavailable", "couldn’t", "couldn't"])
        case .memory:
            return containsAny(lower, ["memory", "remember", "recall", "saved", "unavailable", "couldn’t", "couldn't"])
        case .rag:
            let hasRagTopic = containsAny(lower, ["search", "index", "indexed", "files", "photos", "local"])
            let hasGrounding = containsAny(lower, ["[1]", "[2]", "snippet", "source", "retrieved", "file", "pdf", "note", "module", "modules"])
            let explicitUnavailable = containsAny(lower, ["unavailable", "couldn’t", "couldn't", "no relevant", "no matching"]) 
            return (hasRagTopic && hasGrounding) || explicitUnavailable
        case .trigger:
            return containsAny(lower, ["trigger", "scheduled", "agent", "background", "cancel", "unavailable", "couldn’t", "couldn't"])
        case .alarm:
            return containsAny(lower, ["alarm", "timer", "countdown", "snooze", "pause", "resume", "stop", "authorization", "unavailable", "couldn’t", "couldn't"])
        case .outlook:
            return containsAny(lower, [
                "outlook", "hotmail", "microsoft", "graph", "email", "mail", "message", "inbox", "subject:", "from:", "received:",
                "unread", "attachment", "draft", "sent", "reply", "forward", "archive", "deleted", "moved", "marked", "folder",
                "requires explicit user approval", "not signed in", "sign in", "unavailable", "couldn’t", "couldn't", "failed"
            ])
        case .note:
            return !looksLikeCalendarLeak(lower, unless: false) && !looksLikeWeatherLeak(lower, unless: false)
        case .chat, .unknown:
            return true
        }
    }

    private static func safeMessage(for routing: IntentRoutingDecision) -> String {
        switch routing.intent {
        case .weather:
            return "I couldn’t safely complete the current weather request. Please enable location/weather access or tell me the city."
        case .webSearch:
            return "No direct answer from web search. Try a different phrasing, or provide a URL to fetch directly."
        case .emailDraft:
            return routing.clarificationPrompt ?? "Who should I send it to, and what should it say?"
        case .messageDraft:
            return routing.clarificationPrompt ?? "Who should I message, and what should it say?"
        case .phoneCall:
            return routing.clarificationPrompt ?? "I couldn’t safely complete the phone call request."
        case .contactSearch:
            return "I couldn’t safely complete the contact lookup request."
        case .calendar:
            return "I couldn’t safely complete the calendar request."
        case .reminder:
            return "I couldn’t safely complete the reminder request."
        case .maps:
            return "I couldn’t safely complete the maps/location request."
        case .photos:
            return "I couldn’t safely complete the photo-library request."
        case .camera:
            return "I couldn’t safely complete the camera request."
        case .health:
            return "I couldn’t safely complete the health request."
        case .motion:
            return "I couldn’t safely complete the motion/activity request."
        case .files:
            return "I couldn’t safely complete the file request."
        case .memory:
            return "I couldn’t safely complete the memory request."
        case .rag:
            return "I couldn’t safely complete the local search/indexing request."
        case .trigger:
            return "I couldn’t safely complete the scheduled-agent request."
        case .alarm:
            return "I couldn’t safely complete the alarm/timer request."
        case .outlook:
            return routing.clarificationPrompt ?? "I couldn’t safely complete the Outlook/Hotmail mail request. Make sure Outlook is signed in, then try again."
        case .note:
            return "I couldn’t safely complete the note request."
        case .chat, .unknown:
            return "I hit a routing error. Please try again."
        }
    }

    private static func looksLikeCalendarLeak(_ lower: String, unless allowed: Bool) -> Bool {
        !allowed && containsAny(lower, ["created a new event", "successfully created", "calendar event", "will start in", "starts in 5 minutes"])
    }

    private static func looksLikeWeatherLeak(_ lower: String, unless allowed: Bool) -> Bool {
        !allowed && containsAny(lower, ["weather for", "weather at", "temperature", "humidity", "feels like", "wind ", "clear sky"])
    }

    private static func looksLikeEmailLeak(_ lower: String, unless allowed: Bool) -> Bool {
        !allowed && containsAny(lower, ["dear ", "subject:", "best regards", "sincerely", "i will be in touch soon"])
    }

    private static func looksLikeWebSearchLeak(_ lower: String, unless allowed: Bool) -> Bool {
        !allowed && containsAny(lower, ["web search", "search result", "http://", "https://"])
    }

    private static func containsAny(_ value: String, _ needles: [String]) -> Bool {
        needles.contains { value.contains($0) }
    }
}
