import Foundation

enum LumenIntentPolicy {
    struct HeadlessPromptDecision: Equatable {
        let requiresOpenApp: Bool
        let reason: String?

        static let allow = HeadlessPromptDecision(requiresOpenApp: false, reason: nil)
    }

    static func requiresOpenAppForSensitiveAction(_ action: String) -> Bool {
        headlessPromptDecision(for: action).requiresOpenApp
    }

    static func openAppReason(forHeadlessPrompt prompt: String) -> String? {
        headlessPromptDecision(for: prompt).reason
    }

    static func headlessPromptDecision(for prompt: String) -> HeadlessPromptDecision {
        let normalized = prompt
            .lowercased()
            .replacingOccurrences(of: #"[\s_\-]+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return .allow }

        if let explicitToolReason = explicitSensitiveToolReason(normalized) {
            return HeadlessPromptDecision(requiresOpenApp: true, reason: explicitToolReason)
        }

        if containsExternalNetworkRequest(normalized) {
            return HeadlessPromptDecision(requiresOpenApp: true, reason: "trigger may require external network access")
        }

        let routing = IntentRouter.classify(normalized)
        if let reason = blockedReason(for: routing.intent, prompt: normalized) {
            return HeadlessPromptDecision(requiresOpenApp: true, reason: reason)
        }

        if containsProtectedPersonalDataRequest(normalized) {
            return HeadlessPromptDecision(requiresOpenApp: true, reason: "trigger may read protected personal data")
        }

        if containsMutationRequest(normalized) {
            return HeadlessPromptDecision(requiresOpenApp: true, reason: "trigger may modify local state")
        }

        return .allow
    }

    private static func blockedReason(for intent: UserIntent, prompt: String) -> String? {
        switch intent {
        case .weather, .webSearch, .outlook:
            return "trigger may require external network access"
        case .emailDraft, .messageDraft, .phoneCall, .calendar, .reminder, .trigger, .alarm:
            return "trigger may require approved actions"
        case .contactSearch, .maps, .photos, .camera, .health, .motion:
            return "trigger may read protected personal data"
        case .rag:
            if prompt.contains("photo") || prompt.contains("index") || prompt.contains("reindex") {
                return "trigger may read protected personal data"
            }
            return nil
        case .files, .memory, .note, .chat, .unknown:
            return nil
        }
    }

    private static func explicitSensitiveToolReason(_ prompt: String) -> String? {
        for tool in ToolRegistry.all {
            let canonical = ToolRouteGuard.canonicalToolID(tool.id)
            guard prompt.contains(canonical) || prompt.contains(canonical.replacingOccurrences(of: ".", with: " ")) else {
                continue
            }
            if canonical == "web.search" || canonical == "web.fetch" {
                return "trigger may require external network access"
            }
            if tool.requiresApproval {
                return "trigger may require approved actions"
            }
            if tool.permissionKey != nil {
                return "trigger may read protected personal data"
            }
            switch tool.category {
            case .communication, .location, .media, .health, .productivity:
                if canonical != "trigger.list" && canonical != "alarm.authorization_status" && canonical != "alarm.list" {
                    return "trigger may read protected personal data"
                }
            case .knowledge:
                if canonical == "rag.index_photos" {
                    return "trigger may read protected personal data"
                }
            }
        }
        return nil
    }

    private static func containsExternalNetworkRequest(_ prompt: String) -> Bool {
        prompt.range(of: #"https?://"#, options: .regularExpression) != nil
            || containsAny(prompt, [
                "web search", "search web", "search the web", "internet search", "look online",
                "find online", "fetch url", "open url", "read url", "read website",
                "weather", "forecast", "outlook", "hotmail", "microsoft graph"
            ])
    }

    private static func containsProtectedPersonalDataRequest(_ prompt: String) -> Bool {
        containsAny(prompt, [
            "contact", "contacts", "address book", "phone number", "email address",
            "location", "where am i", "where are we", "near me", "nearby", "closest", "nearest",
            "map", "maps", "directions", "navigate", "route",
            "photo", "photos", "pictures", "camera",
            "health", "heart rate", "steps", "sleep", "motion activity", "walking", "running",
            "calendar", "schedule", "appointments", "meetings", "reminders", "alarms"
        ])
    }

    private static func containsMutationRequest(_ prompt: String) -> Bool {
        let mutationVerbs = [
            "send", "draft", "compose", "call", "dial", "create", "add", "delete", "remove",
            "archive", "move", "reply", "forward", "mark read", "mark unread", "schedule",
            "cancel", "stop", "pause", "resume", "snooze", "notify", "notification",
            "open url", "open maps", "reindex", "index photos"
        ]
        let actionObjects = [
            "email", "mail", "message", "text", "sms", "phone", "calendar", "event",
            "reminder", "trigger", "alarm", "timer", "outlook", "notification", "url",
            "maps", "camera", "photo", "photos"
        ]
        return mutationVerbs.contains { verb in
            prompt.contains(verb) && actionObjects.contains { prompt.contains($0) }
        }
    }

    private static func containsAny(_ prompt: String, _ needles: [String]) -> Bool {
        needles.contains { prompt.contains($0) }
    }
}
