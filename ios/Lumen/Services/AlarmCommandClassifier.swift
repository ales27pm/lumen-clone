import Foundation

nonisolated enum AlarmCommandKind: Sendable, Hashable {
    case authorizationStatus
    case requestAuthorization
    case list
    case schedule
    case countdown
    case cancel
    case pause
    case resume
    case stop
    case snooze
    case unknown
}

nonisolated enum AlarmCommandClassifier {
    static func classifyAlarmCommandKind(_ normalizedText: String) -> AlarmCommandKind {
        let text = normalized(normalizedText)
        guard !text.isEmpty else { return .unknown }

        if containsAny(text, ["auth status", "authorization status", "permission status", "alarm permission status", "alarm auth status"]) {
            return .authorizationStatus
        }
        if containsAny(text, ["request alarm authorization", "request authorization", "request permission", "ask for alarm authorization", "ask for authorization", "ask for permission"]) {
            return .requestAuthorization
        }
        if containsAny(text, ["list active alarms", "show active alarms", "list alarms", "show alarms", "active alarms", "all alarms", "list alarm"]) {
            return .list
        }
        if containsAny(text, ["pause alarm", "pause the alarm"]) || text.hasPrefix("pause ") { return .pause }
        if containsAny(text, ["resume alarm", "resume the alarm"]) || text.hasPrefix("resume ") { return .resume }
        if containsAny(text, ["stop alarm", "stop the alarm"]) || text.hasPrefix("stop ") { return .stop }
        if containsAny(text, ["snooze alarm", "snooze the alarm"]) || text.hasPrefix("snooze ") { return .snooze }
        if containsAny(text, ["cancel alarm", "cancel the alarm", "delete alarm", "delete the alarm", "remove alarm", "remove the alarm"]) || text.hasPrefix("cancel ") {
            return .cancel
        }
        if containsAny(text, ["countdown", "timer", "start a timer", "start timer"]) {
            return .countdown
        }
        if containsAny(text, ["schedule alarm", "schedule an alarm", "set alarm", "set an alarm", "create alarm", "create an alarm", "add alarm", "add an alarm", "wake me", "wake us"]) {
            return .schedule
        }
        if text == "alarm" || text == "alarms" || text == "set alarm" || text == "set an alarm" {
            return .unknown
        }
        return .unknown
    }

    static func firstUUID(in text: String) -> String? {
        firstCapture(in: text, pattern: #"(\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b)"#)
    }

    static func hasTimeOrDate(_ text: String) -> Bool {
        normalized(text).range(
            of: #"(?i)\b(\d{1,2}(:\d{2})?\s*(am|pm)?|noon|midnight|morning|afternoon|evening|tonight|tomorrow|today|in \d+\s+(seconds?|minutes?|hours?|days?))\b"#,
            options: .regularExpression
        ) != nil
    }

    static func hasDuration(_ text: String) -> Bool {
        normalized(text).range(
            of: #"(?i)\b(\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b"#,
            options: .regularExpression
        ) != nil
    }

    static func hasMutationTarget(_ text: String) -> Bool {
        if firstUUID(in: text) != nil { return true }
        let value = normalized(text)
        if firstCapture(in: value, pattern: #"(?i)\b(?:named|called)\s+([^.!?\n]+)"#) != nil { return true }
        let stripped = value
            .replacingOccurrences(of: #"(?i)\b(cancel|delete|remove|pause|resume|stop|snooze)\b"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: #"(?i)\b(the|an|a|my|alarm|timer|please)\b"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return stripped.split(separator: " ").contains { token in
            token.count >= 2 && token.rangeOfCharacter(from: .decimalDigits) == nil
        }
    }

    static func clarificationPrompt(for kind: AlarmCommandKind, text: String) -> String? {
        switch kind {
        case .authorizationStatus, .requestAuthorization, .list:
            return nil
        case .schedule:
            return hasTimeOrDate(text) ? nil : "What time should I use for the alarm?"
        case .countdown:
            return hasDuration(text) ? nil : "What duration should I use for the timer?"
        case .cancel, .pause, .resume, .stop, .snooze:
            return hasMutationTarget(text) ? nil : "Which alarm should I \(mutationVerb(for: kind))?"
        case .unknown:
            return "What time or duration should I use?"
        }
    }

    private static func mutationVerb(for kind: AlarmCommandKind) -> String {
        switch kind {
        case .cancel: return "cancel"
        case .pause: return "pause"
        case .resume: return "resume"
        case .stop: return "stop"
        case .snooze: return "snooze"
        default: return "update"
        }
    }

    private static func normalized(_ text: String) -> String {
        text.lowercased()
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
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
}
