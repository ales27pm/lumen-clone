import Foundation

nonisolated enum IntentClassifierPolicy {
    static func resolve(modelResult: IntentClassificationResult?, deterministic fallback: IntentClassificationResult) -> IntentClassificationResult {
        guard let modelResult else { return sanitized(fallback, source: .deterministicFallback) }
        if modelResult.requiresClarification {
            return sanitized(modelResult, source: .bundledModel)
        }
        if let clarification = clarificationForAmbiguousModelResult(modelResult, fallback: fallback) {
            return sanitized(clarification, source: .policyMerged)
        }
        if modelResult.confidence >= 0.72 {
            return sanitized(modelResult, source: .bundledModel)
        }
        if modelResult.confidence >= 0.50 {
            return resolveMediumConfidence(modelResult: modelResult, fallback: fallback)
        }
        return sanitized(fallback, source: .deterministicFallback)
    }

    private static func resolveMediumConfidence(modelResult: IntentClassificationResult, fallback: IntentClassificationResult) -> IntentClassificationResult {
        if modelResult.intent == fallback.intent {
            return sanitized(IntentClassificationResult(intent: modelResult.intent, confidence: min(0.99, modelResult.confidence + 0.05), alternatives: modelResult.alternatives, requiresClarification: modelResult.requiresClarification || fallback.requiresClarification, clarificationPrompt: modelResult.clarificationPrompt ?? fallback.clarificationPrompt, source: .policyMerged, diagnostics: "merged:agree"), source: .policyMerged)
        }
        if isApprovalSensitive(fallback.intent) {
            return sanitized(fallback, source: .policyMerged)
        }
        if isSemanticNonDestructive(modelResult.intent), modelResult.confidence >= 0.65 {
            return sanitized(IntentClassificationResult(intent: modelResult.intent, confidence: modelResult.confidence, alternatives: modelResult.alternatives, requiresClarification: modelResult.requiresClarification, clarificationPrompt: modelResult.clarificationPrompt, source: .policyMerged, diagnostics: "merged:model_preferred"), source: .policyMerged)
        }
        return sanitized(fallback, source: .policyMerged)
    }

    private static func clarificationForAmbiguousModelResult(_ modelResult: IntentClassificationResult, fallback: IntentClassificationResult) -> IntentClassificationResult? {
        let ranked = rankedAlternatives(for: modelResult, fallback: fallback)
        guard ranked.count >= 2 else { return nil }
        let first = ranked[0]
        let second = ranked[1]
        let gap = first.confidence - second.confidence
        let closeEnough = modelResult.confidence >= 0.50 && gap <= 0.12
        let highConfidenceButClose = modelResult.confidence >= 0.72 && gap <= 0.08
        guard closeEnough || highConfidenceButClose else { return nil }
        guard first.intent != second.intent else { return nil }
        guard shouldClarifyBetween(first.intent, second.intent) else { return nil }

        return IntentClassificationResult(
            intent: first.intent,
            confidence: first.confidence,
            alternatives: ranked,
            requiresClarification: true,
            clarificationPrompt: clarificationPrompt(primary: first.intent, secondary: second.intent),
            source: .policyMerged,
            diagnostics: "merged:ambiguous_intent"
        )
    }

    private static func rankedAlternatives(for modelResult: IntentClassificationResult, fallback: IntentClassificationResult) -> [IntentAlternative] {
        let primary = IntentAlternative(intent: modelResult.intent, confidence: min(max(modelResult.confidence, 0.0), 1.0))
        let fallbackAlternative = IntentAlternative(intent: fallback.intent, confidence: min(max(fallback.confidence, 0.0), 1.0))
        return (modelResult.alternatives + [primary, fallbackAlternative])
            .filter { $0.confidence.isFinite && $0.confidence >= 0 }
            .sorted { $0.confidence > $1.confidence }
            .reduce(into: [IntentAlternative]()) { acc, item in
                if !acc.contains(where: { $0.intent == item.intent }) {
                    acc.append(item)
                }
            }
    }

    private static func shouldClarifyBetween(_ first: UserIntent, _ second: UserIntent) -> Bool {
        if first == .chat || first == .unknown || second == .chat || second == .unknown {
            return false
        }
        return IntentToolMapping.allowedToolIDs(for: first) != IntentToolMapping.allowedToolIDs(for: second)
    }

    private static func clarificationPrompt(primary: UserIntent, secondary: UserIntent) -> String {
        "Do you mean \(clarificationLabel(for: primary)) or \(clarificationLabel(for: secondary))?"
    }

    private static func clarificationLabel(for intent: UserIntent) -> String {
        switch intent {
        case .weather: return "weather information"
        case .webSearch: return "a web search"
        case .emailDraft: return "an email draft"
        case .messageDraft: return "a message draft"
        case .phoneCall: return "a phone call"
        case .contactSearch: return "a contact lookup"
        case .calendar: return "a calendar event"
        case .reminder: return "a reminder"
        case .maps: return "a nearby place or directions"
        case .photos: return "photo-library search"
        case .camera: return "camera capture"
        case .health: return "health data"
        case .motion: return "motion or activity data"
        case .files: return "a local file"
        case .memory, .note: return "memory or notes"
        case .rag: return "local knowledge search"
        case .trigger: return "a scheduled agent run"
        case .alarm: return "an alarm or timer"
        case .outlook: return "Outlook mail"
        case .chat, .unknown: return "a direct answer"
        }
    }

    private static func sanitized(_ result: IntentClassificationResult, source: IntentClassificationResult.Source) -> IntentClassificationResult {
        let boundedConfidence = min(max(result.confidence, 0.0), 1.0)
        let primary = IntentAlternative(intent: result.intent, confidence: boundedConfidence)
        let candidates = result.withAllowedAlternatives().alternatives + [primary]
        let alternatives = candidates
            .filter { $0.confidence.isFinite }
            .sorted { $0.confidence > $1.confidence }
            .reduce(into: [IntentAlternative]()) { acc, item in
                if !acc.contains(where: { $0.intent == item.intent }) {
                    acc.append(item)
                }
            }
        return IntentClassificationResult(
            intent: result.intent,
            confidence: boundedConfidence,
            alternatives: Array(alternatives.prefix(5)),
            requiresClarification: result.requiresClarification,
            clarificationPrompt: result.clarificationPrompt,
            source: source,
            diagnostics: result.diagnostics
        )
    }

    private static func isApprovalSensitive(_ intent: UserIntent) -> Bool {
        switch intent {
        case .calendar, .emailDraft, .messageDraft, .phoneCall, .alarm, .trigger, .outlook:
            return true
        default:
            return false
        }
    }

    private static func isSemanticNonDestructive(_ intent: UserIntent) -> Bool {
        switch intent {
        case .weather, .maps, .webSearch, .rag, .memory, .files, .photos, .health, .motion, .camera:
            return true
        default:
            return false
        }
    }
}

nonisolated enum IntentClarificationPolicy {
    static func apply(_ result: IntentClassificationResult, to text: String) -> IntentClassificationResult {
        if result.requiresClarification { return result }
        guard let prompt = clarificationPrompt(for: result.intent, text: text) else { return result }
        return IntentClassificationResult(
            intent: result.intent,
            confidence: min(result.confidence, 0.64),
            alternatives: result.alternatives.isEmpty ? [IntentAlternative(intent: result.intent, confidence: min(result.confidence, 0.64))] : result.alternatives,
            requiresClarification: true,
            clarificationPrompt: prompt,
            source: result.source,
            diagnostics: appendDiagnostic(result.diagnostics, "slot_clarification")
        )
    }

    private static func clarificationPrompt(for intent: UserIntent, text: String) -> String? {
        let value = normalized(text)
        guard !value.isEmpty else { return nil }

        if isBareReferenceAction(value) {
            return "What would you like me to act on?"
        }

        switch intent {
        case .emailDraft:
            let recipient = hasRecipient(value)
            let body = hasContent(value, excluding: ["draft an email", "draft a email", "write an email", "compose email", "send email"])
            if !recipient && !body { return "Who should I send it to, and what should it say?" }
            if !recipient { return "Who should I send it to?" }
            if !body { return "What should the email say?" }
        case .messageDraft:
            let recipient = hasRecipient(value)
            let body = hasContent(value, excluding: ["draft message", "write a message", "compose message", "send a text", "text message"])
            if !recipient && !body { return "Who should I message, and what should it say?" }
            if !recipient { return "Who should I message?" }
            if !body { return "What should the message say?" }
        case .phoneCall:
            if lacksTarget(after: ["call", "dial", "phone", "start a call to", "place a call to"], in: value) {
                return "Who should I call?"
            }
        case .contactSearch:
            if lacksTarget(after: ["find contact", "search contacts", "contact", "phone number for", "email address for"], in: value) {
                return "Which contact should I look up?"
            }
        case .calendar:
            if isCalendarCreateIntent(value), !hasCalendarSubject(value) {
                return "What should the calendar event be?"
            }
        case .reminder:
            if lacksTarget(after: ["remind me to", "remind me", "create a reminder", "set a reminder", "reminder"], in: value) {
                return "What should I remind you about?"
            }
        case .maps:
            if isBareMapRequest(value) {
                return "What place or destination should I look for?"
            }
        case .webSearch:
            if lacksTarget(after: ["search web", "search the web", "web search", "internet search", "look online", "find online", "fetch url", "open url", "read this url"], in: value) {
                return "What should I search for?"
            }
        case .photos:
            if lacksTarget(after: ["search photos", "find photos", "photo library", "pictures from", "photos from"], in: value) {
                return "Which photos should I look for?"
            }
        case .files:
            if lacksTarget(after: ["read file", "open file", "read document", "open document"], in: value) {
                return "Which file should I open?"
            }
        case .memory, .note:
            if isBareMemoryRequest(value) {
                return "What should I save or recall?"
            }
        case .rag:
            if lacksTarget(after: ["search personal data", "search my files", "search local files", "search my documents", "search my notes", "rag search"], in: value) {
                return "What should I search for?"
            }
        case .trigger:
            if lacksTarget(after: ["schedule agent", "create trigger", "agent run", "background agent"], in: value) {
                return "What should the scheduled agent run do?"
            }
        case .alarm:
            guard matchesAny(value, ["alarm", "timer", "countdown", "snooze", "pause", "resume", "stop", "cancel", "authorization", "permission", "auth status", "list alarms", "active alarms"]) else { return nil }
            let kind = AlarmCommandClassifier.classifyAlarmCommandKind(value)
            return AlarmCommandClassifier.clarificationPrompt(for: kind, text: value)
        case .weather, .camera, .health, .motion, .outlook, .chat, .unknown:
            return nil
        }

        return nil
    }

    private static func isBareReferenceAction(_ text: String) -> Bool {
        let exact = [
            "book that", "book it", "schedule that", "schedule it",
            "cancel that", "cancel it", "send it", "send that",
            "do it", "do that", "open it", "open that", "delete it", "delete that",
            "move it", "move that", "that one", "this one"
        ]
        return exact.contains(text)
    }

    private static func isCalendarCreateIntent(_ text: String) -> Bool {
        matchesAny(text, ["schedule", "create event", "add event", "book", "put on my calendar"])
    }

    private static func hasCalendarSubject(_ text: String) -> Bool {
        let stripped = stripPhrases(["schedule", "create event", "add event", "book", "put on my calendar", "calendar"], from: text)
        let withoutTime = stripped.replacingOccurrences(
            of: #"(?i)\b(today|tonight|tomorrow|next week|next month|at \d{1,2}(:\d{2})?\s*(am|pm)?|on \w+)\b"#,
            with: "",
            options: .regularExpression
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        return hasMeaningfulTarget(withoutTime)
    }

    private static func isBareMapRequest(_ text: String) -> Bool {
        let bare = ["directions", "navigate", "route", "maps", "near me", "nearby", "closest", "nearest", "find nearby", "search nearby"]
        if bare.contains(text) { return true }
        return lacksTarget(after: ["directions to", "navigate to", "route to", "find", "search", "locate", "look for"], in: text)
            && matchesAny(text, ["directions", "navigate", "route", "near me", "nearby", "closest", "nearest"])
    }

    private static func isBareMemoryRequest(_ text: String) -> Bool {
        let bare = ["remember", "remember this", "remember that", "save this", "save memory", "recall memory", "note"]
        return bare.contains(text)
    }

    private static func lacksTarget(after phrases: [String], in text: String) -> Bool {
        if text.rangeOfCharacter(from: .decimalDigits) != nil, matchesAny(text, ["call", "phone", "dial"]) {
            return false
        }
        let stripped = stripPhrases(phrases, from: text)
        return !hasMeaningfulTarget(stripped)
    }

    private static func hasRecipient(_ text: String) -> Bool {
        text.range(of: #"(?i)\b(to|for)\s+[a-z0-9@._+\-]{2,}\b"#, options: .regularExpression) != nil
            || text.range(of: #"(?i)\b(email|mail|message|text|sms|imessage)\s+[a-z0-9@._+\-]{2,}\b"#, options: .regularExpression) != nil
            || text.range(of: #"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b"#, options: .regularExpression) != nil
            || text.range(of: #"\+?\d[\d\s().-]{6,}"#, options: .regularExpression) != nil
    }

    private static func hasContent(_ text: String, excluding phrases: [String]) -> Bool {
        let stripped = stripPhrases(phrases, from: text)
        let withoutPrepositionRecipient = stripped.replacingOccurrences(
            of: #"(?i)\b(to|for)\s+[a-z0-9@._+\-]{2,}\b"#,
            with: "",
            options: .regularExpression
        )
        let withoutActionRecipient = withoutPrepositionRecipient.replacingOccurrences(
            of: #"(?i)\b(email|mail|message|text|sms|imessage)\s+[a-z0-9@._+\-]{2,}\b"#,
            with: "",
            options: .regularExpression
        )
        return hasMeaningfulTarget(stripPhrases(["to", "for"], from: withoutActionRecipient))
    }

    private static func hasTime(_ text: String) -> Bool {
        text.range(of: #"(?i)\b(\d{1,2}(:\d{2})?\s*(am|pm)?|noon|midnight|morning|afternoon|evening|tonight|tomorrow|today|in \d+\s+(minutes?|hours?|days?))\b"#, options: .regularExpression) != nil
    }

    private static func lacksTime(_ text: String) -> Bool {
        !hasTime(text)
    }

    private static func stripPhrases(_ phrases: [String], from text: String) -> String {
        phrases.reduce(text) { partial, phrase in
            partial.replacingOccurrences(of: phrase, with: " ")
        }
        .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func hasMeaningfulTarget(_ text: String) -> Bool {
        let stopWords: Set<String> = [
            "a", "an", "the", "my", "me", "to", "for", "with", "about", "on", "at", "in",
            "this", "that", "it", "them", "please", "can", "you", "i", "want", "need", "help"
        ]
        let words = text
            .split { !$0.isLetter && !$0.isNumber && $0 != "@" && $0 != "." && $0 != "_" && $0 != "-" }
            .map { String($0) }
            .filter { !stopWords.contains($0) }
        return words.contains { $0.count >= 2 || $0.rangeOfCharacter(from: .decimalDigits) != nil }
    }

    private static func normalized(_ text: String) -> String {
        text.lowercased()
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func matchesAny(_ text: String, _ patterns: [String]) -> Bool {
        patterns.contains { text.contains($0) }
    }

    private static func appendDiagnostic(_ existing: String?, _ value: String) -> String {
        guard let existing, !existing.isEmpty else { return value }
        return "\(existing)|\(value)"
    }
}
