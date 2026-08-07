import Foundation

nonisolated struct FinalIntentValidationOutcome: Sendable, Equatable {
    let text: String
    let acceptedCandidate: Bool
    let replacementSource: String
    let rejectionReason: String?
}

nonisolated enum FinalIntentValidator {
    static func validate(_ text: String, routing: IntentRoutingDecision, fallback: String?) -> String {
        validateWithOutcome(text, routing: routing, fallback: fallback).text
    }

    static func validateWithOutcome(_ text: String, routing: IntentRoutingDecision, fallback: String?) -> FinalIntentValidationOutcome {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = clean.lowercased()

        if isValid(clean, lower: lower, for: routing) || isSafeToolObservation(clean, lower: lower, for: routing) {
            return FinalIntentValidationOutcome(
                text: clean,
                acceptedCandidate: true,
                replacementSource: "candidate",
                rejectionReason: nil
            )
        }

        let rejectionReason = replacementReason(for: clean, lower: lower, routing: routing)
        if let fallback {
            let fallbackClean = fallback.trimmingCharacters(in: .whitespacesAndNewlines)
            let fallbackLower = fallbackClean.lowercased()
            if isValid(fallbackClean, lower: fallbackLower, for: routing) || isSafeToolObservation(fallbackClean, lower: fallbackLower, for: routing) {
                emitReplacementDiagnostic(intent: routing.intent, candidateLength: clean.count, replacementSource: "fallback", reason: rejectionReason)
                return FinalIntentValidationOutcome(
                    text: fallbackClean,
                    acceptedCandidate: false,
                    replacementSource: "fallback",
                    rejectionReason: rejectionReason
                )
            }
        }

        let safe = safeMessage(for: routing)
        emitReplacementDiagnostic(intent: routing.intent, candidateLength: clean.count, replacementSource: "safeMessage", reason: rejectionReason)
        return FinalIntentValidationOutcome(
            text: safe,
            acceptedCandidate: false,
            replacementSource: "safeMessage",
            rejectionReason: rejectionReason
        )
    }

    /// Determines whether the candidate text is valid for the given intent routing.
    ///
    /// The text is considered valid if it is non-empty, passes all leak filters, and contains intent-specific keywords or patterns indicating compatibility with the routed intent.
    ///
    /// - Parameters:
    ///   - text: The candidate text to validate.
    ///   - lower: The lowercased version of the candidate text.
    ///   - routing: The intent routing decision that determines which validation rules apply.
    /// - Returns: `true` if the text meets all validation criteria for the intent, `false` otherwise.
    private static func isValid(_ text: String, lower: String, for routing: IntentRoutingDecision) -> Bool {
        guard !text.isEmpty else { return false }
        guard passesLeakFilters(text: text, lower: lower, routing: routing) else { return false }

        if routing.requiresClarification,
           let clarification = routing.clarificationPrompt?.trimmingCharacters(in: .whitespacesAndNewlines),
           !clarification.isEmpty,
           text == clarification {
            return true
        }

        if lower.hasPrefix("approval required for")
            || lower.hasPrefix("this tool requires explicit user approval before it can run:") {
            return isValidApprovalBoundaryFinal(text, lower: lower, routing: routing)
        }

        switch routing.intent {
        case .weather:
            return containsAny(lower, ["weather", "temperature", "humidity", "wind", "feels like", "°c", "rain", "snow", "cloud", "gps", "location access", "network", "timeout", "unreachable", "open-meteo", "service unavailable"])
        case .webSearch:
            return containsAny(lower, ["web", "search", "result", "http", "source", "found", "not available", "no direct answer", "try a different phrasing"])
        case .emailDraft:
            if lower.contains("i will be in touch soon") { return false }
            return !looksLikeCalendarLeak(lower, unless: false) && !looksLikeWeatherLeak(lower, unless: false)
        case .messageDraft:
            return !looksLikeCalendarLeak(lower, unless: false) && !looksLikeWeatherLeak(lower, unless: false) && !looksLikeWebSearchLeak(lower, unless: false)
        case .phoneCall:
            return containsAny(lower, ["call", "phone", "contact", "contact found", "contact search results", "requires", "unavailable", "couldn’t", "couldn't"])
        case .contactSearch:
            return containsAny(lower, ["contact", "contact found", "contact search results", "phone", "email", "found", "unavailable", "couldn’t", "couldn't"])
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
            return containsAny(lower, ["memory", "remember", "recall", "saved", "user's name", "no matching memories", "unavailable", "couldn’t", "couldn't"])
        case .rag:
            let hasRagTopic = containsAny(lower, ["search", "index", "indexed", "files", "photos", "local"])
            let hasGrounding = containsAny(lower, ["[1]", "[2]", "snippet", "source", "retrieved", "file", "pdf", "note", "module", "modules"])
            let hasIndexCompletion = containsAny(lower, ["index updated", "index cleared", "indexed", "reindexed"])
            let explicitUnavailable = containsAny(lower, ["unavailable", "couldn’t", "couldn't", "no relevant", "no matching"])
            return (hasRagTopic && hasGrounding) || hasIndexCompletion || explicitUnavailable
        case .trigger:
            return containsAny(lower, ["trigger", "scheduled", "agent", "background", "cancel", "unavailable", "couldn’t", "couldn't"])
        case .alarm:
            return containsAny(lower, ["alarm", "timer", "countdown", "snooze", "pause", "resume", "stop", "authorization", "unavailable", "couldn’t", "couldn't"])
        case .outlook:
            return containsAny(lower, [
                "outlook", "hotmail", "microsoft", "graph", "email", "mail", "message", "inbox", "subject:", "from:", "received:",
                "unread", "attachment", "draft", "sent", "reply", "forward", "archive", "deleted", "moved", "marked", "folder",
                "requires explicit user approval", "not signed in", "sign in", "missing outlook message context", "unavailable", "couldn’t", "couldn't", "failed"
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
            return routing.clarificationPrompt ?? "Weather tool output could not be validated. Try again or provide a city."
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
            return "I couldn’t safely complete the calendar event request."
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
            return routing.clarificationPrompt ?? "Memory tool output could not be validated."
        case .rag:
            return routing.clarificationPrompt ?? "I couldn’t safely complete the local search/indexing request."
        case .trigger:
            return "I couldn’t safely complete the scheduled-agent request."
        case .alarm:
            return "I couldn’t safely complete the alarm/timer request."
        case .outlook:
            return routing.clarificationPrompt ?? "Outlook tool output could not be validated."
        case .note:
            return "I couldn’t safely complete the note request."
        case .chat, .unknown:
            return "I hit a routing error. Please try again."
        }
    }


    private static func isSafeToolObservation(_ text: String, lower: String, for routing: IntentRoutingDecision) -> Bool {
        guard !text.isEmpty else { return false }
        guard passesLeakFilters(text: text, lower: lower, routing: routing) else { return false }
        switch routing.intent {
        case .weather:
            return containsAny(lower, ["gps signal timeout", "location access was denied", "location permission", "network unreachable", "weather service unavailable", "open-meteo", "geocod", "couldn't get your current location", "couldn’t get your current location"])
        case .calendar:
            return containsAny(lower, ["calendar events:", "calendar event:", "no upcoming events", "no calendar events"])
                || (text.contains("•") && containsAny(lower, [" at ", "am", "pm", "202"]))
        case .maps:
            return containsAny(lower, ["maps search results:", "maps directions:", "current location:"])
        case .motion:
            return containsAny(lower, ["motion activity:", "no motion data", "motion permission", "motion activity is unavailable"])
        case .memory:
            return containsAny(lower, [
                "saved:",
                "memory recall:",
                "i remember that",
                "remember that",
                "no matching memories",
                "memory unavailable",
                "user's name",
                "remembered"
            ])
        case .outlook:
            return containsAny(lower, [
                "outlook is not signed in", "missing outlook message context", "outlook tool failed",
                "authentication expired", "authorization expired", "oauth expired", "oauth sign in",
                "not connected", "no messages", "outlook attachments:", "outlook folders:", "outlook search results:", "outlook status:"
            ])
        case .phoneCall, .contactSearch, .emailDraft, .messageDraft:
            return containsAny(lower, ["contact found:", "contact search results:"])
        default:
            return false
        }
    }

    private static func passesLeakFilters(text: String, lower: String, routing: IntentRoutingDecision) -> Bool {
        guard !AssistantOutputSanitizer.isLeakedToolJSONArtifact(text) else { return false }
        guard !looksLikeCredentialLeak(lower) else { return false }
        guard !looksLikeCalendarLeak(lower, unless: routing.intent == .calendar) else { return false }
        guard !looksLikeWeatherLeak(lower, unless: routing.intent == .weather) else { return false }
        guard !looksLikeEmailLeak(lower, unless: routing.intent == .emailDraft || routing.intent == .outlook) else { return false }
        guard !looksLikeWebSearchLeak(lower, unless: routing.intent == .webSearch) else { return false }
        return true
    }

    private static func isValidApprovalBoundaryFinal(
        _ text: String,
        lower: String,
        routing: IntentRoutingDecision
    ) -> Bool {
        guard passesLeakFilters(text: text, lower: lower, routing: routing) else { return false }
        guard let reference = approvalBoundaryReference(in: lower) else { return false }
        let canonicalToolID = ToolRouteGuard.canonicalToolID(reference.toolID)
        let allowedToolIDs = Set(routing.allowedToolIDs.map(ToolRouteGuard.canonicalToolID))
        guard allowedToolIDs.contains(canonicalToolID) else { return false }
        guard ToolRouteGuard.requiresUserApproval(canonicalToolID) else { return false }
        guard !hasContradictoryExecutionEvidence(reference.postToolText, canonicalToolID: canonicalToolID) else {
            return false
        }
        if reference.usesTrustedGenericHeader,
           hasOnlyBoundaryPunctuation(reference.postToolText) {
            return true
        }
        return hasExplicitNonExecutionEvidence(reference.postToolText, canonicalToolID: canonicalToolID)
    }

    private static func approvalBoundaryReference(
        in lower: String
    ) -> (toolID: String, postToolText: String, usesTrustedGenericHeader: Bool)? {
        let pattern = #"^(approval required for\s+|this tool requires explicit user approval before it can run:\s*)([a-z0-9][a-z0-9._-]*)"#
        guard let expression = try? NSRegularExpression(pattern: pattern) else { return nil }
        let fullRange = NSRange(lower.startIndex..<lower.endIndex, in: lower)
        guard let match = expression.firstMatch(in: lower, range: fullRange),
              let prefixRange = Range(match.range(at: 1), in: lower),
              let toolRange = Range(match.range(at: 2), in: lower) else {
            return nil
        }
        let prefix = String(lower[prefixRange])
        return (
            toolID: String(lower[toolRange]).trimmingCharacters(in: CharacterSet(charactersIn: ".")),
            postToolText: String(lower[toolRange.upperBound...]),
            usesTrustedGenericHeader: prefix.hasPrefix("this tool requires explicit user approval before it can run:")
        )
    }

    private static func hasOnlyBoundaryPunctuation(_ postToolText: String) -> Bool {
        let allowed = CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: ".,:;!?") )
        return postToolText.unicodeScalars.allSatisfy(allowed.contains)
    }

    private static func hasContradictoryExecutionEvidence(_ postToolText: String, canonicalToolID: String) -> Bool {
        let commonSuccessMarkers = [
            "completed successfully",
            "executed successfully",
            "ran successfully",
            "was executed successfully",
            "has been executed",
            "successfully created",
            "successfully sent",
            "successfully scheduled"
        ]
        if containsAny(postToolText, commonSuccessMarkers) {
            return true
        }
        if canonicalToolID == "rag.index_files" || canonicalToolID == "rag.index_photos" {
            return containsAny(postToolText, [
                "index updated",
                "updated successfully",
                "indexing completed",
                "indexing succeeded",
                "indexed successfully",
                "reindex completed",
                "reindexed successfully"
            ])
        }
        return false
    }

    private static func hasExplicitNonExecutionEvidence(_ postToolText: String, canonicalToolID: String) -> Bool {
        if canonicalToolID == "rag.index_files" || canonicalToolID == "rag.index_photos" {
            return containsAny(postToolText, [
                "i did not run it",
                "i didn't run it",
                "i didn’t run it",
                "it was not run",
                "it has not been run"
            ])
        }
        return containsAny(postToolText, [
            "i did not ",
            "i didn't ",
            "i didn’t ",
            "it was not run",
            "it has not been run",
            "after you approve",
            "after approval"
        ])
    }

    private static func replacementReason(for text: String, lower: String, routing: IntentRoutingDecision) -> String {
        if text.isEmpty { return "empty-candidate" }
        if AssistantOutputSanitizer.isLeakedToolJSONArtifact(text) { return "tool-json-leak" }
        if looksLikeCredentialLeak(lower) { return "credential-leak" }
        if looksLikeCalendarLeak(lower, unless: routing.intent == .calendar) { return "calendar-leak" }
        if looksLikeWeatherLeak(lower, unless: routing.intent == .weather) { return "weather-leak" }
        if looksLikeEmailLeak(lower, unless: routing.intent == .emailDraft || routing.intent == .outlook) { return "email-leak" }
        if looksLikeWebSearchLeak(lower, unless: routing.intent == .webSearch) { return "web-leak" }
        return "intent-validation-failed"
    }

    private static func emitReplacementDiagnostic(intent: UserIntent, candidateLength: Int, replacementSource: String, reason: String) {
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .finalIntentCandidateReplaced, values: [
            "intent": intent.rawValue,
            "candidateLength": String(candidateLength),
            "replacementSource": replacementSource,
            "reason": reason
        ]))
        RuntimeFallbackLogger.record(
            source: "final-intent-validator",
            primaryBehavior: "use model candidate as final user-visible response",
            fallbackBehavior: "replace candidate with validated fallback response",
            reason: reason,
            consequence: "primary model final text was not safe or intent-compatible",
            values: [
                "intent": intent.rawValue,
                "candidateLength": String(candidateLength),
                "replacementSource": replacementSource
            ]
        )
    }

    private static func looksLikeCredentialLeak(_ lower: String) -> Bool {
        let secretTerminator = #"(?![a-z0-9._~+/=-])"#
        if lower.range(
            of: ##"(?i)(?:^|[\s{"',])"?(?:access_token|refresh_token|id_token|client_secret|api_key)"?\s*[:=]\s*"?[a-z0-9._~+/=-]{8,}"?"## + secretTerminator,
            options: .regularExpression
        ) != nil {
            return true
        }
        if lower.range(
            of: #"(?i)\bauthorization\s*:\s*bearer\s+[a-z0-9._~+/=-]{16,}"# + secretTerminator,
            options: .regularExpression
        ) != nil {
            return true
        }
        if lower.range(
            of: #"(?i)\bbearer\s+[a-z0-9._~+/=-]{16,}"# + secretTerminator,
            options: .regularExpression
        ) != nil {
            return true
        }
        return lower.range(
            of: #"(?i)\beyj[a-z0-9_-]{8,}\.[a-z0-9_-]{10,}\.[a-z0-9_-]{10,}"# + secretTerminator,
            options: .regularExpression
        ) != nil
    }

    private static func looksLikeCalendarLeak(_ lower: String, unless allowed: Bool) -> Bool {
        !allowed && containsAny(lower, ["created a new event", "successfully created", "calendar event", "will start in", "starts in 5 minutes"])
    }

    private static func looksLikeWeatherLeak(_ lower: String, unless allowed: Bool) -> Bool {
        !allowed && containsAny(lower, ["weather for", "weather at", "temperature", "humidity", "feels like", "wind ", "clear sky"])
    }

    /// Identifies whether the text appears to contain an email draft.
    /// - Parameters:
    ///   - lower: The lowercase text to examine.
    ///   - allowed: If `true`, bypasses the check and returns `false`; used when email content is expected for the current intent.
    /// - Returns: `true` if the text resembles an email draft and the check is enabled, `false` otherwise.
    private static func looksLikeEmailLeak(_ lower: String, unless allowed: Bool) -> Bool {
        !allowed && containsAny(lower, ["dear ", "subject:", "best regards", "sincerely", "i will be in touch soon"])
    }

    /// Determines whether text appears to contain web search results or URLs.
    /// - Parameters:
    ///   - lower: A lowercased string to check.
    ///   - allowed: When `true`, the function returns `false`.
    /// - Returns: `true` if the text contains web search leak patterns and is not explicitly allowed, `false` otherwise.
    private static func looksLikeWebSearchLeak(_ lower: String, unless allowed: Bool) -> Bool {
        !allowed && containsAny(lower, ["web search", "web result", "web results", "http://", "https://"])
    }

    /// Determines whether the string contains any of the provided substrings.
    /// - Parameters:
    ///   - value: The string to search within.
    ///   - needles: The substrings to search for.
    /// - Returns: `true` if the string contains any of the provided substrings, `false` otherwise.
    private static func containsAny(_ value: String, _ needles: [String]) -> Bool {
        needles.contains { value.contains($0) }
    }
}
