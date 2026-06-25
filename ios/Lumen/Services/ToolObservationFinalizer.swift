import Foundation

nonisolated struct ToolObservationFinalizationOutcome: Sendable, Equatable {
    let text: String?
    let accepted: Bool
    let rejectionReason: String?
}

nonisolated enum ToolObservationFinalizer {
    static let supportedToolIDs: Set<String> = [
        "weather", "location.current", "web.search", "web.fetch",
        "outlook.messages.list", "outlook.message.read", "outlook.messages.search",
        "outlook.attachments.list", "outlook.folders.list", "outlook.status",
        "calendar.list", "reminders.list", "contacts.search",
        "memory.save", "memory.recall", "maps.search", "maps.directions",
        "motion.activity", "health.summary", "rag.index_files", "rag.index_photos",
        "rag.search", "alarm.authorization_status", "alarm.list",
        "trigger.create", "trigger.list", "trigger.cancel",
        "files.read", "photos.search"
    ]

    static let noFinalizerNeededToolIDs: Set<String> = []

    static let internalOnlyToolIDs: Set<String> = []

    static func finalizerCoverageKind(for tool: ToolDefinition) -> String? {
        let canonical = ToolRouteGuard.canonicalToolID(tool.id)
        if tool.requiresApproval { return "action-only-approval-boundary" }
        if supportedToolIDs.contains(canonical) { return "finalizer" }
        if noFinalizerNeededToolIDs.contains(canonical) { return "no-finalizer-needed" }
        if internalOnlyToolIDs.contains(canonical) { return "internal-only" }
        return nil
    }

    /// Converts a raw tool observation into a sanitized, intent-appropriate final string.
    /// - Returns: A formatted observation string, or `nil` if the observation is empty, unsafe, or does not match the expected intent.
    static func immediateFinalIfSafe(intent: UserIntent, toolID: String, observation: String, originalPrompt: String) -> String? {
        immediateFinalOutcome(intent: intent, toolID: toolID, observation: observation, originalPrompt: originalPrompt).text
    }

    static func immediateFinalOutcome(
        intent: UserIntent,
        tool: ToolDefinition,
        observation: String,
        originalPrompt: String,
        trustedApprovalCaptured: Bool
    ) -> ToolObservationFinalizationOutcome {
        if tool.requiresApproval && !trustedApprovalCaptured {
            return rejected("approval-required")
        }
        return immediateFinalOutcome(
            intent: intent,
            toolID: tool.id,
            observation: observation,
            originalPrompt: originalPrompt
        )
    }

    static func immediateFinalOutcome(intent: UserIntent, toolID: String, observation: String, originalPrompt: String) -> ToolObservationFinalizationOutcome {
        let canonicalTool = ToolRouteGuard.canonicalToolID(toolID)
        let cleanObservation = ModelOutputSanitizer.stripHiddenBlocksPreservingPayloadMarkers(observation)
        guard !cleanObservation.isEmpty else { return rejected("empty-observation") }
        guard !looksUnsafe(WebRichContentPayload.removingMarkers(from: cleanObservation)) else { return rejected("unsafe-observation") }

        let lowerPrompt = originalPrompt.lowercased()
        let payloadMarkers = WebRichContentPayload.decodeAll(from: cleanObservation).map { $0.encodedMarker() }.joined()
        let plainObservation = WebRichContentPayload.removingMarkers(from: cleanObservation).trimmingCharacters(in: .whitespacesAndNewlines)

        switch canonicalTool {
        case "weather":
            guard intent == .weather else { return rejected("intent-mismatch") }
            return accepted("Weather update: \(plainObservation)\(payloadMarkers)")
        case "location.current":
            guard intent == .weather || intent == .maps else { return rejected("intent-mismatch") }
            return accepted("Current location: \(plainObservation)\(payloadMarkers)")
        case "web.search":
            guard intent == .webSearch else { return rejected("intent-mismatch") }
            if asksForDeepSynthesis(lowerPrompt) { return rejected("deep-synthesis-required") }
            return accepted("Web search results:\n\(compactWebResults(from: cleanObservation, fallback: plainObservation))\(payloadMarkers)")
        case "web.fetch":
            guard intent == .webSearch else { return rejected("intent-mismatch") }
            if asksForDeepSynthesis(lowerPrompt) { return rejected("deep-synthesis-required") }
            return accepted("Fetched page summary:\n\(plainObservation)\(payloadMarkers)")
        case "outlook.messages.list":
            guard intent == .outlook else { return rejected("intent-mismatch") }
            return accepted("Outlook messages:\n\(plainObservation)\(payloadMarkers)")
        case "outlook.message.read":
            guard intent == .outlook else { return rejected("intent-mismatch") }
            return accepted("Outlook message:\n\(plainObservation)\(payloadMarkers)")
        case "outlook.messages.search":
            guard intent == .outlook else { return rejected("intent-mismatch") }
            return accepted("Outlook search results:\n\(plainObservation)\(payloadMarkers)")
        case "outlook.attachments.list":
            guard intent == .outlook else { return rejected("intent-mismatch") }
            return accepted("Outlook attachments:\n\(plainObservation)\(payloadMarkers)")
        case "outlook.folders.list":
            guard intent == .outlook else { return rejected("intent-mismatch") }
            return accepted("Outlook folders:\n\(plainObservation)\(payloadMarkers)")
        case "outlook.status":
            guard intent == .outlook else { return rejected("intent-mismatch") }
            return accepted("Outlook status: \(plainObservation)\(payloadMarkers)")
        case "calendar.list":
            guard intent == .calendar else { return rejected("intent-mismatch") }
            return accepted("Calendar events:\n\(plainObservation)\(payloadMarkers)")
        case "reminders.list":
            guard intent == .reminder else { return rejected("intent-mismatch") }
            return accepted("Reminders:\n\(plainObservation)\(payloadMarkers)")
        case "contacts.search":
            guard intent == .contactSearch || intent == .phoneCall || intent == .emailDraft || intent == .messageDraft else { return rejected("intent-mismatch") }
            if plainObservation.lowercased().contains("no contacts match") {
                return accepted("Contact search results:\n\(plainObservation)\(payloadMarkers)")
            }
            if let single = contactSummaries(from: plainObservation).first, contactSummaries(from: plainObservation).count == 1 {
                return accepted("Contact found: \(single)\(payloadMarkers)")
            }
            return accepted("Contact search results:\n\(plainObservation)\(payloadMarkers)")
        case "memory.save":
            guard intent == .memory || intent == .note else { return rejected("intent-mismatch") }
            return accepted("Saved to memory: \(plainObservation)\(payloadMarkers)")
        case "memory.recall":
            guard intent == .memory || intent == .note else { return rejected("intent-mismatch") }
            return accepted("Memory recall:\n\(plainObservation)\(payloadMarkers)")
        case "maps.search":
            guard intent == .maps else { return rejected("intent-mismatch") }
            return accepted("Maps search results:\n\(plainObservation)\(payloadMarkers)")
        case "maps.directions":
            guard intent == .maps else { return rejected("intent-mismatch") }
            return accepted("Maps directions:\n\(plainObservation)\(payloadMarkers)")
        case "motion.activity":
            guard intent == .motion else { return rejected("intent-mismatch") }
            return accepted("Motion activity:\n\(plainObservation)\(payloadMarkers)")
        case "health.summary":
            guard intent == .health else { return rejected("intent-mismatch") }
            return accepted("Health summary:\n\(plainObservation)\(payloadMarkers)")
        case "rag.index_files":
            guard intent == .rag else { return rejected("intent-mismatch") }
            return accepted("Local file index updated: \(plainObservation)\(payloadMarkers)")
        case "rag.index_photos":
            guard intent == .rag else { return rejected("intent-mismatch") }
            return accepted("Photo index updated: \(plainObservation)\(payloadMarkers)")
        case "rag.search":
            guard intent == .rag else { return rejected("intent-mismatch") }
            return accepted("RAG search results:\n\(groundedRAGObservation(plainObservation))\(payloadMarkers)")
        case "files.read":
            guard intent == .files || intent == .rag else { return rejected("intent-mismatch") }
            let prefix = intent == .rag ? "RAG file result" : "File result"
            return accepted("\(prefix):\n\(plainObservation)\(payloadMarkers)")
        case "photos.search":
            guard intent == .photos || intent == .rag else { return rejected("intent-mismatch") }
            let prefix = intent == .rag ? "RAG photo search results" : "Photo search results"
            return accepted("\(prefix):\n\(plainObservation)\(payloadMarkers)")
        case "alarm.authorization_status":
            guard intent == .alarm else { return rejected("intent-mismatch") }
            return accepted("Alarm authorization status: \(plainObservation)\(payloadMarkers)")
        case "alarm.list":
            guard intent == .alarm else { return rejected("intent-mismatch") }
            return accepted("Active alarms:\n\(plainObservation)\(payloadMarkers)")
        case "trigger.create":
            guard intent == .trigger else { return rejected("intent-mismatch") }
            return accepted("Trigger scheduled: \(plainObservation)\(payloadMarkers)")
        case "trigger.list":
            guard intent == .trigger else { return rejected("intent-mismatch") }
            return accepted("Scheduled triggers:\n\(plainObservation)\(payloadMarkers)")
        case "trigger.cancel":
            guard intent == .trigger else { return rejected("intent-mismatch") }
            return accepted("Trigger cancellation:\n\(plainObservation)\(payloadMarkers)")
        default:
            return rejected("unsupported-tool")
        }
    }

    private static func accepted(_ text: String) -> ToolObservationFinalizationOutcome {
        ToolObservationFinalizationOutcome(text: text, accepted: true, rejectionReason: nil)
    }

    private static func rejected(_ reason: String) -> ToolObservationFinalizationOutcome {
        ToolObservationFinalizationOutcome(text: nil, accepted: false, rejectionReason: reason)
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

    private static func contactSummaries(from observation: String) -> [String] {
        observation
            .split(whereSeparator: \.isNewline)
            .map { line in
                line.trimmingCharacters(in: .whitespacesAndNewlines)
                    .replacingOccurrences(of: #"^\s*[•\-]\s*"#, with: "", options: .regularExpression)
            }
            .filter { !$0.isEmpty }
            .filter { !$0.lowercased().contains("no contacts match") }
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
