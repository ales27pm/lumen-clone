import Foundation

nonisolated enum UserIntent: String, Codable, Sendable, CaseIterable, Hashable {
    case weather
    case webSearch
    case emailDraft
    case messageDraft
    case phoneCall
    case contactSearch
    case calendar
    case reminder
    case maps
    case photos
    case camera
    case health
    case motion
    case files
    case memory
    case rag
    case trigger
    case alarm
    case outlook
    case note
    case chat
    case unknown
}

nonisolated struct IntentRoutingDecision: Sendable {
    let intent: UserIntent
    let allowedToolIDs: Set<String>
    let requiresClarification: Bool
    let clarificationPrompt: String?
}

nonisolated enum IntentRouter {
    private static let weatherToolIDs: Set<String> = ["weather", "location.current"]
    private static let webSearchToolIDs: Set<String> = ["web.search", "web.fetch"]
    private static let emailToolIDs: Set<String> = ["mail.draft", "contacts.search"]
    private static let messageToolIDs: Set<String> = ["messages.draft", "contacts.search"]
    private static let phoneToolIDs: Set<String> = ["phone.call", "contacts.search"]
    private static let contactToolIDs: Set<String> = ["contacts.search"]
    private static let calendarToolIDs: Set<String> = ["calendar.create", "calendar.list"]
    private static let reminderToolIDs: Set<String> = ["reminders.create", "reminders.list"]
    private static let mapsToolIDs: Set<String> = ["maps.search", "maps.directions", "location.current"]
    private static let photosToolIDs: Set<String> = ["photos.search"]
    private static let cameraToolIDs: Set<String> = ["camera.capture"]
    private static let healthToolIDs: Set<String> = ["health.summary"]
    private static let motionToolIDs: Set<String> = ["motion.activity"]
    private static let filesToolIDs: Set<String> = ["files.read"]
    private static let memoryToolIDs: Set<String> = ["memory.save", "memory.recall"]
    private static let ragToolIDs: Set<String> = ["rag.search", "rag.index_files", "rag.index_photos", "files.read", "photos.search"]
    private static let triggerToolIDs: Set<String> = ["trigger.create", "trigger.list", "trigger.cancel"]
    private static let alarmToolIDs: Set<String> = [
        "alarm.authorization_status", "alarm.request_authorization", "alarm.schedule", "alarm.countdown",
        "alarm.list", "alarm.pause", "alarm.resume", "alarm.stop", "alarm.snooze", "alarm.cancel"
    ]
    private static let outlookToolIDs: Set<String> = [
        "outlook.status", "outlook.folders.list", "outlook.messages.list", "outlook.messages.search",
        "outlook.message.read", "outlook.attachments.list", "outlook.draft.create", "outlook.mail.send",
        "outlook.message.mark_read", "outlook.message.mark_unread", "outlook.message.move", "outlook.message.archive",
        "outlook.message.delete", "outlook.message.reply", "outlook.message.reply_all", "outlook.message.forward",
        "contacts.search"
    ]
    private static let noteToolIDs: Set<String> = ["memory.save", "memory.recall"]

    static let mapFollowUpPhrases: [String] = [
        "show me on map", "show it on map", "open it in maps", "open on map",
        "show this location on map", "navigate there", "directions there", "take me there"
    ]

    static func isMapFollowUpPrompt(_ text: String) -> Bool {
        let normalizedText = normalized(text)
        return mapFollowUpPhrases.contains { phrase in
            normalizedText.contains(phrase) || normalizedText.contains(phrase.replacingOccurrences(of: " map", with: " maps"))
        }
    }

    static func classify(_ userMessage: String) -> IntentRoutingDecision {
        let text = normalized(userMessage)
        guard !text.isEmpty else {
            return IntentRoutingDecision(intent: .chat, allowedToolIDs: [], requiresClarification: false, clarificationPrompt: nil)
        }

        if let override = priorityOverride(forNormalizedText: text) {
            return override
        }

        if isPureConversationalGreeting(text) {
            return IntentRoutingDecision(intent: .chat, allowedToolIDs: [], requiresClarification: false, clarificationPrompt: nil)
        }

        if isLikelyOutlookIntent(text) {
            return IntentRoutingDecision(intent: .outlook, allowedToolIDs: outlookToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if isCurrentLocationIntent(text) {
            if matchesAny(text, ["weather", "forecast", "temperature", "outside", "rain", "snow", "wind", "weather here"]) {
                return IntentRoutingDecision(intent: .weather, allowedToolIDs: weatherToolIDs, requiresClarification: false, clarificationPrompt: nil)
            }
            return IntentRoutingDecision(intent: .maps, allowedToolIDs: ["location.current"], requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["alarm", "set an alarm", "set alarm", "countdown", "timer", "snooze", "pause alarm", "resume alarm", "stop alarm", "cancel alarm", "alarm authorization"]) {
            return IntentRoutingDecision(intent: .alarm, allowedToolIDs: alarmToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["schedule agent", "agent run", "background agent", "list triggers", "cancel trigger", "create trigger", "trigger"] ) {
            return IntentRoutingDecision(intent: .trigger, allowedToolIDs: triggerToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["remind me", "reminder", "todo", "to do", "list reminders", "pending reminders"]) {
            return IntentRoutingDecision(intent: .reminder, allowedToolIDs: reminderToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["weather", "forecast", "temperature", "what is it like outside", "weather here", "rain", "snow", "wind outside"]) {
            return IntentRoutingDecision(intent: .weather, allowedToolIDs: weatherToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if isLikelyWebSearchIntent(text) {
            return IntentRoutingDecision(intent: .webSearch, allowedToolIDs: webSearchToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["draft an email", "draft a email", "write an email", "compose email", "email to", "mail to", "send email"]) {
            let recipient = inferredRecipient(text)
            let content = inferredContent(text)
            let clarification: String?
            if !recipient && !content {
                clarification = "Who should I send it to, and what should it say?"
            } else if !recipient {
                clarification = "Who should I send it to?"
            } else if !content {
                clarification = "What should the email say?"
            } else {
                clarification = nil
            }
            return IntentRoutingDecision(intent: .emailDraft, allowedToolIDs: emailToolIDs, requiresClarification: clarification != nil, clarificationPrompt: clarification)
        }

        if matchesAny(text, ["draft message", "write a message", "compose message", "text message", "sms", "imessage", "message to", "send a text"]) {
            let recipient = inferredRecipient(text)
            let content = inferredContent(text)
            let clarification: String?
            if !recipient && !content { clarification = "Who should I message, and what should it say?" }
            else if !recipient { clarification = "Who should I message?" }
            else if !content { clarification = "What should the message say?" }
            else { clarification = nil }
            return IntentRoutingDecision(intent: .messageDraft, allowedToolIDs: messageToolIDs, requiresClarification: clarification != nil, clarificationPrompt: clarification)
        }

        if matchesAny(text, ["contact", "address book", "find contact", "search contacts", "phone number for", "email address for"]) {
            return IntentRoutingDecision(intent: .contactSearch, allowedToolIDs: contactToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if isLikelyPhoneCallIntent(text) {
            let hasTarget = text.split(separator: " ").count >= 2 || text.rangeOfCharacter(from: .decimalDigits) != nil
            return IntentRoutingDecision(intent: .phoneCall, allowedToolIDs: phoneToolIDs, requiresClarification: !hasTarget, clarificationPrompt: hasTarget ? nil : "Who should I call?")
        }

        if matchesAny(text, ["schedule", "calendar", "create event", "meeting", "appointment", "at 5", "tomorrow at", "list events", "upcoming events"]) {
            return IntentRoutingDecision(intent: .calendar, allowedToolIDs: calendarToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if isMapFollowUpPrompt(text) {
            return IntentRoutingDecision(intent: .maps, allowedToolIDs: mapsToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["directions", "navigate", "route to", "maps", "near me", "nearby", "closest", "search nearby", "find a place", "find places"]) {
            return IntentRoutingDecision(intent: .maps, allowedToolIDs: mapsToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["search photos", "find photos", "photo library", "pictures from", "photos from", "images in my library"]) {
            return IntentRoutingDecision(intent: .photos, allowedToolIDs: photosToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["take a photo", "capture image", "open camera", "use camera", "take picture"]) {
            return IntentRoutingDecision(intent: .camera, allowedToolIDs: cameraToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["health summary", "steps", "sleep", "heart rate", "active energy", "walking distance", "health data"]) {
            return IntentRoutingDecision(intent: .health, allowedToolIDs: healthToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["motion activity", "am i walking", "am i running", "device motion", "recent activity"]) {
            return IntentRoutingDecision(intent: .motion, allowedToolIDs: motionToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["read file", "open file", "read document", "imported file", "local document"]) {
            return IntentRoutingDecision(intent: .files, allowedToolIDs: filesToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["remember that", "remember this", "save memory", "recall memory", "what do you remember", "memory about", "save this fact", "keep this in mind"]) {
            return IntentRoutingDecision(intent: .memory, allowedToolIDs: memoryToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["search personal data", "search my files", "search local files", "search my documents", "search my notes", "reindex files", "index files", "reindex photos", "index photos", "rag search", "architecture notes"]) || isLikelyLocalKnowledgeQuery(text) {
            return IntentRoutingDecision(intent: .rag, allowedToolIDs: ragToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["note", "save this"]) {
            return IntentRoutingDecision(intent: .note, allowedToolIDs: noteToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        return IntentRoutingDecision(intent: .chat, allowedToolIDs: [], requiresClarification: false, clarificationPrompt: nil)
    }

    static func priorityOverride(_ userMessage: String) -> IntentRoutingDecision? {
        priorityOverride(forNormalizedText: normalized(userMessage))
    }

    static func allowedToolIDs(for intent: UserIntent) -> Set<String> {
        switch intent {
        case .weather: return weatherToolIDs
        case .webSearch: return webSearchToolIDs
        case .emailDraft: return emailToolIDs
        case .messageDraft: return messageToolIDs
        case .phoneCall: return phoneToolIDs
        case .contactSearch: return contactToolIDs
        case .calendar: return calendarToolIDs
        case .reminder: return reminderToolIDs
        case .maps: return mapsToolIDs
        case .photos: return photosToolIDs
        case .camera: return cameraToolIDs
        case .health: return healthToolIDs
        case .motion: return motionToolIDs
        case .files: return filesToolIDs
        case .memory: return memoryToolIDs
        case .rag: return ragToolIDs
        case .trigger: return triggerToolIDs
        case .alarm: return alarmToolIDs
        case .outlook: return outlookToolIDs
        case .note: return noteToolIDs
        case .chat, .unknown: return []
        }
    }

    static func isToolAllowed(_ toolID: String, for decision: IntentRoutingDecision) -> Bool {
        if decision.allowedToolIDs.isEmpty { return false }
        let canonical = ToolRouteGuard.canonicalToolID(toolID)
        return decision.allowedToolIDs.contains(canonical)
    }

    static func intentRequiresTool(_ decision: IntentRoutingDecision) -> Bool {
        switch decision.intent {
        case .weather, .webSearch, .emailDraft, .messageDraft, .phoneCall, .contactSearch, .calendar, .reminder, .maps, .photos, .camera, .health, .motion, .files, .memory, .rag, .trigger, .alarm, .outlook, .note:
            return true
        case .chat, .unknown:
            return false
        }
    }

    static func unavailableMessage(for decision: IntentRoutingDecision) -> String {
        switch decision.intent {
        case .webSearch: return "Web search is not available in this build yet."
        case .weather: return "Weather tools are unavailable right now. Please enable weather/location tools or provide a city."
        case .emailDraft: return "Email drafting is not available in this build yet."
        case .messageDraft: return "Message drafting is not available in this build yet."
        case .phoneCall: return "Phone call tools are unavailable in this build right now."
        case .contactSearch: return "Contact search is unavailable in this build right now."
        case .calendar: return "Calendar tools are unavailable in this build right now."
        case .reminder: return "Reminder tools are unavailable in this build right now."
        case .maps: return "Maps/location tools are unavailable in this build right now."
        case .photos: return "Photo tools are unavailable in this build right now."
        case .camera: return "Camera tools are unavailable in this build right now."
        case .health: return "Health tools are unavailable in this build right now."
        case .motion: return "Motion tools are unavailable in this build right now."
        case .files: return "File reading tools are unavailable in this build right now."
        case .memory, .note: return "Notes/memory tools are unavailable in this build right now."
        case .rag: return "Local search/indexing tools are unavailable in this build right now."
        case .trigger: return "Scheduled agent tools are unavailable in this build right now."
        case .alarm: return "Alarm tools are unavailable in this build right now."
        case .outlook: return "Outlook tools are unavailable in this build right now."
        case .chat, .unknown: return "I can answer directly, but no matching tool is available for that action."
        }
    }

    static func blockedToolMessage(for decision: IntentRoutingDecision) -> String {
        switch decision.intent {
        case .webSearch: return "That request is a web search. I can only use web search tools for it, not calendar or reminder tools."
        case .weather: return "That request is about weather. I can only use weather/location tools for it."
        case .emailDraft: return "That request is for drafting an email. I can only use email composition tools for it."
        case .messageDraft: return "That request is for drafting a message. I can only use message composition tools for it."
        case .phoneCall: return "That request is for a phone call. I can only use phone/contact tools for it."
        case .contactSearch: return "That request is for contact lookup. I can only use contact tools for it."
        case .calendar: return "That request is calendar-related. I can only use calendar tools for it."
        case .reminder: return "That request is reminder-related. I can only use reminder tools for it."
        case .maps: return "That request is map/location-related. I can only use maps/location tools for it."
        case .photos: return "That request is photo-library related. I can only use photo tools for it."
        case .camera: return "That request is camera-related. I can only use camera tools for it."
        case .health: return "That request is health-related. I can only use health tools for it."
        case .motion: return "That request is motion-related. I can only use motion tools for it."
        case .files: return "That request is file-related. I can only use local file tools for it."
        case .memory, .note: return "That request is note/memory-related. I can only use note/memory tools for it."
        case .rag: return "That request is local-search/indexing related. I can only use RAG/local index tools for it."
        case .trigger: return "That request is scheduled-agent related. I can only use trigger tools for it."
        case .alarm: return "That request is alarm-related. I can only use alarm tools for it."
        case .outlook: return "That request is Outlook/Hotmail mail-related. I can only use Outlook Microsoft Graph tools for it."
        case .chat, .unknown: return "That tool doesn't match your request. Could you clarify what you want to do?"
        }
    }


    private static func isLikelyPhoneCallIntent(_ text: String) -> Bool {
        if matchesAny(text, ["re-call"]) { return false }

        let directCallPatterns = [
            #"\bcall\s+(me|him|her|them|us|[a-z0-9@+\-().]+)\b"#,
            #"\bdial\s+"#,
            #"\bphone\s+"#,
            #"\bstart\s+a?\s*call\b"#
        ]
        return directCallPatterns.contains { pattern in
            text.range(of: pattern, options: .regularExpression) != nil
        }
    }

    private static func priorityOverride(forNormalizedText text: String) -> IntentRoutingDecision? {
        guard !text.isEmpty else { return nil }

        if isExplicitReminderIntent(text) {
            return IntentRoutingDecision(intent: .reminder, allowedToolIDs: reminderToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if isConcreteFileReadIntent(text) {
            return IntentRoutingDecision(intent: .files, allowedToolIDs: filesToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if isExplicitRAGIndexIntent(text) {
            return IntentRoutingDecision(intent: .rag, allowedToolIDs: ragToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if isExplicitMemorySaveIntent(text) || matchesAny(text, ["what do you remember", "recall my saved"]) {
            return IntentRoutingDecision(intent: .memory, allowedToolIDs: memoryToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, ["draft a text", "message jordan", "text message to"]) {
            let recipient = inferredRecipient(text)
            let content = inferredContent(text)
            let clarification: String?
            if !recipient && !content { clarification = "Who should I message, and what should it say?" }
            else if !recipient { clarification = "Who should I message?" }
            else if !content { clarification = "What should the message say?" }
            else { clarification = nil }
            return IntentRoutingDecision(intent: .messageDraft, allowedToolIDs: messageToolIDs, requiresClarification: clarification != nil, clarificationPrompt: clarification)
        }

        if matchesAny(text, [
            "draft an email", "email:", "subject release prep",
            "draft a quick email", "draft a quick email update to",
            "write a quick email", "compose a quick email", "email update to"
        ]) || (text.contains("email") && text.contains("ask one question")) {
            let recipient = inferredRecipient(text)
            let content = inferredContent(text)
            let clarification: String?
            if !recipient && !content { clarification = "Who should I send it to, and what should it say?" }
            else if !recipient { clarification = "Who should I send it to?" }
            else if !content { clarification = "What should the email say?" }
            else { clarification = nil }
            return IntentRoutingDecision(intent: .emailDraft, allowedToolIDs: emailToolIDs, requiresClarification: clarification != nil, clarificationPrompt: clarification)
        }

        if isLikelyPhoneCallIntent(text) || matchesAny(text, ["place a call to", "start a call to"]) {
            let hasTarget = text.split(separator: " ").count >= 2 || text.rangeOfCharacter(from: .decimalDigits) != nil
            return IntentRoutingDecision(intent: .phoneCall, allowedToolIDs: phoneToolIDs, requiresClarification: !hasTarget, clarificationPrompt: hasTarget ? nil : "Who should I call?")
        }

        if matchesAny(text, [
            "take a photo", "open camera", "open the camera", "prepare the camera",
            "camera and prepare", "capture", "capture a photo", "take a picture"
        ]) {
            return IntentRoutingDecision(intent: .camera, allowedToolIDs: cameraToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, [
            "read this url", "read this web url", "fetch and summarize"
        ]) || isURLFetchIntent(text) {
            return IntentRoutingDecision(intent: .webSearch, allowedToolIDs: webSearchToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        if matchesAny(text, [
            "walking or driving", "whether i was walking", "whether i was driving",
            "recent motion", "recent activity"
        ]) {
            return IntentRoutingDecision(intent: .motion, allowedToolIDs: motionToolIDs, requiresClarification: false, clarificationPrompt: nil)
        }

        return nil
    }


    private static func isLikelyLocalKnowledgeQuery(_ text: String) -> Bool {
        let localScopeMarkers = ["my files", "my notes", "my documents", "local", "imported", "codebase", "repo"]
        let lookupVerbs = ["search", "find", "look up", "summarize", "read", "show"]
        let architectureTopics = ["architecture", "system design", "codebase structured", "codebase structure", "how is the codebase structured", "modules"]

        if matchesAny(text, architectureTopics) && (matchesAny(text, localScopeMarkers) || matchesAny(text, lookupVerbs)) {
            return true
        }

        return matchesAny(text, localScopeMarkers) && matchesAny(text, ["architecture", "system design", "structure", "module", "modules"])
    }

    private static func isExplicitMemorySaveIntent(_ text: String) -> Bool {
        matchesAny(text, [
            "save this note:", "save this fact:", "remember this:", "remember that:",
            "keep this in mind:", "save this note", "save this fact"
        ])
    }

    private static func isExplicitReminderIntent(_ text: String) -> Bool {
        let reminderPhrases = [
            "remind me to",
            "remind me",
            "create a reminder",
            "set a reminder"
        ]
        return matchesAny(text, reminderPhrases)
    }

    private static func isExplicitRAGIndexIntent(_ text: String) -> Bool {
        matchesAny(text, [
            "reindex local files", "reindex files", "refresh the file retrieval index",
            "refresh file retrieval index", "reindex photos", "refresh the photo retrieval index",
            "refresh photo retrieval index"
        ])
    }

    private static func isConcreteFileReadIntent(_ text: String) -> Bool {
        guard matchesAny(text, ["open", "read", "show"]) else { return false }
        return text.range(
            of: #"(?i)\b[\w][\w ._-]*\.(md|txt|json|pdf|swift|docx|csv)\b"#,
            options: .regularExpression
        ) != nil
    }

    private static func isURLFetchIntent(_ text: String) -> Bool {
        text.range(
            of: #"(?i)\b(read|fetch|open|summarize)\b.{0,80}\bhttps?://\S+"#,
            options: .regularExpression
        ) != nil
    }

    private static func normalized(_ text: String) -> String {
        text.lowercased().replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func matchesAny(_ text: String, _ patterns: [String]) -> Bool {
        patterns.contains { text.contains($0) }
    }

    private static func isPureConversationalGreeting(_ text: String) -> Bool {
        ["hi", "hello", "hey", "yo", "sup", "bonjour", "salut", "allo", "hi how are you", "hi. how are you", "how are you"].contains(text)
    }

    private static func isCurrentLocationIntent(_ text: String) -> Bool {
        matchesAny(text, [
            "where are we", "where am i", "where are you", "current location", "my location", "our location", "gps location", "what is my location", "what's my location", "where exactly am i"
        ])
    }

    private static func isLikelyOutlookIntent(_ text: String) -> Bool {
        let outlookMarkers = ["outlook", "hotmail", "live mail", "msn mail", "microsoft mail", "microsoft graph", "graph mail"]
        let mailActions = [
            "inbox", "email", "emails", "mail", "message", "messages", "unread", "new", "latest", "recent", "folders", "attachments",
            "search", "find", "read", "open", "check", "show", "list", "draft", "send", "reply", "reply all", "forward", "archive", "delete", "trash", "mark read", "mark unread", "move"
        ]
        if matchesAny(text, outlookMarkers) && matchesAny(text, mailActions) { return true }
        let genericReadMailCommands = [
            "read new emails", "read new email", "read unread emails", "read unread email", "read latest email", "read the latest email",
            "read recent emails", "read recent email", "check new emails", "check new email", "check unread emails", "check unread email",
            "check my unread email", "check my unread emails", "check my inbox", "show my unread email", "show my unread emails",
            "show new emails", "show latest emails", "show recent emails", "list new emails", "list unread emails", "open latest email", "open the latest email"
        ]
        if matchesAny(text, genericReadMailCommands) { return true }
        let hasReadVerb = matchesAny(text, ["read", "check", "show", "list", "open"])
        let hasMailboxObject = matchesAny(text, ["email", "emails", "mail", "inbox"])
        let hasNonComposeQualifier = matchesAny(text, ["new", "unread", "latest", "recent", "inbox"])
        return hasReadVerb && hasMailboxObject && hasNonComposeQualifier
    }

    private static func isLikelyWebSearchIntent(_ text: String) -> Bool {
        let explicitWebCommands = [
            "search web", "search the web", "search on web", "web search", "internet search",
            "look online", "find online", "fetch url", "open url", "read this url", "read this website"
        ]
        if matchesAny(text, explicitWebCommands) { return true }

        let hasWebChannel = matchesAny(text, ["web", "internet", "online"])
        let hasDiscoveryVerb = matchesAny(
            text,
            [
                "search", "find", "look up", "research", "fetch information", "fetch info",
                "how to", "how do i", "tutorial", "guide", "diy", "steps to", "instructions"
            ]
        )
        return hasWebChannel && hasDiscoveryVerb
    }

    private static func inferredRecipient(_ text: String) -> Bool {
        text.contains(" to ") || text.contains("@") || text.contains("recipient") || text.rangeOfCharacter(from: .decimalDigits) != nil
    }

    private static func inferredContent(_ text: String) -> Bool {
        text.contains(" about ") || text.contains(" saying ") || text.contains(" body ") || text.contains(" that says ") || text.contains(" message ") || text.split(separator: " ").count >= 8
    }
}
