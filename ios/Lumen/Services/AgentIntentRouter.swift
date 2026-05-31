import Foundation

/// Narrows the agent's action space before generation. It also detects low-
/// confidence or underspecified requests early so Lumen can ask one concise
/// clarification instead of guessing or inventing tool results.
nonisolated enum AgentIntentRouter {
    enum Intent: Sendable, Equatable, CaseIterable {
        case conversation
        case answerWithContext
        case clarify
        case webSearch
        case fetchURL
        case calendarList
        case calendarCreate
        case reminderList
        case reminderCreate
        case contactSearch
        case draftMessage
        case draftMail
        case phoneCall
        case location
        case weather
        case mapsSearch
        case mapsDirections
        case photosSearch
        case cameraCapture
        case healthSummary
        case motionActivity
        case fileRead
        case memorySave
        case memoryRecall
        case ragSearch
        case ragIndex
        case triggerCreate
        case triggerList
        case triggerCancel
        case alarmStatus
        case alarmRequestAuthorization
        case alarmSchedule
        case alarmControl
        case outlookStatus
        case outlookFoldersList
        case outlookMessagesList
        case outlookMessagesSearch
        case outlookMessageRead
        case outlookAttachmentsList
        case outlookDraftCreate
        case outlookMailSend
        case outlookMessageMarkRead
        case outlookMessageMarkUnread
        case outlookMessageMove
        case outlookMessageArchive
        case outlookMessageDelete
        case outlookMessageReply
        case outlookMessageReplyAll
        case outlookMessageForward
    }

    struct Decision: Sendable, Equatable {
        let intent: Intent
        let confidence: Int
        let reason: String
        let allowedToolIDs: Set<String>
        let requiresUserApproval: Bool
        let shouldAskClarification: Bool
        let clarificationQuestion: String?
        let alternatives: [Intent]

        var allowsTools: Bool { !allowedToolIDs.isEmpty }
    }

    private struct Candidate: Sendable {
        let intent: Intent
        let score: Int
        let reason: String
    }

    private static let modelConfidenceThreshold = 85

    static func decide(userMessage: String, attachments: [ChatAttachment] = []) -> Decision {
        let text = normalized(userMessage)
        let compact = compacted(userMessage)

        if text.isEmpty {
            return makeDecision(.conversation, confidence: 100, reason: "empty message")
        }

        if isPureConversation(text: text, compact: compact) {
            return makeDecision(.conversation, confidence: 100, reason: "pure conversational turn")
        }

        if !attachments.isEmpty {
            return makeDecision(.answerWithContext, confidence: 92, reason: "attachments are already in prompt context")
        }

        if let deterministic = deterministicDecision(for: text) {
            logRouting(predictedIntent: deterministic.intent, usedFallback: false, fallbackReason: nil)
            return deterministic
        }

        var candidates: [Candidate] = []
        func add(_ intent: Intent, _ score: Int, _ reason: String) {
            guard score > 0 else { return }
            candidates.append(Candidate(intent: intent, score: score, reason: reason))
        }

        let hasURL = containsAny(text, ["http://", "https://", "www."])
        if hasURL {
            add(.fetchURL, score(text, strong: ["read", "open", "fetch", "summarize", "analyze", "check", "review"], weak: ["url", "link", "page", "site"]) + 8, "explicit URL/link handling")
        }

        add(.weather, score(text, strong: ["weather", "forecast", "temperature"], weak: ["rain", "snow", "wind", "humid", "feels like", "outside"]), "weather terms")
        add(.location, score(text, strong: ["where am i", "current location", "my location", "gps location"], weak: ["coordinates"]), "current location terms")
        add(.mapsDirections, score(text, strong: ["directions to", "navigate to", "route to", "take me to", "open maps to"], weak: ["drive to", "walk to"]), "navigation terms")
        add(.mapsSearch, score(text, strong: ["near me", "nearby", "closest", "around me", "around here", "in my area"], weak: ["local", "address of"]), "local search terms")

        let calendarMentions = containsAny(text, ["calendar", "event", "meeting", "appointment"])
        if calendarMentions || containsAny(text, ["schedule", "book me", "put it on"]) {
            add(.calendarCreate, score(text, strong: ["create", "add", "schedule", "book", "put", "make", "set up"], weak: ["tomorrow", "today", "next week", "at "]) + 4, "calendar write intent")
            add(.calendarList, score(text, strong: ["show", "list", "what", "when", "next", "upcoming"], weak: ["today", "tomorrow", "this week"]) + 4, "calendar read intent")
        }

        let reminderMentions = containsAny(text, ["reminder", "remind me", "todo", "to do"])
        if reminderMentions {
            add(.reminderCreate, score(text, strong: ["create", "add", "set", "remind me", "make"], weak: ["tomorrow", "later", "at "]) + 4, "reminder write intent")
            add(.reminderList, score(text, strong: ["show", "list", "what", "pending"], weak: ["reminders", "todos"]) + 4, "reminder read intent")
        }

        add(.draftMessage, score(text, strong: ["send a text", "draft message", "imessage", "sms"], weak: ["text ", "message "]), "message draft intent")
        add(.draftMail, score(text, strong: ["draft email", "compose email", "send email", "write an email"], weak: ["email", "mail"]), "email draft intent")
        add(.phoneCall, score(text, strong: ["call ", "phone ", "dial "], weak: ["ring"]), "phone call intent")
        add(.contactSearch, score(text, strong: ["contact", "address book", "phone number of", "email address of"], weak: ["number for", "email for"]), "contact lookup intent")

        let outlookContext = containsAny(text, ["outlook", "hotmail", "live mail", "msn mail", "microsoft mail", "microsoft graph", "graph mail"])
        let outlookMailContext = outlookContext || containsAny(text, ["my inbox", "inbox", "unread email", "unread mail", "email inbox", "mail inbox"])
        if outlookMailContext {
            add(.outlookStatus, score(text, strong: ["outlook status", "signed in", "connected", "connection", "auth status", "account status"], weak: ["outlook", "hotmail", "microsoft"]), "Outlook account status intent")
            add(.outlookFoldersList, score(text, strong: ["folders", "mail folders", "list folders", "show folders"], weak: ["folder", "labels"]), "Outlook folder list intent")
            add(.outlookMessagesList, score(text, strong: ["inbox", "latest emails", "recent emails", "recent mail", "list emails", "show emails", "check email", "check mail", "unread emails", "unread mail"], weak: ["messages", "mail", "email"]) + (outlookContext ? 5 : 0), "Outlook message list intent")
            add(.outlookMessagesSearch, score(text, strong: ["search email", "search mail", "find email", "find mail", "look for email", "look for mail", "email about", "mail about"], weak: ["search", "find", "from", "subject", "invoice", "receipt"]) + (outlookContext ? 5 : 0), "Outlook message search intent")
            add(.outlookMessageRead, score(text, strong: ["read email", "read mail", "open email", "open mail", "show message", "read message"], weak: ["message id", "email id", "body"]), "Outlook message read intent")
            add(.outlookAttachmentsList, score(text, strong: ["attachments", "attachment", "paperclip", "files attached"], weak: ["attached", "file"]), "Outlook attachment list intent")
            add(.outlookDraftCreate, score(text, strong: ["draft outlook", "draft hotmail", "create outlook draft", "save draft", "draft an outlook email"], weak: ["draft email", "compose email"]), "Outlook draft create intent")
            add(.outlookMailSend, score(text, strong: ["send outlook", "send hotmail", "send email from outlook", "send email through outlook", "send an outlook email"], weak: ["send email", "email to"]), "Outlook send mail intent")
            add(.outlookMessageMarkRead, score(text, strong: ["mark read", "mark as read"], weak: ["read"]), "Outlook mark read intent")
            add(.outlookMessageMarkUnread, score(text, strong: ["mark unread", "mark as unread"], weak: ["unread"]), "Outlook mark unread intent")
            add(.outlookMessageMove, score(text, strong: ["move email", "move mail", "move message", "put email in folder"], weak: ["move", "folder"]), "Outlook move message intent")
            add(.outlookMessageArchive, score(text, strong: ["archive email", "archive mail", "archive message"], weak: ["archive"]), "Outlook archive intent")
            add(.outlookMessageDelete, score(text, strong: ["delete email", "delete mail", "delete message", "trash email", "trash mail"], weak: ["delete", "trash"]), "Outlook delete intent")
            add(.outlookMessageReply, score(text, strong: ["reply to email", "reply to mail", "reply outlook", "respond to email"], weak: ["reply", "respond"]), "Outlook reply intent")
            add(.outlookMessageReplyAll, score(text, strong: ["reply all", "reply-all", "respond to all"], weak: ["everyone"]), "Outlook reply-all intent")
            add(.outlookMessageForward, score(text, strong: ["forward email", "forward mail", "forward message"], weak: ["forward"]), "Outlook forward intent")
        }

        let photoScore = score(text, strong: ["photo", "picture", "camera roll", "image library"], weak: ["image", "album"])
        if photoScore > 0 {
            if containsAny(text, ["take", "capture", "shoot", "camera"]) { add(.cameraCapture, photoScore + 5, "camera capture intent") }
            else { add(.photosSearch, photoScore, "photo library intent") }
        }

        add(.healthSummary, score(text, strong: ["heart rate", "sleep", "health", "calories", "distance walked"], weak: ["steps"]), "health summary intent")
        add(.motionActivity, score(text, strong: ["motion activity", "am i walking", "am i running"], weak: ["walking", "running", "cycling"]), "motion activity intent")
        add(.memorySave, score(text, strong: ["remember that", "save this memory", "remember this", "store this"], weak: ["keep this in mind"]), "memory save intent")
        add(.memoryRecall, score(text, strong: ["what do you remember", "memory about", "saved memory"], weak: ["recall"]), "memory recall intent")
        add(.ragSearch, score(text, strong: ["my files", "local files", "indexed files", "personal data", "my documents", "rag search"], weak: ["my notes", "my pdfs"]), "personal data search intent")
        add(.ragIndex, score(text, strong: ["reindex", "index files", "index photos", "rebuild index"], weak: ["refresh index"]), "index rebuild intent")
        add(.fileRead, score(text, strong: ["read file", "open file", "imported file"], weak: ["file named"]), "local imported file intent")

        let triggerMentions = containsAny(text, ["trigger", "scheduled agent", "run later", "automation"])
        if triggerMentions {
            add(.triggerCreate, score(text, strong: ["create", "schedule", "add", "run later"], weak: ["every", "when"]) + 4, "trigger create intent")
            add(.triggerCancel, score(text, strong: ["cancel", "delete", "stop"], weak: ["remove"]) + 4, "trigger cancel intent")
            add(.triggerList, score(text, strong: ["show", "list", "what"], weak: ["active"]) + 4, "trigger list intent")
        }

        let alarmMentions = containsAny(text, ["alarm", "countdown", "timer"])
        if alarmMentions {
            add(.alarmRequestAuthorization, score(text, strong: ["request authorization", "ask permission"], weak: ["permission"]), "alarm permission request")
            add(.alarmStatus, score(text, strong: ["authorization", "auth status", "permission"], weak: ["status"]), "alarm status intent")
            add(.alarmSchedule, score(text, strong: ["set", "schedule", "start", "create", "wake me"], weak: ["in ", "at "]) + 4, "alarm schedule intent")
            add(.alarmControl, score(text, strong: ["pause", "resume", "stop", "snooze", "cancel", "list"], weak: ["alarms"]), "alarm control intent")
        }

        add(.webSearch, score(text, strong: ["search the web", "web search", "look up", "find online", "research online"], weak: ["google", "internet", "latest", "current", "documentation", "research"]), "web knowledge intent")

        let ranked = candidates.sorted { lhs, rhs in
            if lhs.score != rhs.score { return lhs.score > rhs.score }
            return priority(lhs.intent) > priority(rhs.intent)
        }

        guard let best = ranked.first, best.score >= 7 else {
            if looksLikeUnderspecifiedCommand(text) {
                return makeClarificationDecision(question: "What do you want me to do with that?", reason: "underspecified command", alternatives: [])
            }
            return makeDecision(.answerWithContext, confidence: 82, reason: "no tool intent above threshold")
        }

        let closeAlternatives = ranked
            .dropFirst()
            .filter { abs($0.score - best.score) <= 3 && $0.score >= 7 }
            .map(\.intent)

        if shouldClarifyAmbiguity(best: best, alternatives: closeAlternatives, text: text) {
            return makeClarificationDecision(
                question: clarificationQuestion(best: best.intent, alternatives: closeAlternatives, text: text),
                reason: "ambiguous intent: \(best.intent) vs \(closeAlternatives.map(String.init(describing:)).joined(separator: ", "))",
                alternatives: [best.intent] + closeAlternatives
            )
        }

        let intent = normalizeConflicts(best.intent, text: text)
        if let missing = missingRequiredDetails(for: intent, text: text) {
            return makeClarificationDecision(question: missing, reason: "missing required details for \(intent)", alternatives: [intent])
        }

        let predictedDecision = makeDecision(intent, confidence: min(100, best.score * 10), reason: best.reason)
        if predictedDecision.confidence < modelConfidenceThreshold, let fallback = deterministicDecision(for: text) {
            logRouting(predictedIntent: predictedDecision.intent, usedFallback: true, fallbackReason: "low_confidence_\(predictedDecision.confidence)_below_\(modelConfidenceThreshold) -> \(fallback.intent)")
            return fallback
        }

        logRouting(predictedIntent: predictedDecision.intent, usedFallback: false, fallbackReason: nil)
        return predictedDecision
    }

    static func inferIntent(from userMessage: String, attachments: [ChatAttachment] = []) -> Intent {
        decide(userMessage: userMessage, attachments: attachments).intent
    }

    static func filteredTools(from enabledTools: [ToolDefinition], userMessage: String, attachments: [ChatAttachment] = []) -> [ToolDefinition] {
        let decision = decide(userMessage: userMessage, attachments: attachments)
        guard decision.allowsTools, !decision.shouldAskClarification else { return [] }
        return enabledTools.filter { decision.allowedToolIDs.contains($0.id) }
    }

    static func routingSystemNote(for decision: Decision) -> String {
        let compatibility = """
        Tool-call compatibility rules:
        - If you call a tool, emit exactly one JSON object and no prose around it.
        - Use either {"tool":"tool.id","args":{...}} or {"action":{"tool":"tool.id","args":{...}}}.
        - Args may contain normal JSON values: strings, numbers, booleans, arrays, objects, or null.
        - Do not emit privacy or anonymizer placeholder tokens. Use available user text or ask one concise follow-up.
        """

        if decision.shouldAskClarification {
            return "\n\nRouting: The user's intent is not clear enough to act. Ask exactly one concise clarification question. Do not use tools. Do not invent completed actions.\n\n\(compatibility)"
        }
        if decision.allowedToolIDs.isEmpty {
            return "\n\nRouting: No tools are available for this turn. Answer directly in natural language. Do not invent tool results or claim that actions were performed. If the user's request is unclear, ask one concise clarification question.\n\n\(compatibility)"
        }
        let tools = decision.allowedToolIDs.sorted().joined(separator: ", ")
        return "\n\nRouting: The user's inferred intent is \(decision.intent) with confidence \(decision.confidence). Only these tools are available for this turn: \(tools). Use a tool only if it is necessary. If key details are missing, ask one concise clarification question before acting. Do not invent tool results. Tools that require approval must not be described as completed until they actually return a successful result.\n\n\(compatibility)"
    }

    static func allowedToolIDs(for intent: Intent) -> Set<String> {
        switch intent {
        case .conversation, .answerWithContext, .clarify: []
        case .webSearch: ["web.search"]
        case .fetchURL: ["web.fetch", "web.search"]
        case .calendarList: ["calendar.list"]
        case .calendarCreate: ["calendar.create", "calendar.list"]
        case .reminderList: ["reminders.list"]
        case .reminderCreate: ["reminders.create", "reminders.list"]
        case .contactSearch: ["contacts.search"]
        case .draftMessage: ["messages.draft", "contacts.search"]
        case .draftMail: ["mail.draft", "contacts.search"]
        case .phoneCall: ["phone.call", "contacts.search"]
        case .location: ["location.current"]
        case .weather: ["weather", "location.current"]
        case .mapsSearch: ["maps.search", "location.current"]
        case .mapsDirections: ["maps.directions", "maps.search", "location.current"]
        case .photosSearch: ["photos.search"]
        case .cameraCapture: ["camera.capture"]
        case .healthSummary: ["health.summary"]
        case .motionActivity: ["motion.activity"]
        case .fileRead: ["files.read"]
        case .memorySave: ["memory.save"]
        case .memoryRecall: ["memory.recall"]
        case .ragSearch: ["rag.search"]
        case .ragIndex: ["rag.index_files", "rag.index_photos"]
        case .triggerCreate: ["trigger.create", "trigger.list"]
        case .triggerList: ["trigger.list"]
        case .triggerCancel: ["trigger.cancel", "trigger.list"]
        case .alarmStatus: ["alarm.authorization_status", "alarm.list"]
        case .alarmRequestAuthorization: ["alarm.request_authorization", "alarm.authorization_status"]
        case .alarmSchedule: ["alarm.schedule", "alarm.countdown", "alarm.list"]
        case .alarmControl: ["alarm.list", "alarm.pause", "alarm.resume", "alarm.stop", "alarm.snooze", "alarm.cancel"]
        case .outlookStatus: ["outlook.status"]
        case .outlookFoldersList: ["outlook.folders.list"]
        case .outlookMessagesList: ["outlook.messages.list", "outlook.folders.list"]
        case .outlookMessagesSearch: ["outlook.messages.search", "outlook.messages.list"]
        case .outlookMessageRead: ["outlook.message.read", "outlook.messages.search", "outlook.messages.list"]
        case .outlookAttachmentsList: ["outlook.attachments.list", "outlook.message.read"]
        case .outlookDraftCreate: ["outlook.draft.create", "contacts.search"]
        case .outlookMailSend: ["outlook.mail.send", "contacts.search"]
        case .outlookMessageMarkRead: ["outlook.message.mark_read", "outlook.message.read"]
        case .outlookMessageMarkUnread: ["outlook.message.mark_unread", "outlook.message.read"]
        case .outlookMessageMove: ["outlook.message.move", "outlook.folders.list"]
        case .outlookMessageArchive: ["outlook.message.archive"]
        case .outlookMessageDelete: ["outlook.message.delete"]
        case .outlookMessageReply: ["outlook.message.reply", "outlook.message.read"]
        case .outlookMessageReplyAll: ["outlook.message.reply_all", "outlook.message.read"]
        case .outlookMessageForward: ["outlook.message.forward", "contacts.search"]
        }
    }

    private static func makeDecision(_ intent: Intent, confidence: Int, reason: String) -> Decision {
        let allowed = allowedToolIDs(for: intent)
        let requiresApproval = allowed.contains { ToolRegistry.find(id: $0)?.requiresApproval == true }
        return Decision(intent: intent, confidence: confidence, reason: reason, allowedToolIDs: allowed, requiresUserApproval: requiresApproval, shouldAskClarification: false, clarificationQuestion: nil, alternatives: [])
    }

    private static func makeClarificationDecision(question: String, reason: String, alternatives: [Intent]) -> Decision {
        Decision(intent: .clarify, confidence: 50, reason: reason, allowedToolIDs: [], requiresUserApproval: false, shouldAskClarification: true, clarificationQuestion: question, alternatives: alternatives)
    }

    private static func normalizeConflicts(_ intent: Intent, text: String) -> Intent {
        if intent == .mapsSearch, containsAny(text, ["how to", "tutorial", "guide", "plans", "blueprint", "documentation"]) { return .webSearch }
        if intent == .calendarCreate, !containsAny(text, ["calendar", "event", "meeting", "appointment", "schedule", "book"]) { return .answerWithContext }
        if intent == .draftMail, containsAny(text, ["outlook", "hotmail", "microsoft graph", "microsoft mail"]) { return .outlookDraftCreate }
        return intent
    }

    private static func missingRequiredDetails(for intent: Intent, text: String) -> String? {
        switch intent {
        case .calendarCreate:
            let hasTime = containsAny(text, ["today", "tomorrow", "am", "pm", " at ", "next ", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
            let hasSubject = text.replacingOccurrences(of: "calendar", with: "").replacingOccurrences(of: "event", with: "").split(separator: " ").count > 3
            if !hasSubject && !hasTime { return "What event should I create, and when should it start?" }
            if !hasSubject { return "What should I call the event?" }
            if !hasTime { return "When should the event start?" }
        case .reminderCreate:
            if !containsAny(text, ["remind me", "to ", "about "]) { return "What should I remind you about?" }
        case .draftMail:
            if !containsAny(text, [" to ", "email "]) { return "Who should the email go to, and what should it say?" }
        case .draftMessage:
            if !containsAny(text, [" to ", "text ", "message "]) { return "Who should the message go to, and what should it say?" }
        case .phoneCall:
            if text.split(separator: " ").count <= 2 { return "Who do you want to call?" }
        case .mapsDirections:
            if text.split(separator: " ").count <= 3 { return "Where do you want directions to?" }
        case .outlookMessagesSearch:
            if !containsAny(text, ["search", "find", "about", "from", "subject", "invoice", "receipt"]) { return "What should I search for in Outlook?" }
        case .outlookMessageRead, .outlookAttachmentsList, .outlookMessageMarkRead, .outlookMessageMarkUnread, .outlookMessageMove, .outlookMessageArchive, .outlookMessageDelete, .outlookMessageReply, .outlookMessageReplyAll, .outlookMessageForward:
            if !containsAny(text, ["message id", "email id", "id:", "this email", "that email", "selected email", "latest", "last email"]) {
                return "Which Outlook message should I use? Give me the message id, or ask me to search/list messages first."
            }
        case .outlookDraftCreate, .outlookMailSend:
            if !containsAny(text, [" to ", "email ", "@"]) { return "Who should the Outlook email go to, and what should it say?" }
        default:
            break
        }
        return nil
    }


    private static func deterministicDecision(for text: String) -> Decision? {
        if containsAny(text, ["call ", "dial ", "phone "]) {
            return makeDecision(.phoneCall, confidence: 100, reason: "deterministic action verb: call/dial/phone")
        }
        if containsAny(text, ["email", "mail "]) {
            return makeDecision(.draftMail, confidence: 100, reason: "deterministic action verb: email")
        }
        if containsAny(text, ["message", "text ", "sms", "imessage"]) {
            return makeDecision(.draftMessage, confidence: 100, reason: "deterministic action verb: message")
        }
        if containsAny(text, ["schedule", "book", "set up"]) {
            return makeDecision(.calendarCreate, confidence: 100, reason: "deterministic action verb: schedule")
        }

        if containsAny(text, ["remember", "store", "what did i tell you", "what do you remember"]) {
            let intent: Intent = containsAny(text, ["what did i tell you", "what do you remember"]) ? .memoryRecall : .memorySave
            return makeDecision(intent, confidence: 100, reason: "deterministic memory rule")
        }

        if containsAny(text, ["explain", "compare", "summarize conceptually"]) {
            return makeDecision(.conversation, confidence: 100, reason: "deterministic conceptual chat rule")
        }

        return nil
    }

    private static func logRouting(predictedIntent: Intent, usedFallback: Bool, fallbackReason: String?) {
        if usedFallback {
            print("[AgentIntentRouter] predicted=\(predictedIntent) fallback=deterministic reason=\(fallbackReason ?? "unknown")")
        } else {
            print("[AgentIntentRouter] predicted=\(predictedIntent) fallback=none")
        }
    }

    private static func shouldClarifyAmbiguity(best: Candidate, alternatives: [Intent], text: String) -> Bool {
        guard !alternatives.isEmpty else { return false }
        let actionIntents: Set<Intent> = [
            .calendarCreate, .reminderCreate, .draftMail, .draftMessage, .phoneCall, .triggerCreate, .alarmSchedule, .mapsDirections,
            .outlookDraftCreate, .outlookMailSend, .outlookMessageMarkRead, .outlookMessageMarkUnread, .outlookMessageMove,
            .outlookMessageArchive, .outlookMessageDelete, .outlookMessageReply, .outlookMessageReplyAll, .outlookMessageForward
        ]
        if actionIntents.contains(best.intent) || alternatives.contains(where: { actionIntents.contains($0) }) { return true }
        return best.score < 12 && text.split(separator: " ").count < 5
    }

    private static func clarificationQuestion(best: Intent, alternatives: [Intent], text: String) -> String {
        let options = ([best] + alternatives).prefix(3).map(userFacingName(for:)).joined(separator: ", ")
        return "Do you want me to \(options)?"
    }

    private static func userFacingName(for intent: Intent) -> String {
        switch intent {
        case .calendarCreate: "create a calendar event"
        case .calendarList: "check your calendar"
        case .reminderCreate: "create a reminder"
        case .reminderList: "list reminders"
        case .draftMail: "draft an email"
        case .draftMessage: "draft a message"
        case .phoneCall: "start a call"
        case .mapsDirections: "get directions"
        case .mapsSearch: "search nearby places"
        case .webSearch: "search the web"
        case .weather: "check the weather"
        case .outlookStatus: "check Outlook sign-in"
        case .outlookFoldersList: "list Outlook folders"
        case .outlookMessagesList: "check your Outlook inbox"
        case .outlookMessagesSearch: "search Outlook mail"
        case .outlookMessageRead: "read an Outlook email"
        case .outlookAttachmentsList: "list Outlook attachments"
        case .outlookDraftCreate: "create an Outlook draft"
        case .outlookMailSend: "send an Outlook email"
        case .outlookMessageMarkRead: "mark an Outlook email as read"
        case .outlookMessageMarkUnread: "mark an Outlook email as unread"
        case .outlookMessageMove: "move an Outlook email"
        case .outlookMessageArchive: "archive an Outlook email"
        case .outlookMessageDelete: "delete an Outlook email"
        case .outlookMessageReply: "reply to an Outlook email"
        case .outlookMessageReplyAll: "reply-all to an Outlook email"
        case .outlookMessageForward: "forward an Outlook email"
        default: "continue"
        }
    }

    private static func looksLikeUnderspecifiedCommand(_ text: String) -> Bool {
        text.split(separator: " ").count <= 4 && containsAny(text, ["do it", "make it", "fix it", "that", "this", "go", "run it", "continue"])
    }

    private static func isPureConversation(text: String, compact: String) -> Bool {
        if ["hi", "hello", "hey", "yo", "sup", "bonjour", "salut", "allo", "ok", "okay", "thanks", "thankyou", "merci", "yes", "no"].contains(compact) { return true }
        if text.count < 30, containsAny(text, ["how are you", "what's up", "whats up", "good morning", "good evening"]) { return true }
        return false
    }

    private static func score(_ text: String, strong: [String], weak: [String]) -> Int {
        strong.reduce(0) { $0 + (text.contains($1) ? 8 : 0) } + weak.reduce(0) { $0 + (text.contains($1) ? 3 : 0) }
    }

    private static func priority(_ intent: Intent) -> Int {
        switch intent {
        case .conversation: 100
        case .calendarCreate, .reminderCreate, .alarmSchedule, .triggerCreate: 80
        case .outlookMailSend, .outlookDraftCreate, .outlookMessageReply, .outlookMessageReplyAll, .outlookMessageForward, .outlookMessageDelete, .outlookMessageArchive, .outlookMessageMove: 78
        case .outlookMessagesList, .outlookMessagesSearch, .outlookMessageRead, .outlookAttachmentsList, .outlookFoldersList, .outlookStatus: 76
        case .mapsDirections, .draftMail, .draftMessage, .phoneCall: 70
        case .weather, .webSearch, .fetchURL: 60
        default: 50
        }
    }

    private static func containsAny(_ text: String, _ needles: [String]) -> Bool { needles.contains { text.contains($0) } }
    private static func normalized(_ text: String) -> String { text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased().replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression) }
    private static func compacted(_ text: String) -> String { text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased().replacingOccurrences(of: #"[^a-z0-9]+"#, with: "", options: .regularExpression) }
}
