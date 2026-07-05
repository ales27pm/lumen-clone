import Foundation

#if canImport(AlarmKit)
import AlarmKit
#endif

nonisolated enum DeterministicToolPlanner {
    private static let outlookMessageReferenceToolIDs: Set<String> = [
        "outlook.message.read",
        "outlook.attachments.list",
        "outlook.message.mark_read",
        "outlook.message.mark_unread",
        "outlook.message.move",
        "outlook.message.archive",
        "outlook.message.delete",
        "outlook.message.reply",
        "outlook.message.reply_all",
        "outlook.message.forward"
    ]

    static func planForSpecificTool(toolID: String, prompt: String, availableToolIDs: Set<String>) -> AgentAction? {
        let canonical = ToolRouteGuard.canonicalToolID(toolID)
        guard availableToolIDs.contains(canonical) else { return nil }
        let text = normalized(prompt)
        switch canonical {
        case "camera.capture", "location.current", "outlook.status", "outlook.folders.list", "calendar.list", "reminders.list", "health.summary", "motion.activity", "trigger.list", "alarm.authorization_status", "alarm.list":
            return AgentAction(tool: canonical, args: [:])
        case "maps.search":
            let query = extractNearbySearchQuery(from: prompt) ?? extractDestination(from: prompt) ?? ""
            return AgentAction(tool: canonical, args: ["query": .string(query)])
        case "maps.directions":
            guard let destination = extractDestination(from: prompt), !destination.isEmpty else { return nil }
            return AgentAction(tool: canonical, args: ["destination": .string(destination)])
        case "weather":
            if let destination = extractDestination(from: prompt) { return AgentAction(tool: canonical, args: ["location": .string(destination)]) }
            return AgentAction(tool: canonical, args: [:])
        case "web.search":
            let query = extractWebQuery(from: prompt)
            return query.isEmpty ? nil : AgentAction(tool: canonical, args: ["query": .string(query)])
        case "web.fetch":
            guard let url = firstURL(in: prompt) else { return nil }
            return AgentAction(tool: canonical, args: ["url": .string(url)])
        case "outlook.messages.list":
            var args: AgentJSONArguments = ["limit": .string("10")]
            if text.contains("unread") { args["unreadOnly"] = .string("true") }
            return AgentAction(tool: canonical, args: args)
        case "outlook.message.read", "outlook.attachments.list", "outlook.message.mark_read", "outlook.message.mark_unread", "outlook.message.archive", "outlook.message.delete":
            return AgentAction(tool: canonical, args: outlookMessageReadArgs(extractOutlookMessageReference(from: text) ?? "latest"))
        case "outlook.message.reply", "outlook.message.reply_all":
            var args = outlookMessageReadArgs(extractOutlookMessageReference(from: text) ?? "latest")
            args["body"] = .string(extractOutlookBody(from: prompt) ?? "")
            return AgentAction(tool: canonical, args: args)
        case "outlook.message.forward":
            var args = outlookMessageReadArgs(extractOutlookMessageReference(from: text) ?? "latest")
            if let to = extractEmailAddress(from: prompt) { args["to"] = .string(to) }
            if let body = extractOutlookBody(from: prompt), !body.isEmpty { args["body"] = .string(body) }
            return AgentAction(tool: canonical, args: args)
        case "outlook.message.move":
            var args = outlookMessageReadArgs(extractOutlookMessageReference(from: text) ?? "latest")
            if let destination = extractOutlookDestinationFolder(from: text) { args["destination"] = .string(destination) }
            return AgentAction(tool: canonical, args: args)
        case "outlook.messages.search":
            let query = extractOutlookSearchQuery(from: prompt)
            return query.isEmpty ? nil : AgentAction(tool: canonical, args: ["query": .string(query), "limit": .string("10")])
        case "outlook.draft.create", "outlook.mail.send":
            var args: AgentJSONArguments = ["subject": .string(extractOutlookSubject(from: prompt)), "body": .string(extractOutlookBody(from: prompt) ?? "")]
            if let to = extractEmailAddress(from: prompt) { args["to"] = .string(to) }
            return AgentAction(tool: canonical, args: args)
        case "messages.draft":
            var args: AgentJSONArguments = ["body": .string(extractCommunicationBody(from: prompt))]
            if let phone = firstPhoneNumber(in: prompt) {
                args["to"] = .string(phone)
            } else if let recipient = extractRecipientName(from: prompt), !recipient.isEmpty {
                args["to"] = .string(recipient)
            }
            return AgentAction(tool: canonical, args: args)
        default:
            return AgentAction(tool: canonical, args: [:])
        }
    }

    /// Produces a sequence of actions to execute a user request, with optional prerequisite context actions.
    /// - Parameters:
    ///   - routing: The routing decision that determines the user's intent.
    ///   - prompt: The user's input text.
    ///   - availableToolIDs: Tools currently available for execution.
    /// - Returns: An array of `AgentAction` objects to execute in order; empty if no plan can be made.
    static func planSteps(routing: IntentRoutingDecision, prompt: String, availableToolIDs: Set<String>) -> [AgentAction] {
        let text = normalized(prompt)

        if routing.intent == .memory, let memoryPlan = MemoryCommandPlan.saveThenRecall(from: prompt) {
            var actions: [AgentAction] = []
            if let save = plan(routing: routing, prompt: prompt, availableToolIDs: availableToolIDs), ToolRouteGuard.canonicalToolID(save.tool) == "memory.save" {
                actions.append(save)
            } else if availableToolIDs.contains("memory.save") {
                actions.append(AgentAction(tool: "memory.save", args: [
                    "content": .string(memoryPlan.saveContent),
                    "kind": .string("fact")
                ]))
            }
            if availableToolIDs.contains("memory.recall") {
                actions.append(AgentAction(tool: "memory.recall", args: ["query": .string(memoryPlan.recallQuery)]))
            }
            if !actions.isEmpty { return actions }
        }

        if routing.intent == .outlook, let single = plan(routing: routing, prompt: prompt, availableToolIDs: availableToolIDs) {
            let canonical = ToolRouteGuard.canonicalToolID(single.tool)
            if outlookMessageReferenceToolIDs.contains(canonical), availableToolIDs.contains("outlook.messages.list"), needsFreshOutlookMessageContext(action: single, prompt: text) {
                return [
                    AgentAction(tool: "outlook.messages.list", args: ["limit": .string("1")]),
                    single
                ]
            }
            if ToolRouteGuard.requiresUserApproval(canonical) {
                return [single]
            }
            return [single]
        }

        if routing.intent == .webSearch {
            if let url = firstURL(in: prompt), availableToolIDs.contains("web.fetch") {
                return [AgentAction(tool: "web.fetch", args: ["url": .string(url)])]
            }
            if ToolRouteGuard.shouldUseWebSearchForDynamicPublicLookup(text), availableToolIDs.contains("web.search") {
                var actions: [AgentAction] = []
                if availableToolIDs.contains("location.current") {
                    actions.append(AgentAction(tool: "location.current", args: [:]))
                }
                actions.append(AgentAction(tool: "web.search", args: ["query": .string(dynamicPublicLookupWebQuery(from: prompt))]))
                return actions
            }
        }

        if routing.intent == .maps,
           availableToolIDs.contains("location.current"),
           isNearbyMapSearchIntent(text),
           let single = plan(routing: routing, prompt: prompt, availableToolIDs: availableToolIDs),
           ToolRouteGuard.canonicalToolID(single.tool) == "maps.search" {
            return [
                AgentAction(tool: "location.current", args: [:]),
                single
            ]
        }

        if let single = plan(routing: routing, prompt: prompt, availableToolIDs: availableToolIDs) {
            return [single]
        }
        return []
    }

    /// Plans a single tool action based on the user's intent and prompt, respecting tool availability.
    /// - Returns: An `AgentAction` for the matched tool, or `nil` if no valid action can be planned.
    static func plan(routing: IntentRoutingDecision, prompt: String, availableToolIDs: Set<String>) -> AgentAction? {
        let text = normalized(prompt)

        func has(_ tool: String) -> Bool { availableToolIDs.contains(tool) }
        func action(_ tool: String, _ args: AgentJSONArguments = [:]) -> AgentAction? {
            guard has(tool) else { return nil }
            return AgentAction(tool: tool, args: args)
        }

        switch routing.intent {
        case .webSearch:
            if has("web.fetch"), let url = firstURL(in: prompt) { return AgentAction(tool: "web.fetch", args: ["url": .string(url)]) }
            guard has("web.search") else { return nil }
            if ToolRouteGuard.shouldUseWebSearchForDynamicPublicLookup(text) {
                return AgentAction(tool: "web.search", args: ["query": .string(dynamicPublicLookupWebQuery(from: prompt))])
            }
            let query = extractWebQuery(from: prompt)
            return query.isEmpty ? nil : AgentAction(tool: "web.search", args: ["query": .string(query)])
        case .emailDraft:
            var args: AgentJSONArguments = ["subject": .string(extractEmailSubject(from: prompt)), "body": .string(extractCommunicationBody(from: prompt))]
            if let email = extractEmailAddress(from: prompt) { args["to"] = .string(email) }
            else if let recipient = extractRecipientName(from: prompt), !recipient.isEmpty { args["to"] = .string(recipient) }
            return action("mail.draft", args)
        case .messageDraft:
            var args: AgentJSONArguments = ["body": .string(extractCommunicationBody(from: prompt))]
            if let phone = firstPhoneNumber(in: prompt) {
                args["to"] = .string(phone)
            } else if let recipient = extractRecipientName(from: prompt), !recipient.isEmpty {
                args["to"] = .string(recipient)
            }
            return action("messages.draft", args)
        case .phoneCall:
            if let phone = firstPhoneNumber(in: prompt) { return action("phone.call", ["number": .string(phone)]) }
            if let query = extractCallTarget(from: prompt), !query.isEmpty { return action("contacts.search", ["query": .string(query)]) }
            return nil
        case .outlook:
            return planOutlook(text: text, prompt: prompt, availableToolIDs: availableToolIDs)
        case .weather:
            if has("weather") {
                if let destination = extractDestination(from: prompt) { return action("weather", ["location": .string(destination)]) }
                return action("weather")
            }
            return action("location.current")
        case .maps:
            if routing.allowedToolIDs == ["location.current"] || containsAny(text, ["where are we", "where am i", "current location", "my location"]) { return action("location.current") }
            if containsAny(text, ["directions", "navigate", "route"]) {
                guard let destination = extractDestination(from: prompt), !destination.isEmpty else { return nil }
                return action("maps.directions", ["destination": .string(destination)])
            }
            if isNearbyMapSearchIntent(text) {
                let query = extractNearbySearchQuery(from: prompt) ?? extractDestination(from: prompt) ?? ""
                return action("maps.search", ["query": .string(query)])
            }
            return nil
        case .calendar:
            if isCalendarCreateIntent(text) {
                return action("calendar.create", [
                    "title": .string(extractCalendarTitle(from: prompt)),
                    "startsInMinutes": .string(String(calendarStartOffsetMinutes(from: text) ?? 60))
                ])
            }
            if isCalendarReadIntent(text) { return action("calendar.list") }
            if has("calendar.list") { return action("calendar.list") }
            return nil
        case .reminder:
            if containsAny(text, ["list", "show", "pending"]) { return action("reminders.list") }
            if containsAny(text, ["create", "add", "remind me"]), let body = extractOutlookBody(from: prompt), !body.isEmpty { return action("reminders.create", ["title": .string(body)]) }
            return nil
        case .contactSearch:
            if let q = extractContactQuery(from: prompt), !q.isEmpty { return action("contacts.search", ["query": .string(q)]) }
            return nil
        case .photos:
            let query = extractPhotoQuery(from: prompt)
            return query.isEmpty ? nil : action("photos.search", ["query": .string(query)])
        case .camera:
            return action("camera.capture")
        case .health:
            return action("health.summary")
        case .motion:
            return action("motion.activity")
        case .files:
            if let name = extractFileName(from: prompt) { return action("files.read", ["name": .string(name)]) }
            if containsAny(text, ["attachment", "attached", "this file", "this document", "read file", "read document"]) { return action("files.read") }
            return nil
        case .memory, .note:
            if isPersonalProfileRecallIntent(text) { return action("memory.recall", ["query": .string("user name")]) }
            if containsAny(text, ["what do you remember", "recall", "remember about", "tell me what style i asked you to use"]) {
                return action("memory.recall", ["query": .string(extractMemoryRecallQuery(from: prompt))])
            }
            if containsAny(text, ["remember", "save", "note", "keep this in mind", "keep in mind", "my name is", "call me"]) {
                return action("memory.save", [
                    "content": .string(extractMemoryFact(from: prompt)),
                    "kind": .string("fact")
                ])
            }
            return nil
        case .rag:
            if containsAny(text, ["index photos", "index photo", "reindex photos", "reindex photo", "photo metadata", "photo retrieval index"]) { return action("rag.index_photos") }
            if containsAny(text, ["reindex", "index files", "file retrieval index", "refresh retrieval index"]) { return action("rag.index_files") }
            if containsAny(text, ["search", "summarize", "read", "show", "find"]) {
                let query = expandRAGQueryIfNeeded(originalPrompt: prompt)
                return action("rag.search", ["query": .string(query)])
            }
            return nil
        case .alarm:
            switch AlarmCommandClassifier.classifyAlarmCommandKind(text) {
            case .requestAuthorization:
                return action("alarm.request_authorization")
            case .authorizationStatus:
                return action("alarm.authorization_status")
            case .list:
                return action("alarm.list")
            case .pause:
                let args = alarmMutationArgs(from: prompt)
                guard args["id"] != nil else { return nil }
                return action("alarm.pause", args)
            case .resume:
                let args = alarmMutationArgs(from: prompt)
                guard args["id"] != nil else { return nil }
                return action("alarm.resume", args)
            case .stop:
                let args = alarmMutationArgs(from: prompt)
                guard args["id"] != nil else { return nil }
                return action("alarm.stop", args)
            case .snooze:
                let args = alarmMutationArgs(from: prompt)
                guard args["id"] != nil else { return nil }
                return action("alarm.snooze", args)
            case .cancel:
                let args = alarmMutationArgs(from: prompt)
                guard args["id"] != nil else { return nil }
                return action("alarm.cancel", args)
            case .countdown:
                guard let seconds = countdownDurationSeconds(from: text) else { return nil }
                return action("alarm.countdown", [
                    "title": .string(extractAlarmTitle(from: prompt, fallback: "Countdown")),
                    "durationSeconds": .string(String(seconds))
                ])
            case .schedule:
                if let duration = RelativeDuration.parse(from: text), duration.unit == .seconds {
                    return action("alarm.countdown", [
                        "title": .string(extractAlarmTitle(from: prompt, fallback: "Alarm")),
                        "durationSeconds": .string(String(duration.seconds))
                    ])
                }
                guard let minutes = calendarStartOffsetMinutes(from: text) else { return nil }
                return action("alarm.schedule", [
                    "title": .string(extractAlarmTitle(from: prompt, fallback: "Alarm")),
                    "inMinutes": .string(String(minutes))
                ])
            case .unknown:
                return nil
            }
        case .trigger:
            if containsAny(text, ["list", "show", "scheduled agent runs", "scheduled runs", "active triggers", "active scheduled", "what triggers"]) {
                return action("trigger.list")
            }
            if text.contains("cancel") {
                let token = extractTriggerCancelIdentifier(from: prompt)
                guard !token.isEmpty else { return nil }
                return action("trigger.cancel", ["id": .string(token)])
            }
            if has("trigger.create") {
                var args: AgentJSONArguments = [
                    "title": .string(extractTriggerTitle(from: prompt)),
                    "prompt": .string(extractTriggerPrompt(from: prompt)),
                    "schedule": .string("once")
                ]
                if text.contains("tonight") { args["inMinutes"] = .string("120") }
                else if let minutes = calendarStartOffsetMinutes(from: text) { args["inMinutes"] = .string(String(minutes)) }
                else { args["inMinutes"] = .string("60") }
                return action("trigger.create", args)
            }
            return nil
        default:
            return nil
        }
    }

    private static func outlookMessageReadArgs(_ messageID: String) -> AgentJSONArguments {
        let value = messageID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "latest" : messageID
        return ["messageId": .string(value), "id": .string(value)]
    }

    private static func needsFreshOutlookMessageContext(action: AgentAction, prompt: String) -> Bool {
        let args = action.args.stringCoerced
        let raw = args["messageId"] ?? args["id"] ?? args["message"] ?? ""
        let normalizedReference = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalizedReference.isEmpty { return true }
        if ["latest", "last", "newest", "recent", "first", "#1", "this", "that", "it", "current", "selected"].contains(normalizedReference) { return true }
        return containsAny(prompt, ["latest email", "latest outlook", "last email", "last outlook", "newest email", "recent email", "first email"])
    }

    private static func planOutlook(text: String, prompt: String, availableToolIDs: Set<String>) -> AgentAction? {
        func can(_ tool: String) -> Bool { availableToolIDs.contains(tool) }
        func action(_ tool: String, _ args: AgentJSONArguments = [:]) -> AgentAction? { can(tool) ? AgentAction(tool: tool, args: args) : nil }

        if containsAny(text, ["status", "connected", "signed in", "auth"]) { return action("outlook.status") }
        if containsAny(text, ["folder", "folders"]) { return action("outlook.folders.list") }
        if containsAny(text, ["attachment", "attachments", "paperclip"]) { return action("outlook.attachments.list", outlookMessageReadArgs(extractOutlookMessageReference(from: text) ?? "latest")) }
        if containsAny(text, ["search", "find", "invoice"]) {
            let q = extractOutlookSearchQuery(from: prompt)
            if !q.isEmpty { return action("outlook.messages.search", ["query": .string(q), "limit": .string("10")]) }
        }
        if !text.contains("move") && containsAny(text, ["new emails", "new email", "unread emails", "unread email", "inbox"]) {
            var args: AgentJSONArguments = ["limit": .string("10")]
            if text.contains("unread") { args["unreadOnly"] = .string("true") }
            return action("outlook.messages.list", args)
        }
        if containsAny(text, ["mark", "set"]) && text.contains("unread") { return action("outlook.message.mark_unread", outlookMessageReadArgs(extractOutlookMessageReference(from: text) ?? "latest")) }
        if containsAny(text, ["mark", "set"]) && text.contains("read") { return action("outlook.message.mark_read", outlookMessageReadArgs(extractOutlookMessageReference(from: text) ?? "latest")) }
        if !containsAny(text, ["move", "archive", "delete", "trash", "mark", "set", "reply", "respond", "forward"]) && isLatestOutlookReadIntent(text) {
            return action("outlook.message.read", outlookMessageReadArgs("latest"))
                ?? action("outlook.messages.list", ["limit": .string("1")])
        }
        if containsAny(text, ["reply all", "reply-all", "respond to all"]) {
            var args = outlookMessageReadArgs(extractOutlookMessageReference(from: text) ?? "latest")
            args["body"] = .string(extractOutlookBody(from: prompt) ?? "")
            return action("outlook.message.reply_all", args)
        }
        if containsAny(text, ["reply", "respond"]) {
            var args = outlookMessageReadArgs(extractOutlookMessageReference(from: text) ?? "latest")
            args["body"] = .string(extractOutlookBody(from: prompt) ?? "")
            return action("outlook.message.reply", args)
        }
        if text.contains("forward") {
            var args = outlookMessageReadArgs(extractOutlookMessageReference(from: text) ?? "latest")
            if let to = extractEmailAddress(from: prompt) { args["to"] = .string(to) }
            if let body = extractOutlookBody(from: prompt), !body.isEmpty { args["body"] = .string(body) }
            return action("outlook.message.forward", args)
        }
        if text.contains("move") {
            guard let destination = extractOutlookDestinationFolder(from: text) else { return nil }
            var args = outlookMessageReadArgs(extractOutlookMessageReference(from: text) ?? "latest")
            args["destination"] = .string(destination)
            return action("outlook.message.move", args)
        }
        if text.contains("archive") { return action("outlook.message.archive", outlookMessageReadArgs(extractOutlookMessageReference(from: text) ?? "latest")) }
        if containsAny(text, ["delete", "trash"]) { return action("outlook.message.delete", outlookMessageReadArgs(extractOutlookMessageReference(from: text) ?? "latest")) }
        if text.contains("send") && containsAny(text, ["email", "mail", "outlook", "hotmail"]) {
            var args: AgentJSONArguments = ["subject": .string(extractOutlookSubject(from: prompt)), "body": .string(extractOutlookBody(from: prompt) ?? "")]
            if let to = extractEmailAddress(from: prompt) { args["to"] = .string(to) }
            return action("outlook.mail.send", args)
        }
        if containsAny(text, ["draft", "compose", "write an email"]) {
            var args: AgentJSONArguments = ["subject": .string(extractOutlookSubject(from: prompt)), "body": .string(extractOutlookBody(from: prompt) ?? "")]
            if let to = extractEmailAddress(from: prompt) { args["to"] = .string(to) }
            return action("outlook.draft.create", args)
        }
        var args: AgentJSONArguments = ["limit": .string("10")]
        if text.contains("unread") { args["unreadOnly"] = .string("true") }
        return action("outlook.messages.list", args)
    }

    private static func normalized(_ text: String) -> String { text.lowercased().replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression).trimmingCharacters(in: .whitespacesAndNewlines) }
    private static func containsAny(_ value: String, _ needles: [String]) -> Bool { needles.contains { value.contains($0) } }

    private static func expandRAGQueryIfNeeded(originalPrompt: String) -> String {
        let base = extractWebQuery(from: originalPrompt)
        let normalizedBase = base.trimmingCharacters(in: .whitespacesAndNewlines)
        let fallback = normalizedBase.isEmpty ? originalPrompt : normalizedBase
        let lower = normalized(fallback)
        let architectureTerms = ["architecture", "module", "service", "component", "package"]
        guard containsAny(lower, ["architecture", "module", "service", "component", "package", "design", "system"]) else { return fallback }
        var query = fallback
        for term in architectureTerms where !lower.contains(term) { query += " " + term }
        return query
    }

    static func extractWebQuery(from text: String) -> String { SlotAgentService.shared_extractWebQuery(text) }
    static func extractOutlookSearchQuery(from text: String) -> String { SlotAgentService.shared_extractOutlookSearchQuery(text) }
    static func extractOutlookMessageReference(from text: String) -> String? { SlotAgentService.shared_extractOutlookMessageReference(text) }
    static func extractOutlookBody(from text: String) -> String? { let b = SlotAgentService.shared_extractOutlookBody(text); return b.isEmpty ? nil : b }
    static func firstURL(in text: String) -> String? { SlotAgentService.shared_firstURL(text) }

    private static func extractMinutes(from text: String) -> Int? {
        if let duration = RelativeDuration.parse(from: text) {
            return duration.minutesCeiled
        }
        let nsRange = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let regex = try? NSRegularExpression(pattern: #"(\d+)\s*(?:minute|minutes|min|mins)"#),
              let match = regex.firstMatch(in: text, range: nsRange),
              match.numberOfRanges > 1,
              let range = Range(match.range(at: 1), in: text),
              let value = Int(text[range]) else { return nil }
        return max(1, min(value, 24 * 60))
    }

    private static func countdownDurationSeconds(from text: String) -> Int? {
        if let duration = RelativeDuration.parse(from: text) { return duration.seconds }
        if let seconds = firstCapture(in: text, pattern: #"(?i)\b(\d+)\s*(?:second|seconds|sec|secs)\b"#).flatMap(Int.init) { return max(1, min(seconds, 24 * 60 * 60)) }
        if let minutes = extractMinutes(from: text) { return minutes * 60 }
        return nil
    }

    private static func isAlarmScheduleIntent(_ text: String) -> Bool {
        let hasAlarmTarget = containsAny(text, ["alarm", "wake me", "wake us"])
        guard hasAlarmTarget else { return false }
        return containsAny(text, ["schedule", "set an alarm", "set alarm", "create an alarm", "create alarm", "add an alarm", "add alarm", "wake me", "wake us"])
    }

    private static func alarmMutationArgs(from prompt: String) -> AgentJSONArguments {
        if let uuid = firstUUID(in: prompt) { return ["id": .string(uuid)] }
        let title = extractAlarmTitle(from: prompt, fallback: "")
        if !title.isEmpty, let uuid = lookupAlarmUUIDByTitle(title) { return ["id": .string(uuid)] }
        return [:]
    }

    private static func lookupAlarmUUIDByTitle(_ title: String) -> String? {
#if canImport(AlarmKit)
        if #available(iOS 26.0, *) {
            do {
                let alarms = try AlarmManager.shared.alarms
                let matches = alarms.filter { alarm in
                    String(describing: alarm).localizedCaseInsensitiveContains(title) ||
                    (Mirror(reflecting: alarm).children.first { $0.label == "title" }?.value as? String)?.localizedCaseInsensitiveCompare(title) == .orderedSame
                }
                if matches.count == 1, let alarm = matches.first,
                   let id = Mirror(reflecting: alarm).children.first(where: { $0.label == "id" })?.value as? UUID {
                    return id.uuidString
                }
            } catch { return nil }
        }
#endif
        return nil
    }

    private static func firstUUID(in text: String) -> String? { AlarmCommandClassifier.firstUUID(in: text) }

    private static func extractAlarmTitle(from prompt: String, fallback: String) -> String {
        if let named = firstCapture(in: prompt, pattern: #"(?i)\b(?:named|called)\s+([^.!?\n]+)"#) {
            var cleaned = cleanCapturedValue(named)
            if let range = cleaned.range(of: #"(?i)\s+(?:for|in)\s+\d+\s+(?:seconds?|minutes?|hours?|days?)\b"#, options: .regularExpression) {
                cleaned = String(cleaned[..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if !cleaned.isEmpty { return cleaned }
        }
        return fallback
    }

    private static func extractTriggerCancelIdentifier(from prompt: String) -> String {
        if let uuid = firstUUID(in: prompt) { return uuid }
        if let named = firstCapture(in: prompt, pattern: #"(?i)\b(?:named|called)\s+([^.!?\n]+)"#) { return cleanCapturedValue(named) }
        if let trigger = firstCapture(in: prompt, pattern: #"(?i)\bcancel\s+(?:the\s+)?(?:trigger|scheduled run|agent run)\s+([^.!?\n]+)"#) { return cleanCapturedValue(trigger) }
        return ""
    }

    private static func extractTriggerTitle(from prompt: String) -> String {
        let lower = normalized(prompt)
        if lower.contains("reminder") { return "Reminder summary" }
        if lower.contains("digest") { return "Scheduled digest" }
        return "Scheduled agent run"
    }

    private static func isPersonalProfileRecallIntent(_ text: String) -> Bool { IntentRouter.isPersonalProfileRecallIntent(text) }
    private static func isLatestOutlookReadIntent(_ text: String) -> Bool { containsAny(text, ["latest email", "last email", "read latest", "open latest", "open email", "latest outlook email", "last outlook email", "read my latest email", "read outlook message"]) }
    /// Determines whether text contains phrases indicating a save-then-recall memory pattern.
/// - Parameters:
///   - text: The text to evaluate.
/// - Returns: `true` if the text contains both save-related and recall-related phrases, `false` otherwise.
private static func isMemorySaveThenRecallIntent(_ text: String) -> Bool { MemoryCommandPlan.saveThenRecall(from: text) != nil }
    /// Determines whether text represents a nearby or proximity-based map search.
/// - Returns: `true` if the text contains proximity keywords, `false` otherwise.
private static func isNearbyMapSearchIntent(_ text: String) -> Bool { containsAny(text, ["nearby", "near me", "closest", "nearest", "around me", "around here", "in my area"]) }

    /// Produces a normalized web query for dynamic public lookup.
    ///
    /// Removes leading "where is/are" phrasing, corrects "Alcoholic Anonymous" capitalization, and appends "near me" to locality-based queries. The result is capped at 300 characters.
    ///
    /// - Parameter prompt: The original user prompt.
    /// - Returns: The cleaned web query.
    private static func dynamicPublicLookupWebQuery(from prompt: String) -> String {
        var query = extractWebQuery(from: prompt)
        query = query.replacingOccurrences(
            of: #"(?i)^\s*where\s+(?:is|are)\s+"#,
            with: "",
            options: .regularExpression
        )
        query = query.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters))
        let lower = normalized(query)

        if lower.contains("alcoholic anonymous") && !lower.contains("alcoholics anonymous") {
            query = query.replacingOccurrences(of: "alcoholic anonymous", with: "Alcoholics Anonymous", options: [.caseInsensitive])
        }
        if containsAny(lower, ["nearest", "closest", "around me", "around here", "in my area"])
            && !containsAny(lower, ["near me", "nearby"]) {
            query += " near me"
        }
        return String(query.trimmingCharacters(in: .whitespacesAndNewlines).prefix(300))
    }

    /// Determines whether a prompt contains keywords indicating a request to read or view calendar information.
    /// - Returns: `true` if the text contains calendar read keywords, `false` otherwise.
    private static func isCalendarReadIntent(_ text: String) -> Bool {
        containsAny(text, ["list", "show", "search", "find", "read", "check", "upcoming", "what's on", "what is on", "calendar", "event", "events", "appointment", "appointments", "meeting", "meetings", "schedule", "today", "tomorrow", "next", "do i have", "any"])
    }

    private static func isCalendarCreateIntent(_ text: String) -> Bool {
        if containsAny(text, ["what's on", "what is on", "do i have", "when is", "next meeting", "next event", "show", "list", "search my calendar", "read my calendar"]) { return false }
        if text.contains("my schedule") && !text.hasPrefix("schedule ") { return false }
        return containsAny(text, ["set an appointment", "set appointment", "schedule", "book "]) || (containsAny(text, ["create", "add", "put "]) && containsAny(text, ["event", "appointment", "meeting", "calendar"]))
    }

    private static func extractCalendarTitle(from prompt: String) -> String {
        if let called = firstCapture(in: prompt, pattern: #"(?i)\bcalled\s+([^.!?\n]+)"#) {
            let cleaned = cleanCapturedValue(called)
            if !cleaned.isEmpty { return cleaned }
        }
        let lower = normalized(prompt)
        if lower.contains("appointment") { return "Appointment" }
        if lower.contains("meeting") { return "Meeting" }
        return "Event"
    }

    private static func calendarStartOffsetMinutes(from text: String) -> Int? {
        if let duration = RelativeDuration.parse(from: text) {
            return duration.minutesCeiled
        }
        if RelativeDuration.containsExplicitRelativeSyntax(text) {
            return nil
        }
        if let explicitMinutes = extractMinutes(from: text) { return explicitMinutes }
        let now = Date()
        var calendar = Calendar.current
        calendar.locale = Locale(identifier: "en_US_POSIX")
        var target = now
        if text.contains("tomorrow") { target = calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: now)) ?? now.addingTimeInterval(24 * 60 * 60) }
        let hour = calendarHour(from: text) ?? (text.contains("morning") ? 9 : nil) ?? (text.contains("afternoon") ? 13 : nil) ?? (text.contains("evening") ? 18 : nil)
        if let hour {
            let base = text.contains("tomorrow") ? target : now
            let start = calendar.startOfDay(for: base)
            target = calendar.date(byAdding: .hour, value: hour, to: start) ?? target
        } else if !text.contains("tomorrow") {
            return nil
        }
        if target <= now { target = calendar.date(byAdding: .day, value: 1, to: target) ?? now.addingTimeInterval(60 * 60) }
        return max(1, Int(target.timeIntervalSince(now) / 60))
    }

    private static func calendarHour(from text: String) -> Int? {
        if let value = firstCapture(in: text, pattern: #"(?i)\bat\s+([0-9]{1,2})(?::[0-9]{2})?\s*(?:am|pm)?"#), let raw = Int(value) {
            let pm = text.contains("pm") || text.contains("afternoon") || text.contains("evening")
            if pm, raw < 12 { return raw + 12 }
            return raw == 12 && text.contains("am") ? 0 : min(max(raw, 0), 23)
        }
        let words = ["one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12]
        for (word, hour) in words where text.contains(" at \(word)") {
            let pm = text.contains("pm") || text.contains("afternoon") || text.contains("evening")
            return pm && hour < 12 ? hour + 12 : hour
        }
        return nil
    }

    private static func extractMemoryRecallQuery(from prompt: String) -> String {
        MemoryCommandPlan.extractMemoryRecallQuery(from: prompt)
    }

    static func extractMemoryFact(from prompt: String) -> String {
        MemoryCommandPlan.extractMemoryFact(from: prompt)
    }

    private static func firstCapture(in text: String, pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = text as NSString
        guard let match = regex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)), match.numberOfRanges > 1 else { return nil }
        return ns.substring(with: match.range(at: 1))
    }

    private static func cleanCapturedValue(_ value: String) -> String { value.trimmingCharacters(in: CharacterSet(charactersIn: " \t\n\r\"'.,!?")) }
    private static func normalizeFactSentence(_ value: String) -> String {
        let cleaned = cleanCapturedValue(value)
        guard !cleaned.isEmpty else { return value.trimmingCharacters(in: .whitespacesAndNewlines) }
        return cleaned
    }

    private static func extractTriggerPrompt(from prompt: String) -> String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "Run the scheduled agent task." }
        let lower = normalized(trimmed)
        if let range = lower.range(of: "to ") {
            let offset = lower.distance(from: lower.startIndex, to: range.upperBound)
            let start = trimmed.index(trimmed.startIndex, offsetBy: offset, limitedBy: trimmed.endIndex) ?? trimmed.endIndex
            let task = String(trimmed[start...]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !task.isEmpty { return task }
        }
        return trimmed
    }

    static func extractDestination(from text: String) -> String? {
        let lower = normalized(text)
        for marker in [" to ", " near ", " for ", " in ", " at "] {
            if let r = lower.range(of: marker) { return String(text[r.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines) }
        }
        return nil
    }

    private static func extractPhotoQuery(from prompt: String) -> String {
        let text = normalized(prompt)
        if containsAny(text, ["latest selfie", "newest selfie", "recent selfies", "recent selfie", "selfie picture"]) {
            return "latest selfie"
        }
        if containsAny(text, ["latest photo", "latest picture", "newest photo", "newest picture", "recent photo", "recent picture"]) {
            return "latest photo"
        }
        if text.contains("selfie") { return "selfie" }
        let destination = extractDestination(from: prompt)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return destination?.isEmpty == false ? destination! : "photos"
    }

    static func extractContactQuery(from text: String) -> String? { extractDestination(from: text) }

    static func extractNearbySearchQuery(from text: String) -> String? {
        let lower = normalized(text)
        if let range = lower.range(of: "nearby ") {
            let query = String(text[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            return query.isEmpty ? nil : query
        }
        if let range = lower.range(of: "closest ") {
            let query = String(text[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            return query.isEmpty ? nil : query
        }
        if let range = lower.range(of: "nearest ") {
            let query = String(text[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            return query.isEmpty ? nil : query
        }
        if let range = lower.range(of: " near me") {
            let head = String(text[..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            let cleaned = head.replacingOccurrences(of: #"(?i)^(find|show|search|locate)\s+"#, with: "", options: .regularExpression).trimmingCharacters(in: .whitespacesAndNewlines)
            return cleaned.isEmpty ? nil : cleaned
        }
        return nil
    }

    static func extractEmailAddress(from text: String) -> String? {
        let pattern = #"[A-Z0-9a-z._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = text as NSString
        guard let match = regex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) else { return nil }
        return ns.substring(with: match.range)
    }

    static func extractOutlookSubject(from text: String) -> String {
        let lower = text.lowercased()
        for marker in [" subject ", " subject:"] {
            if let range = lower.range(of: marker) {
                let remainder = String(text[range.upperBound...])
                if let bodyRange = remainder.lowercased().range(of: " body ") { return String(remainder[..<bodyRange.lowerBound]).trimmingCharacters(in: CharacterSet(charactersIn: "\"' :.,!?")) }
                return remainder.trimmingCharacters(in: CharacterSet(charactersIn: "\"' :.,!?"))
            }
        }
        return ""
    }

    static func extractOutlookDestinationFolder(from text: String) -> String? {
        let lower = normalized(text)
        if lower.contains("junk") || lower.contains("spam") { return "junkemail" }
        if lower.contains("trash") || lower.contains("deleted") { return "deleteditems" }
        if lower.contains("archive") { return "archive" }
        if lower.contains("inbox") { return "inbox" }
        if lower.contains("sent") { return "sentitems" }
        if lower.contains("draft") { return "drafts" }
        return nil
    }

    static func extractFileName(from text: String) -> String? {
        let pattern = #"[A-Za-z0-9][A-Za-z0-9._-]*\.(?:md|txt|json|pdf|swift|docx|csv)"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = text as NSString
        guard let match = regex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) else { return nil }
        return ns.substring(with: match.range)
    }

    private static func firstPhoneNumber(in text: String) -> String? {
        let pattern = #"\+?[0-9][0-9\s\-\(\)]{6,}[0-9]"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = text as NSString
        guard let match = regex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) else { return nil }
        return ns.substring(with: match.range).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func extractRecipientName(from text: String) -> String? {
        extractNameAfterMarkers(in: text, markers: [" to ", " for "], terminators: [" about ", " saying ", " that says ", " body ", " message ", " and ask ", " from contacts"])
    }

    private static func extractCallTarget(from text: String) -> String? {
        extractNameAfterMarkers(in: text, markers: ["place a call to ", "start a call to ", "call ", "phone "], terminators: [" from contacts", " using contacts", " in contacts"])
    }

    private static func extractNameAfterMarkers(in text: String, markers: [String], terminators: [String]) -> String? {
        let lower = text.lowercased()
        for marker in markers {
            guard let range = lower.range(of: marker) else { continue }
            var remainder = String(text[range.upperBound...])
            let lowerRemainder = remainder.lowercased()
            if let terminator = terminators.compactMap({ lowerRemainder.range(of: $0)?.lowerBound }).min() { remainder = String(remainder[..<terminator]) }
            let cleaned = remainder.trimmingCharacters(in: CharacterSet(charactersIn: "\"' :.,!?"))
            return cleaned.isEmpty ? nil : cleaned
        }
        return nil
    }

    private static func extractCommunicationBody(from text: String) -> String {
        let lower = text.lowercased()
        for marker in [" saying ", " that says ", " body ", " body:", " message ", " about "] {
            if let range = lower.range(of: marker) { return String(text[range.upperBound...]).trimmingCharacters(in: CharacterSet(charactersIn: "\"' :.,!?")) }
        }
        if lower.hasPrefix("text ") || lower.hasPrefix("message ") || lower.hasPrefix("sms ") || lower.hasPrefix("imessage ") {
            if let range = lower.range(of: " that ") {
                let body = String(text[range.upperBound...]).trimmingCharacters(in: CharacterSet(charactersIn: "\"' :.,!?"))
                if !body.isEmpty { return body }
            }
        }
        return ""
    }

    private static func extractEmailSubject(from text: String) -> String {
        let body = extractCommunicationBody(from: text)
        if !body.isEmpty { return String(body.prefix(48)) }
        return ""
    }
}
