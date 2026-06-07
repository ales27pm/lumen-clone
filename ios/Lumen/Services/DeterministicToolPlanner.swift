import Foundation

nonisolated enum DeterministicToolPlanner {
    static func planForSpecificTool(toolID: String, prompt: String, availableToolIDs: Set<String>) -> AgentAction? {
        let canonical = ToolRouteGuard.canonicalToolID(toolID)
        guard availableToolIDs.contains(canonical) else { return nil }
        let text = normalized(prompt)
        switch canonical {
        case "camera.capture":
            return AgentAction(tool: canonical, args: [:])
        case "location.current":
            return AgentAction(tool: canonical, args: [:])
        case "maps.search":
            let query = extractNearbySearchQuery(from: prompt) ?? extractDestination(from: prompt) ?? ""
            return AgentAction(tool: canonical, args: ["query": .string(query)])
        case "maps.directions":
            guard let destination = extractDestination(from: prompt), !destination.isEmpty else { return nil }
            return AgentAction(tool: canonical, args: ["destination": .string(destination)])
        case "outlook.status":
            return AgentAction(tool: canonical, args: [:])
        case "outlook.messages.list":
            var args: AgentJSONArguments = ["limit": .string("10")]
            if text.contains("unread") { args["unreadOnly"] = .string("true") }
            return AgentAction(tool: canonical, args: args)
        case "outlook.message.read":
            return AgentAction(tool: canonical, args: ["message": .string(extractOutlookMessageReference(from: text) ?? "latest")])
        default:
            return AgentAction(tool: canonical, args: [:])
        }
    }

    static func planSteps(routing: IntentRoutingDecision, prompt: String, availableToolIDs: Set<String>) -> [AgentAction] {
        let text = normalized(prompt)
        if routing.intent == .memory, isMemorySaveThenRecallIntent(text) {
            var actions: [AgentAction] = []
            if let save = plan(routing: routing, prompt: prompt, availableToolIDs: availableToolIDs), ToolRouteGuard.canonicalToolID(save.tool) == "memory.save" {
                actions.append(save)
            } else if availableToolIDs.contains("memory.save") {
                actions.append(AgentAction(tool: "memory.save", args: [
                    "content": .string(extractMemoryFact(from: prompt)),
                    "kind": .string("fact")
                ]))
            }
            if availableToolIDs.contains("memory.recall") {
                actions.append(AgentAction(tool: "memory.recall", args: ["query": .string(extractMemoryRecallQuery(from: prompt))]))
            }
            if !actions.isEmpty { return actions }
        }
        if routing.intent == .outlook, isLatestOutlookReadIntent(text) {
            if availableToolIDs.contains("outlook.messages.list"),
               availableToolIDs.contains("outlook.message.read") {
                return [
                    AgentAction(tool: "outlook.messages.list", args: ["limit": .string("1")]),
                    AgentAction(tool: "outlook.message.read", args: ["message": .string("latest")])
                ]
            }
            if availableToolIDs.contains("outlook.message.read") {
                return [AgentAction(tool: "outlook.message.read", args: ["message": .string("latest")])]
            }
        }
        if let single = plan(routing: routing, prompt: prompt, availableToolIDs: availableToolIDs) {
            return [single]
        }
        return []
    }

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
            let query = extractWebQuery(from: prompt)
            return query.isEmpty ? nil : AgentAction(tool: "web.search", args: ["query": .string(query)])
        case .emailDraft:
            var args: AgentJSONArguments = ["subject": .string(extractEmailSubject(from: prompt)), "body": .string(extractCommunicationBody(from: prompt))]
            if let email = extractEmailAddress(from: prompt) {
                args["to"] = .string(email)
            } else if let recipient = extractRecipientName(from: prompt), !recipient.isEmpty {
                args["to"] = .string(recipient)
            }
            return action("mail.draft", args)
        case .messageDraft:
            var args: AgentJSONArguments = ["body": .string(extractCommunicationBody(from: prompt))]
            if let recipient = extractRecipientName(from: prompt), !recipient.isEmpty { args["to"] = .string(recipient) }
            return action("messages.draft", args)
        case .phoneCall:
            if let phone = firstPhoneNumber(in: prompt) { return action("phone.call", ["number": .string(phone)]) }
            if let query = extractCallTarget(from: prompt), !query.isEmpty {
                return action("contacts.search", ["query": .string(query)])
            }
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
            if containsAny(text, ["nearby", "near me", "closest"]) {
                let query = extractNearbySearchQuery(from: prompt) ?? extractDestination(from: prompt) ?? ""
                return action("maps.search", ["query": .string(query)])
            }
            return nil
        case .calendar:
            if containsAny(text, ["list", "show", "upcoming", "today", "tomorrow"]) { return action("calendar.list") }
            return nil
        case .reminder:
            if containsAny(text, ["list", "show", "pending"]) { return action("reminders.list") }
            if containsAny(text, ["create", "add", "remind me"]), let body = extractOutlookBody(from: prompt), !body.isEmpty { return action("reminders.create", ["title": .string(body)]) }
            return nil
        case .contactSearch:
            if let q = extractContactQuery(from: prompt), !q.isEmpty { return action("contacts.search", ["query": .string(q)]) }
            return nil
        case .photos: return action("photos.search", ["query": .string(extractDestination(from: prompt) ?? "")])
        case .health: return action("health.summary")
        case .motion: return action("motion.activity")
        case .files:
            if let name = extractFileName(from: prompt) { return action("files.read", ["name": .string(name)]) }
            return nil
        case .memory:
            if isPersonalProfileRecallIntent(text) { return action("memory.recall", ["query": .string("user name")]) }
            if containsAny(text, ["what do you remember", "recall", "remember about"]) { return action("memory.recall", ["query": .string(extractContactQuery(from: prompt) ?? "")]) }
            if containsAny(text, ["remember", "save", "my name is", "call me"]) {
                return action("memory.save", [
                    "content": .string(extractMemoryFact(from: prompt)),
                    "kind": .string("fact")
                ])
            }
            return nil
        case .rag:
            if containsAny(text, ["index photos", "reindex photos", "photo retrieval index"]) { return action("rag.index_photos") }
            if containsAny(text, ["reindex", "index files", "file retrieval index", "refresh retrieval index"]) { return action("rag.index_files") }
            if containsAny(text, ["search"]) {
                let query = expandRAGQueryIfNeeded(originalPrompt: prompt)
                return action("rag.search", ["query": .string(query)])
            }
            return nil
        case .alarm:
            if containsAny(text, ["list", "status"]) { return action("alarm.list") ?? action("alarm.authorization_status") }
            return nil
        case .trigger:
            if text.contains("list") { return action("trigger.list") }
            if text.contains("cancel"), let token = extractContactQuery(from: prompt), !token.isEmpty { return action("trigger.cancel", ["id": .string(token)]) }
            if has("trigger.create") {
                var args: AgentJSONArguments = [
                    "title": .string(extractTriggerTitle(from: prompt)),
                    "prompt": .string(extractTriggerPrompt(from: prompt)),
                    "schedule": .string("once")
                ]
                if text.contains("tonight") {
                    args["inMinutes"] = .string("120")
                } else if let minutes = extractMinutes(from: text) {
                    args["inMinutes"] = .string(String(minutes))
                } else {
                    args["inMinutes"] = .string("60")
                }
                return action("trigger.create", args)
            }
            return nil
        default:
            return nil
        }
    }

    private static func planOutlook(text: String, prompt: String, availableToolIDs: Set<String>) -> AgentAction? {
        func can(_ tool: String) -> Bool { availableToolIDs.contains(tool) }
        func action(_ tool: String, _ args: AgentJSONArguments = [:]) -> AgentAction? { can(tool) ? AgentAction(tool: tool, args: args) : nil }
        if containsAny(text, ["search", "find", "invoice"]) {
            let q = extractOutlookSearchQuery(from: prompt)
            if !q.isEmpty { return action("outlook.messages.search", ["query": .string(q), "limit": .string("10")]) }
        }
        if containsAny(text, ["new emails", "new email", "unread emails", "unread email", "inbox"]) {
            var args: AgentJSONArguments = ["limit": .string("10")]
            if text.contains("unread") { args["unreadOnly"] = .string("true") }
            return action("outlook.messages.list", args)
        }
        if isLatestOutlookReadIntent(text) {
            return action("outlook.messages.list", ["limit": .string("1")])
                ?? action("outlook.message.read", ["message": .string("latest")])
        }
        if containsAny(text, ["reply all", "reply-all", "respond to all"]) {
            return action("outlook.message.reply_all", ["message": .string(extractOutlookMessageReference(from: text) ?? "latest"), "body": .string(extractOutlookBody(from: prompt) ?? "")])
        }
        if containsAny(text, ["reply", "respond"]) {
            return action("outlook.message.reply", ["message": .string(extractOutlookMessageReference(from: text) ?? "latest"), "body": .string(extractOutlookBody(from: prompt) ?? "")])
        }
        if text.contains("forward") {
            var args: AgentJSONArguments = ["message": .string(extractOutlookMessageReference(from: text) ?? "latest")]
            if let to = extractEmailAddress(from: prompt) { args["to"] = .string(to) }
            if let body = extractOutlookBody(from: prompt), !body.isEmpty { args["body"] = .string(body) }
            return action("outlook.message.forward", args)
        }
        if text.contains("archive") { return action("outlook.message.archive", ["message": .string(extractOutlookMessageReference(from: text) ?? "latest")]) }
        if containsAny(text, ["delete", "trash"]) { return action("outlook.message.delete", ["message": .string(extractOutlookMessageReference(from: text) ?? "latest")]) }
        if text.contains("mark") && text.contains("unread") { return action("outlook.message.mark_unread", ["message": .string(extractOutlookMessageReference(from: text) ?? "latest")]) }
        if text.contains("mark") && text.contains("read") { return action("outlook.message.mark_read", ["message": .string(extractOutlookMessageReference(from: text) ?? "latest")]) }
        if text.contains("move") {
            guard let destination = extractOutlookDestinationFolder(from: text) else { return nil }
            return action("outlook.message.move", [
                "message": .string(extractOutlookMessageReference(from: text) ?? "latest"),
                "destination": .string(destination)
            ])
        }
        if containsAny(text, ["status", "connected", "signed in", "auth"]) { return action("outlook.status") }
        if containsAny(text, ["folder", "folders"]) { return action("outlook.folders.list") }
        if containsAny(text, ["attachment", "attachments", "paperclip"]) { return action("outlook.attachments.list", ["message": .string(extractOutlookMessageReference(from: text) ?? "latest")]) }
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
        guard containsAny(lower, ["architecture", "module", "service", "component", "package", "design", "system"]) else {
            return fallback
        }
        var query = fallback
        for term in architectureTerms where !lower.contains(term) {
            query += " " + term
        }
        return query
    }
    static func extractWebQuery(from text: String) -> String { SlotAgentService.shared_extractWebQuery(text) }
    static func extractOutlookSearchQuery(from text: String) -> String { SlotAgentService.shared_extractOutlookSearchQuery(text) }
    static func extractOutlookMessageReference(from text: String) -> String? { SlotAgentService.shared_extractOutlookMessageReference(text) }
    static func extractOutlookBody(from text: String) -> String? { let b = SlotAgentService.shared_extractOutlookBody(text); return b.isEmpty ? nil : b }
    static func firstURL(in text: String) -> String? { SlotAgentService.shared_firstURL(text) }
    private static func extractMinutes(from text: String) -> Int? {
        let nsRange = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let regex = try? NSRegularExpression(pattern: #"(\d+)\s*(?:minute|minutes|min|mins)"#),
              let match = regex.firstMatch(in: text, range: nsRange),
              match.numberOfRanges > 1,
              let range = Range(match.range(at: 1), in: text),
              let value = Int(text[range]) else {
            return nil
        }
        return max(1, min(value, 24 * 60))
    }

    private static func extractTriggerTitle(from prompt: String) -> String {
        let lower = normalized(prompt)
        if lower.contains("reminder") { return "Reminder summary" }
        if lower.contains("digest") { return "Scheduled digest" }
        return "Scheduled agent run"
    }


    private static func isPersonalProfileRecallIntent(_ text: String) -> Bool {
        IntentRouter.isPersonalProfileRecallIntent(text)
    }

    private static func isLatestOutlookReadIntent(_ text: String) -> Bool {
        containsAny(text, ["latest email", "last email", "read latest", "open latest", "open email", "latest outlook email", "last outlook email", "read my latest email"])
    }

    private static func isMemorySaveThenRecallIntent(_ text: String) -> Bool {
        containsAny(text, ["remember", "save", "note", "keep this in mind"])
            && containsAny(text, ["tell me what", "what you remembered", "what did you remember", "repeat it back", "then tell"])
    }

    private static func extractMemoryRecallQuery(from prompt: String) -> String {
        let fact = extractMemoryFact(from: prompt)
        let cleaned = fact
            .replacingOccurrences(of: "User's name is ", with: "", options: [.caseInsensitive])
            .replacingOccurrences(of: "User prefers to be called ", with: "", options: [.caseInsensitive])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if cleaned.lowercased().contains("prefer concise bullet points") { return "prefer concise bullet points" }
        if !cleaned.isEmpty { return cleaned }
        return prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func extractMemoryFact(from prompt: String) -> String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if let name = firstCapture(in: trimmed, pattern: #"(?i)\b(?:can you\s+)?(?:please\s+)?(?:remember|save|note)\s+(?:that\s+)?my name is\s+([^.!?\n]+)"#)
            ?? firstCapture(in: trimmed, pattern: #"(?i)\bmy name is\s+([^.!?\n]+)"#) {
            return "User's name is \(cleanCapturedValue(name))"
        }
        if let name = firstCapture(in: trimmed, pattern: #"(?i)\bcall me\s+([^.!?\n]+)"#) {
            return "User prefers to be called \(cleanCapturedValue(name))"
        }
        if let fact = firstCapture(in: trimmed, pattern: #"(?i)\bremember that\s+(.+)"#)
            ?? firstCapture(in: trimmed, pattern: #"(?i)\bsave this fact:?\s+(.+)"#)
            ?? firstCapture(in: trimmed, pattern: #"(?i)\bkeep this in mind:?\s+(.+)"#) {
            return normalizeFactSentence(fact)
        }
        return normalizeFactSentence(trimmed)
    }

    private static func firstCapture(in text: String, pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = text as NSString
        guard let match = regex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)), match.numberOfRanges > 1 else { return nil }
        return ns.substring(with: match.range(at: 1))
    }

    private static func cleanCapturedValue(_ value: String) -> String {
        value.trimmingCharacters(in: CharacterSet(charactersIn: " \t\n\r\"'.,!?"))
    }

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
        if let range = lower.range(of: " near me") {
            let head = String(text[..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            let cleaned = head.replacingOccurrences(of: #"(?i)^(find|show|search|locate)\s+"#, with: "", options: .regularExpression)
                .trimmingCharacters(in: .whitespacesAndNewlines)
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
                if let bodyRange = remainder.lowercased().range(of: " body ") {
                    return String(remainder[..<bodyRange.lowerBound]).trimmingCharacters(in: CharacterSet(charactersIn: "\"' :.,!?"))
                }
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
        extractNameAfterMarkers(
            in: text,
            markers: [" to ", " for "],
            terminators: [" about ", " saying ", " that says ", " body ", " message ", " and ask ", " from contacts"]
        )
    }

    private static func extractCallTarget(from text: String) -> String? {
        extractNameAfterMarkers(
            in: text,
            markers: ["place a call to ", "start a call to ", "call ", "phone "],
            terminators: [" from contacts", " using contacts", " in contacts"]
        )
    }

    private static func extractNameAfterMarkers(in text: String, markers: [String], terminators: [String]) -> String? {
        let lower = text.lowercased()
        for marker in markers {
            guard let range = lower.range(of: marker) else { continue }
            var remainder = String(text[range.upperBound...])
            let lowerRemainder = remainder.lowercased()
            if let terminator = terminators.compactMap({ lowerRemainder.range(of: $0)?.lowerBound }).min() {
                remainder = String(remainder[..<terminator])
            }
            let cleaned = remainder.trimmingCharacters(in: CharacterSet(charactersIn: "\"' :.,!?"))
            return cleaned.isEmpty ? nil : cleaned
        }
        return nil
    }

    private static func extractCommunicationBody(from text: String) -> String {
        let lower = text.lowercased()
        for marker in [" saying ", " that says ", " body ", " body:", " message ", " about "] {
            if let range = lower.range(of: marker) {
                return String(text[range.upperBound...]).trimmingCharacters(in: CharacterSet(charactersIn: "\"' :.,!?"))
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
