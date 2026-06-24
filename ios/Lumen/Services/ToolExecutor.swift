import Foundation

nonisolated enum ToolExecutionApproval: Sendable {
    case autonomous
    case pending
    case userApproved
}

nonisolated enum ToolRouteGuard {
    enum PermissionGateDecision: Equatable, Sendable {
        case allowed
        case request
        case denied(String)
    }

    @MainActor
    static func ensurePermissionIfNeeded(
        for canonicalToolID: String,
        arguments: [String: String],
        isForeground: Bool = true
    ) async -> String? {
        if canonicalToolID == "weather" {
            let value = (arguments["location"] ?? arguments["city"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if !value.isEmpty, value != "here", value != "current", value != "current location" {
                return nil
            }
        }
        guard let tool = ToolRegistry.find(id: canonicalToolID),
              let permissionKind = tool.permissionKind else {
            return nil
        }

        let permissions = PermissionsCenter.shared
        let initial = permissions.state(permissionKind)
        switch permissionGateDecision(for: permissionKind, state: initial, isForeground: isForeground) {
        case .allowed:
            return nil
        case .request:
            await permissions.request(permissionKind)
            let updated = permissions.state(permissionKind)
            if updated == .granted || updated == .limited {
                return nil
            }
            return permissionUnavailableMessage(for: permissionKind)
        case .denied(let message):
            return message
        }
    }

    static func permissionGateDecision(
        for permissionKind: PermissionKind,
        state: PermissionState,
        isForeground: Bool
    ) -> PermissionGateDecision {
        switch state {
        case .granted, .limited:
            return .allowed
        case .notDetermined:
            return isForeground ? .request : .denied(permissionUnavailableMessage(for: permissionKind))
        case .denied, .restricted, .unavailable:
            return .denied(permissionUnavailableMessage(for: permissionKind))
        }
    }

    private static func permissionUnavailableMessage(for permissionKind: PermissionKind) -> String {
        "I need \(permissionKind.title.lowercased()) access to do that. Please enable it in Settings or provide an alternative."
    }

    static func canonicalToolID(_ raw: String) -> String {
        let id = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            .replacingOccurrences(of: "-", with: ".")
            .replacingOccurrences(of: " ", with: ".")
        switch id {
        case "weather", "weather.current", "current.weather", "forecast.current", "weather.get", "get.weather", "getweather", "currentweather":
            return "weather"
        case "search", "internet.search", "web", "web.search", "websearch", "browser.search", "google.search", "google", "search.web", "searchweb":
            return "web.search"
        case "fetch", "web.fetch", "browser.fetch", "url.fetch", "fetch.url", "open.url", "read.url", "read.website":
            return "web.fetch"
        case "maps", "map", "map.search", "maps.search", "nearby.search", "local.search", "places.search", "place.search", "google.maps", "google.maps.api", "googlemaps", "googlemapsapi", "maps.api", "mapsapi", "nearest.place", "find.nearby":
            return "maps.search"
        case "maps.directions", "map.directions", "directions", "navigation", "navigate", "route", "route.to", "open.maps":
            return "maps.directions"
        case "location", "gps", "current.location", "location.get", "get.location", "currentlocation":
            return "location.current"
        case "calendar", "calendar.create", "create.event", "event.create", "schedule.event":
            return "calendar.create"
        case "calendar.list", "list.events", "events.list":
            return "calendar.list"
        case "reminder", "reminders.create", "reminder.create", "create.reminder":
            return "reminders.create"
        case "reminders.list", "reminder.list", "list.reminders":
            return "reminders.list"
        case "mail", "email", "email.draft", "mail.draft", "compose.email":
            return "mail.draft"
        case "message", "messages.draft", "sms", "sms.draft", "compose.message", "imessage":
            return "messages.draft"
        case "phone", "phone.call", "call", "dial":
            return "phone.call"
        case "contacts", "contacts.search", "contact.search", "search.contacts":
            return "contacts.search"
        case "contacts.lookup":
            return "contacts.search"
        case "calendar.read":
            return "calendar.list"
        case "location.snapshot":
            return "location.current"
        case "memory.search":
            return "memory.recall"
        case "rag.search.secure":
            return "rag.search"
        case "outlook", "outlook.status", "microsoft.outlook.status", "hotmail.status", "graph.status":
            return "outlook.status"
        case "outlook.folders", "outlook.folder.list", "outlook.folders.list", "hotmail.folders", "mail.folders.list":
            return "outlook.folders.list"
        case "outlook.messages", "outlook.inbox", "outlook.mail.list", "outlook.messages.list", "hotmail.inbox", "hotmail.messages", "graph.mail.list":
            return "outlook.messages.list"
        case "outlook.search", "outlook.messages.search", "outlook.mail.search", "hotmail.search", "search.outlook", "search.email", "email.search":
            return "outlook.messages.search"
        case "outlook.read", "outlook.message.read", "outlook.mail.read", "read.outlook", "read.email":
            return "outlook.message.read"
        case "outlook.attachments", "outlook.attachments.list", "outlook.message.attachments", "email.attachments":
            return "outlook.attachments.list"
        case "outlook.draft", "outlook.draft.create", "outlook.create.draft", "outlook.mail.draft", "hotmail.draft":
            return "outlook.draft.create"
        case "outlook.send", "outlook.mail.send", "hotmail.send", "send.outlook", "send.email.graph":
            return "outlook.mail.send"
        case "outlook.mark.read", "outlook.message.mark.read", "outlook.message.mark_read", "email.mark.read":
            return "outlook.message.mark_read"
        case "outlook.mark.unread", "outlook.message.mark.unread", "outlook.message.mark_unread", "email.mark.unread":
            return "outlook.message.mark_unread"
        case "outlook.move", "outlook.message.move", "email.move":
            return "outlook.message.move"
        case "outlook.archive", "outlook.message.archive", "email.archive":
            return "outlook.message.archive"
        case "outlook.delete", "outlook.message.delete", "email.delete":
            return "outlook.message.delete"
        case "outlook.reply", "outlook.message.reply", "email.reply":
            return "outlook.message.reply"
        case "outlook.reply.all", "outlook.replyall", "outlook.message.reply.all", "outlook.message.reply_all", "email.reply.all":
            return "outlook.message.reply_all"
        case "outlook.forward", "outlook.message.forward", "email.forward":
            return "outlook.message.forward"
        case "alarm.auth.status", "alarm.authorization", "alarm.authorization.status", "alarm.authorization_status", "alarm.status", "alarm.permission.status":
            return "alarm.authorization_status"
        case "alarm.request.auth", "alarm.request.authorization", "alarm.request_authorization", "request.alarm.authorization", "request.alarm.permission":
            return "alarm.request_authorization"
        case "alarm.schedule", "schedule.alarm", "create.alarm", "set.alarm", "alarm.create":
            return "alarm.schedule"
        case "alarm.countdown", "countdown.alarm", "start.countdown", "timer.start", "start.timer":
            return "alarm.countdown"
        case "alarm.list", "list.alarms", "alarms.list", "show.alarms":
            return "alarm.list"
        case "alarm.pause", "pause.alarm":
            return "alarm.pause"
        case "alarm.resume", "resume.alarm":
            return "alarm.resume"
        case "alarm.stop", "stop.alarm":
            return "alarm.stop"
        case "alarm.snooze", "snooze.alarm":
            return "alarm.snooze"
        case "alarm.cancel", "cancel.alarm", "delete.alarm":
            return "alarm.cancel"
        default:
            return id
        }
    }

    static func normalizedArguments(for canonicalToolID: String, rawToolID: String, arguments: [String: String]) -> [String: String] {
        var out = arguments
        let loweredValues = arguments.mapValues { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }

        switch canonicalToolID {
        case "maps.search":
            if out["query"] == nil {
                out["query"] = arguments["location"] ?? arguments["destination"] ?? arguments["place"] ?? arguments["nearby"]
            }
            let q = (out["query"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if q.isEmpty || q == "current location" || q == "current" || q == "here" || q == "near me" {
                out["query"] = "nearest place near me"
            }
            if q.contains("airport") && !q.contains("near") {
                out["query"] = "nearest airport near me"
            }
        case "maps.directions":
            if out["destination"] == nil {
                out["destination"] = arguments["query"] ?? arguments["location"] ?? arguments["place"]
            }
        case "weather":
            if out["location"] == nil {
                out["location"] = arguments["query"] ?? arguments["city"]
            }
        case "web.search":
            if out["query"] == nil {
                out["query"] = arguments["q"] ?? arguments["term"] ?? arguments["search"]
            }
        case "web.fetch":
            if out["url"] == nil {
                out["url"] = arguments["uri"] ?? arguments["link"] ?? arguments["query"]
            }
        case "outlook.messages.search":
            if out["query"] == nil {
                out["query"] = arguments["q"] ?? arguments["term"] ?? arguments["search"] ?? arguments["subject"] ?? arguments["from"]
            }
        case "outlook.message.read", "outlook.attachments.list", "outlook.message.mark_read", "outlook.message.mark_unread", "outlook.message.move", "outlook.message.archive", "outlook.message.delete", "outlook.message.reply", "outlook.message.reply_all":
            if out["messageId"] == nil {
                out["messageId"] = arguments["id"] ?? arguments["messageID"] ?? arguments["message"]
            }
            if out["id"] == nil {
                out["id"] = out["messageId"] ?? arguments["messageID"] ?? arguments["message"]
            }
        case "outlook.draft.create", "outlook.mail.send", "outlook.message.forward":
            if out["to"] == nil {
                out["to"] = arguments["recipient"] ?? arguments["recipients"] ?? arguments["email"]
            }
            if out["body"] == nil {
                out["body"] = arguments["message"] ?? arguments["text"] ?? arguments["content"] ?? arguments["comment"]
            }
            if canonicalToolID == "outlook.message.forward", out["messageId"] == nil {
                out["messageId"] = arguments["id"] ?? arguments["messageID"] ?? arguments["message"]
            }
        default:
            break
        }

        if canonicalToolID == "maps.search", loweredValues["location"] == "current location", out["query"]?.lowercased().contains("airport") != true {
            out["query"] = "nearest airport near me"
        }
        return out
    }

    static func canExecuteTool(_ canonicalToolID: String, arguments: [String: String], approval: ToolExecutionApproval) -> Bool {
        if requiresUserApproval(canonicalToolID), approval != .userApproved {
            return false
        }

        if canonicalToolID == "calendar.create" {
            return isExplicitCalendarCreateIntent(arguments: arguments)
        }
        return true
    }

    static func approvalRequiredMessage(for canonicalToolID: String) -> String {
        if canonicalToolID == "calendar.create" {
            return "Calendar event creation requires explicit user approval. I did not create an event."
        }
        return "This tool requires explicit user approval before it can run: \(canonicalToolID)."
    }

    static func requiresUserApproval(_ canonicalToolID: String) -> Bool {
        ToolRegistry.find(id: canonicalToolID)?.requiresApproval ?? false
    }

    static func isExplicitCalendarCreateIntent(arguments: [String: String]) -> Bool {
        let title = arguments["title"]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !title.isEmpty else { return false }

        let startsIn = arguments["startsInMinutes"]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let hasValidStart = Int(startsIn).map { $0 >= 0 } ?? false
        guard hasValidStart else { return false }

        let suspiciousGreetingTitles = ["hi", "hello", "hey", "hi lumen", "hello lumen", "hey lumen"]
        return !suspiciousGreetingTitles.contains(title.lowercased())
    }

    /// Determines whether a query should use web search instead of nearby search.
    /// - Parameter query: The search query to analyze.
    /// - Returns: `true` if the query indicates web search intent, `false` otherwise.
    static func shouldUseWebSearchInsteadOfNearbySearch(query: String) -> Bool {
        let normalized = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return false }

        if shouldUseWebSearchForDynamicPublicLookup(normalized) {
            return true
        }

        let localIntentMarkers = [
            "near me", "nearby", "closest", "around me", "around here", "in my area",
            "directions", "route to", "open maps", "address of", "store near",
            "restaurant near", "coffee near", "gas station", "pharmacy near", "airport near", "nearest airport", "nearest"
        ]
        if localIntentMarkers.contains(where: { normalized.contains($0) }) {
            return false
        }

        let webIntentMarkers = [
            "diy", "how to", "tutorial", "guide", "research", "internet", "web",
            "article", "manual", "documentation", "plans", "blueprint", "build"
        ]
        if webIntentMarkers.contains(where: { normalized.contains($0) }) {
            return true
        }

        if normalized.hasPrefix("search ") || normalized.hasPrefix("search for ") || normalized.hasPrefix("look up ") {
            return true
        }

        return false
    }

    /// Identifies queries that describe time-sensitive, location-specific events or services.
    /// - Parameter value: The query string to evaluate.
    /// - Returns: `true` if the query combines temporal, location-scope, and dynamic-subject references; `false` otherwise.
    static func shouldUseWebSearchForDynamicPublicLookup(_ value: String) -> Bool {
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return false }

        if isScheduledSupportGroupMeetingLookup(normalized) {
            return true
        }

        let timeMarkers = [
            "today", "tonight", "tomorrow", "this morning", "this afternoon", "this evening",
            "this weekend", "this week", "next week", "monday", "tuesday", "wednesday",
            "thursday", "friday", "saturday", "sunday", "open now", "open late", "closing time",
            "hours", "schedule", "timetable"
        ]
        let dynamicSubjects = [
            "meeting", "event", "class", "session", "showtime", "movie time", "screening",
            "clinic", "clinic hours", "walk-in", "walk in", "appointment availability", "store hours",
            "opening hours", "hours for", "hours of", "bus schedule", "train schedule",
            "ferry schedule", "flight status", "price", "ticket", "sale", "concert"
        ]
        let localScopeMarkers = [
            "near me", "nearby", "closest", "nearest", "around me", "around here", "in my area",
            "near us", "close to me", "where is", "where are"
        ]

        let hasClockTime = normalized.range(of: #"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b"#, options: .regularExpression) != nil
        let hasTime = timeMarkers.contains { normalized.contains($0) } || hasClockTime
        let hasDynamicSubject = dynamicSubjects.contains { normalized.contains($0) }
        let hasLocalScope = localScopeMarkers.contains { normalized.contains($0) }

        return hasTime && hasDynamicSubject && hasLocalScope
    }

    /// Determines if a string represents a query for a scheduled support group meeting.
    /// - Returns: `true` if the string contains a recovery program marker, includes "meeting", and specifies a day or time, `false` otherwise.
    static func isScheduledSupportGroupMeetingLookup(_ value: String) -> Bool {
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return false }

        let recoveryProgramMarkers = [
            "alcoholics anonymous", "alcoholic anonymous", "aa meeting", "a.a. meeting",
            "narcotics anonymous", "na meeting", "n.a. meeting", "smart recovery",
            "recovery meeting", "support group meeting"
        ]
        guard recoveryProgramMarkers.contains(where: { normalized.contains($0) }) else {
            return false
        }

        return normalized.contains("meeting") && normalized.range(
            of: #"\b(today|tonight|tomorrow|this week|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"#,
            options: .regularExpression
        ) != nil
    }

    /// Validates a Maps destination string to prevent prompt injection and description leak.
    /// - Parameter value: The destination string to validate.
    /// - Returns: The trimmed destination if valid, `nil` if empty or containing suspicious markers.
    static func sanitizedMapsDestination(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let lowered = trimmed.lowercased()
        let descriptionLeakMarkers = [
            "args:", "[runtime policy]", "[available local tools]",
            "use only for navigation/route requests", "find nearby/local places",
            "- maps.search", "- maps.directions", "<!-- lumen_grounding_v1 -->"
        ]
        if descriptionLeakMarkers.contains(where: { lowered.contains($0) }) {
            return nil
        }

        return trimmed
    }
}
