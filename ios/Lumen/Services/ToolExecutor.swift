import Foundation

nonisolated enum ToolExecutionApproval: Sendable {
    case autonomous
    case pending
    case userApproved
}

@MainActor
final class ToolExecutor {
    static let shared = ToolExecutor()

    private init() {}

    func execute(
        _ toolID: String,
        arguments: AgentJSONArguments,
        approval: ToolExecutionApproval = .autonomous
    ) async -> String {
        let id = ToolRouteGuard.canonicalToolID(toolID)
        let stringArguments = ToolRouteGuard.normalizedArguments(for: id, rawToolID: toolID, arguments: arguments.stringCoerced)

        guard ToolRouteGuard.canExecuteTool(id, arguments: stringArguments, approval: approval) else {
            return ToolRouteGuard.approvalRequiredMessage(for: id)
        }
        if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: id, arguments: stringArguments) {
            return permissionFailure
        }

        switch id {
        case "calendar.create":
            return await CalendarTools.createEvent(
                title: stringArguments["title"] ?? "New Event",
                startsInMinutes: Int(stringArguments["startsInMinutes"] ?? "60") ?? 60
            )
        case "calendar.list":
            return await CalendarTools.listEvents()
        case "reminders.create":
            return await CalendarTools.createReminder(title: stringArguments["title"] ?? "Reminder")
        case "reminders.list":
            return await CalendarTools.listReminders()
        case "contacts.search":
            return await ContactsTools.searchContacts(query: stringArguments["query"] ?? "")
        case "messages.draft":
            return await ContactsTools.composeMessage(arguments: stringArguments)
        case "mail.draft":
            return await ContactsTools.composeMail(arguments: stringArguments)
        case "phone.call":
            return await ContactsTools.call(number: stringArguments["number"] ?? "")
        case "location.current":
            return await LocationTools.currentLocation()
        case "weather":
            return await WeatherTools.currentWeather(location: stringArguments["location"] ?? stringArguments["city"] ?? stringArguments["query"])
        case "maps.directions":
            return LocationTools.openDirections(destination: stringArguments["destination"] ?? stringArguments["query"] ?? "")
        case "maps.search":
            let query = stringArguments["query"] ?? stringArguments["location"] ?? stringArguments["destination"] ?? ""
            if ToolRouteGuard.shouldUseWebSearchInsteadOfNearbySearch(query: query) {
                return await WebTools.webSearch(query: query)
            }
            return await LocationTools.searchNearby(query: query)
        case "photos.search":
            return await PhotosTools.searchPhotos(query: stringArguments["query"] ?? "")
        case "camera.capture":
            return await PhotosTools.captureImage()
        case "health.summary":
            return await HealthTools.healthSummary()
        case "motion.activity":
            return await MotionTools.shared.motionActivity()
        case "web.search":
            return await WebTools.webSearch(query: stringArguments["query"] ?? "")
        case "web.fetch":
            return await WebTools.webFetch(url: stringArguments["url"] ?? "")
        case "files.read":
            return await FilesTools.readImportedFile(name: stringArguments["name"] ?? "")
        case "memory.save":
            return await MemoryTools.save(content: stringArguments["content"] ?? "", kind: stringArguments["kind"] ?? "fact")
        case "memory.recall":
            return await MemoryTools.recall(query: stringArguments["query"] ?? "")
        case "rag.search":
            return await MemoryTools.ragSearch(query: stringArguments["query"] ?? "", limit: Int(stringArguments["limit"] ?? "5") ?? 5)
        case "rag.index_files":
            return await MemoryTools.ragIndexFiles()
        case "rag.index_photos":
            return await MemoryTools.ragIndexPhotos(months: Int(stringArguments["months"] ?? "6") ?? 6)
        case "trigger.create":
            return await TriggerTools.create(args: stringArguments)
        case "trigger.list":
            return await TriggerTools.list()
        case "trigger.cancel":
            return await TriggerTools.cancel(title: stringArguments["title"] ?? stringArguments["id"] ?? "")
        case "alarm.authorization_status":
            return await AlarmTools.authorizationStatus()
        case "alarm.request_authorization":
            return await AlarmTools.requestAuthorization()
        case "alarm.schedule":
            return await AlarmTools.schedule(args: stringArguments)
        case "alarm.countdown":
            return await AlarmTools.countdown(args: stringArguments)
        case "alarm.list":
            return await AlarmTools.list()
        case "alarm.pause":
            return await AlarmTools.pause(id: stringArguments["id"] ?? "")
        case "alarm.resume":
            return await AlarmTools.resume(id: stringArguments["id"] ?? "")
        case "alarm.stop":
            return await AlarmTools.stop(id: stringArguments["id"] ?? "")
        case "alarm.snooze":
            return await AlarmTools.snooze(id: stringArguments["id"] ?? "")
        case "alarm.cancel":
            return await AlarmTools.cancel(id: stringArguments["id"] ?? stringArguments["title"] ?? "")
        case "outlook.status":
            return await OutlookTools.status()
        case "outlook.folders.list":
            return await OutlookTools.listFolders(args: stringArguments)
        case "outlook.messages.list":
            return await OutlookTools.listMessages(args: stringArguments)
        case "outlook.messages.search":
            return await OutlookTools.searchMessages(args: stringArguments)
        case "outlook.message.read":
            return await OutlookTools.readMessage(args: stringArguments)
        case "outlook.attachments.list":
            return await OutlookTools.listAttachments(args: stringArguments)
        case "outlook.draft.create":
            return await OutlookTools.createDraft(args: stringArguments)
        case "outlook.mail.send":
            return await OutlookTools.sendMail(args: stringArguments)
        case "outlook.message.mark_read":
            return await OutlookTools.markRead(args: stringArguments, isRead: true)
        case "outlook.message.mark_unread":
            return await OutlookTools.markRead(args: stringArguments, isRead: false)
        case "outlook.message.move":
            return await OutlookTools.moveMessage(args: stringArguments)
        case "outlook.message.archive":
            var args = stringArguments
            args["destination"] = "archive"
            return await OutlookTools.moveMessage(args: args)
        case "outlook.message.delete":
            return await OutlookTools.deleteMessage(args: stringArguments)
        case "outlook.message.reply":
            return await OutlookTools.reply(args: stringArguments, replyAll: false)
        case "outlook.message.reply_all":
            return await OutlookTools.reply(args: stringArguments, replyAll: true)
        case "outlook.message.forward":
            return await OutlookTools.forward(args: stringArguments)
        default:
            return "Unknown tool: \(toolID). Available tools include weather, web.search, maps.search, maps.directions, location.current, outlook.messages.list, outlook.messages.search, outlook.message.read, outlook.draft.create, outlook.mail.send, outlook.message.reply, outlook.message.forward, outlook.message.archive."
        }
    }

    func execute(
        _ toolID: String,
        arguments: [String: String],
        approval: ToolExecutionApproval = .autonomous
    ) async -> String {
        let typedArguments = AgentJSONArguments(stringDictionary: arguments)
        return await execute(toolID, arguments: typedArguments, approval: approval)
    }

}

nonisolated enum ToolRouteGuard {
    @MainActor
    static func ensurePermissionIfNeeded(for canonicalToolID: String, arguments: [String: String]) async -> String? {
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
        switch initial {
        case .granted:
            return nil
        case .notDetermined:
            await permissions.request(permissionKind)
            let updated = permissions.state(permissionKind)
            if updated == .granted || updated == .limited {
                return nil
            }
            return permissionUnavailableMessage(for: permissionKind)
        case .limited:
            return nil
        case .denied, .restricted, .unavailable:
            return permissionUnavailableMessage(for: permissionKind)
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

    static func shouldUseWebSearchInsteadOfNearbySearch(query: String) -> Bool {
        let normalized = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return false }

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
}
