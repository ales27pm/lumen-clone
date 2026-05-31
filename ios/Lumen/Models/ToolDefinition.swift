import Foundation
import SwiftUI

nonisolated struct ToolDefinition: Identifiable, Hashable, Sendable {
    let id: String
    let name: String
    let category: ToolCategory
    let description: String
    let icon: String
    let tint: String
    let requiresApproval: Bool
    let permissionKey: String?

    var color: Color {
        switch tint {
        case "blue": .blue
        case "green": .green
        case "orange": .orange
        case "pink": .pink
        case "purple": .purple
        case "red": .red
        case "yellow": .yellow
        case "mint": .mint
        case "indigo": .indigo
        case "teal": .teal
        default: .cyan
        }
    }

    var permissionKind: PermissionKind? {
        if let key = permissionKey, let kind = PermissionKind(usageDescriptionKey: key) {
            return kind
        }

        switch id {
        case "trigger.create", "trigger.list", "trigger.cancel":
            return .notifications
        default:
            return nil
        }
    }
}

nonisolated enum ToolCategory: String, CaseIterable, Sendable {
    case productivity = "Productivity"
    case communication = "Communication"
    case location = "Location"
    case media = "Media"
    case health = "Health & Motion"
    case knowledge = "Knowledge"
}

nonisolated enum ToolRegistry {
    static let all: [ToolDefinition] = [
        ToolDefinition(id: "calendar.create", name: "Create Event", category: .productivity, description: "Add an event to your calendar. Args: title, startsInMinutes.", icon: "calendar.badge.plus", tint: "red", requiresApproval: true, permissionKey: "NSCalendarsFullAccessUsageDescription"),
        ToolDefinition(id: "calendar.list", name: "List Events", category: .productivity, description: "Read upcoming calendar events. Args: none.", icon: "calendar", tint: "red", requiresApproval: false, permissionKey: "NSCalendarsFullAccessUsageDescription"),
        ToolDefinition(id: "reminders.create", name: "Add Reminder", category: .productivity, description: "Create a new reminder. Args: title.", icon: "checklist", tint: "orange", requiresApproval: true, permissionKey: "NSRemindersFullAccessUsageDescription"),
        ToolDefinition(id: "reminders.list", name: "List Reminders", category: .productivity, description: "Read pending reminders. Args: none.", icon: "list.bullet.rectangle", tint: "orange", requiresApproval: false, permissionKey: "NSRemindersFullAccessUsageDescription"),
        ToolDefinition(id: "contacts.search", name: "Search Contacts", category: .communication, description: "Find a contact by name. Args: query. Only use for the user's address book, not web people search.", icon: "person.crop.circle.fill", tint: "blue", requiresApproval: false, permissionKey: "NSContactsUsageDescription"),
        ToolDefinition(id: "messages.draft", name: "Draft Message", category: .communication, description: "Compose an iMessage/SMS draft. Args: to or recipient or number, body or message or text.", icon: "message.fill", tint: "green", requiresApproval: true, permissionKey: nil),
        ToolDefinition(id: "mail.draft", name: "Draft Email", category: .communication, description: "Compose an email draft using the system mail composer. Args: to or recipient or email, subject, body or message or text.", icon: "envelope.fill", tint: "indigo", requiresApproval: true, permissionKey: nil),

        ToolDefinition(id: "outlook.status", name: "Outlook Status", category: .communication, description: "Check whether the Microsoft Graph Outlook account is signed in. Args: none.", icon: "envelope.badge.shield.half.filled", tint: "indigo", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "outlook.folders.list", name: "List Outlook Folders", category: .communication, description: "List Outlook/Hotmail mail folders with unread and total counts. Args: optional includeHidden true/false.", icon: "folder.fill", tint: "indigo", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "outlook.messages.list", name: "List Outlook Messages", category: .communication, description: "List recent Outlook messages. Args: optional folder or folderId, limit, unreadOnly.", icon: "tray.full.fill", tint: "indigo", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "outlook.messages.search", name: "Search Outlook Messages", category: .communication, description: "Search Outlook mail with Microsoft Graph. Args: query, optional folder/folderId, limit.", icon: "envelope.open.fill", tint: "indigo", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "outlook.message.read", name: "Read Outlook Message", category: .communication, description: "Read one Outlook message body by id. Args: messageId or id.", icon: "doc.text.magnifyingglass", tint: "indigo", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "outlook.attachments.list", name: "List Outlook Attachments", category: .communication, description: "List attachment metadata for one Outlook message. Args: messageId or id.", icon: "paperclip", tint: "indigo", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "outlook.draft.create", name: "Create Outlook Draft", category: .communication, description: "Create a saved Outlook draft through Microsoft Graph. Args: to, subject, body.", icon: "square.and.pencil", tint: "indigo", requiresApproval: true, permissionKey: nil),
        ToolDefinition(id: "outlook.mail.send", name: "Send Outlook Email", category: .communication, description: "Send an Outlook email through Microsoft Graph. Args: to, subject, body. Requires explicit approval.", icon: "paperplane.fill", tint: "indigo", requiresApproval: true, permissionKey: nil),
        ToolDefinition(id: "outlook.message.mark_read", name: "Mark Outlook Read", category: .communication, description: "Mark an Outlook message as read. Args: messageId or id. Requires explicit approval.", icon: "envelope.open", tint: "indigo", requiresApproval: true, permissionKey: nil),
        ToolDefinition(id: "outlook.message.mark_unread", name: "Mark Outlook Unread", category: .communication, description: "Mark an Outlook message as unread. Args: messageId or id. Requires explicit approval.", icon: "envelope.badge", tint: "indigo", requiresApproval: true, permissionKey: nil),
        ToolDefinition(id: "outlook.message.move", name: "Move Outlook Message", category: .communication, description: "Move an Outlook message to a folder. Args: messageId or id, destination or destinationId. Requires explicit approval.", icon: "folder.badge.gearshape", tint: "indigo", requiresApproval: true, permissionKey: nil),
        ToolDefinition(id: "outlook.message.archive", name: "Archive Outlook Message", category: .communication, description: "Move an Outlook message to Archive. Args: messageId or id. Requires explicit approval.", icon: "archivebox.fill", tint: "indigo", requiresApproval: true, permissionKey: nil),
        ToolDefinition(id: "outlook.message.delete", name: "Delete Outlook Message", category: .communication, description: "Delete an Outlook message. Args: messageId or id. Requires explicit approval.", icon: "trash.fill", tint: "red", requiresApproval: true, permissionKey: nil),
        ToolDefinition(id: "outlook.message.reply", name: "Reply Outlook Message", category: .communication, description: "Reply to an Outlook message. Args: messageId or id, body/comment. Requires explicit approval.", icon: "arrowshape.turn.up.left.fill", tint: "indigo", requiresApproval: true, permissionKey: nil),
        ToolDefinition(id: "outlook.message.reply_all", name: "Reply All Outlook Message", category: .communication, description: "Reply-all to an Outlook message. Args: messageId or id, body/comment. Requires explicit approval.", icon: "arrowshape.turn.up.left.2.fill", tint: "indigo", requiresApproval: true, permissionKey: nil),
        ToolDefinition(id: "outlook.message.forward", name: "Forward Outlook Message", category: .communication, description: "Forward an Outlook message. Args: messageId or id, to, optional body/comment. Requires explicit approval.", icon: "arrowshape.turn.up.forward.fill", tint: "indigo", requiresApproval: true, permissionKey: nil),

        ToolDefinition(id: "phone.call", name: "Start Call", category: .communication, description: "Open the phone dialer for a number. Args: number. Never use for general lookup.", icon: "phone.fill", tint: "green", requiresApproval: true, permissionKey: nil),
        ToolDefinition(id: "location.current", name: "Current Location", category: .location, description: "Get the user's current GPS location. Args: none. Use before nearby/local map searches when location context is needed.", icon: "location.fill", tint: "teal", requiresApproval: false, permissionKey: "NSLocationWhenInUseUsageDescription"),
        ToolDefinition(id: "weather", name: "Current Weather", category: .location, description: "Get current weather using GPS or a city. Args: optional location or city. Use when the user asks weather, temperature, rain, snow, wind, forecast now, or conditions.", icon: "cloud.sun.fill", tint: "blue", requiresApproval: false, permissionKey: "NSLocationWhenInUseUsageDescription"),
        ToolDefinition(id: "maps.directions", name: "Get Directions", category: .location, description: "Open Apple Maps directions to a real destination. Args: destination. Use only for navigation/route requests.", icon: "map.fill", tint: "teal", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "maps.search", name: "Search Nearby", category: .location, description: "Find nearby/local places in Apple Maps. Args: query. Use only for local places like coffee near me, pharmacy nearby, closest hardware store, addresses, or directions. Do not use for DIY, tutorials, research, articles, or general web search.", icon: "mappin.and.ellipse", tint: "teal", requiresApproval: false, permissionKey: "NSLocationWhenInUseUsageDescription"),
        ToolDefinition(id: "photos.search", name: "Search Photos", category: .media, description: "Search the user's photo library by date/category terms. Args: query. Not for web image search.", icon: "photo.on.rectangle.angled", tint: "purple", requiresApproval: false, permissionKey: "NSPhotoLibraryUsageDescription"),
        ToolDefinition(id: "camera.capture", name: "Capture Image", category: .media, description: "Take a photo with the device camera. Args: none.", icon: "camera.fill", tint: "pink", requiresApproval: true, permissionKey: "NSCameraUsageDescription"),
        ToolDefinition(id: "health.summary", name: "Health Summary", category: .health, description: "Read steps, sleep, heart rate, energy, and distance. Args: none.", icon: "heart.text.square.fill", tint: "red", requiresApproval: false, permissionKey: "NSHealthShareUsageDescription"),
        ToolDefinition(id: "motion.activity", name: "Motion Activity", category: .health, description: "Detect recent device motion activity such as walking/running. Args: none.", icon: "figure.walk", tint: "mint", requiresApproval: false, permissionKey: "NSMotionUsageDescription"),
        ToolDefinition(id: "web.search", name: "Web Search", category: .knowledge, description: "Search the web for general knowledge, fresh information, tutorials, DIY guides, plans, research, articles, or documentation. Args: query. Use this for `search for ...` unless the user explicitly wants nearby/local places.", icon: "globe", tint: "blue", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "web.fetch", name: "Fetch URL", category: .knowledge, description: "Fetch and read a specific web page. Args: url. Use only when the user gives a URL or a prior web search returns one to inspect.", icon: "link", tint: "blue", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "files.read", name: "Read File", category: .knowledge, description: "Read a previously imported local document by name. Args: name. Do not use for attached files already visible in the current prompt.", icon: "doc.text.fill", tint: "yellow", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "memory.save", name: "Save Memory", category: .knowledge, description: "Store a user fact or preference for future recall. Args: content, kind.", icon: "brain.head.profile", tint: "purple", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "memory.recall", name: "Recall Memory", category: .knowledge, description: "Search stored memories about the user. Args: query. Not for web search.", icon: "sparkle.magnifyingglass", tint: "purple", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "rag.search", name: "Search Personal Data", category: .knowledge, description: "Semantic search across indexed local files, PDFs, notes, and photo metadata. Args: query, optional limit. Not for internet search.", icon: "doc.text.magnifyingglass", tint: "yellow", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "rag.index_files", name: "Reindex Files", category: .knowledge, description: "Rebuild the index for imported files and PDFs. Args: none.", icon: "arrow.triangle.2.circlepath.doc.on.clipboard", tint: "yellow", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "rag.index_photos", name: "Reindex Photos", category: .knowledge, description: "Rebuild the monthly photo metadata index. Args: months.", icon: "photo.stack", tint: "purple", requiresApproval: false, permissionKey: "NSPhotoLibraryUsageDescription"),
        ToolDefinition(id: "trigger.create", name: "Schedule Agent Run", category: .productivity, description: "Schedule a background agent run. Args: title, prompt, schedule, plus inMinutes/atTime/intervalSeconds/beforeMinutes depending on schedule.", icon: "alarm", tint: "orange", requiresApproval: true, permissionKey: nil),
        ToolDefinition(id: "trigger.list", name: "List Triggers", category: .productivity, description: "Show active scheduled agent runs. Args: none.", icon: "list.bullet.clipboard", tint: "orange", requiresApproval: false, permissionKey: nil),
        ToolDefinition(id: "trigger.cancel", name: "Cancel Trigger", category: .productivity, description: "Cancel a scheduled agent run. Args: title or id.", icon: "xmark.circle", tint: "red", requiresApproval: true, permissionKey: nil),
        ToolDefinition(id: "alarm.authorization_status", name: "Alarm Auth Status", category: .productivity, description: "Check AlarmKit authorization state. Args: none.", icon: "checkmark.shield", tint: "orange", requiresApproval: false, permissionKey: "NSAlarmKitUsageDescription"),
        ToolDefinition(id: "alarm.request_authorization", name: "Request Alarm Auth", category: .productivity, description: "Request permission to use AlarmKit alarms. Args: none.", icon: "lock.open.display", tint: "orange", requiresApproval: true, permissionKey: "NSAlarmKitUsageDescription"),
        ToolDefinition(id: "alarm.schedule", name: "Schedule Alarm", category: .productivity, description: "Schedule an AlarmKit alarm. Args: title, inMinutes or timestamp, optional repeats, snoozeMinutes.", icon: "alarm.waves.left.and.right.fill", tint: "orange", requiresApproval: true, permissionKey: "NSAlarmKitUsageDescription"),
        ToolDefinition(id: "alarm.countdown", name: "Start Countdown", category: .productivity, description: "Create a countdown alarm. Args: title, durationSeconds.", icon: "timer", tint: "orange", requiresApproval: true, permissionKey: "NSAlarmKitUsageDescription"),
        ToolDefinition(id: "alarm.list", name: "List Alarms", category: .productivity, description: "List active AlarmKit alarms. Args: none.", icon: "list.bullet", tint: "orange", requiresApproval: false, permissionKey: "NSAlarmKitUsageDescription"),
        ToolDefinition(id: "alarm.pause", name: "Pause Alarm", category: .productivity, description: "Pause an alarm. Args: id UUID.", icon: "pause.circle", tint: "orange", requiresApproval: true, permissionKey: "NSAlarmKitUsageDescription"),
        ToolDefinition(id: "alarm.resume", name: "Resume Alarm", category: .productivity, description: "Resume a paused alarm. Args: id UUID.", icon: "play.circle", tint: "orange", requiresApproval: true, permissionKey: "NSAlarmKitUsageDescription"),
        ToolDefinition(id: "alarm.stop", name: "Stop Alarm", category: .productivity, description: "Stop an alerting alarm. Args: id UUID.", icon: "stop.circle", tint: "red", requiresApproval: true, permissionKey: "NSAlarmKitUsageDescription"),
        ToolDefinition(id: "alarm.snooze", name: "Snooze Alarm", category: .productivity, description: "Snooze an alerting alarm. Args: id UUID.", icon: "moon.zzz", tint: "orange", requiresApproval: true, permissionKey: "NSAlarmKitUsageDescription"),
        ToolDefinition(id: "alarm.cancel", name: "Cancel Alarm", category: .productivity, description: "Cancel a scheduled alarm. Args: id UUID or title fallback.", icon: "alarm", tint: "red", requiresApproval: true, permissionKey: "NSAlarmKitUsageDescription"),
    ]

    static func find(id: String) -> ToolDefinition? {
        let canonical = ToolRouteGuard.canonicalToolID(id)
        return all.first { $0.id == canonical }
    }
}
