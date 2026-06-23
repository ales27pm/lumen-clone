import Foundation
import EventKit

@MainActor
enum CalendarTools {
    private enum CalendarOperation {
        case read
        case write
    }

    struct CalendarEventRecord: Equatable, Sendable {
        let title: String
        let start: Date
        let end: Date
        let calendarTitle: String
        let location: String?
    }

    struct CalendarListQuery: Equatable, Sendable {
        let start: Date
        let end: Date
        let limit: Int
        let titleFilter: String?
    }

    struct CalendarToolResponse: Equatable, Sendable {
        let status: ToolResultStatus
        let displayText: String
        let modelText: String
        let structuredPayload: [String: String]?
        let metricsSummary: String
        let errorCode: String?
    }

    protocol EventProvider {
        func authorizationState() -> AssistantPermissionState
        func events(start: Date, end: Date, limit: Int, titleFilter: String?) async throws -> [CalendarEventRecord]
        func createEvent(title: String, startsInMinutes: Int, now: Date) async throws -> CalendarEventRecord
    }

    struct EventKitProvider: EventProvider {
        func authorizationState() -> AssistantPermissionState {
            guard Self.hasCalendarUsageDescription else { return .unavailable }
            switch EKEventStore.authorizationStatus(for: .event) {
            case .fullAccess: return .granted
            case .writeOnly: return .limited
            case .denied: return .denied
            case .restricted: return .restricted
            case .notDetermined: return .notDetermined
            @unknown default: return .unavailable
            }
        }

        func events(start: Date, end: Date, limit: Int, titleFilter: String?) async throws -> [CalendarEventRecord] {
            let store = EKEventStore()
            let predicate = store.predicateForEvents(withStart: start, end: end, calendars: nil)
            return store.events(matching: predicate)
                .filter { event in
                    guard let titleFilter, !titleFilter.isEmpty else { return true }
                    return event.calendar.title.localizedCaseInsensitiveContains(titleFilter)
                }
                .sorted { $0.startDate < $1.startDate }
                .prefix(limit)
                .map {
                    CalendarEventRecord(
                        title: $0.title ?? "Untitled",
                        start: $0.startDate,
                        end: $0.endDate,
                        calendarTitle: $0.calendar.title,
                        location: $0.location
                    )
                }
        }

        func createEvent(title: String, startsInMinutes: Int, now: Date) async throws -> CalendarEventRecord {
            let store = EKEventStore()
            guard let calendar = store.defaultCalendarForNewEvents else {
                throw CalendarProviderError.providerUnavailable
            }
            let event = EKEvent(eventStore: store)
            event.title = title
            event.startDate = now.addingTimeInterval(TimeInterval(startsInMinutes * 60))
            event.endDate = event.startDate.addingTimeInterval(3600)
            event.calendar = calendar
            try store.save(event, span: .thisEvent)
            return CalendarEventRecord(
                title: event.title ?? title,
                start: event.startDate,
                end: event.endDate,
                calendarTitle: calendar.title,
                location: event.location
            )
        }

        nonisolated static func hasCalendarUsageDescription(infoDictionary: [String: Any]) -> Bool {
            infoDictionary["NSCalendarsFullAccessUsageDescription"] != nil
                || infoDictionary["NSCalendarsWriteOnlyAccessUsageDescription"] != nil
                || infoDictionary["NSCalendarsUsageDescription"] != nil
        }

        private static var hasCalendarUsageDescription: Bool {
            Self.hasCalendarUsageDescription(infoDictionary: Bundle.main.infoDictionary ?? [:])
        }
    }

    enum CalendarProviderError: Error {
        case providerUnavailable
    }

    static func createEvent(
        title: String,
        startsInMinutes: Int
    ) async -> String {
        await createEvent(title: title, startsInMinutes: startsInMinutes, provider: EventKitProvider())
    }

    static func createEvent(
        title: String,
        startsInMinutes: Int,
        provider: EventProvider,
        now: Date = Date()
    ) async -> String {
        await createEventResult(title: title, startsInMinutes: startsInMinutes, provider: provider, now: now).displayText
    }

    static func createEventResult(title: String, startsInMinutes: Int) async -> CalendarToolResponse {
        await createEventResult(title: title, startsInMinutes: startsInMinutes, provider: EventKitProvider())
    }

    static func createEventResult(
        title: String,
        startsInMinutes: Int,
        provider: EventProvider,
        now: Date = Date()
    ) async -> CalendarToolResponse {
        let trimmedTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedTitle.isEmpty, startsInMinutes >= 0 else {
            return invalidArgumentsResponse()
        }
        let authorizationState = provider.authorizationState()
        guard let unavailable = unavailableResponse(for: authorizationState, operation: .write) else {
            do {
                let event = try await provider.createEvent(title: trimmedTitle, startsInMinutes: startsInMinutes, now: now)
                let text = "Created event \"\(event.title)\" starting \(event.start.formatted(date: .abbreviated, time: .shortened))."
                let availability = authorizationState == .limited ? "limited" : "granted"
                return CalendarToolResponse(status: .success, displayText: text, modelText: text, structuredPayload: ["availability": availability, "created": "true"], metricsSummary: "calendar_create_success_\(availability)", errorCode: nil)
            } catch {
                return providerFailureResponse(action: "create")
            }
        }
        return unavailable
    }

    static func listEvents() async -> String {
        await listEventsResult(provider: EventKitProvider()).displayText
    }

    static func listEventsResult(arguments: [String: String] = [:], now: Date = Date()) async -> CalendarToolResponse {
        await listEventsResult(arguments: arguments, provider: EventKitProvider(), now: now)
    }

    static func listEventsResult(
        arguments: [String: String] = [:],
        provider: EventProvider,
        now: Date = Date()
    ) async -> CalendarToolResponse {
        let query: CalendarListQuery
        do {
            query = try parseListArguments(arguments, now: now)
        } catch {
            return invalidArgumentsResponse()
        }

        guard let unavailable = unavailableResponse(for: provider.authorizationState(), operation: .read) else {
            do {
                let events = try await provider.events(start: query.start, end: query.end, limit: query.limit, titleFilter: query.titleFilter)
                if events.isEmpty {
                    return CalendarToolResponse(
                        status: .success,
                        displayText: "No calendar events found in that range.",
                        modelText: "No calendar events found in that range.",
                        structuredPayload: ["availability": "granted", "count": "0"],
                        metricsSummary: "calendar_list_success_empty_granted",
                        errorCode: nil
                    )
                }
                let text = events.map {
                    "- \($0.title) | \(isoFormatter.string(from: $0.start)) to \(isoFormatter.string(from: $0.end)) | \($0.calendarTitle)"
                }.joined(separator: "\n")
                return CalendarToolResponse(
                    status: .success,
                    displayText: text,
                    modelText: text,
                    structuredPayload: ["availability": "granted", "count": "\(events.count)"],
                    metricsSummary: "calendar_list_success_granted",
                    errorCode: nil
                )
            } catch {
                return providerFailureResponse(action: "list")
            }
        }
        return unavailable
    }

    nonisolated static func parseListArguments(_ arguments: [String: String], now: Date = Date()) throws -> CalendarListQuery {
        let argumentDateFormatter = ISO8601DateFormatter()
        let limit = Int(arguments["limit"] ?? "10") ?? 10
        guard (1...20).contains(limit) else { throw ToolExecutionError.invalidArguments("limit") }

        let start: Date
        if let rawStart = arguments["startDate"] {
            guard let parsed = argumentDateFormatter.date(from: rawStart) else { throw ToolExecutionError.invalidArguments("startDate") }
            start = parsed
        } else {
            start = now
        }

        let end: Date
        if let rawEnd = arguments["endDate"] {
            guard let parsed = argumentDateFormatter.date(from: rawEnd) else { throw ToolExecutionError.invalidArguments("endDate") }
            end = parsed
        } else {
            end = now.addingTimeInterval(7 * 24 * 3600)
        }

        guard end > start else { throw ToolExecutionError.invalidArguments("endDate") }
        guard end.timeIntervalSince(start) <= 31 * 24 * 3600 else { throw ToolExecutionError.invalidArguments("date range max 31d") }

        let titleFilter = arguments["calendarTitleFilter"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let titleFilter, titleFilter.count > 120 {
            throw ToolExecutionError.invalidArguments("calendarTitleFilter")
        }
        return CalendarListQuery(start: start, end: end, limit: limit, titleFilter: titleFilter?.isEmpty == true ? nil : titleFilter)
    }

    private static let isoFormatter = ISO8601DateFormatter()

    private static func unavailableResponse(for state: AssistantPermissionState, operation: CalendarOperation) -> CalendarToolResponse? {
        switch state {
        case .granted:
            return nil
        case .notDetermined:
            return CalendarToolResponse(
                status: .denied,
                displayText: "Calendar access has not been granted yet. Open Lumen permissions and allow calendar access, then try again.",
                modelText: "Calendar permission is not determined.",
                structuredPayload: ["availability": "notDetermined"],
                metricsSummary: "calendar_permission_not_determined",
                errorCode: "calendar_permission_not_determined"
            )
        case .denied:
            return CalendarToolResponse(
                status: .denied,
                displayText: "Calendar access is denied. Enable calendar access in iOS Settings to use this calendar tool.",
                modelText: "Calendar permission is denied.",
                structuredPayload: ["availability": "denied"],
                metricsSummary: "calendar_permission_denied",
                errorCode: "calendar_permission_denied"
            )
        case .restricted:
            return CalendarToolResponse(
                status: .denied,
                displayText: "Calendar access is restricted on this device, so I cannot read calendar events.",
                modelText: "Calendar permission is restricted.",
                structuredPayload: ["availability": "restricted"],
                metricsSummary: "calendar_permission_restricted",
                errorCode: "calendar_permission_restricted"
            )
        case .limited:
            if operation == .write {
                return nil
            }
            return CalendarToolResponse(
                status: .denied,
                displayText: "Calendar access is write-only, so I cannot read calendar events.",
                modelText: "Calendar permission is limited.",
                structuredPayload: ["availability": "limited"],
                metricsSummary: "calendar_permission_limited",
                errorCode: "calendar_permission_limited"
            )
        case .unavailable, .unknown:
            return CalendarToolResponse(
                status: .unavailable,
                displayText: "Calendar events are unavailable on this device or build.",
                modelText: "Calendar provider is unavailable.",
                structuredPayload: ["availability": state.rawValue],
                metricsSummary: "calendar_provider_unavailable",
                errorCode: "calendar_provider_unavailable"
            )
        }
    }

    private static func invalidArgumentsResponse() -> CalendarToolResponse {
        CalendarToolResponse(
            status: .failed,
            displayText: "The calendar request is missing valid date or limit arguments.",
            modelText: "Calendar arguments are invalid.",
            structuredPayload: ["availability": "unknown", "failure": "invalidArguments"],
            metricsSummary: "calendar_invalid_arguments",
            errorCode: "calendar_invalid_arguments"
        )
    }

    private static func providerFailureResponse(action: String) -> CalendarToolResponse {
        CalendarToolResponse(
            status: .failed,
            displayText: "I couldn't \(action) calendar events right now. Try again later.",
            modelText: "Calendar provider failure.",
            structuredPayload: ["availability": "granted", "failure": "providerFailure"],
            metricsSummary: "calendar_provider_failure",
            errorCode: "calendar_provider_failure"
        )
    }

    static func createReminder(title: String) async -> String {
        let store = EKEventStore()
        do {
            let granted = try await store.requestFullAccessToReminders()
            guard granted else { return "Reminders access was denied." }
            let reminder = EKReminder(eventStore: store)
            reminder.title = title
            reminder.calendar = store.defaultCalendarForNewReminders()
            try store.save(reminder, commit: true)
            return "Added reminder: \"\(title)\"."
        } catch {
            return "Couldn't add reminder: \(error.localizedDescription)"
        }
    }

    static func listReminders() async -> String {
        let store = EKEventStore()
        do {
            let granted = try await store.requestFullAccessToReminders()
            guard granted else { return "Reminders access was denied." }
            let predicate = store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: nil, calendars: nil)
            return await withCheckedContinuation { cont in
                store.fetchReminders(matching: predicate) { reminders in
                    let items = (reminders ?? []).prefix(5)
                    if items.isEmpty { cont.resume(returning: "No pending reminders.") }
                    else { cont.resume(returning: items.map { "• \($0.title ?? "Untitled")" }.joined(separator: "\n")) }
                }
            }
        } catch {
            return "Couldn't load reminders: \(error.localizedDescription)"
        }
    }
}
