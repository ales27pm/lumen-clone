import Foundation
import EventKit

@MainActor
enum CalendarTools {
    static func createEvent(title: String, startsInMinutes: Int) async -> String {
        let store = EKEventStore()
        do {
            let granted = try await store.requestFullAccessToEvents()
            guard granted else { return "Calendar access was denied." }
            let event = EKEvent(eventStore: store)
            event.title = title
            event.startDate = Date().addingTimeInterval(TimeInterval(startsInMinutes * 60))
            event.endDate = event.startDate.addingTimeInterval(3600)
            event.calendar = store.defaultCalendarForNewEvents
            try store.save(event, span: .thisEvent)
            return "Created event \"\(title)\" starting \(event.startDate.formatted(date: .abbreviated, time: .shortened))."
        } catch {
            return "Couldn't create event: \(error.localizedDescription)"
        }
    }

    static func listEvents() async -> String {
        let store = EKEventStore()
        do {
            let granted = try await store.requestFullAccessToEvents()
            guard granted else { return "Calendar access was denied." }
            let predicate = store.predicateForEvents(withStart: Date(), end: Date().addingTimeInterval(86400 * 7), calendars: nil)
            let events = store.events(matching: predicate).prefix(5)
            if events.isEmpty { return "No events in the next 7 days." }
            return events.map { "• \($0.title ?? "Untitled") — \($0.startDate.formatted(date: .abbreviated, time: .shortened))" }.joined(separator: "\n")
        } catch {
            return "Couldn't load events: \(error.localizedDescription)"
        }
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
