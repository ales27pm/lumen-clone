import Foundation
import SwiftData

@MainActor
enum TriggerTools {
    static func create(args: [String: String]) async -> String {
        guard let container = SharedContainer.shared else { return "Store unavailable." }
        let ctx = ModelContext(container)
        let title = args["title"] ?? "Scheduled run"
        let prompt = args["prompt"] ?? title
        let schedule = normalizedScheduleType(from: args["schedule"])
        let trigger: Trigger
        switch schedule {
        case .once:
            let minutes = Int(args["inMinutes"] ?? "60") ?? 60
            let fire = Date().addingTimeInterval(TimeInterval(minutes * 60))
            trigger = Trigger(title: title, prompt: prompt, scheduleType: .once, fireDate: fire)
        case .daily:
            let hhmm = args["atTime"] ?? "09:00"
            let parts = hhmm.split(separator: ":").compactMap { Int($0) }
            let mins = (parts.first ?? 9) * 60 + (parts.count > 1 ? parts[1] : 0)
            trigger = Trigger(title: title, prompt: prompt, scheduleType: .daily, timeOfDayMinutes: mins)
        case .interval:
            let seconds = TimeInterval(Int(args["intervalSeconds"] ?? "3600") ?? 3600)
            trigger = Trigger(title: title, prompt: prompt, scheduleType: .interval, intervalSeconds: seconds)
        case .beforeNextEvent:
            let before = Int(args["beforeMinutes"] ?? "15") ?? 15
            trigger = Trigger(title: title, prompt: prompt, scheduleType: .beforeNextEvent, beforeNextEventMinutes: before)
        }
        trigger.nextFireAt = trigger.computeNextFire()
        ctx.insert(trigger)
        do {
            try ctx.save()
        } catch {
            return triggerSaveFailureMessage(operation: "create", error: error)
        }
        await TriggerScheduler.shared.requestPermission()
        TriggerScheduler.shared.scheduleBackgroundRefresh()
        let when = trigger.nextFireAt?.formatted(date: .abbreviated, time: .shortened) ?? "background"
        return "Scheduled \"\(title)\" (\(schedule.label)) — next run: \(when)."
    }

    private static func normalizedScheduleType(from raw: String?) -> TriggerScheduleType {
        switch raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "relative", "once", "one-time", "one time":
            return .once
        case "absolute", "daily":
            return .daily
        case "interval":
            return .interval
        case "beforenextevent", "before_next_event", "before-next-event":
            return .beforeNextEvent
        default:
            return .once
        }
    }

    static func list() async -> String {
        guard let container = SharedContainer.shared else { return "Store unavailable." }
        let ctx = ModelContext(container)
        let all: [Trigger]
        do {
            all = try ctx.fetch(FetchDescriptor<Trigger>())
        } catch {
            return triggerFetchFailureMessage(error: error)
        }
        if all.isEmpty { return "No scheduled runs." }
        return all.map { t in
            let next = t.nextFireAt?.formatted(date: .abbreviated, time: .shortened) ?? (t.isPaused ? "paused" : "—")
            return "• \(t.title) — \(t.kind.label) — next: \(next) — UUID: \(t.id.uuidString)"
        }.joined(separator: "\n")
    }

    static func cancel(title: String) async -> String {
        let token = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            return "Missing trigger id or exact title."
        }
        guard let container = SharedContainer.shared else { return "Store unavailable." }
        let ctx = ModelContext(container)
        let all: [Trigger]
        do {
            all = try ctx.fetch(FetchDescriptor<Trigger>())
        } catch {
            return triggerFetchFailureMessage(error: error)
        }
        let matches: [Trigger]
        if UUID(uuidString: token) != nil {
            matches = all.filter { $0.id.uuidString.caseInsensitiveCompare(token) == .orderedSame }
        } else {
            matches = all.filter { $0.title.caseInsensitiveCompare(token) == .orderedSame }
        }
        guard matches.count == 1, let m = matches.first else {
            if matches.count > 1 {
                return "Multiple triggers match \"\(token)\". Use the UUID from the list to disambiguate."
            }
            return "No trigger matching \"\(token)\"."
        }
        ctx.delete(m)
        do {
            try ctx.save()
        } catch {
            return triggerSaveFailureMessage(operation: "cancel", error: error)
        }
        return "Cancelled \"\(m.title)\"."
    }

    static func triggerFetchFailureMessage(error: Error) -> String {
        "Trigger fetch failed (\(RuntimeMetricErrorSanitizer.code(for: error)))."
    }

    static func triggerSaveFailureMessage(operation: String, error: Error) -> String {
        "Trigger \(operation) failed: persistence save failed (\(RuntimeMetricErrorSanitizer.code(for: error)))."
    }
}
