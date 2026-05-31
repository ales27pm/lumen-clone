import Foundation
import SwiftData

@MainActor
enum TriggerTools {
    static func create(args: [String: String]) async -> String {
        guard let container = SharedContainer.shared else { return "Store unavailable." }
        let ctx = ModelContext(container)
        let title = args["title"] ?? "Scheduled run"
        let prompt = args["prompt"] ?? title
        let schedule = TriggerScheduleType(rawValue: args["schedule"] ?? "once") ?? .once
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
        try? ctx.save()
        await TriggerScheduler.shared.requestPermission()
        TriggerScheduler.shared.scheduleBackgroundRefresh()
        let when = trigger.nextFireAt?.formatted(date: .abbreviated, time: .shortened) ?? "background"
        return "Scheduled \"\(title)\" (\(schedule.label)) — next run: \(when)."
    }

    static func list() async -> String {
        guard let container = SharedContainer.shared else { return "Store unavailable." }
        let ctx = ModelContext(container)
        let all = (try? ctx.fetch(FetchDescriptor<Trigger>())) ?? []
        if all.isEmpty { return "No scheduled runs." }
        return all.map { t in
            let next = t.nextFireAt?.formatted(date: .abbreviated, time: .shortened) ?? (t.isPaused ? "paused" : "—")
            return "• \(t.title) — \(t.kind.label) — next: \(next)"
        }.joined(separator: "\n")
    }

    static func cancel(title: String) async -> String {
        guard let container = SharedContainer.shared else { return "Store unavailable." }
        let ctx = ModelContext(container)
        let all = (try? ctx.fetch(FetchDescriptor<Trigger>())) ?? []
        let match = all.first { $0.title.localizedCaseInsensitiveContains(title) || $0.id.uuidString == title }
        guard let m = match else { return "No trigger matching \"\(title)\"." }
        ctx.delete(m)
        try? ctx.save()
        return "Cancelled \"\(m.title)\"."
    }
}
