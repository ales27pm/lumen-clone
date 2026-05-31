import Foundation
import SwiftData

@Model
final class Trigger {
    var id: UUID = UUID()
    var title: String = ""
    var prompt: String = ""
    var scheduleType: String = "once"
    var fireDate: Date?
    var timeOfDayMinutes: Int?
    var intervalSeconds: TimeInterval?
    var weekdayMask: Int = 0
    var beforeNextEventMinutes: Int?
    var isPaused: Bool = false
    var createdAt: Date = Date()
    var lastRunAt: Date?
    var lastResult: String?
    var nextFireAt: Date?
    var conversationID: UUID?

    init(title: String, prompt: String, scheduleType: TriggerScheduleType, fireDate: Date? = nil, timeOfDayMinutes: Int? = nil, intervalSeconds: TimeInterval? = nil, weekdayMask: Int = 0, beforeNextEventMinutes: Int? = nil) {
        self.title = title
        self.prompt = prompt
        self.scheduleType = scheduleType.rawValue
        self.fireDate = fireDate
        self.timeOfDayMinutes = timeOfDayMinutes
        self.intervalSeconds = intervalSeconds
        self.weekdayMask = weekdayMask
        self.beforeNextEventMinutes = beforeNextEventMinutes
    }

    var kind: TriggerScheduleType { TriggerScheduleType(rawValue: scheduleType) ?? .once }

    func computeNextFire(from now: Date = Date()) -> Date? {
        let cal = Calendar.current
        switch kind {
        case .once:
            return (fireDate ?? now) > now ? fireDate : nil
        case .daily:
            guard let m = timeOfDayMinutes else { return nil }
            var comps = cal.dateComponents([.year, .month, .day], from: now)
            comps.hour = m / 60
            comps.minute = m % 60
            var candidate = cal.date(from: comps) ?? now
            if candidate <= now { candidate = cal.date(byAdding: .day, value: 1, to: candidate) ?? candidate }
            if weekdayMask != 0 {
                for _ in 0..<8 {
                    let wd = cal.component(.weekday, from: candidate)
                    if (weekdayMask & (1 << (wd - 1))) != 0 { return candidate }
                    candidate = cal.date(byAdding: .day, value: 1, to: candidate) ?? candidate
                }
            }
            return candidate
        case .interval:
            guard let s = intervalSeconds, s > 60 else { return nil }
            let base = lastRunAt ?? createdAt
            var t = base.addingTimeInterval(s)
            if t < now { t = now.addingTimeInterval(min(s, 3600)) }
            return t
        case .beforeNextEvent:
            return nil
        }
    }
}

enum TriggerScheduleType: String, Codable, CaseIterable, Sendable {
    case once, daily, interval, beforeNextEvent

    var label: String {
        switch self {
        case .once: "One-time"
        case .daily: "Daily"
        case .interval: "Interval"
        case .beforeNextEvent: "Before next event"
        }
    }

    var icon: String {
        switch self {
        case .once: "clock"
        case .daily: "sun.max"
        case .interval: "arrow.triangle.2.circlepath"
        case .beforeNextEvent: "calendar.badge.clock"
        }
    }
}
