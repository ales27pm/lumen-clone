import Foundation
import SwiftData

nonisolated struct TriggerCreateArguments: Equatable, Sendable {
    let title: String
    let prompt: String
    let schedule: TriggerScheduleType
    let inMinutes: Int
    let timeOfDayMinutes: Int
    let intervalSeconds: Int
    let beforeMinutes: Int
}

nonisolated enum TriggerCreateArgumentError: Error, Equatable, Sendable {
    case invalidArgument(String)
    case scheduleUnavailable(TriggerScheduleUnavailability)
}

@MainActor
enum TriggerTools {
    nonisolated static let maximumScheduleDelayMinutes = ToolArgumentValueDomains.maximumScheduleDelayMinutes
    nonisolated static let minimumIntervalSeconds = TriggerScheduleContract.minimumIntervalSeconds
    nonisolated static let maximumIntervalSeconds = ToolArgumentValueDomains.maximumTriggerIntervalSeconds
    nonisolated static let maximumBeforeEventMinutes = ToolArgumentValueDomains.maximumBeforeEventMinutes

    static func create(args: [String: String]) async -> String {
        let arguments: TriggerCreateArguments
        switch createArguments(from: args) {
        case .success(let validated):
            arguments = validated
        case .failure(let error):
            return invalidCreateArgumentsMessage(error)
        }

        guard let container = SharedContainer.shared else { return "Store unavailable." }
        let ctx = ModelContext(container)
        let trigger: Trigger
        switch arguments.schedule {
        case .once:
            guard let seconds = seconds(fromMinutes: arguments.inMinutes) else {
                return invalidCreateArgumentsMessage(.invalidArgument("inMinutes"))
            }
            let fire = Date().addingTimeInterval(TimeInterval(seconds))
            guard fire.timeIntervalSinceReferenceDate.isFinite else {
                return invalidCreateArgumentsMessage(.invalidArgument("inMinutes"))
            }
            trigger = Trigger(title: arguments.title, prompt: arguments.prompt, scheduleType: .once, fireDate: fire)
        case .daily:
            trigger = Trigger(
                title: arguments.title,
                prompt: arguments.prompt,
                scheduleType: .daily,
                timeOfDayMinutes: arguments.timeOfDayMinutes
            )
        case .interval:
            trigger = Trigger(
                title: arguments.title,
                prompt: arguments.prompt,
                scheduleType: .interval,
                intervalSeconds: TimeInterval(arguments.intervalSeconds)
            )
        case .beforeNextEvent:
            return invalidCreateArgumentsMessage(.scheduleUnavailable(
                .beforeNextEventRequiresForegroundCalendarIntegration
            ))
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
        return "Scheduled \"\(arguments.title)\" (\(arguments.schedule.label)) — next run: \(when)."
    }

    nonisolated static func createArguments(from args: [String: String]) -> Result<TriggerCreateArguments, TriggerCreateArgumentError> {
        guard let schedule = normalizedScheduleType(from: args["schedule"]) else {
            return .failure(.invalidArgument("schedule"))
        }
        if let unavailability = schedule.creationUnavailability {
            return .failure(.scheduleUnavailable(unavailability))
        }
        guard let inMinutes = integerArgument(
            args["inMinutes"],
            defaultValue: 60,
            domain: ToolArgumentValueDomains.triggerDelayMinutes
        ) else {
            return .failure(.invalidArgument("inMinutes"))
        }
        guard let timeOfDayMinutes = ToolArgumentValueDomains.clockTime24Hour.clockTimeMinutes(
            from: args["atTime"] ?? "09:00"
        ) else {
            return .failure(.invalidArgument("atTime"))
        }
        guard let intervalSeconds = integerArgument(
            args["intervalSeconds"],
            defaultValue: 3_600,
            domain: ToolArgumentValueDomains.triggerIntervalSeconds
        ) else {
            return .failure(.invalidArgument("intervalSeconds"))
        }
        guard let beforeMinutes = integerArgument(
            args["beforeMinutes"],
            defaultValue: 15,
            domain: ToolArgumentValueDomains.triggerBeforeEventMinutes
        ) else {
            return .failure(.invalidArgument("beforeMinutes"))
        }

        let title = args["title"] ?? "Scheduled run"
        let prompt = args["prompt"] ?? title
        return .success(TriggerCreateArguments(
            title: title,
            prompt: prompt,
            schedule: schedule,
            inMinutes: inMinutes,
            timeOfDayMinutes: timeOfDayMinutes,
            intervalSeconds: intervalSeconds,
            beforeMinutes: beforeMinutes
        ))
    }

    nonisolated static func seconds(fromMinutes minutes: Int) -> Int? {
        guard (1...maximumScheduleDelayMinutes).contains(minutes) else { return nil }
        let result = minutes.multipliedReportingOverflow(by: 60)
        return result.overflow ? nil : result.partialValue
    }

    nonisolated static func invalidCreateArgumentsMessage(_ error: TriggerCreateArgumentError) -> String {
        switch error {
        case .invalidArgument(let name):
            return "Trigger create failed: invalid argument `\(name)`."
        case .scheduleUnavailable(let unavailability):
            return "Trigger create unavailable: \(unavailability.message)"
        }
    }

    private nonisolated static func integerArgument(
        _ rawValue: String?,
        defaultValue: Int,
        domain: ToolArgumentValueDomain
    ) -> Int? {
        guard let rawValue else { return defaultValue }
        return domain.integerValue(from: rawValue)
    }

    private nonisolated static func normalizedScheduleType(from raw: String?) -> TriggerScheduleType? {
        switch raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "relative", "once", "one-time", "one time":
            return .once
        case "absolute", "daily":
            return .daily
        case "interval":
            return .interval
        case "beforenextevent", "before_next_event", "before-next-event":
            return .beforeNextEvent
        case nil, "":
            return .once
        default:
            return nil
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
