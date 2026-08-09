import Foundation

#if canImport(AlarmKit)
import AlarmKit
import SwiftUI

private struct LumenAlarmMetadata: AlarmMetadata {
    let title: String
}
#endif

nonisolated struct AlarmScheduleArguments: Equatable, Sendable {
    let title: String
    let fireDate: Date
    let repeats: Bool
    let snoozeMinutes: Int
}

nonisolated enum AlarmScheduleArgumentError: Error, Equatable, Sendable {
    case missingSchedule
    case invalidArgument(String)
}

nonisolated enum AlarmCountdownArgumentError: Error, Equatable, Sendable {
    case missingDuration
    case invalidArgument(String)
}

@MainActor
enum AlarmTools {
    nonisolated static let unavailableMessage = "AlarmKit availability: unavailable (requires iOS 26.0+ and an AlarmKit-capable device runtime)."
    nonisolated static let maximumScheduleDelayMinutes = ToolArgumentValueDomains.maximumScheduleDelayMinutes
    nonisolated static let maximumSnoozeMinutes = ToolArgumentValueDomains.maximumSnoozeMinutes
    nonisolated static let maximumDurationSeconds = ToolArgumentValueDomains.maximumCountdownDurationSeconds

    nonisolated static func isRuntimeUnavailableText(_ text: String) -> Bool {
        let lower = text.lowercased()
        return lower.contains("alarmkit availability: unavailable")
            || lower.contains("alarmkit requires")
            || lower.contains("alarmkit-capable")
            || lower.contains("alarmkit capable")
    }

    static func authorizationStatus() async -> String {
        guard alarmUsageDescriptionPresent() else { return missingUsageDescriptionMessage }
#if canImport(AlarmKit)
        if #available(iOS 26.0, *) {
            return "Alarm authorization status: \(String(describing: AlarmManager.shared.authorizationState))."
        }
#endif
        return unavailableMessage
    }

    static func requestAuthorization() async -> String {
        guard alarmUsageDescriptionPresent() else { return missingUsageDescriptionMessage }
#if canImport(AlarmKit)
        if #available(iOS 26.0, *) {
            do {
                let state = try await AlarmManager.shared.requestAuthorization()
                return "Alarm authorization result: \(String(describing: state))."
            } catch {
                return authorizationFailureMessage()
            }
        }
#endif
        return unavailableMessage
    }

    static func schedule(args: [String: String]) async -> String {
        let arguments: AlarmScheduleArguments
        switch scheduleArguments(from: args) {
        case .success(let validated):
            arguments = validated
        case .failure(let error):
            return invalidScheduleArgumentsMessage(error)
        }

        if arguments.repeats {
            return "Alarm scheduling failed: repeating alarms are not supported by this tool path yet."
        }

        return await scheduleAlarm(
            title: arguments.title,
            fireDate: arguments.fireDate,
            repeats: arguments.repeats,
            snoozeMinutes: arguments.snoozeMinutes
        )
    }

    static func countdown(args: [String: String]) async -> String {
        let title = args["title"]?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? args["title"]!
            : "Countdown"
        let duration: Int
        switch countdownDurationSeconds(from: args) {
        case .success(let validated):
            duration = validated
        case .failure(let error):
            return invalidCountdownArgumentsMessage(error)
        }
        return await scheduleCountdown(title: title, durationSeconds: duration)
    }

    nonisolated static func countdownDurationSeconds(
        from args: [String: String]
    ) -> Result<Int, AlarmCountdownArgumentError> {
        guard let rawDuration = args["durationSeconds"] else {
            return .failure(.missingDuration)
        }
        guard let duration = ToolArgumentValueDomains.alarmCountdownSeconds.integerValue(from: rawDuration) else {
            return .failure(.invalidArgument("durationSeconds"))
        }
        return .success(duration)
    }

    nonisolated static func scheduleArguments(
        from args: [String: String],
        now: Date = Date()
    ) -> Result<AlarmScheduleArguments, AlarmScheduleArgumentError> {
        let title = args["title"]?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? args["title"]!
            : "Alarm"
        guard let snoozeMinutes = integerArgument(
            args["snoozeMinutes"],
            defaultValue: 5,
            domain: ToolArgumentValueDomains.alarmSnoozeMinutes
        ) else {
            return .failure(.invalidArgument("snoozeMinutes"))
        }
        let repeats = (args["repeats"] ?? "false").trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "true"

        let fireDate: Date
        if let rawMinutes = args["inMinutes"] {
            guard let inMinutes = ToolArgumentValueDomains.alarmScheduleDelayMinutes.integerValue(from: rawMinutes),
                  let seconds = seconds(fromScheduleDelayMinutes: inMinutes) else {
                return .failure(.invalidArgument("inMinutes"))
            }
            fireDate = now.addingTimeInterval(TimeInterval(seconds))
            guard fireDate.timeIntervalSinceReferenceDate.isFinite else {
                return .failure(.invalidArgument("inMinutes"))
            }
        } else if let rawTimestamp = args["timestamp"] {
            guard let unix = TimeInterval(rawTimestamp), unix.isFinite else {
                return .failure(.invalidArgument("timestamp"))
            }
            fireDate = Date(timeIntervalSince1970: unix)
            guard fireDate.timeIntervalSinceReferenceDate.isFinite else {
                return .failure(.invalidArgument("timestamp"))
            }
        } else {
            return .failure(.missingSchedule)
        }

        return .success(AlarmScheduleArguments(
            title: title,
            fireDate: fireDate,
            repeats: repeats,
            snoozeMinutes: snoozeMinutes
        ))
    }

    nonisolated static func seconds(fromScheduleDelayMinutes minutes: Int) -> Int? {
        seconds(fromMinutes: minutes, maximum: maximumScheduleDelayMinutes)
    }

    nonisolated static func seconds(fromSnoozeMinutes minutes: Int) -> Int? {
        seconds(fromMinutes: minutes, maximum: maximumSnoozeMinutes)
    }

    private nonisolated static func seconds(fromMinutes minutes: Int, maximum: Int) -> Int? {
        guard (1...maximum).contains(minutes) else { return nil }
        let result = minutes.multipliedReportingOverflow(by: 60)
        return result.overflow ? nil : result.partialValue
    }

    nonisolated static func invalidScheduleArgumentsMessage(_ error: AlarmScheduleArgumentError) -> String {
        switch error {
        case .missingSchedule:
            return "Missing schedule. Provide `inMinutes` or `timestamp` (Unix seconds)."
        case .invalidArgument(let argument):
            return "Alarm scheduling failed: invalid argument `\(argument)`."
        }
    }

    nonisolated static func invalidCountdownArgumentsMessage(_ error: AlarmCountdownArgumentError) -> String {
        switch error {
        case .missingDuration:
            return "Missing duration. Provide `durationSeconds` greater than 0."
        case .invalidArgument(let argument):
            return "Alarm countdown failed: invalid argument `\(argument)`."
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

    static func list() async -> String {
        guard alarmUsageDescriptionPresent() else { return missingUsageDescriptionMessage }
#if canImport(AlarmKit)
        if #available(iOS 26.0, *) {
            do {
                let alarms = try AlarmManager.shared.alarms
                if alarms.isEmpty { return "No active alarms." }
                return "Active alarms:\n" + alarms.map { alarm in
                    "• id=\(alarm.id.uuidString); state=\(String(describing: alarm.state))"
                }.joined(separator: "\n")
            } catch {
                return readFailureMessage()
            }
        }
#endif
        return unavailableMessage
    }

    static func cancel(id: String) async -> String {
        let trimmed = id.trimmingCharacters(in: .whitespacesAndNewlines)
        if UUID(uuidString: trimmed) != nil {
            return await mutateAlarm(id: trimmed, actionName: "cancel") {
#if canImport(AlarmKit)
                if #available(iOS 26.0, *) {
                    try AlarmManager.shared.cancel(id: $0)
                }
#endif
            }
        }
        return "Invalid alarm id. Provide the alarm UUID from `alarm.list` in `id`."
    }

    static func pause(id: String) async -> String {
        await mutateAlarm(id: id, actionName: "pause") {
#if canImport(AlarmKit)
            if #available(iOS 26.0, *) {
                try AlarmManager.shared.pause(id: $0)
            }
#endif
        }
    }

    static func resume(id: String) async -> String {
        await mutateAlarm(id: id, actionName: "resume") {
#if canImport(AlarmKit)
            if #available(iOS 26.0, *) {
                try AlarmManager.shared.resume(id: $0)
            }
#endif
        }
    }

    static func stop(id: String) async -> String {
        await mutateAlarm(id: id, actionName: "stop") {
#if canImport(AlarmKit)
            if #available(iOS 26.0, *) {
                try AlarmManager.shared.stop(id: $0)
            }
#endif
        }
    }

    static func snooze(id: String) async -> String {
        await mutateAlarm(id: id, actionName: "snooze") {
#if canImport(AlarmKit)
            if #available(iOS 26.0, *) {
                try AlarmManager.shared.countdown(id: $0)
            }
#endif
        }
    }

    private static func mutateAlarm(id: String, actionName: String, _ operation: (UUID) throws -> Void) async -> String {
        guard let uuid = UUID(uuidString: id) else {
            return "Invalid alarm id. Provide a UUID string in `id`."
        }
        guard alarmUsageDescriptionPresent() else { return missingUsageDescriptionMessage }
#if canImport(AlarmKit)
        if #available(iOS 26.0, *) {
            do {
                try operation(uuid)
                return "Alarm \(actionName) completed for \(uuid.uuidString)."
            } catch {
                return mutationFailureMessage(actionName: actionName)
            }
        }
#endif
        return unavailableMessage
    }

    private static func scheduleAlarm(title: String, fireDate: Date, repeats: Bool, snoozeMinutes: Int) async -> String {
        guard let snoozeSeconds = seconds(fromSnoozeMinutes: snoozeMinutes) else {
            return invalidScheduleArgumentsMessage(.invalidArgument("snoozeMinutes"))
        }
        guard alarmUsageDescriptionPresent() else { return missingUsageDescriptionMessage }
#if canImport(AlarmKit)
        if #available(iOS 26.0, *) {
            do {
                let id = UUID()
                let configuration = AlarmManager.AlarmConfiguration<LumenAlarmMetadata>(
                    countdownDuration: Alarm.CountdownDuration(
                        preAlert: nil,
                        postAlert: TimeInterval(snoozeSeconds)
                    ),
                    schedule: .fixed(fireDate),
                    attributes: alarmAttributes(title: title)
                )
                let alarm = try await AlarmManager.shared.schedule(id: id, configuration: configuration)
                return "Alarm scheduled: id=\(alarm.id.uuidString); title=\"\(title)\"; fireDate=\(fireDate.formatted(date: .abbreviated, time: .shortened)); state=\(String(describing: alarm.state))."
            } catch {
                return schedulingFailureMessage()
            }
        }
#endif
        _ = repeats
        _ = snoozeMinutes
        return "\(unavailableMessage) Requested \"\(title)\" for \(fireDate.formatted(date: .abbreviated, time: .shortened))."
    }

    private static func scheduleCountdown(title: String, durationSeconds: Int) async -> String {
        guard alarmUsageDescriptionPresent() else { return missingUsageDescriptionMessage }
#if canImport(AlarmKit)
        if #available(iOS 26.0, *) {
            do {
                let id = UUID()
                let configuration = AlarmManager.AlarmConfiguration<LumenAlarmMetadata>.timer(
                    duration: TimeInterval(durationSeconds),
                    attributes: alarmAttributes(title: title)
                )
                let alarm = try await AlarmManager.shared.schedule(id: id, configuration: configuration)
                return "Alarm countdown scheduled: id=\(alarm.id.uuidString); title=\"\(title)\"; durationSeconds=\(durationSeconds); state=\(String(describing: alarm.state))."
            } catch {
                return countdownFailureMessage()
            }
        }
#endif
        return "\(unavailableMessage) Requested countdown \"\(title)\" for \(durationSeconds) seconds."
    }

#if canImport(AlarmKit)
    @available(iOS 26.0, *)
    private static func alarmAttributes(title: String) -> AlarmAttributes<LumenAlarmMetadata> {
        AlarmAttributes(
            presentation: alarmPresentation(title: title),
            metadata: LumenAlarmMetadata(title: title),
            tintColor: .orange
        )
    }

    @available(iOS 26.0, *)
    private static func alarmPresentation(title: String) -> AlarmPresentation {
        let displayTitle = title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Alarm" : title
        let localizedTitle = LocalizedStringResource(stringLiteral: displayTitle)
        let pauseButton = AlarmButton(text: "Pause", textColor: .orange, systemImageName: "pause.fill")
        let resumeButton = AlarmButton(text: "Resume", textColor: .orange, systemImageName: "play.fill")
        if #available(iOS 26.1, *) {
            let secondaryButton = AlarmButton(text: "Snooze", textColor: .orange, systemImageName: "zzz")
            return AlarmPresentation(
                alert: .init(title: localizedTitle, secondaryButton: secondaryButton, secondaryButtonBehavior: .countdown),
                countdown: .init(title: localizedTitle, pauseButton: pauseButton),
                paused: .init(title: localizedTitle, resumeButton: resumeButton)
            )
        } else {
            let stopButton = AlarmButton(text: "Stop", textColor: .orange, systemImageName: "stop.fill")
            return AlarmPresentation(
                alert: .init(title: localizedTitle, stopButton: stopButton),
                countdown: .init(title: localizedTitle, pauseButton: pauseButton),
                paused: .init(title: localizedTitle, resumeButton: resumeButton)
            )
        }
    }
#endif

    nonisolated static var missingUsageDescriptionMessage: String {
        "AlarmKit availability: unavailable (missing NSAlarmKitUsageDescription in the installed app bundle)."
    }

    nonisolated static func authorizationFailureMessage() -> String {
        "Alarm authorization failed. Try again later."
    }

    nonisolated static func readFailureMessage() -> String {
        "Alarm read failed. Try again later."
    }

    nonisolated static func mutationFailureMessage(actionName: String) -> String {
        "Alarm \(actionName) failed. Try again later."
    }

    nonisolated static func schedulingFailureMessage() -> String {
        "Alarm scheduling failed. Try again later."
    }

    nonisolated static func countdownFailureMessage() -> String {
        "Alarm countdown failed. Try again later."
    }

    private nonisolated static func alarmUsageDescriptionPresent() -> Bool {
        PermissionsCenter.alarmUsageDescriptionPresent()
    }
}
