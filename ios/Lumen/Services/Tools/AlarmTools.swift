import Foundation

#if canImport(AlarmKit)
import AlarmKit
import SwiftUI

private struct LumenAlarmMetadata: AlarmMetadata {
    let title: String
}
#endif

@MainActor
enum AlarmTools {
    nonisolated static let unavailableMessage = "AlarmKit availability: unavailable (requires iOS 26.0+ and an AlarmKit-capable device runtime)."

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
                return "Alarm authorization failed: \(error.localizedDescription)"
            }
        }
#endif
        return unavailableMessage
    }

    static func schedule(args: [String: String]) async -> String {
        let title = args["title"]?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? args["title"]!
            : "Alarm"
        let snoozeMinutes = Int(args["snoozeMinutes"] ?? "5") ?? 5
        let repeats = (args["repeats"] ?? "false").lowercased() == "true"
        if repeats {
            return "Alarm scheduling failed: repeating alarms are not supported by this tool path yet."
        }

        if let inMinutes = Int(args["inMinutes"] ?? "") {
            let fireDate = Date().addingTimeInterval(TimeInterval(max(1, inMinutes) * 60))
            return await scheduleAlarm(
                title: title,
                fireDate: fireDate,
                repeats: repeats,
                snoozeMinutes: max(1, snoozeMinutes)
            )
        }

        if let unix = TimeInterval(args["timestamp"] ?? "") {
            let fireDate = Date(timeIntervalSince1970: unix)
            return await scheduleAlarm(
                title: title,
                fireDate: fireDate,
                repeats: repeats,
                snoozeMinutes: max(1, snoozeMinutes)
            )
        }

        return "Missing schedule. Provide `inMinutes` or `timestamp` (Unix seconds)."
    }

    static func countdown(args: [String: String]) async -> String {
        let title = args["title"]?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? args["title"]!
            : "Countdown"
        let duration = Int(args["durationSeconds"] ?? "") ?? 0
        guard duration > 0 else {
            return "Missing duration. Provide `durationSeconds` greater than 0."
        }
        return await scheduleCountdown(title: title, durationSeconds: duration)
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
                return "Unable to read alarms: \(error.localizedDescription)"
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
                return "Alarm \(actionName) failed: \(error.localizedDescription)"
            }
        }
#endif
        return unavailableMessage
    }

    private static func scheduleAlarm(title: String, fireDate: Date, repeats: Bool, snoozeMinutes: Int) async -> String {
        guard alarmUsageDescriptionPresent() else { return missingUsageDescriptionMessage }
#if canImport(AlarmKit)
        if #available(iOS 26.0, *) {
            do {
                let id = UUID()
                let configuration = AlarmManager.AlarmConfiguration<LumenAlarmMetadata>(
                    countdownDuration: Alarm.CountdownDuration(
                        preAlert: nil,
                        postAlert: TimeInterval(max(1, snoozeMinutes) * 60)
                    ),
                    schedule: .fixed(fireDate),
                    attributes: alarmAttributes(title: title)
                )
                let alarm = try await AlarmManager.shared.schedule(id: id, configuration: configuration)
                return "Alarm scheduled: id=\(alarm.id.uuidString); title=\"\(title)\"; fireDate=\(fireDate.formatted(date: .abbreviated, time: .shortened)); state=\(String(describing: alarm.state))."
            } catch {
                return "Alarm scheduling failed: \(error.localizedDescription)"
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
                return "Alarm countdown failed: \(error.localizedDescription)"
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

    private nonisolated static func alarmUsageDescriptionPresent() -> Bool {
        PermissionsCenter.alarmUsageDescriptionPresent()
    }
}
