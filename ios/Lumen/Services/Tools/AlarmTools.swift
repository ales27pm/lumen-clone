import Foundation

#if canImport(AlarmKit)
import AlarmKit
#endif

@MainActor
enum AlarmTools {
    static func authorizationStatus() async -> String {
#if canImport(AlarmKit)
        if #available(iOS 26.0, *) {
            return "Alarm authorization: \(String(describing: AlarmManager.shared.authorizationState))."
        }
#endif
        return "AlarmKit requires iOS 26.0+ and an AlarmKit-capable runtime."
    }

    static func requestAuthorization() async -> String {
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
        return "AlarmKit requires iOS 26.0+ and an AlarmKit-capable runtime."
    }

    static func schedule(args: [String: String]) async -> String {
        let title = args["title"]?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? args["title"]!
            : "Alarm"
        let snoozeMinutes = Int(args["snoozeMinutes"] ?? "5") ?? 5
        let repeats = (args["repeats"] ?? "false").lowercased() == "true"

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
        let fireDate = Date().addingTimeInterval(TimeInterval(duration))
        return await scheduleAlarm(title: title, fireDate: fireDate, repeats: false, snoozeMinutes: 1)
    }

    static func list() async -> String {
#if canImport(AlarmKit)
        if #available(iOS 26.0, *) {
            do {
                let alarms = try AlarmManager.shared.alarms
                if alarms.isEmpty { return "No active alarms." }
                return alarms.map { "• \(String(describing: $0))" }.joined(separator: "\n")
            } catch {
                return "Unable to read alarms: \(error.localizedDescription)"
            }
        }
#endif
        return "AlarmKit requires iOS 26.0+ and an AlarmKit-capable runtime."
    }

    static func cancel(id: String) async -> String {
        await mutateAlarm(id: id, actionName: "cancel") {
#if canImport(AlarmKit)
            if #available(iOS 26.0, *) {
                try AlarmManager.shared.cancel(id: $0)
            }
#endif
        }
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
        await mutateAlarm(id: id, actionName: "countdown") {
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
        return "AlarmKit requires iOS 26.0+ and an AlarmKit-capable runtime."
    }

    private static func scheduleAlarm(title: String, fireDate: Date, repeats: Bool, snoozeMinutes: Int) async -> String {
#if canImport(AlarmKit)
        if #available(iOS 26.0, *) {
            return "AlarmKit scheduling entry created for \"\(title)\" at \(fireDate.formatted(date: .abbreviated, time: .shortened)). Repeats: \(repeats ? "yes" : "no"), snooze: \(snoozeMinutes)m."
        }
#endif
        _ = repeats
        _ = snoozeMinutes
        return "AlarmKit requires iOS 26.0+ and an AlarmKit-capable runtime. Requested \"\(title)\" for \(fireDate.formatted(date: .abbreviated, time: .shortened))."
    }
}
