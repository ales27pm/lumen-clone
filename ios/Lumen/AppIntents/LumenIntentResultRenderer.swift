import Foundation

enum LumenIntentResultRenderer {
    static func openAppRequired(_ reason: String) -> String { "Open Lumen to approve: \(reason)" }
    static func degraded(_ reason: String) -> String { "Unavailable right now: \(reason)" }
}
