import Foundation

struct EntitlementAuditWarning: Sendable, Equatable {
    let code: String
    let message: String
}

enum BackgroundEntitlementValidator {
    static let requiredTaskIDs: Set<String> = [
        TriggerScheduler.refreshIdentifier,
        TriggerScheduler.processIdentifier
    ]

    static func validate(infoDictionary: [String: Any]) -> [EntitlementAuditWarning] {
        var warnings: [EntitlementAuditWarning] = []
        let permitted: Set<String>
        if let values = infoDictionary["BGTaskSchedulerPermittedIdentifiers"] as? [String] {
            permitted = Set(values)
        } else if let value = infoDictionary["BGTaskSchedulerPermittedIdentifiers"] as? String {
            permitted = Set(value.split { $0 == " " || $0 == ";" || $0 == "," }.map(String.init))
        } else {
            permitted = []
        }
        for required in requiredTaskIDs where !permitted.contains(required) {
            warnings.append(.init(code: "missing_bg_task_id", message: "Missing BG task identifier: \(required)"))
        }

        let keyChecks: [[String]] = [
            ["NSMicrophoneUsageDescription"],
            ["NSSpeechRecognitionUsageDescription"],
            ["NSCalendarsUsageDescription", "NSCalendarsFullAccessUsageDescription"],
            ["NSContactsUsageDescription"]
        ]
        for alternatives in keyChecks where !alternatives.contains(where: { (infoDictionary[$0] as? String)?.isEmpty == false }) {
            warnings.append(.init(code: "missing_usage_description", message: "Missing usage description: \(alternatives.joined(separator: " or "))"))
        }
        return warnings
    }
}
