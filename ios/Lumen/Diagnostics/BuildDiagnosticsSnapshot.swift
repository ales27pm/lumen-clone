import Foundation

struct BuildDiagnosticsSnapshot: Sendable {
    let bundleIdentifier: String
    let bundleVersion: String
    let buildSourceIdentifier: String
    let gitSHA: String
    let configuration: String
    let scheme: String
    let alarmKitUsageDescription: String?

    static func current(infoDictionary: [String: Any] = Bundle.main.infoDictionary ?? [:], bundleIdentifier: String? = Bundle.main.bundleIdentifier) -> BuildDiagnosticsSnapshot {
        BuildDiagnosticsSnapshot(
            bundleIdentifier: bundleIdentifier ?? "unknown",
            bundleVersion: stringValue(infoDictionary["CFBundleVersion"]) ?? "unknown",
            buildSourceIdentifier: stringValue(infoDictionary["LumenBuildSourceIdentifier"]) ?? stringValue(infoDictionary["CFBundleVersion"]) ?? "unknown",
            gitSHA: stringValue(infoDictionary["LumenGitSHA"]) ?? "unknown",
            configuration: stringValue(infoDictionary["LumenBuildConfiguration"]) ?? "unknown",
            scheme: stringValue(infoDictionary["LumenBuildScheme"]) ?? "unknown",
            alarmKitUsageDescription: stringValue(infoDictionary["NSAlarmKitUsageDescription"])
        )
    }

    private static func stringValue(_ value: Any?) -> String? {
        guard let string = value as? String else { return nil }
        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
