import Foundation

struct PermissionDiagnostics: Sendable {
    let statuses: [PermissionDomain: AssistantPermissionState]
    let entitlementWarnings: [EntitlementAuditWarning]

    static func collect(registry: PermissionRegistry = .shared, infoDictionary: [String: Any] = Bundle.main.infoDictionary ?? [:]) async -> PermissionDiagnostics {
        let statuses = await registry.diagnostics()
        let warnings = BackgroundEntitlementValidator.validate(infoDictionary: infoDictionary)
        return PermissionDiagnostics(statuses: statuses, entitlementWarnings: warnings)
    }
}
