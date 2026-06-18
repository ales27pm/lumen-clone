import Foundation

struct PermissionDiagnostics: Sendable {
    let statuses: [PermissionDomain: AssistantPermissionState]
    let entitlementWarnings: [EntitlementAuditWarning]

    static func collect(registry: PermissionRegistry? = nil, infoDictionary: [String: Any] = Bundle.main.infoDictionary ?? [:]) async -> PermissionDiagnostics {
        let resolvedRegistry: PermissionRegistry
        if let providedRegistry = registry {
            resolvedRegistry = providedRegistry
        } else {
            resolvedRegistry = await MainActor.run { PermissionRegistry.shared }
        }
        let statuses = await resolvedRegistry.diagnostics()
        let warnings = BackgroundEntitlementValidator.validate(infoDictionary: infoDictionary)
        return PermissionDiagnostics(statuses: statuses, entitlementWarnings: warnings)
    }
}
