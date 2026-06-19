import Foundation

struct PermissionDiagnostics: Sendable {
    let statuses: [PermissionDomain: AssistantPermissionState]
    let entitlementWarnings: [EntitlementAuditWarning]

    /// Collects current permission diagnostics and entitlement warnings.
    /// - Parameters:
    ///   - registry: The permission registry to use. If `nil`, resolves to `PermissionRegistry.shared`.
    ///   - infoDictionary: A dictionary used for entitlement validation.
    /// - Returns: A `PermissionDiagnostics` instance containing permission statuses and entitlement warnings.
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
