import Foundation

struct PermissionGateDecision: Sendable, Equatable { let allowed: Bool; let reason: String? }

enum PermissionGate {
    static func evaluate(domain: PermissionDomain, state: AssistantPermissionState, isForeground: Bool) -> PermissionGateDecision {
        if !isForeground && state == .notDetermined { return .init(allowed: false, reason: "Permission requests are blocked in background") }
        switch state {
        case .granted: return .init(allowed: true, reason: nil)
        case .notDetermined: return .init(allowed: false, reason: "Permission not granted yet")
        case .limited: return .init(allowed: false, reason: "Permission limited")
        case .denied: return .init(allowed: false, reason: "Permission denied")
        case .restricted: return .init(allowed: false, reason: "Permission restricted")
        case .unavailable: return .init(allowed: false, reason: "Capability unavailable")
        case .unknown: return .init(allowed: false, reason: "Permission state unknown")
        }
    }
}
