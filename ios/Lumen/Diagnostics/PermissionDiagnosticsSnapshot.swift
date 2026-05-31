import Foundation

struct PermissionDiagnosticsSnapshot: Sendable {
    let domains: [(domain: String, state: String)]
}
