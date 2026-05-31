import Foundation

struct BackgroundDiagnosticsSnapshot: Sendable {
    let permittedIdentifiers: [String]
    let entitlementWarnings: [String]
}
