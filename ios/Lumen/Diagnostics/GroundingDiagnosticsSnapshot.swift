import Foundation

struct GroundingDiagnosticsSnapshot: Sendable {
    let contextSource: String
    let degradedReasons: [String]
    let sectionCounts: [String: Int]
    let doubleGroundingNormalized: Bool
}
