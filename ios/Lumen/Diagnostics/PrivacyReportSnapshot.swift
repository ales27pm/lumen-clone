import Foundation

struct PrivacyReportSnapshot: Sendable {
    let networkToolsEnabled: Bool?
    let networkAccessState: String
    let recentToolCategories: [String]
    let appIntentLimitations: [String]
}
