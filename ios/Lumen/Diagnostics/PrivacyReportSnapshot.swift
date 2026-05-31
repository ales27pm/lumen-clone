import Foundation

struct PrivacyReportSnapshot: Sendable {
    let localOnlyMode: Bool
    let networkAccessState: String
    let recentToolCategories: [String]
    let appIntentLimitations: [String]
}
