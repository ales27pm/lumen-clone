import Foundation

enum GroundingPrivacyLevel: String, Codable, Sendable { case low, moderate, sensitive }

struct PromptGroundingSection: Codable, Sendable {
    let title: String
    let content: String
    let estimatedChars: Int
    let sourceIDs: [String]
    let privacyLevel: GroundingPrivacyLevel
}
