import Foundation

enum MemoryUserExplicitness: String, Codable, Sendable { case explicitPreference, correction, repeatedFact, inferred, projectFact, transient }
enum MemorySensitivity: String, Codable, Sendable { case normal, personal, credentialLike, healthOrLegal, financial }

struct MemoryCandidate: Codable, Sendable {
    let text: String
    let kind: String
    let topics: [String]
    let conversationID: UUID?
    let messageID: UUID?
    let createdAt: Date
    let confidence: Double
    let extractionReason: String
    let userExplicitness: MemoryUserExplicitness
    let sensitivity: MemorySensitivity
}
