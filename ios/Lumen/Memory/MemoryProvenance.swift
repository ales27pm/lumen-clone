import Foundation

struct MemoryProvenance: Codable, Sendable, Equatable {
    let memoryID: UUID
    let conversationID: UUID?
    let messageID: UUID?
    let source: String
    let createdAt: Date
}
