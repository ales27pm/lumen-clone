import Foundation

struct LLMChatMessage: Codable, Sendable, Equatable, Identifiable {
    enum Role: String, Codable, Sendable, Equatable, CaseIterable {
        case system
        case user
        case assistant
        case tool
    }

    let id: UUID
    let role: Role
    let content: String
    let name: String?
    let toolCallID: String?
    let createdAt: Date

    init(
        id: UUID = UUID(),
        role: Role,
        content: String,
        name: String? = nil,
        toolCallID: String? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.name = name
        self.toolCallID = toolCallID
        self.createdAt = createdAt
    }
}
