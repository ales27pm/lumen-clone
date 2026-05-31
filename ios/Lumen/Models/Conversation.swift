import Foundation
import SwiftData

@Model
final class Conversation {
    var id: UUID = UUID()
    var title: String = "New Chat"
    var createdAt: Date = Date()
    var updatedAt: Date = Date()
    var isPinned: Bool = false
    var modelName: String?
    var systemPrompt: String?
    @Relationship(deleteRule: .cascade, inverse: \ChatMessage.conversation)
    var messages: [ChatMessage] = []

    init(title: String = "New Chat", systemPrompt: String? = nil, modelName: String? = nil) {
        self.title = title
        self.systemPrompt = systemPrompt
        self.modelName = modelName
    }

    var sortedMessages: [ChatMessage] {
        messages.sorted { $0.createdAt < $1.createdAt }
    }

    var preview: String {
        sortedMessages.last(where: { $0.role != "system" })?.content ?? "No messages yet"
    }
}
