import Foundation
import SwiftData

@Model
final class MemoryItem {
    var id: UUID = UUID()
    var content: String = ""
    var source: String = "manual"
    var createdAt: Date = Date()
    var embedding: [Double] = []
    var kind: String = "fact"
    var isPinned: Bool = false
    var topic: String?
    var expiresAt: Date?
    var freshnessClass: String?

    init(
        content: String,
        kind: MemoryKind = .fact,
        source: String = "manual",
        embedding: [Double] = [],
        topic: String? = nil,
        expiresAt: Date? = nil,
        freshnessClass: MemoryFreshnessClass? = nil
    ) {
        self.content = content
        self.kind = kind.rawValue
        self.source = source
        self.embedding = embedding
        self.topic = topic
        self.expiresAt = expiresAt
        self.freshnessClass = freshnessClass?.rawValue
    }

    var memoryKind: MemoryKind { MemoryKind(rawValue: kind) ?? .fact }
}

enum MemoryFreshnessClass: String, Codable, CaseIterable, Sendable {
    case volatile
    case shortLived
    case durable
    case timeless
}

enum MemoryKind: String, Codable, CaseIterable, Sendable {
    case fact, preference, conversation, person, project

    var label: String {
        switch self {
        case .fact: "Fact"
        case .preference: "Preference"
        case .conversation: "Chat"
        case .person: "Person"
        case .project: "Project"
        }
    }

    var icon: String {
        switch self {
        case .fact: "sparkle"
        case .preference: "heart.fill"
        case .conversation: "bubble.left.and.bubble.right.fill"
        case .person: "person.crop.circle.fill"
        case .project: "folder.fill"
        }
    }

    var tintName: String {
        switch self {
        case .fact: "cyan"
        case .preference: "pink"
        case .conversation: "blue"
        case .person: "orange"
        case .project: "green"
        }
    }
}
