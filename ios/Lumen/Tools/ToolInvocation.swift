import Foundation

enum ToolInvocationSource: String, Codable, Sendable {
    case modelProposed
    case userInitiated
    case userApproved
    case backgroundTrigger
    case appIntent
    case system
}

extension ToolInvocationSource {
    var allowsPermissionPrompts: Bool {
        switch self {
        case .appIntent, .backgroundTrigger, .system:
            return false
        case .modelProposed, .userInitiated, .userApproved:
            return true
        }
    }
}

struct ToolInvocation: Codable, Sendable {
    let id: UUID
    let toolID: ToolID
    let arguments: [String: String]
    let source: ToolInvocationSource
    let conversationID: UUID?
    let turnID: UUID?
    let createdAt: Date
}
