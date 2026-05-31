import Foundation

enum ToolInvocationSource: String, Codable, Sendable { case modelProposed, userInitiated, backgroundTrigger, appIntent, system }

struct ToolInvocation: Codable, Sendable {
    let id: UUID
    let toolID: ToolID
    let arguments: [String: String]
    let source: ToolInvocationSource
    let conversationID: UUID?
    let turnID: UUID?
    let createdAt: Date
}
