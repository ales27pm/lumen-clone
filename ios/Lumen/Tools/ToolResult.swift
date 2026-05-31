import Foundation

enum ToolResultStatus: String, Codable, Sendable { case success, denied, requiresApproval, failed, unavailable }
enum ToolResultPrivacyLevel: String, Codable, Sendable { case low, moderate, sensitive }

struct ToolResult: Codable, Sendable {
    let invocationID: UUID
    let status: ToolResultStatus
    let displayText: String
    let modelText: String
    let structuredPayload: [String: String]?
    let privacyLevel: ToolResultPrivacyLevel
    let metricsSummary: String
    let errorCode: String?
}
