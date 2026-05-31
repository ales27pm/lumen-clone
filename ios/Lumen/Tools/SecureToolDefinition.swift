import Foundation

struct SecureToolDefinition: Codable, Sendable, Equatable {
    let id: ToolID
    let displayName: String
    let description: String
    let category: SecureToolCategory
    let requiredPermissions: [PermissionDomain]
    let supportsBackgroundExecution: Bool
    let requiresUserApproval: Bool
    let argumentSchemaDescription: String
    let resultPrivacyLevel: ToolResultPrivacyLevel
    let maxOutputCharacters: Int
}
