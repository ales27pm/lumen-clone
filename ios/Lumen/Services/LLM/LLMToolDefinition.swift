import Foundation

struct LLMToolDefinition: Codable, Sendable, Equatable {
    let name: String
    let description: String
    let jsonSchema: String
    let isDestructive: Bool
    let requiresUserApproval: Bool

    init(
        name: String,
        description: String,
        jsonSchema: String,
        isDestructive: Bool,
        requiresUserApproval: Bool
    ) {
        self.name = name
        self.description = description
        self.jsonSchema = jsonSchema
        self.isDestructive = isDestructive
        self.requiresUserApproval = requiresUserApproval
    }
}
