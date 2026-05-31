import Foundation

struct ToolSecuritySnapshot: Sendable {
    struct ToolRow: Sendable {
        let id: String
        let category: String
        let requiredPermissions: [String]
        let supportsBackground: Bool
        let requiresApproval: Bool
    }
    let tools: [ToolRow]
}
