import Foundation

enum SecureToolCategory: String, Codable, Sendable { case readOnly, permissionRead, userVisibleAction, sensitiveAction, destructiveAction, externalNetwork }
