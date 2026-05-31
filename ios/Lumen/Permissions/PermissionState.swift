import Foundation

enum AssistantPermissionState: String, Codable, Sendable, Equatable {
    case notDetermined, granted, limited, denied, restricted, unavailable, unknown
}
