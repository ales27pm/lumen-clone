import Foundation

struct PermissionRequestResult: Sendable, Equatable {
    let domain: PermissionDomain
    let state: AssistantPermissionState
    let message: String
}
