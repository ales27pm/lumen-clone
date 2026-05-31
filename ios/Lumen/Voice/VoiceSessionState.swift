import Foundation

enum VoiceSessionState: Equatable, Sendable {
    case idle
    case requestingPermissions
    case listening
    case processing
    case speaking
    case interrupted
    case denied(String)
    case failed(String)
}
