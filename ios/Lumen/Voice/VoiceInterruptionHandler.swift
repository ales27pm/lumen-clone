import Foundation

@MainActor
struct VoiceInterruptionHandler {
    static func shouldInterruptOnBackground() -> Bool { true }
}
