import Foundation

@MainActor
struct VoiceInterruptionHandler {
    nonisolated static func shouldInterruptOnBackground() -> Bool { true }
}
