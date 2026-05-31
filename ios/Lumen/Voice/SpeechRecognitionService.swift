import Foundation

@MainActor
final class SpeechRecognitionService {
    func requestPermissions() async -> Bool {
        let speech = await PermissionRegistry.shared.request(.speechRecognition)
        guard speech.state == .granted else { return false }
        let mic = await PermissionRegistry.shared.request(.microphone)
        return mic.state == .granted
    }
}
