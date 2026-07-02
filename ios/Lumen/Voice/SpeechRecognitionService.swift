import Foundation

@MainActor
final class SpeechRecognitionService {
    private let requestPermissionsHandler: (() async -> Bool)?

    init(requestPermissionsHandler: (() async -> Bool)? = nil) {
        self.requestPermissionsHandler = requestPermissionsHandler
    }

    func requestPermissions() async -> Bool {
        if let requestPermissionsHandler {
            return await requestPermissionsHandler()
        }
        let speech = await PermissionRegistry.shared.request(.speechRecognition)
        guard speech.state == .granted else { return false }
        let mic = await PermissionRegistry.shared.request(.microphone)
        return mic.state == .granted
    }
}
