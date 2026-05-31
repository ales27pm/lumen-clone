import Foundation

enum LumenIntentPolicy {
    static func requiresOpenAppForSensitiveAction(_ action: String) -> Bool {
        let lowered = action.lowercased()
        return lowered.contains("calendar") || lowered.contains("contacts") || lowered.contains("location") || lowered.contains("notify") || lowered.contains("open.url")
    }
}
