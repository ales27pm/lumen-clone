import Foundation

nonisolated enum ToolResultStatusClassifier {
    static func status(from text: String) -> ToolResultStatus {
        let lower = text.lowercased()

        if containsAny(lower, [
            "approval required",
            "requires approval",
            "requires explicit user approval",
            "requires user approval",
            "before it can run",
            "i did not create",
            "not create an event"
        ]) {
            return .requiresApproval
        }

        if containsAny(lower, [
            "permission denied",
            "missing permission",
            "missing required permission",
            "access denied",
            "please enable it in settings",
            "enable it in settings",
            "i need calendar access",
            "i need reminders access",
            "i need contacts access",
            "i need location access",
            "i need photo library access",
            "i need camera access",
            "i need health access",
            "i need motion access",
            "tool denied",
            "denied by",
            " is disabled",
            "tool disabled"
        ]) {
            return .denied
        }

        if containsAny(lower, [
            "unknown tool",
            "tool unavailable",
            "unavailable pending",
            "not available",
            "not configured",
            "not signed in",
            "unsupported native"
        ]) {
            return .unavailable
        }

        if containsAny(lower, [
            "failed",
            "error",
            "couldn't",
            "could not",
            "unable to"
        ]) {
            return .failed
        }

        return .success
    }

    private static func containsAny(_ text: String, _ needles: [String]) -> Bool {
        needles.contains { text.contains($0) }
    }
}
