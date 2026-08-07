import Foundation

nonisolated enum CarPlayVoiceSessionState: String, Sendable, Equatable {
    case idle
    case requestingPermission
    case listening
    case thinking
    case speaking
    case unavailable

    var acceptsAsk: Bool {
        self == .idle
    }
}

nonisolated enum CarPlayVoiceSessionPolicy {
    static let listeningTimeoutSeconds: TimeInterval = 12
    static let voiceStateActivationMinimumInterval: TimeInterval = 0.35
    static let maxSpokenAnswerCharacters = 420
    static let thermalRetryMessage = "Device is too warm; cool iPhone and retry."
    static let emptyTranscriptMessage = "I didn’t catch that. Please try again."

    static func compactAlertTitle(title: String, message: String, maxCharacters: Int = 180) -> String {
        let compact = "\(title): \(message)"
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard compact.count > maxCharacters else { return compact }
        return String(compact.prefix(max(0, maxCharacters - 1))).trimmingCharacters(in: .whitespacesAndNewlines) + "…"
    }

    static func blocksModelRun(thermalState: ProcessInfo.ThermalState) -> Bool {
        thermalState == .serious || thermalState == .critical
    }

    static func spokenAnswer(from raw: String, maxCharacters: Int = maxSpokenAnswerCharacters) -> String {
        let sanitized = FinalOutputSanitizer.sanitizeUserVisibleText(raw).text
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard sanitized.count > maxCharacters else { return sanitized }
        return String(sanitized.prefix(maxCharacters)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func headlessDenialMessage(for prompt: String) -> String? {
        guard let reason = LumenIntentPolicy.openAppReason(forHeadlessPrompt: prompt) else {
            return nil
        }
        return "Open Lumen on iPhone to approve: \(reason)."
    }

    static func acceptsAsk(in state: CarPlayVoiceSessionState) -> Bool {
        state.acceptsAsk
    }
}

nonisolated enum CarPlayVoiceStateActivationDecision: Equatable, Sendable {
    case activateNow
    case delay(TimeInterval)
    case skipDuplicate
}

nonisolated enum CarPlayVoiceStateActivationPolicy {
    static func decision(
        requestedStateID: String,
        previousStateID: String?,
        lastActivationUptime: TimeInterval?,
        nowUptime: TimeInterval,
        minimumInterval: TimeInterval = CarPlayVoiceSessionPolicy.voiceStateActivationMinimumInterval
    ) -> CarPlayVoiceStateActivationDecision {
        if previousStateID == requestedStateID {
            return .skipDuplicate
        }
        guard let lastActivationUptime else {
            return .activateNow
        }
        let elapsed = max(0, nowUptime - lastActivationUptime)
        guard elapsed < minimumInterval else {
            return .activateNow
        }
        return .delay(max(0, minimumInterval - elapsed))
    }
}
