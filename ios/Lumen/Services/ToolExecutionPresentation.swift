import Foundation

struct ToolExecutionPresentation: Equatable {
    let status: ToolStatus
    let message: String

    static func presentation(for rawToolID: String, rawResult: String) -> ToolExecutionPresentation {
        let canonicalToolID = ToolRouteGuard.canonicalToolID(rawToolID)
        let trimmedRaw = rawResult.trimmingCharacters(in: .whitespacesAndNewlines)
        let redacted = ToolArgumentRedactor.redactDisplayContent(rawResult)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let visible = redacted.isEmpty && !trimmedRaw.isEmpty ? "(no displayable result)" : redacted

        if let authMessage = outlookAuthenticationMessage(for: canonicalToolID, result: visible) {
            return ToolExecutionPresentation(status: .failed, message: authMessage)
        }

        if isFailureResult(canonicalToolID: canonicalToolID, result: visible) {
            return ToolExecutionPresentation(status: .failed, message: visible)
        }

        return ToolExecutionPresentation(status: .completed, message: visible)
    }

    private static func isFailureResult(canonicalToolID: String, result: String) -> Bool {
        let lowered = result.lowercased()
        if lowered.isEmpty { return false }

        let failurePrefixes = [
            "outlook tool failed:",
            "tool failed:",
            "unknown tool:",
            "missing ",
            "i need ",
            "denied by user",
            "calendar event creation requires explicit user approval",
            "this tool requires explicit user approval"
        ]
        if failurePrefixes.contains(where: { lowered.hasPrefix($0) }) { return true }

        let outlookFailureSignatures = [
            " not signed in",
            " sign in first",
            "authorization failed",
            "authentication failed",
            "unauthorized or expired",
            "permission denied",
            "requires explicit user approval",
            "access to do that",
            "invalid_grant",
            "interaction_required",
            "consent_required",
            "invalid_scope",
            "aadsts70000"
        ]
        if canonicalToolID.hasPrefix("outlook."), outlookFailureSignatures.contains(where: { lowered.contains($0) }) {
            return true
        }

        return false
    }

    static func outlookAuthenticationMessage(for canonicalToolID: String, result: String) -> String? {
        guard canonicalToolID.hasPrefix("outlook.") else { return nil }
        let lowered = result.lowercased()
        let isAuthFailure = lowered.contains("aadsts70000")
            || lowered.contains("unauthorized or expired")
            || lowered.contains("must first sign in")
            || lowered.contains("grant the client application access")
            || lowered.contains("invalid_grant")
            || lowered.contains("interaction_required")
            || lowered.contains("consent_required")
            || lowered.contains("invalid_scope")

        guard isAuthFailure else { return nil }
        return "Outlook needs to be reconnected before this can run. Sign in again and grant the requested mail permission, then retry. No Outlook email was sent."
    }
}

enum ToolArgumentRedactor {
    private static let sensitiveKeys = [
        "pendingActionID",
        "pending_action_id",
        "pendingAction",
        "approvalRequestID",
        "approval_request_id",
        "access_token",
        "accessToken",
        "refresh_token",
        "refreshToken",
        "authorization"
    ]

    static func redactDisplayContent(_ content: String) -> String {
        var output = content
        for key in sensitiveKeys {
            output = removeLooseKeyValuePair(key: key, from: output)
            output = removeEmbeddedJSONSensitivePair(key: key, from: output)
        }
        return normalizeSeparators(output).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func removeLooseKeyValuePair(key: String, from content: String) -> String {
        let escapedKey = NSRegularExpression.escapedPattern(for: key)
        let uuidOrToken = #"[A-Za-z0-9_./+=:-]{8,}"#
        let patterns = [
            #"(?i)(^|,\s*|\n\s*)"# + escapedKey + #"\s*[:=]\s*"# + uuidOrToken + #"\s*(?=,|\n|$)"#,
            #"(?i)(^|,\s*|\n\s*)"# + escapedKey + #"\s*[:=]\s*"[^"\n]*"\s*(?=,|\n|$)"#
        ]
        return patterns.reduce(content) { partial, pattern in
            partial.replacingOccurrences(of: pattern, with: "$1", options: .regularExpression)
        }
    }

    private static func removeEmbeddedJSONSensitivePair(key: String, from content: String) -> String {
        let escapedKey = NSRegularExpression.escapedPattern(for: key)
        let quotedKey = "\"" + escapedKey + "\""
        let patterns = [
            "(?i),?\\s*" + quotedKey + "\\s*:\\s*\"[^\"]*\"",
            "(?i),?\\s*" + quotedKey + "\\s*:\\s*[^,}\\]]+"
        ]
        return patterns.reduce(content) { partial, pattern in
            partial.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
        }
    }

    private static func normalizeSeparators(_ content: String) -> String {
        content
            .replacingOccurrences(of: #",\s*,"#, with: ",", options: .regularExpression)
            .replacingOccurrences(of: #"\n\s*\n\s*\n+"#, with: "\n\n", options: .regularExpression)
            .replacingOccurrences(of: "{,", with: "{")
            .replacingOccurrences(of: "[,", with: "[")
            .replacingOccurrences(of: ",}", with: "}")
            .replacingOccurrences(of: ",]", with: "]")
    }
}
