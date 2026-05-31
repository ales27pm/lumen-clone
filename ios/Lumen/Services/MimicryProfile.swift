import Foundation

nonisolated struct MimicryProfile: Sendable, Hashable {
    let state: String
    let style: String
    let constraints: [String]

    var promptFragment: String {
        let rules = constraints.map { "- \($0)" }.joined(separator: "\n")
        return """
        Mimicry v1 tone profile:
        - user state: \(state)
        - response style: \(style)
        \(rules)
        """
    }
}

nonisolated enum MimicryProfiler {
    static func profile(userMessage: String, settings: SettingsSnapshot) -> MimicryProfile {
        let lower = userMessage.lowercased()
        let state = detectedState(from: lower)
        let style = detectedStyle(from: lower, settings: settings)
        return MimicryProfile(
            state: state,
            style: style,
            constraints: constraints(for: state, style: style, settings: settings)
        )
    }

    private static func detectedState(from lower: String) -> String {
        if containsAny(lower, ["crash", "broken", "doesn't work", "does not work", "error", "failed", "fail", "stuck"]) {
            return "blocked-or-debugging"
        }
        if containsAny(lower, ["continue", "implement", "ship", "finish", "complete", "do it"]) {
            return "execution-focused"
        }
        if containsAny(lower, ["why", "explain", "understand", "what is missing", "how does"]) {
            return "analysis-focused"
        }
        return "neutral"
    }

    private static func detectedStyle(from lower: String, settings: SettingsSnapshot) -> String {
        if containsAny(lower, ["short", "quick", "concise", "no preamble", "direct"]) {
            return "direct-concise"
        }
        if containsAny(lower, ["detail", "exhaustive", "full", "complete", "blueprint"]) {
            return "structured-detailed"
        }
        if selectedPresetHint(from: settings) == "coder" {
            return "technical-precise"
        }
        return "calm-direct"
    }

    private static func constraints(for state: String, style: String, settings: SettingsSnapshot) -> [String] {
        var rules: [String] = [
            "Do not add ceremonial introductions.",
            "Prefer concrete next actions over vague reassurance.",
            "Do not claim background work; only report completed actions."
        ]

        if state == "blocked-or-debugging" {
            rules.append("Prioritize root cause, fix path, and verification.")
        }
        if state == "execution-focused" {
            rules.append("Report implementation progress tersely and keep momentum.")
        }
        if style == "direct-concise" {
            rules.append("Keep the final answer compact unless code or commands are required.")
        }
        if style == "structured-detailed" {
            rules.append("Use clear sections and preserve important implementation details.")
        }
        if !settings.agentModeEnabled {
            rules.append("Agent mode is disabled; do not imply autonomous tool execution.")
        }

        return rules
    }

    private static func selectedPresetHint(from settings: SettingsSnapshot) -> String {
        let prompt = settings.systemPrompt.lowercased()
        if prompt.contains("coder mode") { return "coder" }
        if prompt.contains("researcher mode") { return "researcher" }
        return "general"
    }

    private static func containsAny(_ value: String, _ needles: [String]) -> Bool {
        needles.contains { value.contains($0) }
    }
}
