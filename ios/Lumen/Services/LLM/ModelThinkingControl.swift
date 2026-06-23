import Foundation

nonisolated enum ModelThinkingControl {
    static let noHiddenReasoningOnlyInstruction = "Do not output hidden reasoning, <think> blocks, chain-of-thought, or internal analysis."
    static let noHiddenReasoningInstruction = "Do not output hidden reasoning, <think> blocks, chain-of-thought, or internal analysis. Return only the final answer."
    private static let reasoningCaptureInstruction = "If reasoning capture is enabled, put any internal reasoning inside <think>...</think> and put the final user-visible answer after </think>. Do not include hidden reasoning in the final answer text."

    #if DEBUG
    static func developerInstruction(reasoningCaptureEnabled: Bool) -> String {
        return reasoningCaptureEnabled ? reasoningCaptureInstruction : noHiddenReasoningInstruction
    }
    #else
    static func developerInstruction(reasoningCaptureEnabled _: Bool) -> String {
        return noHiddenReasoningInstruction
    }
    #endif

    /// Constructs a system prompt by appending reasoning control instructions to a base prompt.
    /// - Parameters:
    ///   - base: The base prompt text.
    ///   - reasoningCaptureEnabled: When `true` in DEBUG builds, enables reasoning capture; ignored in non-DEBUG builds.
    ///   - requireFinalAnswerOnly: Determines the instruction set: when `true`, requires only the final answer; when `false`, allows hidden reasoning without requiring a final answer. Defaults to `true`.
    /// - Returns: The base prompt with reasoning control instructions appended, or the base unchanged if it already contains the selected instructions.
    static func systemPrompt(_ base: String, reasoningCaptureEnabled: Bool, requireFinalAnswerOnly: Bool = true) -> String {
        let trimmed = base.trimmingCharacters(in: .whitespacesAndNewlines)
        let rule = thinkingRule(
            reasoningCaptureEnabled: reasoningCaptureEnabled,
            requireFinalAnswerOnly: requireFinalAnswerOnly
        )
        guard !trimmed.lowercased().contains(rule.lowercased()) else { return base }
        guard !trimmed.isEmpty else { return rule }
        return "\(trimmed)\n\n\(rule)"
    }

    /// Appends a Qwen thinking directive to a user message if requested and not already present.
    /// - Parameters:
    ///   - reasoningCaptureEnabled: Determines the directive type in DEBUG builds.
    ///   - useQwenThinkingDirective: Whether to append the directive.
    /// - Returns: The user message with a directive appended, or unchanged if the directive already exists or appending is disabled.
    static func userMessage(_ base: String, reasoningCaptureEnabled: Bool, useQwenThinkingDirective: Bool) -> String {
        guard useQwenThinkingDirective else { return base }
        let directive = thinkingDirective(reasoningCaptureEnabled: reasoningCaptureEnabled)
        let lower = base.lowercased()
        guard !lower.contains("/think"), !lower.contains("/no_think") else { return base }
        let trimmed = base.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return directive }
        return "\(trimmed)\n\n\(directive)"
    }

    #if DEBUG
    private static func thinkingRule(reasoningCaptureEnabled: Bool, requireFinalAnswerOnly: Bool) -> String {
        if reasoningCaptureEnabled && requireFinalAnswerOnly {
            return reasoningCaptureInstruction
        }
        return requireFinalAnswerOnly ? noHiddenReasoningInstruction : noHiddenReasoningOnlyInstruction
    }

    private static func thinkingDirective(reasoningCaptureEnabled: Bool) -> String {
        reasoningCaptureEnabled ? "/think" : "/no_think"
    }
    #else
    private static func thinkingRule(reasoningCaptureEnabled _: Bool, requireFinalAnswerOnly: Bool) -> String {
        requireFinalAnswerOnly ? noHiddenReasoningInstruction : noHiddenReasoningOnlyInstruction
    }

    private static func thinkingDirective(reasoningCaptureEnabled _: Bool) -> String {
        "/no_think"
    }
    #endif
}
