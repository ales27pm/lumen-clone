import Foundation

nonisolated enum ModelThinkingControl {
    static let noHiddenReasoningOnlyInstruction = "Do not output hidden reasoning, <think> blocks, chain-of-thought, or internal analysis."
    static let noHiddenReasoningInstruction = "Do not output hidden reasoning, <think> blocks, chain-of-thought, or internal analysis. Return only the final answer."
    private static let reasoningCaptureInstruction = "If reasoning capture is enabled, put any internal reasoning inside <think>...</think> and put the final user-visible answer after </think>. Do not include hidden reasoning in the final answer text."

    static func developerInstruction(reasoningCaptureEnabled: Bool) -> String {
        reasoningCaptureEnabled ? reasoningCaptureInstruction : noHiddenReasoningInstruction
    }

    static func systemPrompt(_ base: String, reasoningCaptureEnabled: Bool, requireFinalAnswerOnly: Bool = true) -> String {
        let trimmed = base.trimmingCharacters(in: .whitespacesAndNewlines)
        let rule = reasoningCaptureEnabled
            ? reasoningCaptureInstruction
            : (requireFinalAnswerOnly ? noHiddenReasoningInstruction : noHiddenReasoningOnlyInstruction)
        guard !trimmed.lowercased().contains(rule.lowercased()) else { return base }
        guard !trimmed.isEmpty else { return rule }
        return "\(trimmed)\n\n\(rule)"
    }

    static func userMessage(_ base: String, reasoningCaptureEnabled: Bool, useQwenThinkingDirective: Bool) -> String {
        guard useQwenThinkingDirective else { return base }
        let directive = reasoningCaptureEnabled ? "/think" : "/no_think"
        let lower = base.lowercased()
        guard !lower.contains("/think"), !lower.contains("/no_think") else { return base }
        let trimmed = base.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return directive }
        return "\(trimmed)\n\n\(directive)"
    }
}
