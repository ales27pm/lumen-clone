import Foundation

nonisolated enum ModelThinkingControl {
    static let noHiddenReasoningOnlyInstruction = "Do not output hidden reasoning, <think> blocks, chain-of-thought, or internal analysis."
    static let noHiddenReasoningInstruction = "Do not output hidden reasoning, <think> blocks, chain-of-thought, or internal analysis. Return only the final answer."
    static let qwen3AssistantGenerationMarker = "<|im_start|>assistant\n"
    static let qwen3NonThinkingGenerationPrefix = "<think>\n\n</think>\n\n"
    private static let reasoningCaptureInstruction = "If reasoning capture is enabled, put any internal reasoning inside <think>...</think> and put the final user-visible answer after </think>. Do not include hidden reasoning in the final answer text."

    enum Qwen3PromptContractError: Error, Equatable, LocalizedError, Sendable {
        case missingControlledDirective
        case unexpectedAssistantGenerationSuffix

        var errorDescription: String? {
            switch self {
            case .missingControlledDirective:
                return "The final Qwen3 user message is missing its controlled thinking directive."
            case .unexpectedAssistantGenerationSuffix:
                return "The rendered Qwen3 prompt does not end at the controlled assistant generation boundary."
            }
        }
    }

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

    /// Preserves the byte-exact system prompt used by the Qwen3 training
    /// contract when non-thinking generation is already enforced by `/no_think`
    /// and the hard assistant prefix. Other models retain the explicit system
    /// instruction, and DEBUG reasoning capture still receives its developer
    /// trace instruction.
    static func runtimeSystemPrompt(
        _ base: String,
        reasoningCaptureEnabled: Bool,
        requireFinalAnswerOnly: Bool = true,
        useQwenNonThinkingContract: Bool
    ) -> String {
        if useQwenNonThinkingContract && !reasoningCaptureEnabled {
            return base
        }
        return systemPrompt(
            base,
            reasoningCaptureEnabled: reasoningCaptureEnabled,
            requireFinalAnswerOnly: requireFinalAnswerOnly
        )
    }

    /// Appends the selected Qwen thinking directive to a user message.
    /// - Parameters:
    ///   - reasoningCaptureEnabled: Determines the directive type in DEBUG builds.
    ///   - useQwenThinkingDirective: Whether to append the directive.
    /// - Returns: The user message with exactly one controlled terminal directive, or unchanged if appending is disabled.
    static func userMessage(_ base: String, reasoningCaptureEnabled: Bool, useQwenThinkingDirective: Bool) -> String {
        guard useQwenThinkingDirective else { return base }
        let directive = thinkingDirective(reasoningCaptureEnabled: reasoningCaptureEnabled)
        let trimmed = strippingTerminalThinkingDirective(from: base)
        guard !trimmed.isEmpty else { return directive }
        return "\(trimmed)\n\n\(directive)"
    }

    /// Reconstructs the hard `enable_thinking=false` generation prefix that the
    /// pinned Qwen3 tokenizer emits during SFT, DPO, and frozen evaluation. The
    /// bundled SwiftLlama/llama.cpp template API cannot pass template kwargs, so
    /// the shared Qwen3 runtime must add the prefix after template rendering and
    /// before tokenization.
    static func finalizeQwen3Prompt(
        _ renderedPrompt: String,
        finalUserMessage: String?
    ) throws -> String {
        guard let finalUserMessage,
              let terminalDirective = terminalThinkingDirective(in: finalUserMessage) else {
            throw Qwen3PromptContractError.missingControlledDirective
        }

        #if DEBUG
        if terminalDirective == "/think" {
            return renderedPrompt
        }
        #endif

        guard terminalDirective == "/no_think" else {
            throw Qwen3PromptContractError.missingControlledDirective
        }
        let controlledSuffix = qwen3AssistantGenerationMarker + qwen3NonThinkingGenerationPrefix
        if renderedPrompt.hasSuffix(controlledSuffix) {
            return renderedPrompt
        }
        guard renderedPrompt.hasSuffix(qwen3AssistantGenerationMarker) else {
            throw Qwen3PromptContractError.unexpectedAssistantGenerationSuffix
        }
        return renderedPrompt + qwen3NonThinkingGenerationPrefix
    }

    private static func strippingTerminalThinkingDirective(from value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let directive = terminalThinkingDirective(in: trimmed) else { return trimmed }
        let directiveStart = trimmed.index(trimmed.endIndex, offsetBy: -directive.count)
        return String(trimmed[..<directiveStart]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func terminalThinkingDirective(in value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()
        for directive in ["/no_think", "/think"] {
            guard lower.hasSuffix(directive) else { continue }
            let start = lower.index(lower.endIndex, offsetBy: -directive.count)
            guard start == lower.startIndex || lower[lower.index(before: start)].isWhitespace else {
                continue
            }
            return directive
        }
        return nil
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
