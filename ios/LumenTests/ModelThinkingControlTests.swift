import Foundation
import Testing
@testable import Lumen

struct ModelThinkingControlTests {
    @Test func qwen3NonThinkingRuntimePreservesTrainingSystemPromptBytes() {
        let trainingPrompt = "You are Executor, Lumen's structured routing executor."

        let runtimePrompt = ModelThinkingControl.runtimeSystemPrompt(
            trainingPrompt,
            reasoningCaptureEnabled: false,
            requireFinalAnswerOnly: false,
            useQwenNonThinkingContract: true
        )

        #expect(runtimePrompt == trainingPrompt)
    }

    @Test func nonQwenRuntimeRetainsExplicitReasoningControl() {
        let runtimePrompt = ModelThinkingControl.runtimeSystemPrompt(
            "System",
            reasoningCaptureEnabled: false,
            useQwenNonThinkingContract: false
        )

        #expect(runtimePrompt.contains(ModelThinkingControl.noHiddenReasoningInstruction))
    }

    @Test func nonThinkingDirectiveIsCanonicalAndTerminal() {
        let controlled = ModelThinkingControl.userMessage(
            "Explain the literal /think switch.\n\n/think",
            reasoningCaptureEnabled: false,
            useQwenThinkingDirective: true
        )

        #expect(controlled == "Explain the literal /think switch.\n\n/no_think")
    }

    @Test func existingNonThinkingDirectiveIsIdempotent() {
        let controlled = ModelThinkingControl.userMessage(
            "Return JSON.\n\n/no_think\n",
            reasoningCaptureEnabled: false,
            useQwenThinkingDirective: true
        )

        #expect(controlled == "Return JSON.\n\n/no_think")
    }

    @Test func qwen3HardNonThinkingPrefixMatchesTrainingPrompt() throws {
        let rendered = "<|im_start|>user\nHello\n\n/no_think<|im_end|>\n<|im_start|>assistant\n"

        let finalized = try ModelThinkingControl.finalizeQwen3Prompt(
            rendered,
            finalUserMessage: "Hello\n\n/no_think"
        )

        #expect(finalized == rendered + "<think>\n\n</think>\n\n")
        #expect(finalized.hasSuffix(ModelThinkingControl.qwen3NonThinkingGenerationPrefix))
    }

    @Test func qwen3HardNonThinkingPrefixIsIdempotent() throws {
        let rendered = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

        let finalized = try ModelThinkingControl.finalizeQwen3Prompt(
            rendered,
            finalUserMessage: "/no_think"
        )

        #expect(finalized == rendered)
    }

    @Test func qwen3PromptRejectsMissingControlledDirective() {
        #expect(throws: ModelThinkingControl.Qwen3PromptContractError.missingControlledDirective) {
            try ModelThinkingControl.finalizeQwen3Prompt(
                "<|im_start|>assistant\n",
                finalUserMessage: "Hello"
            )
        }
    }

    @Test func qwen3PromptRejectsUnexpectedTemplateSuffix() {
        #expect(throws: ModelThinkingControl.Qwen3PromptContractError.unexpectedAssistantGenerationSuffix) {
            try ModelThinkingControl.finalizeQwen3Prompt(
                "<|im_start|>assistant",
                finalUserMessage: "Hello\n\n/no_think"
            )
        }
    }
}
