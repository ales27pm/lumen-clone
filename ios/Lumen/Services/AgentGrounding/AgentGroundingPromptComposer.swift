import Foundation

nonisolated enum AgentGroundingPromptComposer {
    static func composeSystemPrompt(for slot: LumenModelSlot, fallbackSystemPrompt: String) -> String {
        let fallback = fallbackSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)

        let store = BundledAgentGroundingStore(bundle: .main)
        guard let bundledPrompt = try? store.systemPrompt(for: slot.rawValue)
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !bundledPrompt.isEmpty else {
            return fallbackSystemPrompt
        }

        let loadedRuntimeGrounding = (try? store.loadRuntimeGroundingPrompt(maxCharacters: 6_000))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let runtimeGrounding = (loadedRuntimeGrounding?.isEmpty == false) ? loadedRuntimeGrounding : nil

        let groundedPrompt: String
        if let runtimeGrounding {
            groundedPrompt = """
            \(bundledPrompt)

            Bundled runtime grounding:
            \(runtimeGrounding)
            """
        } else {
            groundedPrompt = bundledPrompt
        }

        guard !fallback.isEmpty else {
            return groundedPrompt
        }

        return """
        \(groundedPrompt)

        Runtime caller context and user configuration:
        \(fallback)
        """
    }
}

extension GenerateRequest {
    nonisolated func groundingSystemPrompt(for slot: LumenModelSlot) -> GenerateRequest {
        if preservesRawStructuredAgentOutput {
            return self
        }
        let callerPrompt = hostSelfModel?.removing(from: systemPrompt) ?? systemPrompt
        let groundedPrompt = AgentGroundingPromptComposer.composeSystemPrompt(for: slot, fallbackSystemPrompt: callerPrompt)
        return GenerateRequest(
            id: id,
            sessionID: sessionID,
            systemPrompt: hostSelfModel?.injecting(into: groundedPrompt) ?? groundedPrompt,
            history: history,
            userMessage: userMessage,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            maxTokens: maxTokens,
            modelName: modelName,
            relevantMemories: relevantMemories,
            attachments: attachments,
            responseFormat: responseFormat,
            seed: seed,
            developerTraceModeEnabled: developerTraceModeEnabled,
            reasoningCaptureEnabled: reasoningCaptureEnabled,
            reasoningTraceBudgetCharacters: reasoningTraceBudgetCharacters,
            allowsMemoryPressureContinuation: allowsMemoryPressureContinuation,
            hostSelfModel: hostSelfModel
        )
    }
}
