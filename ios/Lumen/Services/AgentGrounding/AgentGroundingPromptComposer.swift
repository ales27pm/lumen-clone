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
        GenerateRequest(
            id: id,
            sessionID: sessionID,
            systemPrompt: AgentGroundingPromptComposer.composeSystemPrompt(
                for: slot,
                fallbackSystemPrompt: systemPrompt
            ),
            history: history,
            userMessage: userMessage,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            maxTokens: maxTokens,
            modelName: modelName,
            relevantMemories: relevantMemories,
            attachments: attachments,
            seed: seed,
            developerTraceModeEnabled: developerTraceModeEnabled,
            reasoningCaptureEnabled: reasoningCaptureEnabled,
            reasoningTraceBudgetCharacters: reasoningTraceBudgetCharacters
        )
    }
}
