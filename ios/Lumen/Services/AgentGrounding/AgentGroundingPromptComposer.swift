import Foundation

nonisolated enum AgentGroundingPromptComposer {
    static func composeSystemPrompt(for slot: LumenModelSlot, fallbackSystemPrompt: String) -> String {
        let fallback = fallbackSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let bundledPrompt = try? BundledAgentGroundingStore(bundle: .main).systemPrompt(for: slot.rawValue)
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !bundledPrompt.isEmpty else {
            return fallbackSystemPrompt
        }

        guard !fallback.isEmpty else {
            return bundledPrompt
        }

        return """
        \(bundledPrompt)

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
