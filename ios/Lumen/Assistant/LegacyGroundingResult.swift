import Foundation

struct LegacyGroundingResult: Sendable {
    let systemPrompt: String
    let userMessage: String
    let grounding: AssistantGroundingContext?
    let sections: [PromptGroundingSection]
    let bridgedTools: [ToolDefinition]
    let degradedReasons: [String]
    let metricsSummary: String
    let truncationOccurred: Bool
}
