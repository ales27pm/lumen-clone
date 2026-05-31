import Foundation
import SwiftData
#if canImport(AppIntents)
import AppIntents

@available(iOS 16.0, *)
struct LumenAskIntent: AppIntent {
    static var title: LocalizedStringResource = "Ask Lumen"
    static var description = IntentDescription("Ask Lumen for a bounded local answer.")
    static var openAppWhenRun = false

    @Parameter(title: "Question") var question: String

    @MainActor
    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let trimmed = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .result(value: "Question is required.") }
        guard trimmed.count <= 1_000 else { return .result(value: "Question is too long (max 1000 characters).") }
        guard let container = SharedContainer.shared else {
            return .result(value: LumenIntentResultRenderer.degraded("model context unavailable"))
        }
        let ctx = ModelContext(container)
        let settings = SettingsSnapshot.loadFromDisk()
        let result = await AgentRunner.runHeadless(prompt: trimmed, settings: settings, context: ctx, maxSteps: min(2, settings.maxAgentSteps))
        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let bounded = String((text.isEmpty ? LumenIntentResultRenderer.degraded("no response") : text).prefix(500))
        return .result(value: bounded)
    }
}
#endif
