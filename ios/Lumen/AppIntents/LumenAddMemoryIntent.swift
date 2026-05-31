import Foundation
import SwiftData
#if canImport(AppIntents)
import AppIntents

@available(iOS 16.0, *)
struct LumenAddMemoryIntent: AppIntent {
    static var title: LocalizedStringResource = "Add Lumen Memory"
    static var openAppWhenRun = false

    @Parameter(title: "Memory Text") var text: String

    @MainActor
    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let body = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !body.isEmpty, body.count <= 1000 else { return .result(value: "Memory text must be 1...1000 characters.") }
        if let policyMessage = Self.policyMessage(for: body) {
            return .result(value: policyMessage)
        }
        guard let container = SharedContainer.shared else {
            return .result(value: LumenIntentResultRenderer.degraded("memory store unavailable"))
        }
        let ctx = ModelContext(container)
        let candidate = MemoryCandidate(text: body, kind: "fact", topics: [], conversationID: nil, messageID: UUID(), createdAt: Date(), confidence: 0.7, extractionReason: "app-intent", userExplicitness: .explicitPreference, sensitivity: .normal)
        let score = MemoryScorer.score(candidate: candidate)
        guard score.decision == .save else {
            return .result(value: "Memory not saved: did not meet save policy.")
        }
        do {
            try await MemoryStore.remember(body, kind: .fact, source: "app-intent", topic: nil, context: ctx)
            return .result(value: "Memory saved.")
        } catch {
            return .result(value: LumenIntentResultRenderer.degraded("memory save failed"))
        }
    }

    static func policyMessage(for body: String) -> String? {
        let lower = body.lowercased()
        if lower.contains("password") || lower.contains("api key") || lower.contains("secret") {
            return "Memory rejected: credential-like content is not allowed."
        }
        if lower.contains("medical") || lower.contains("legal") || lower.contains("bank") || lower.contains("financial") {
            return LumenIntentResultRenderer.openAppRequired("sensitive memory requires in-app approval")
        }
        return nil
    }
}
#endif
