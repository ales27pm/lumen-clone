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
        let drain = await MemoryCaptureQueue.drain(context: ctx, maxItems: 3, allowPromotion: true)
        do {
            try await MemoryStore.remember(body, kind: .fact, source: "app-intent", topic: nil, context: ctx)
            return .result(value: Self.savedMessage(drained: drain.promoted))
        } catch {
            do {
                let queued = try MemoryCaptureQueue.enqueue(content: body, kind: .fact, source: "app-intent-pending")
                let pending = (try? MemoryCaptureQueue.pendingCount()) ?? 1
                return .result(value: Self.queuedMessage(pendingCount: pending, retryCount: queued.retryCount))
            } catch {
                return .result(value: LumenIntentResultRenderer.degraded("memory capture failed"))
            }
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

    static func savedMessage(drained: Int) -> String {
        guard drained > 0 else { return "Memory saved." }
        return "Memory saved. Also indexed \(drained) pending memory capture\(drained == 1 ? "" : "s")."
    }

    static func queuedMessage(pendingCount: Int, retryCount: Int) -> String {
        let retrySuffix = retryCount > 0 ? " Retry count: \(retryCount)." : ""
        return "Memory captured locally for later indexing. Pending captures: \(max(1, pendingCount)).\(retrySuffix)"
    }
}
#endif
