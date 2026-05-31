import Foundation
import SwiftData
#if canImport(AppIntents)
import AppIntents

@available(iOS 16.0, *)
struct LumenMemorySearchIntent: AppIntent {
    static var title: LocalizedStringResource = "Search Lumen Memory"
    static var openAppWhenRun = true

    @Parameter(title: "Query") var query: String
    @Parameter(title: "Limit", default: 5) var limit: Int

    @MainActor
    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard (1...300).contains(q.count) else { return .result(value: "Query must be 1...300 characters.") }
        let capped = max(1, min(limit, 10))
        guard let container = SharedContainer.shared else {
            return .result(value: LumenIntentResultRenderer.degraded("memory store unavailable"))
        }
        let ctx = ModelContext(container)
        let items = await MemoryEngine().search(query: q, limit: capped, context: ctx)
        let lines = items.prefix(capped).map { "- \($0.content.prefix(100))" }
        let out = lines.isEmpty ? "No memories found." : lines.joined(separator: "\n")
        return .result(value: String(out.prefix(700)))
    }
}
#endif
