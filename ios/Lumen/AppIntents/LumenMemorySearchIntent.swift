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
        let result = await MemoryEngine().searchWithDiagnostics(query: q, limit: capped, context: ctx)
        return .result(value: Self.renderSearchResult(result, limit: capped))
    }

    static func renderSearchResult(_ result: MemoryEngine.SearchResult, limit: Int) -> String {
        let capped = max(1, min(limit, 10))
        let diagnostic = sanitizedDiagnostic(result.diagnostic)
        if result.mode == "failed" {
            return LumenIntentResultRenderer.degraded("memory search failed\(diagnosticSuffix(diagnostic))")
        }

        let lines = result.items.prefix(capped).map { "- \($0.content.prefix(100))" }
        if lines.isEmpty {
            guard let diagnostic, !isEmptyMemoryDiagnostic(diagnostic) else {
                return "No memories found."
            }
            return LumenIntentResultRenderer.degraded("memory search degraded\(diagnosticSuffix(diagnostic))")
        }

        var output = lines.joined(separator: "\n")
        if let diagnostic, !isEmptyMemoryDiagnostic(diagnostic) {
            output += "\nSearch diagnostic: \(diagnostic)"
        }
        return String(output.prefix(700))
    }

    private static func diagnosticSuffix(_ diagnostic: String?) -> String {
        guard let diagnostic, !diagnostic.isEmpty else { return "." }
        return ": \(diagnostic)."
    }

    private static func sanitizedDiagnostic(_ diagnostic: String?) -> String? {
        guard let diagnostic = diagnostic?.trimmingCharacters(in: .whitespacesAndNewlines),
              !diagnostic.isEmpty else { return nil }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_-:;."))
        let sanitized = String(diagnostic.unicodeScalars.map { allowed.contains($0) ? Character($0) : "_" })
        return String(sanitized.prefix(160))
    }

    private static func isEmptyMemoryDiagnostic(_ diagnostic: String) -> Bool {
        diagnostic == "empty_store" || diagnostic == "empty_query" || diagnostic == "lexical_empty_query"
    }
}
#endif
