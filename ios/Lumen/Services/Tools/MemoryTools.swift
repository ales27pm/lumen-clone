import Foundation
import SwiftData

@MainActor
enum MemoryTools {
    struct RAGIndexExecution: Sendable, Equatable {
        let text: String
        let status: ToolResultStatus
        let diagnostic: String?
    }

    static func save(content: String, kind: String) async -> String {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "Need content." }
        let k = MemoryKind(rawValue: kind) ?? .fact
        guard let container = SharedContainer.shared else { return "Memory unavailable." }
        let ctx = ModelContext(container)
        let result = await MemoryStore.rememberWithDiagnostics(trimmed, kind: k, source: "agent", context: ctx)
        return saveMessage(from: result)
    }

    static func recall(query: String) async -> String {
        guard let container = SharedContainer.shared else { return "Memory unavailable." }
        let ctx = ModelContext(container)
        let result = await MemoryStore.recallWithDiagnostics(query: query, context: ctx, limit: 5)
        let items = result.items
        if items.isEmpty {
            if result.mode == "failed", let diagnostic = result.diagnostic {
                return "Memory recall failed. Diagnostic: \(diagnostic)."
            }
            if let diagnostic = result.diagnostic, diagnostic != "empty_store" {
                return "No matching memories. Diagnostic: \(diagnostic)."
            }
            return result.diagnostic == "empty_store" ? "No memories saved yet." : "No matching memories."
        }
        return items.map { "• \($0.content)" }.joined(separator: "\n")
    }

    static func ragSearch(query: String, limit: Int) async -> String {
        let trimmed = query.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return "Need a search query." }
        guard let container = SharedContainer.shared else {
            return "RAG storage unavailable. Diagnostic: swiftdata_shared_container_unavailable."
        }
        let ctx = ModelContext(container)
        let expandedQuery = expandRAGQueryIfNeeded(trimmed)
        let retrieval = await RAGEngine().retrieveWithDiagnostics(
            query: expandedQuery,
            relevanceQuery: trimmed,
            limit: limit,
            context: ctx
        )
        let results = retrieval.results
        if results.isEmpty {
            if retrieval.mode == "failed" || isFailureDiagnostic(retrieval.diagnostic) {
                return "RAG search failed. Diagnostic: \(diagnosticText(retrieval.diagnostic))."
            }
            let countResult = RAGStore.countsWithDiagnostics(context: ctx)
            if countResult.mode == "failed" || isFailureDiagnostic(countResult.diagnostic) {
                return "RAG search failed while checking the index. Diagnostic: \(diagnosticText(countResult.diagnostic))."
            }
            let totalIndexed = countResult.counts.values.reduce(0, +)
            if totalIndexed == 0 {
                return "No matching files found. Your local index appears empty. Import or create local files/notes, then run reindex files."
            }
            if let diagnostic = retrieval.diagnostic {
                return "No matching files found. Search completed with diagnostic: \(diagnostic)."
            }
            return "No matching files found. Try a narrower query (file name, module name, or service/component keywords), or add more project notes before searching again."
        }
        return results.enumerated().map { idx, r in
            let kind = RAGSourceType(rawValue: r.source.type)?.label ?? r.source.type
            let src = "\(kind) · \(r.source.title)"
            return "[\(idx + 1)] \(src) · score \(String(format: "%.2f", r.score))\n\(r.excerpt)"
        }.joined(separator: "\n\n")
    }


    private static func expandRAGQueryIfNeeded(_ query: String) -> String {
        let lower = query.lowercased()
        let shouldExpand = ["architecture notes", "architecture", "module", "service", "component", "package"].contains { lower.contains($0) }
        guard shouldExpand else { return query }
        let expansionTerms = ["architecture", "module", "service", "component", "package"]
        return query + " " + expansionTerms.joined(separator: " ")
    }

    static func ragIndexFiles() async -> String {
        await ragIndexFilesExecution().text
    }

    static func ragIndexFilesExecution() async -> RAGIndexExecution {
        guard let container = SharedContainer.shared else {
            return RAGIndexExecution(
                text: "RAG storage unavailable. Diagnostic: swiftdata_shared_container_unavailable.",
                status: .unavailable,
                diagnostic: "swiftdata_shared_container_unavailable"
            )
        }
        let ctx = ModelContext(container)
        let embeddingReady = await RAGStore.embeddingRuntimeAvailable()
        guard embeddingReady else {
            return RAGIndexExecution(
                text: "RAG indexing failed: embedding model is unavailable. Load a local embedding model, then run reindex files.",
                status: .unavailable,
                diagnostic: "embedding_model_unavailable"
            )
        }
        let result = await RAGStore.indexImportedFilesWithDiagnostics(context: ctx)
        return ragIndexExecution(from: result, text: ragIndexFilesMessage(from: result))
    }

    static func ragIndexPhotos(months: Int) async -> String {
        await ragIndexPhotosExecution(months: months).text
    }

    static func ragIndexPhotosExecution(months: Int) async -> RAGIndexExecution {
        guard let container = SharedContainer.shared else {
            return RAGIndexExecution(
                text: "RAG storage unavailable. Diagnostic: swiftdata_shared_container_unavailable.",
                status: .unavailable,
                diagnostic: "swiftdata_shared_container_unavailable"
            )
        }
        let ctx = ModelContext(container)
        let embeddingReady = await RAGStore.embeddingRuntimeAvailable()
        guard embeddingReady else {
            return RAGIndexExecution(
                text: "RAG photo indexing failed: embedding model is unavailable. Load a local embedding model, then try again.",
                status: .unavailable,
                diagnostic: "embedding_model_unavailable"
            )
        }
        let result = await RAGStore.indexPhotosWithDiagnostics(monthsBack: max(1, months), context: ctx)
        return ragIndexExecution(from: result, text: ragIndexPhotosMessage(from: result))
    }

    static func ragIndexExecution(from result: RAGStore.IndexResult, text: String) -> RAGIndexExecution {
        let status: ToolResultStatus
        switch result.mode {
        case .indexed, .cleared:
            status = .success
        case .skipped:
            status = .unavailable
        case .partial, .failed:
            status = .failed
        }
        return RAGIndexExecution(text: text, status: status, diagnostic: result.diagnostic)
    }

    static func ragIndexFilesMessage(from result: RAGStore.IndexResult) -> String {
        switch result.mode {
        case .indexed:
            return "Indexed \(result.indexedCount) chunks from imported files."
        case .cleared:
            return "No imported files remain; the previous file index was cleared. Diagnostic: \(diagnosticText(result.diagnostic))."
        case .skipped:
            return "RAG indexing skipped. Diagnostic: \(diagnosticText(result.diagnostic))."
        case .partial:
            return "RAG indexing partially completed: indexed \(result.indexedCount) chunks. Diagnostic: \(diagnosticText(result.diagnostic))."
        case .failed:
            return "RAG indexing failed. Diagnostic: \(diagnosticText(result.diagnostic))."
        }
    }

    static func ragIndexPhotosMessage(from result: RAGStore.IndexResult) -> String {
        switch result.mode {
        case .indexed:
            return "Indexed \(result.indexedCount) monthly photo summaries."
        case .cleared:
            return "No photos remain in the selected range; the previous photo index was cleared. Diagnostic: \(diagnosticText(result.diagnostic))."
        case .skipped:
            return "RAG photo indexing skipped. Diagnostic: \(diagnosticText(result.diagnostic))."
        case .partial:
            return "RAG photo indexing partially completed: indexed \(result.indexedCount) monthly summaries. Diagnostic: \(diagnosticText(result.diagnostic))."
        case .failed:
            return "RAG photo indexing failed. Diagnostic: \(diagnosticText(result.diagnostic))."
        }
    }

    static func saveMessage(from result: MemoryStore.RememberResult) -> String {
        switch result.mode {
        case "stored":
            return "Saved memory."
        case "skipped" where result.diagnostic == "duplicate_memory":
            return "Memory already saved."
        case "skipped" where result.diagnostic == "empty_content":
            return "Need content."
        case "skipped":
            return "Memory save skipped. Diagnostic: \(diagnosticText(result.diagnostic))."
        case "failed":
            return "Memory save failed. Diagnostic: \(diagnosticText(result.diagnostic))."
        default:
            return "Memory save did not complete. Diagnostic: \(diagnosticText(result.diagnostic))."
        }
    }

    static func diagnosticText(_ diagnostic: String?) -> String {
        let trimmed = diagnostic?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? "unknown" : trimmed
    }

    private static func isFailureDiagnostic(_ diagnostic: String?) -> Bool {
        guard let diagnostic else { return false }
        return diagnostic.contains("fetch_failed")
            || diagnostic.contains("persist_failed")
            || diagnostic.contains("permission_denied")
            || diagnostic.contains("corrupt")
    }
}
