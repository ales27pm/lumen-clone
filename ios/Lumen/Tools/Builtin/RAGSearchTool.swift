import Foundation
import SwiftData
import CryptoKit

struct RAGSearchTool: LocalTool {
    let definition = SecureToolDefinition(id: "rag.search.secure", displayName: "Search RAG", description: "Search indexed local chunks", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "{query:string,limit?:1...12,sourceScope?:all|documents|notes|photos,sourceFilter?:string,minimumScore?:0...1}", resultPrivacyLevel: .moderate, maxOutputCharacters: 1800)

    func validateArguments(_ arguments: [String : String]) throws { _ = try parse(arguments) }
    private func parse(_ a:[String:String]) throws -> (String,Int,RAGSourceScope,String?,Double?) {
        let q = (a["query"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard (1...500).contains(q.count) else { throw ToolExecutionError.invalidArguments("query") }
        let limit = Int(a["limit"] ?? "6") ?? 6
        guard (1...12).contains(limit) else { throw ToolExecutionError.invalidArguments("limit") }
        let rawScope = (a["sourceScope"] ?? RAGSourceScope.all.rawValue).trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let sourceScope = RAGSourceScope(rawValue: rawScope) else {
            throw ToolExecutionError.invalidArguments("sourceScope")
        }
        let source = a["sourceFilter"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let source, source.count > 120 { throw ToolExecutionError.invalidArguments("sourceFilter") }
        let min = a["minimumScore"].flatMap(Double.init)
        if let min, !(0...1).contains(min) { throw ToolExecutionError.invalidArguments("minimumScore") }
        return (q,limit,sourceScope,source?.isEmpty==true ? nil:source,min)
    }

    /// Executes a RAG search query and returns the results.
    /// - Returns: A tool result with the formatted search results, or status indicating unavailability or invalid arguments.
    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        do {
            let (q, limitRaw, sourceScope, source, minScore) = try parse(invocation.arguments)
            let limit = context.isForeground ? limitRaw : min(limitRaw, 6)
            guard let mc = context.modelContext else {
                return .init(
                    invocationID: invocation.id,
                    status: .unavailable,
                    displayText: "RAG storage unavailable.",
                    modelText: "RAG storage unavailable.",
                    structuredPayload: [
                        "diagnostic": "swiftdata_model_context_unavailable",
                        "mode": "no_model_context"
                    ],
                    privacyLevel: .moderate,
                    metricsSummary: "no_model_context",
                    errorCode: "swiftdata_model_context_unavailable"
                )
            }
            let output = await Self.searchRows(
                query: q,
                limit: limit,
                sourceScope: sourceScope,
                source: source,
                minScore: minScore,
                outputBudgetChars: definition.maxOutputCharacters,
                modelContext: mc
            )
            if let failureDiagnostic = output.failureDiagnostic {
                return .init(
                    invocationID: invocation.id,
                    status: .failed,
                    displayText: "RAG search failed.",
                    modelText: "RAG search failed.",
                    structuredPayload: ["diagnostic": failureDiagnostic, "mode": output.mode],
                    privacyLevel: .moderate,
                    metricsSummary: output.mode,
                    errorCode: "rag_search_failed"
                )
            }
            let rows = output.rows
            let mode = output.mode
            let txt: String
            if output.scopedCorpusEmpty {
                txt = sourceScope.emptyCorpusMessage(query: q)
            } else {
                txt = rows.isEmpty ? "No matching RAG chunks found." : rows.joined(separator: "\n")
            }
            var payload = output.diagnosticsPayload
            payload["mode"] = mode
            payload["count"] = "\(rows.count)"
            payload["sourceScope"] = sourceScope.rawValue
            return SafeToolOutputLimiter.limit(result: .init(invocationID: invocation.id, status: .success, displayText: txt, modelText: txt, structuredPayload: payload, privacyLevel: .moderate, metricsSummary: mode, errorCode: nil), maxOutput: definition.maxOutputCharacters)
        } catch {
            return .init(invocationID: invocation.id, status: .failed, displayText: "Invalid RAG query.", modelText: "RAG input invalid.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "invalid_args", errorCode: "invalid")
        }
    }

    /// Searches local RAG chunks and returns ranked, deduplicated results.
    /// - Returns: A tuple containing formatted chunk rows and the search mode ("semantic" or "lexical").
    @MainActor
    private static func searchRows(query: String, limit: Int, sourceScope: RAGSourceScope, source: String?, minScore: Double?, outputBudgetChars: Int, modelContext: ModelContext) async -> (rows: [String], mode: String, diagnosticsPayload: [String: String], failureDiagnostic: String?, scopedCorpusEmpty: Bool) {
        let retrieval = await RAGEngine().retrieveWithDiagnostics(
            query: query,
            limit: limit,
            sourceTypes: sourceScope.sourceTypes,
            context: modelContext
        )
        var results = retrieval.results
        let mode = retrieval.mode
        if let source {
            results = results.filter { $0.source.title.localizedCaseInsensitiveContains(source) || ($0.source.ref?.localizedCaseInsensitiveContains(source) ?? false) }
        }
        if let minScore { results = results.filter { $0.score >= minScore } }
        var failureDiagnostic = isFailureDiagnostic(retrieval.diagnostic) ? (retrieval.diagnostic ?? "unknown") : nil
        var scopedCorpusEmpty = false
        if results.isEmpty, let sourceTypes = sourceScope.sourceTypes {
            let counts = RAGStore.countsWithDiagnostics(context: modelContext)
            if counts.mode == "failed" || isFailureDiagnostic(counts.diagnostic) {
                failureDiagnostic = counts.diagnostic ?? "counts_failed"
            } else {
                scopedCorpusEmpty = sourceTypes.reduce(0) { $0 + (counts.counts[$1] ?? 0) } == 0
            }
        }

        var seen = Set<String>()
        let deduped = results.filter { item in
            let keyData = Data((item.source.title + item.excerpt).utf8)
            let key = SHA256.hash(data: keyData).compactMap { String(format: "%02x", $0) }.joined()
            let inserted = !seen.contains(key)
            if inserted { seen.insert(key) }
            return inserted
        }
        let context = RAGContextBuilder.build(results: Array(deduped.prefix(limit)), budgetChars: outputBudgetChars)

        let rows = context.selected.map { e in
            return "- [\(e.chunkID.uuidString.prefix(8))] \(e.source.title) | score=\(String(format:"%.2f", e.score)) | \(e.excerpt)"
        }
        var payload = diagnosticsPayload(for: context, dedupedCount: deduped.count)
        if let diagnostic = retrieval.diagnostic {
            payload["diagnostic"] = diagnostic
        }
        if scopedCorpusEmpty {
            payload["diagnostic"] = "scoped_index_empty:\(sourceScope.rawValue)"
        }
        return (rows, mode, payload, failureDiagnostic, scopedCorpusEmpty)
    }

    private static func diagnosticsPayload(for context: RAGContextResult, dedupedCount: Int) -> [String: String] {
        [
            "candidateCount": "\(context.candidateCount)",
            "dedupedCount": "\(dedupedCount)",
            "selectedSourceCount": "\(context.selectedSourceCount)",
            "diversityPassApplied": context.diversityPassApplied ? "true" : "false",
            "estimatedChars": "\(context.totalChars)",
            "estimatedTokens": "\(context.totalTokens)",
            "confidence": String(format: "%.2f", context.confidence)
        ]
    }

    private static func isFailureDiagnostic(_ diagnostic: String?) -> Bool {
        guard let diagnostic else { return false }
        return diagnostic.contains("fetch_failed")
            || diagnostic.contains("persist_failed")
            || diagnostic.contains("permission_denied")
            || diagnostic.contains("corrupt")
    }
}

private extension RAGSourceScope {
    func emptyCorpusMessage(query: String) -> String {
        switch self {
        case .all:
            return "No matching personal data found for '\(query)'. Your local index appears empty. Import or create local content, then reindex."
        case .documents:
            return "No matching documents found for '\(query)'. Your local document index appears empty. Import local files, then reindex."
        case .notes:
            return "No matching notes found for '\(query)'. Your local notes index appears empty. Create or import notes, then reindex."
        case .photos:
            return "No matching photo metadata found for '\(query)'. Your local photo index appears empty. Index the photo library, then try again."
        }
    }
}
