import Foundation
import SwiftData
import CryptoKit

struct RAGSearchTool: LocalTool {
    let definition = SecureToolDefinition(id: "rag.search.secure", displayName: "Search RAG", description: "Search indexed local chunks", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "{query:string,limit?:1...12,sourceFilter?:string,minimumScore?:0...1}", resultPrivacyLevel: .moderate, maxOutputCharacters: 1800)

    func validateArguments(_ arguments: [String : String]) throws { _ = try parse(arguments) }
    private func parse(_ a:[String:String]) throws -> (String,Int,String?,Double?) {
        let q = (a["query"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard (1...500).contains(q.count) else { throw ToolExecutionError.invalidArguments("query") }
        let limit = Int(a["limit"] ?? "6") ?? 6
        guard (1...12).contains(limit) else { throw ToolExecutionError.invalidArguments("limit") }
        let source = a["sourceFilter"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let source, source.count > 120 { throw ToolExecutionError.invalidArguments("sourceFilter") }
        let min = a["minimumScore"].flatMap(Double.init)
        if let min, !(0...1).contains(min) { throw ToolExecutionError.invalidArguments("minimumScore") }
        return (q,limit,source?.isEmpty==true ? nil:source,min)
    }

    /// Executes a RAG search query and returns the results.
    /// - Returns: A tool result with the formatted search results, or status indicating unavailability or invalid arguments.
    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        do {
            let (q, limitRaw, source, minScore) = try parse(invocation.arguments)
            let limit = context.isForeground ? limitRaw : min(limitRaw, 6)
            guard let mc = context.modelContext else { return .init(invocationID: invocation.id, status: .unavailable, displayText: "RAG storage unavailable.", modelText: "RAG unavailable.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "no_model_context", errorCode: "unavailable") }
            let output = await Self.searchRows(query: q, limit: limit, source: source, minScore: minScore, modelContext: mc)
            let rows = output.rows
            let mode = output.mode
            let txt = rows.isEmpty ? "No matching RAG chunks found." : rows.joined(separator: "\n")
            return SafeToolOutputLimiter.limit(result: .init(invocationID: invocation.id, status: .success, displayText: txt, modelText: txt, structuredPayload: ["mode": mode, "count": "\(rows.count)"], privacyLevel: .moderate, metricsSummary: mode, errorCode: nil), maxOutput: definition.maxOutputCharacters)
        } catch {
            return .init(invocationID: invocation.id, status: .failed, displayText: "Invalid RAG query.", modelText: "RAG input invalid.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "invalid_args", errorCode: "invalid")
        }
    }

    /// Searches local RAG chunks and returns ranked, deduplicated results.
    /// - Returns: A tuple containing formatted chunk rows and the search mode ("semantic" or "lexical").
    @MainActor
    private static func searchRows(query: String, limit: Int, source: String?, minScore: Double?, modelContext: ModelContext) async -> (rows: [String], mode: String) {
        let engine = RAGEngine()
        let semanticResults = await engine.retrieve(query: query, limit: limit, context: modelContext)
        var results: [(chunk: RAGChunk, score: Double)] = []
        var mode = "semantic"
        if !semanticResults.isEmpty {
            let chunks = (try? modelContext.fetch(FetchDescriptor<RAGChunk>())) ?? []
            let map = Dictionary(uniqueKeysWithValues: chunks.map { ($0.id, $0) })
            results = semanticResults.compactMap { r in map[r.chunkID].map { (chunk: $0, score: r.score) } }
        }
        if results.isEmpty {
            mode = "lexical"
            let all = (try? modelContext.fetch(FetchDescriptor<RAGChunk>())) ?? []
            let terms = query.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init)
            results = all.compactMap { c in
                if let source, !(c.sourceName.localizedCaseInsensitiveContains(source) || (c.sourceRef?.localizedCaseInsensitiveContains(source) ?? false)) { return nil }
                let text = c.content.lowercased()
                let hits = terms.filter { text.contains($0) }.count
                guard hits > 0 else { return nil }
                return (chunk: c, score: Double(hits) / Double(max(1, terms.count)))
            }.sorted { $0.score > $1.score }.prefix(limit).map { $0 }
        }
        if let source {
            results = results.filter { $0.chunk.sourceName.localizedCaseInsensitiveContains(source) || ($0.chunk.sourceRef?.localizedCaseInsensitiveContains(source) ?? false) }
        }
        if let minScore { results = results.filter { $0.score >= minScore } }

        var seen = Set<String>()
        let dedup = results.filter { item in
            let keyData = Data((item.chunk.sourceName + item.chunk.content).utf8)
            let key = SHA256.hash(data: keyData).compactMap { String(format: "%02x", $0) }.joined()
            let inserted = !seen.contains(key)
            if inserted { seen.insert(key) }
            return inserted
        }.prefix(limit)

        let rows = dedup.map { e in
            let excerpt = e.chunk.content.count > 140 ? String(e.chunk.content.prefix(140)) + "..." : e.chunk.content
            return "- [\(e.chunk.id.uuidString.prefix(8))] \(e.chunk.sourceName) | score=\(String(format:"%.2f", e.score)) | \(excerpt)"
        }
        return (rows, mode)
    }
}
