import Foundation
import SwiftData

struct MemorySearchTool: LocalTool {
    struct Args { let query: String; let limit: Int; let pinnedOnly: Bool; let topic: String? }

    let definition = SecureToolDefinition(id: "memory.search", displayName: "Search Memory", description: "Search local memory items", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "{query:string,limit?:1...20,includePinnedOnly?:bool,topic?:string}", resultPrivacyLevel: .moderate, maxOutputCharacters: 1800)

    func validateArguments(_ arguments: [String : String]) throws {
        _ = try parse(arguments)
    }

    private func parse(_ a: [String: String]) throws -> Args {
        let query = (a["query"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard (1...300).contains(query.count) else { throw ToolExecutionError.invalidArguments("query must be 1...300 chars") }
        let limit = Int(a["limit"] ?? "8") ?? 8
        guard (1...20).contains(limit) else { throw ToolExecutionError.invalidArguments("limit must be 1...20") }
        let pinnedOnly = (a["includePinnedOnly"] ?? "false").lowercased() == "true"
        let topic = a["topic"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let topic, topic.count > 80 { throw ToolExecutionError.invalidArguments("topic too long") }
        return Args(query: query, limit: limit, pinnedOnly: pinnedOnly, topic: topic?.isEmpty == true ? nil : topic)
    }

    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        do {
            let args = try parse(invocation.arguments)
            guard let mc = context.modelContext else { return .init(invocationID: invocation.id, status: .unavailable, displayText: "Memory storage unavailable.", modelText: "Memory search unavailable.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "no_model_context", errorCode: "unavailable") }
            let engine = MemoryEngine()
            let now = Date()
            let initial = await engine.search(query: args.query, limit: args.limit * 2, context: mc)
            let filtered = initial.filter { item in
                if args.pinnedOnly && !item.isPinned { return false }
                if let topic = args.topic, !(item.topic?.localizedCaseInsensitiveContains(topic) ?? false) { return false }
                return !MemoryStore.isExpired(item, now: now)
            }
            let q = args.query.lowercased()
            let scored = filtered.map { item -> (MemoryItem, Double) in
                let text = item.content.lowercased()
                var s = 0.0
                if text == q { s += 5 }
                if text.hasPrefix(q) { s += 2 }
                if text.contains(q) { s += 1 }
                if item.isPinned { s += 0.5 }
                return (item, s)
            }.sorted { $0.1 > $1.1 }.prefix(args.limit)

            let rows = scored.map { item, score in
                let excerpt = item.content.count > 120 ? String(item.content.prefix(120)) + "…" : item.content
                return "- [\(item.id.uuidString.prefix(8))] \(excerpt) | kind=\(item.kind) | score=\(String(format: "%.2f", score)) | source=\(item.source)"
            }
            let text = rows.isEmpty ? "No matching memories found." : rows.joined(separator: "\n")
            return SafeToolOutputLimiter.limit(result: .init(invocationID: invocation.id, status: .success, displayText: text, modelText: text, structuredPayload: ["count": "\(rows.count)"], privacyLevel: .moderate, metricsSummary: "lexical", errorCode: nil), maxOutput: definition.maxOutputCharacters)
        } catch let e as ToolExecutionError {
            return .init(invocationID: invocation.id, status: .failed, displayText: "Invalid memory query.", modelText: "Memory search input invalid.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "invalid_args", errorCode: "\(e)")
        } catch {
            return .init(invocationID: invocation.id, status: .failed, displayText: "Memory search failed.", modelText: "Memory search failed.", structuredPayload: nil, privacyLevel: .moderate, metricsSummary: "failed", errorCode: "failed")
        }
    }
}
