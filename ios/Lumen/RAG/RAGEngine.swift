import Foundation
import SwiftData

struct RAGMaintenanceResult: Sendable, Equatable {
    let success: Bool
    let metricSummary: String
}

@MainActor
final class RAGEngine {
    private let indexer = RAGIndexer()

    func retrieve(query: String, limit: Int, context: ModelContext) async -> [RAGRetrievalResult] {
        let semantic = await RAGStore.search(query: query, context: context, limit: limit)
        let mapped = semantic.map { item in
            let ref = item.chunk.sourceRef ?? item.chunk.id.uuidString
            return RAGRetrievalResult(chunkID: item.chunk.id, source: .init(id: ref, type: item.chunk.sourceType, title: item.chunk.sourceName, ref: item.chunk.sourceRef), excerpt: String(item.chunk.content.prefix(220)), score: item.score, retrievalMode: "semantic", offsetStart: nil, offsetEnd: nil)
        }
        var seen = Set<String>()
        return mapped.filter { result in
            let key = "\(result.source.id)#\(result.chunkID.uuidString)"
            if seen.contains(key) { return false }
            seen.insert(key)
            return true
        }
    }

    func buildContext(query: String, budget: Int, context: ModelContext) async -> RAGContextResult {
        let r = await retrieve(query: query, limit: 12, context: context)
        return RAGContextBuilder.build(results: r, budgetChars: budget)
    }

    func index(source: RAGSource, title: String, text: String, metadata: [String:String], context: ModelContext) async throws -> Int {
        try await indexer.indexText(source: source, title: title, text: text, metadata: metadata, context: context)
    }

    func maintenance(context: ModelContext) async -> RAGMaintenanceResult {
        do {
            var descriptor = FetchDescriptor<RAGChunk>()
            descriptor.fetchLimit = 1
            let hasChunks = try !context.fetch(descriptor).isEmpty
            return .init(success: true, metricSummary: hasChunks ? "maintenance_success_work_done" : "maintenance_success_empty")
        } catch {
            return .init(success: false, metricSummary: "maintenance_failed")
        }
    }
}
