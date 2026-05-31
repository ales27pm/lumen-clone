import Foundation
import SwiftData

@MainActor
final class RAGIndexer {
    func indexText(source: RAGSource, title: String, text: String, metadata: [String:String], context: ModelContext) async throws -> Int {
        let type = RAGSourceType(rawValue: source.type) ?? .note
        let chunks = ChunkingStrategy.chunk(text, type: .plain)
        var pending: [RAGChunk] = []
        for (i,c) in chunks.enumerated() {
            let embedding: [Double]
            do {
                embedding = try await AppLlamaService.shared.embed(c.text)
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                throw RAGIndexingError.embeddingFailed
            }
            guard !embedding.isEmpty else { throw RAGIndexingError.emptyEmbedding }
            let chunk = RAGChunk(content: c.text, sourceType: type, sourceName: title, sourceRef: source.ref, chunkIndex: i, embedding: embedding)
            context.insert(chunk)
            pending.append(chunk)
        }
        try context.save()
        for chunk in pending where !chunk.embedding.isEmpty {
            RAGVectorIndex.shared.append(id: chunk.persistentModelID, bucket: chunk.sourceType, vector: chunk.embedding)
        }
        return pending.count
    }
}

enum RAGIndexingError: Error, Sendable, Equatable {
    case embeddingFailed
    case emptyEmbedding
}
