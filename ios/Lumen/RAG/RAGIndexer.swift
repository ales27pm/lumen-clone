import Foundation
import SwiftData

@MainActor
final class RAGIndexer {
    func indexText(
        source: RAGSource,
        title: String,
        text: String,
        metadata: [String:String],
        context: ModelContext,
        embed: (EmbeddingRequest) async throws -> EmbeddingRuntimeResult = { request in
            try await AssistantKernel.runEmbeddingWithIdentity(text: request.text, purpose: request.purpose, dimensions: request.dimensions)
        }
    ) async throws -> Int {
        guard let writeEpoch = EmbeddingIndexLifecycle.captureWrite() else {
            throw RAGIndexingError.embeddingIndexReset
        }
        let type = RAGSourceType(rawValue: source.type) ?? .note
        let chunks = ChunkingStrategy.chunk(text, type: .plain)
        var pending: [RAGChunk] = []
        var activeEmbeddingMetadata: RAGEmbeddingIndexMetadata?
        for (i,c) in chunks.enumerated() {
            guard EmbeddingIndexLifecycle.permitsWrite(writeEpoch) else {
                throw RAGIndexingError.embeddingIndexReset
            }
            let embeddingResult: EmbeddingRuntimeResult
            do {
                let embeddingText = SemanticEmbeddingText.documentBody(
                    content: c.text,
                    sourceName: title,
                    sourceType: type.rawValue,
                    chunkIndex: i
                )
                embeddingResult = try await embed(EmbeddingRequest(text: embeddingText, purpose: .retrievalDocument(title: title)))
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                guard EmbeddingIndexLifecycle.permitsWrite(writeEpoch) else {
                    throw RAGIndexingError.embeddingIndexReset
                }
                throw RAGIndexingError.embeddingFailed
            }
            try Task.checkCancellation()
            guard EmbeddingIndexLifecycle.permitsWrite(writeEpoch) else {
                throw RAGIndexingError.embeddingIndexReset
            }
            let embedding = embeddingResult.vector
            guard !embedding.isEmpty else { throw RAGIndexingError.emptyEmbedding }
            let embeddingMetadata = RAGEmbeddingIndexMetadata(
                formatVersion: embeddingResult.formatVersion,
                modelIdentifier: embeddingResult.modelIdentifier,
                dimension: embedding.count
            )
            if let activeEmbeddingMetadata, activeEmbeddingMetadata != embeddingMetadata {
                RAGVectorIndex.shared.invalidate()
                pending.removeAll(keepingCapacity: true)
                throw RAGIndexingError.embeddingIdentityChanged
            }
            if activeEmbeddingMetadata == nil {
                _ = RAGVectorIndex.shared.ensureLoaded(context: context, metadata: embeddingMetadata)
                activeEmbeddingMetadata = embeddingMetadata
            }
            let chunk = RAGChunk(
                content: c.text,
                sourceType: type,
                sourceName: title,
                sourceRef: source.ref,
                chunkIndex: i,
                embedding: embedding,
                embeddingFormatVersion: embeddingMetadata.formatVersion,
                embeddingModelIdentifier: embeddingMetadata.modelIdentifier,
                embeddingDimension: embeddingMetadata.dimension
            )
            pending.append(chunk)
        }
        try Task.checkCancellation()
        guard EmbeddingIndexLifecycle.permitsWrite(writeEpoch) else {
            throw RAGIndexingError.embeddingIndexReset
        }
        for chunk in pending {
            context.insert(chunk)
        }
        do {
            try context.save()
        } catch {
            for chunk in pending {
                context.delete(chunk)
            }
            throw error
        }
        var appendedAll = true
        for chunk in pending where !chunk.embedding.isEmpty {
            guard RAGVectorIndex.shared.append(
                id: chunk.persistentModelID,
                bucket: chunk.sourceType,
                vector: chunk.embedding,
                metadata: RAGEmbeddingIndexMetadata(
                    formatVersion: chunk.embeddingFormatVersion,
                    modelIdentifier: chunk.embeddingModelIdentifier,
                    dimension: chunk.embeddingDimension
                )
            ) else {
                appendedAll = false
                break
            }
        }
        if !appendedAll {
            guard let activeEmbeddingMetadata else {
                throw RAGIndexingError.vectorIndexUnavailable
            }
            RAGVectorIndex.shared.invalidate()
            let reload = RAGVectorIndex.shared.ensureLoaded(context: context, metadata: activeEmbeddingMetadata)
            guard reload.mode != "failed" else {
                throw RAGIndexingError.vectorIndexUnavailable
            }
        }
        return pending.count
    }
}

enum RAGIndexingError: Error, Sendable, Equatable {
    case embeddingFailed
    case emptyEmbedding
    case embeddingIdentityChanged
    case embeddingIndexReset
    case vectorIndexUnavailable
}
