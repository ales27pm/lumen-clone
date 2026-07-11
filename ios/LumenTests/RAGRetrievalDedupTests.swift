import XCTest
import SwiftData
@testable import Lumen

final class RAGRetrievalDedupTests: XCTestCase {
    func testEmbeddingModelIdentifierUsesArtifactContentsNotFilename() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("rag-model-identity-\(UUID().uuidString)")
        let firstDirectory = root.appendingPathComponent("first")
        let secondDirectory = root.appendingPathComponent("second")
        try FileManager.default.createDirectory(at: firstDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: secondDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let first = firstDirectory.appendingPathComponent("embedding.gguf")
        let second = secondDirectory.appendingPathComponent("embedding.gguf")
        try Data("model-a".utf8).write(to: first)
        try Data("model-b".utf8).write(to: second)

        let firstIdentifier = try RAGEmbeddingMetadata.modelIdentifier(forFileURL: first)
        let secondIdentifier = try RAGEmbeddingMetadata.modelIdentifier(forFileURL: second)
        XCTAssertNotEqual(firstIdentifier, secondIdentifier)

        try Data("model-a".utf8).write(to: second)
        XCTAssertEqual(firstIdentifier, try RAGEmbeddingMetadata.modelIdentifier(forFileURL: second))
    }

    @MainActor func testVectorIndexExcludesUnversionedEmbeddingChunks() {
        let current = RAGChunk(content: "current", sourceType: .note, sourceName: "current", embedding: [1, 0])
        let legacy = RAGChunk(
            content: "legacy",
            sourceType: .note,
            sourceName: "legacy",
            embedding: [1, 0],
            embeddingFormatVersion: 0,
            embeddingModelIdentifier: "",
            embeddingDimension: 0
        )
        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }

        let result = RAGVectorIndex.shared.ensureLoadedForTests { [current, legacy] }

        XCTAssertEqual(result.loadedCount, 1)
        XCTAssertEqual(RAGVectorIndex.shared.count, 1)
    }

    @MainActor func testVectorIndexAppendEnforcesLoadedEmbeddingMetadata() throws {
        let container = try ModelContainer(for: RAGChunk.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let context = ModelContext(container)
        let chunk = RAGChunk(content: "row", sourceType: .note, sourceName: "row", embedding: [1, 0])
        context.insert(chunk)
        let metadata = RAGEmbeddingIndexMetadata(
            formatVersion: SemanticEmbeddingText.formatVersion,
            modelIdentifier: "llama:sha256:abc",
            dimension: 2
        )
        RAGVectorIndex.shared.invalidate()
        defer { RAGVectorIndex.shared.invalidate() }
        _ = RAGVectorIndex.shared.ensureLoadedForTests(
            formatVersion: metadata.formatVersion,
            modelIdentifier: metadata.modelIdentifier,
            dimension: metadata.dimension,
            fetch: { [] }
        )

        XCTAssertEqual(RAGVectorIndex.shared.dimension, 2)
        XCTAssertTrue(RAGVectorIndex.shared.append(
            id: chunk.persistentModelID,
            bucket: chunk.sourceType,
            vector: chunk.embedding,
            metadata: metadata
        ))
        XCTAssertEqual(RAGVectorIndex.shared.count, 1)

        let mismatched = RAGEmbeddingIndexMetadata(
            formatVersion: metadata.formatVersion,
            modelIdentifier: "llama:sha256:different",
            dimension: metadata.dimension
        )
        XCTAssertFalse(RAGVectorIndex.shared.append(
            id: chunk.persistentModelID,
            bucket: chunk.sourceType,
            vector: chunk.embedding,
            metadata: mismatched
        ))
        XCTAssertEqual(RAGVectorIndex.shared.count, 0)
        XCTAssertEqual(RAGVectorIndex.shared.dimension, 0)
    }

    @MainActor func testDedupKeepsDistinctChunksWithSameExcerpt() async throws {
        let schema = Schema([RAGChunk.self]); let c = try ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(c)
        ctx.insert(RAGChunk(content: "same content", sourceType: .file, sourceName: "doc", sourceRef: "doc-a"))
        ctx.insert(RAGChunk(content: "same content", sourceType: .file, sourceName: "doc", sourceRef: "doc-a"))
        try ctx.save()
        let out = await RAGEngine().retrieve(query: "same", limit: 10, context: ctx)
        XCTAssertEqual(out.count, 2)
    }

    @MainActor func testRetrieveUsesFocusedExcerptAroundQueryTerms() async throws {
        let schema = Schema([RAGChunk.self])
        let c = try ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(c)
        let prefix = String(repeating: "background context ", count: 40)
        ctx.insert(RAGChunk(content: "\(prefix) executor preflight adapter contract should stay visible in the excerpt", sourceType: .file, sourceName: "runtime-notes", sourceRef: "runtime-notes"))
        try ctx.save()

        let out = await RAGEngine().retrieve(query: "adapter contract", limit: 1, context: ctx)

        XCTAssertEqual(out.count, 1)
        XCTAssertTrue(out[0].excerpt.contains("adapter contract"))
        XCTAssertTrue(out[0].retrievalMode.contains("local_rerank"))
        XCTAssertGreaterThan(out[0].offsetStart ?? 0, 0)
    }

    @MainActor func testRetrieveKeepsPublicLimitAfterExpandedCandidatePool() async throws {
        let schema = Schema([RAGChunk.self])
        let c = try ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(c)
        for index in 0..<8 {
            ctx.insert(RAGChunk(content: "adapter contract note \(index)", sourceType: .file, sourceName: "doc-\(index)", sourceRef: "doc-\(index)"))
        }
        try ctx.save()

        let out = await RAGEngine().retrieve(query: "adapter contract", limit: 3, context: ctx)

        XCTAssertEqual(out.count, 3)
    }
}
