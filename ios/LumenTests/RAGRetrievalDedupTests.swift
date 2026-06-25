import XCTest
import SwiftData
@testable import Lumen

final class RAGRetrievalDedupTests: XCTestCase {
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
