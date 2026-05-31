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
}
