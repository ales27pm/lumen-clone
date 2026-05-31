import XCTest
import SwiftData
@testable import Lumen

final class RAGSearchToolTests: XCTestCase {
    @MainActor func testLexicalFallbackAndDedupe() async {
        let schema = Schema([RAGChunk.self]); let container = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(container)
        ctx.insert(RAGChunk(content: "swift memory search", sourceType: .file, sourceName: "a"))
        ctx.insert(RAGChunk(content: "swift memory search", sourceType: .file, sourceName: "a"))
        try! ctx.save()
        let tool = RAGSearchTool()
        let inv = ToolInvocation(id: UUID(), toolID: "rag.search.secure", arguments: ["query":"swift","limit":"6"], source: .system, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await tool.execute(invocation: inv, context: .init(isForeground: true, appState: nil, modelContext: ctx, permissionRegistry: .shared, metricsStore: .shared))
        XCTAssertEqual(res.status, .success)
        XCTAssertEqual(res.structuredPayload?["mode"], "lexical")
        XCTAssertEqual(res.structuredPayload?["count"], "1")
        XCTAssertTrue(res.modelText.contains("swift memory search"))
    }
}
