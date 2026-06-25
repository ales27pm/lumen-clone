import XCTest
import SwiftData
import SwiftUI
@testable import Lumen

final class RAGSearchToolTests: XCTestCase {
    @MainActor func testLexicalFallbackAndDedupe() async {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let schema = Schema([RAGChunk.self]); let container = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(container)
        ctx.insert(RAGChunk(content: "swift memory search", sourceType: .file, sourceName: "a"))
        ctx.insert(RAGChunk(content: "swift memory search", sourceType: .file, sourceName: "a"))
        try! ctx.save()
        let tool = RAGSearchTool()
        let inv = ToolInvocation(id: UUID(), toolID: "rag.search.secure", arguments: ["query":"swift","limit":"6"], source: .system, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await tool.execute(invocation: inv, context: .init(isForeground: true, appState: nil, modelContext: ctx, permissionRegistry: .shared, metricsStore: .shared))
        XCTAssertEqual(res.status, .success)
        XCTAssertEqual(res.structuredPayload?["mode"], "lexical_fallback")
        XCTAssertEqual(res.structuredPayload?["count"], "1")
        XCTAssertTrue(res.modelText.contains("swift memory search"))
    }

    @MainActor func testRAGStoreFallsBackToLexicalWhenEmbeddingBudgetDenied() async {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let schema = Schema([RAGChunk.self])
        let container = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(container)
        ctx.insert(RAGChunk(content: "architecture module notes for local runtime", sourceType: .note, sourceName: "notes"))
        try? ctx.save()

        let result = await RAGStore.searchWithDiagnostics(query: "architecture module", context: ctx, limit: 3)

        XCTAssertEqual(result.mode, "lexical_fallback")
        XCTAssertEqual(result.diagnostic, "rag.search: lowPowerMode=true")
        XCTAssertEqual(result.matches.count, 1)
        XCTAssertEqual(result.matches.first?.chunk.sourceName, "notes")
    }
}
