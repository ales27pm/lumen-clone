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
        XCTAssertEqual(res.structuredPayload?["mode"], "lexical_fallback+local_rerank")
        XCTAssertEqual(res.structuredPayload?["count"], "1")
        XCTAssertEqual(res.structuredPayload?["dedupedCount"], "1")
        XCTAssertEqual(res.structuredPayload?["selectedSourceCount"], "1")
        XCTAssertEqual(res.structuredPayload?["diversityPassApplied"], "false")
        XCTAssertNotNil(res.structuredPayload?["estimatedTokens"])
        XCTAssertNotNil(res.structuredPayload?["confidence"])
        XCTAssertTrue(res.modelText.contains("swift memory search"))
    }

    @MainActor func testToolUsesFocusedRerankedExcerpt() async {
        let schema = Schema([RAGChunk.self])
        let container = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(container)
        let prefix = String(repeating: "general notes ", count: 35)
        ctx.insert(RAGChunk(content: "\(prefix) adapter contract diagnostics should remain visible", sourceType: .file, sourceName: "runtime"))
        try! ctx.save()

        let tool = RAGSearchTool()
        let inv = ToolInvocation(id: UUID(), toolID: "rag.search.secure", arguments: ["query":"adapter contract","limit":"1"], source: .system, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await tool.execute(invocation: inv, context: .init(isForeground: true, appState: nil, modelContext: ctx, permissionRegistry: .shared, metricsStore: .shared))

        XCTAssertEqual(res.status, .success)
        XCTAssertTrue(res.structuredPayload?["mode"]?.contains("local_rerank") == true)
        XCTAssertNotNil(res.structuredPayload?["estimatedChars"])
        XCTAssertNotNil(res.structuredPayload?["estimatedTokens"])
        XCTAssertNotNil(res.structuredPayload?["confidence"])
        XCTAssertTrue(res.modelText.contains("adapter contract"))
    }

    @MainActor func testToolPayloadReportsSourceDiversity() async {
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
        ctx.insert(RAGChunk(content: "adapter contract diagnostics alpha", sourceType: .file, sourceName: "runtime-a", sourceRef: "runtime-a"))
        ctx.insert(RAGChunk(content: "adapter contract diagnostics beta", sourceType: .file, sourceName: "runtime-a", sourceRef: "runtime-a"))
        ctx.insert(RAGChunk(content: "adapter contract diagnostics gamma", sourceType: .file, sourceName: "runtime-a", sourceRef: "runtime-a"))
        ctx.insert(RAGChunk(content: "adapter contract diagnostics delta", sourceType: .file, sourceName: "runtime-b", sourceRef: "runtime-b"))
        try! ctx.save()

        let tool = RAGSearchTool()
        let inv = ToolInvocation(id: UUID(), toolID: "rag.search.secure", arguments: ["query":"adapter contract diagnostics","limit":"4"], source: .system, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await tool.execute(invocation: inv, context: .init(isForeground: true, appState: nil, modelContext: ctx, permissionRegistry: .shared, metricsStore: .shared))

        XCTAssertEqual(res.status, .success)
        XCTAssertEqual(res.structuredPayload?["selectedSourceCount"], "2")
        XCTAssertEqual(res.structuredPayload?["diversityPassApplied"], "true")
        XCTAssertEqual(res.structuredPayload?["dedupedCount"], "4")
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
