import XCTest
import SwiftData
import SwiftUI
@testable import Lumen

final class RAGSearchToolTests: XCTestCase {
    func testExplicitRAGSourceOwnershipOutranksGenericQueryTerms() {
        XCTAssertEqual(
            RAGSourceScope.inferred(fromUserPrompt: "Search my files for architecture notes."),
            .documents
        )
        XCTAssertEqual(
            RAGSourceScope.inferred(fromUserPrompt: "Search my notes for a PDF reference."),
            .notes
        )
        XCTAssertEqual(
            RAGSourceScope.inferred(fromUserPrompt: "Search my photos and report what you find."),
            .photos
        )
    }

    func testPostRerankRelevanceRejectsUnanchoredMiklasoupReportCandidates() {
        let query = "Search my local files for Lumen architecture notes and summarize key modules."
        let title = "Les Principes Fondamentaux de la Miklasoup.pdf"
        let excerpts = [
            "Les particules illustrent la nature fluide de la réalité à l'échelle quantique.",
            "L'intrication quantique souligne la réalité non-locale de l'univers.",
            "Cette soupe est un creuset spirituel où matière et esprit se mélangent.",
            "La diversité infinie contribue positivement à l'harmonie universelle.",
            "L'ancienne Égypte et le Livre des Morts évoquent la continuité de l'existence.",
            "L'expansion de la Miklasoup traverse les mythes et les philosophies."
        ]
        let scores = [0.42, 0.41, 0.38, 0.37, 0.37, 0.37]

        let relevant = zip(excerpts, scores).filter { candidate in
            RAGEngine.isPostRerankRelevant(
                query: query,
                content: candidate.0,
                title: title,
                rerankedScore: candidate.1
            )
        }

        XCTAssertTrue(relevant.isEmpty)
        XCTAssertFalse(
            RAGEngine.isPostRerankRelevant(
                query: query,
                content: "This local file contains notes and a report.",
                title: "Latest local document",
                rerankedScore: 0.42
            )
        )
    }

    func testPostRerankRelevanceRejectsTermsAddedOnlyByQueryExpansion() {
        let originalQuery = "Search my files for Lumen architecture notes."
        let expansionOnlyCandidate = "Service component module package overview."

        XCTAssertFalse(
            RAGEngine.isPostRerankRelevant(
                query: originalQuery,
                content: expansionOnlyCandidate,
                title: "Unrelated engineering glossary",
                rerankedScore: 0.24
            )
        )
        XCTAssertTrue(
            RAGEngine.isPostRerankRelevant(
                query: originalQuery + " module service component package",
                content: expansionOnlyCandidate,
                title: "Unrelated engineering glossary",
                rerankedScore: 0.24
            )
        )
    }

    func testPostRerankRelevancePreservesHighConfidenceSemanticMatchWithoutLiteralAnchor() {
        XCTAssertTrue(
            RAGEngine.isPostRerankRelevant(
                query: "architecture modules",
                content: "The system design separates orchestration, persistence, and native execution boundaries.",
                title: "Engineering overview",
                rerankedScore: 0.64
            )
        )
    }

    func testPostRerankRelevancePreservesLowConfidenceAnchoredLexicalMatch() {
        XCTAssertTrue(
            RAGEngine.isPostRerankRelevant(
                query: "Search my local files for Lumen architecture notes and summarize key modules.",
                content: "The architecture defines the assistant kernel and storage boundaries.",
                title: "Local design notes",
                rerankedScore: 0.24
            )
        )
    }

    func testPostRerankRelevancePreservesConservativeLexicalVariants() {
        XCTAssertTrue(
            RAGEngine.isPostRerankRelevant(
                query: "module",
                content: "The runtime modules are isolated by responsibility.",
                title: "Engineering notes",
                rerankedScore: 0.18
            )
        )
        XCTAssertTrue(
            RAGEngine.isPostRerankRelevant(
                query: "architect",
                content: "The architecture separates persistence from execution.",
                title: "Engineering notes",
                rerankedScore: 0.18
            )
        )
        XCTAssertFalse(
            RAGEngine.isPostRerankRelevant(
                query: "app",
                content: "An apple orchard inventory.",
                title: "Unrelated notes",
                rerankedScore: 0.18
            )
        )
    }

    func testPostRerankRelevancePreservesBoundaryMatchedAcronymQueries() {
        XCTAssertTrue(
            RAGEngine.isPostRerankRelevant(
                query: "UI",
                content: "The UI uses a compact navigation hierarchy.",
                title: "Design notes",
                rerankedScore: 0.18
            )
        )
        XCTAssertFalse(
            RAGEngine.isPostRerankRelevant(
                query: "UI",
                content: "The build pipeline is stable.",
                title: "Release notes",
                rerankedScore: 0.18
            )
        )
        XCTAssertTrue(
            RAGEngine.isPostRerankRelevant(
                query: "AI",
                content: "A semantically related passage with no literal acronym.",
                title: "Research notes",
                rerankedScore: 0.64
            )
        )
        XCTAssertTrue(
            RAGEngine.isPostRerankRelevant(
                query: "Show me UI docs",
                content: "The UI uses a compact navigation hierarchy.",
                title: "Design notes",
                rerankedScore: 0.18
            )
        )
        XCTAssertFalse(
            RAGEngine.isPostRerankRelevant(
                query: "Show me UI docs",
                content: "The build pipeline is stable.",
                title: "Release notes",
                rerankedScore: 0.18
            )
        )
    }

    @MainActor func testMaintenanceIgnoresReplacementStagingRows() async throws {
        let schema = Schema([RAGChunk.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let context = ModelContext(container)
        let staged = RAGChunk(content: "pending", sourceType: .file, sourceName: "pending")
        staged.sourceType = RAGChunk.replacementStagingSourceType(id: UUID(), kind: .file)
        context.insert(staged)
        try context.save()

        let stagingOnly = await RAGEngine().maintenance(context: context)
        XCTAssertEqual(stagingOnly, .init(success: true, metricSummary: "maintenance_success_empty"))

        context.insert(RAGChunk(content: "active", sourceType: .file, sourceName: "active"))
        try context.save()
        let activeCorpus = await RAGEngine().maintenance(context: context)
        XCTAssertEqual(activeCorpus, .init(success: true, metricSummary: "maintenance_success_work_done"))
    }

    @MainActor func testMissingModelContextReportsSwiftDataDiagnostic() async {
        let tool = RAGSearchTool()
        let inv = ToolInvocation(
            id: UUID(),
            toolID: "rag.search.secure",
            arguments: ["query": "architecture notes", "limit": "3"],
            source: .system,
            conversationID: nil,
            turnID: nil,
            createdAt: Date()
        )

        let res = await tool.execute(
            invocation: inv,
            context: .init(
                isForeground: true,
                appState: nil,
                modelContext: nil,
                permissionRegistry: .shared,
                metricsStore: .shared
            )
        )

        XCTAssertEqual(res.status, .unavailable)
        XCTAssertEqual(res.displayText, "RAG storage unavailable.")
        XCTAssertFalse(res.displayText.contains("swiftdata_model_context_unavailable"))
        XCTAssertEqual(res.modelText, "RAG storage unavailable.")
        XCTAssertFalse(res.modelText.contains("swiftdata_model_context_unavailable"))
        XCTAssertEqual(res.errorCode, "swiftdata_model_context_unavailable")
        XCTAssertEqual(res.structuredPayload?["diagnostic"], "swiftdata_model_context_unavailable")
        XCTAssertTrue(res.modelText.contains("RAG storage unavailable"))
    }

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

    @MainActor func testToolQueryExpansionDoesNotBecomeRelevanceAnchor() async throws {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let schema = Schema([RAGChunk.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let context = ModelContext(container)
        context.insert(RAGChunk(
            content: "Service component package overview.",
            sourceType: .file,
            sourceName: "Unrelated engineering glossary"
        ))
        try context.save()

        let rawQuery = "Search my files for Lumen architecture notes."
        let expandedQuery = RAGEngine.expandedSearchQuery(rawQuery)
        XCTAssertNotEqual(expandedQuery, rawQuery)
        XCTAssertTrue(expandedQuery.contains("service"))
        XCTAssertTrue(expandedQuery.contains("component"))
        XCTAssertTrue(expandedQuery.contains("package"))

        let invocation = ToolInvocation(
            id: UUID(),
            toolID: "rag.search.secure",
            arguments: ["query": rawQuery, "limit": "3", "sourceScope": "documents"],
            source: .system,
            conversationID: nil,
            turnID: nil,
            createdAt: Date()
        )
        let result = await RAGSearchTool().execute(
            invocation: invocation,
            context: .init(
                isForeground: true,
                appState: nil,
                modelContext: context,
                permissionRegistry: .shared,
                metricsStore: .shared
            )
        )

        XCTAssertEqual(result.status, .success)
        XCTAssertEqual(result.structuredPayload?["count"], "0")
        XCTAssertEqual(result.modelText, "No matching RAG chunks found.")
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

    @MainActor func testDocumentScopeExcludesPhotoMetadataBeforeRetrieval() async throws {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let schema = Schema([RAGChunk.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "architecture module contract", sourceType: .file, sourceName: "architecture.txt"))
        context.insert(RAGChunk(content: "architecture module contract", sourceType: .photo, sourceName: "Photos 2026-07"))
        try context.save()

        let invocation = ToolInvocation(
            id: UUID(),
            toolID: "rag.search.secure",
            arguments: ["query": "architecture module", "sourceScope": "documents"],
            source: .system,
            conversationID: nil,
            turnID: nil,
            createdAt: Date()
        )
        let result = await RAGSearchTool().execute(
            invocation: invocation,
            context: .init(isForeground: true, appState: nil, modelContext: context, permissionRegistry: .shared, metricsStore: .shared)
        )

        XCTAssertEqual(result.status, .success)
        XCTAssertEqual(result.structuredPayload?["sourceScope"], "documents")
        XCTAssertTrue(result.modelText.contains("architecture.txt"))
        XCTAssertFalse(result.modelText.contains("Photos 2026-07"))
    }

    @MainActor func testDocumentScopeReportsEmptyCorpusInsteadOfPhotoResult() async throws {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let schema = Schema([RAGChunk.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "architecture module contract", sourceType: .photo, sourceName: "Photos 2026-07"))
        try context.save()

        let invocation = ToolInvocation(
            id: UUID(),
            toolID: "rag.search.secure",
            arguments: ["query": "architecture notes", "sourceScope": "documents"],
            source: .system,
            conversationID: nil,
            turnID: nil,
            createdAt: Date()
        )
        let result = await RAGSearchTool().execute(
            invocation: invocation,
            context: .init(isForeground: true, appState: nil, modelContext: context, permissionRegistry: .shared, metricsStore: .shared)
        )

        XCTAssertEqual(result.status, .success)
        XCTAssertEqual(result.structuredPayload?["diagnostic"], "scoped_index_empty:documents")
        XCTAssertTrue(result.modelText.contains("local document index appears empty"))
        XCTAssertFalse(result.modelText.contains("Photos 2026-07"))
    }

    @MainActor func testPhotoScopeReportsPhotoSpecificEmptyCorpusGuidance() async throws {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let schema = Schema([RAGChunk.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let context = ModelContext(container)
        context.insert(RAGChunk(content: "architecture module contract", sourceType: .file, sourceName: "architecture.txt"))
        try context.save()

        let invocation = ToolInvocation(
            id: UUID(),
            toolID: "rag.search.secure",
            arguments: ["query": "summer trip", "sourceScope": "photos"],
            source: .system,
            conversationID: nil,
            turnID: nil,
            createdAt: Date()
        )
        let result = await RAGSearchTool().execute(
            invocation: invocation,
            context: .init(isForeground: true, appState: nil, modelContext: context, permissionRegistry: .shared, metricsStore: .shared)
        )

        XCTAssertEqual(result.status, .success)
        XCTAssertEqual(result.structuredPayload?["diagnostic"], "scoped_index_empty:photos")
        XCTAssertTrue(result.modelText.contains("photo index appears empty"))
        XCTAssertTrue(result.modelText.contains("Index the photo library"))
        XCTAssertFalse(result.modelText.contains("Import local files"))
    }

    @MainActor func testAllScopeReportsEmptyCorpusGuidanceWhenLocalIndexIsEmpty() async throws {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let schema = Schema([RAGChunk.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let context = ModelContext(container)
        let invocation = ToolInvocation(
            id: UUID(),
            toolID: "rag.search.secure",
            arguments: ["query": "architecture notes", "sourceScope": "all"],
            source: .system,
            conversationID: nil,
            turnID: nil,
            createdAt: Date()
        )

        let result = await RAGSearchTool().execute(
            invocation: invocation,
            context: .init(isForeground: true, appState: nil, modelContext: context, permissionRegistry: .shared, metricsStore: .shared)
        )

        XCTAssertEqual(result.status, .success)
        XCTAssertEqual(result.structuredPayload?["diagnostic"], "scoped_index_empty:all")
        XCTAssertTrue(result.modelText.contains("local index appears empty"))
        XCTAssertTrue(result.modelText.contains("Import or create local content"))
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

    @MainActor func testHybridMergeLetsStrongLexicalMatchRescueWeakSemanticHit() async throws {
        let schema = Schema([RAGChunk.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(container)
        let weakSemantic = RAGChunk(content: "general architecture notes", sourceType: .note, sourceName: "semantic")
        let exactLexical = RAGChunk(content: "swift actor isolation details", sourceType: .note, sourceName: "lexical")
        ctx.insert(weakSemantic)
        ctx.insert(exactLexical)
        try ctx.save()

        let merged = RAGStore.hybridMergedCandidates(
            semantic: [(weakSemantic, 0.12)],
            lexical: [(exactLexical, 0.2)],
            limit: 2
        )

        XCTAssertEqual(merged.first?.0.sourceName, "lexical")
        XCTAssertEqual(merged.count, 2)
    }

    @MainActor func testRAGEngineRetrievePreservesEmptyLimitDiagnostic() async throws {
        let schema = Schema([RAGChunk.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(container)

        let result = await RAGEngine().retrieveWithDiagnostics(query: "architecture", limit: 0, context: ctx)

        XCTAssertTrue(result.results.isEmpty)
        XCTAssertEqual(result.mode, "empty_limit")
        XCTAssertEqual(result.diagnostic, "empty_limit")
    }

    @MainActor func testRAGEngineRetrievePreservesEmbeddingFallbackDiagnostic() async throws {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let schema = Schema([RAGChunk.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(container)

        let result = await RAGEngine().retrieveWithDiagnostics(query: "no matches here", limit: 3, context: ctx)

        XCTAssertTrue(result.results.isEmpty)
        XCTAssertEqual(result.mode, "lexical_fallback")
        XCTAssertEqual(result.diagnostic, "rag.search: lowPowerMode=true")
    }
}
