import Foundation
import Testing
@testable import Lumen

struct SourcesViewApprovalTests {
    @Test func reindexMenuRoutesThroughForegroundDestructiveConfirmation() throws {
        let source = try sourcesViewSource()

        #expect(source.contains("Button { pendingReindex = .files }"))
        #expect(source.contains("Button { pendingReindex = .photos }"))
        #expect(!source.contains("Button { reindexFiles() }"))
        #expect(!source.contains("Button { reindexPhotos() }"))
        #expect(source.contains(".confirmationDialog("))
        #expect(source.contains("Button(target.confirmationButtonTitle, role: .destructive)"))
        #expect(source.contains("confirmReindex(target)"))
        #expect(source.contains("clear and rebuild the existing local file index"))
        #expect(source.contains("clear and rebuild the existing local photo metadata index"))
    }

    @Test func unavailableReindexStorageKeepsTypedDiagnostic() throws {
        let source = try sourcesViewSource()
        let expected = "RAG storage unavailable. Diagnostic: swiftdata_shared_container_unavailable."

        #expect(source.components(separatedBy: expected).count - 1 == 2)
    }

    @Test func replacementStagingRowsAreExcludedFromVisibleCounts() throws {
        let source = try sourcesViewSource()

        #expect(source.contains("filter: #Predicate<RAGChunk>"))
        #expect(source.contains("!$0.sourceType.starts(with: \"__lumen_rag_replacement_staging__:\")"))
        #expect(source.contains("for c in chunks"))
        #expect(source.contains("Text(\"\\(chunks.count) chunks indexed\")"))
    }

    private func sourcesViewSource() throws -> String {
        let iosRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: iosRoot.appendingPathComponent("Lumen/Views/SourcesView.swift"),
            encoding: .utf8
        )
    }
}
