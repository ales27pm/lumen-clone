import Testing
@testable import Lumen

@MainActor
struct MemoryToolsTests {
    @Test func saveRejectsEmptyContent() async {
        let result = await MemoryTools.save(content: "   \n\t  ", kind: "fact")
        #expect(result == "Need content.")
    }

    @Test func saveMessageDoesNotEchoSavedContent() {
        let sensitiveContent = "my launch password is swordfish"
        let message = MemoryTools.saveMessage(from: MemoryStore.RememberResult(mode: "stored", diagnostic: nil))

        #expect(message == "Saved memory.")
        #expect(!message.contains(sensitiveContent))
    }

    @Test func saveMessageSurfacesSanitizedFailureDiagnostic() {
        let message = MemoryTools.saveMessage(
            from: MemoryStore.RememberResult(mode: "failed", diagnostic: "remember_failed:embeddingUnavailable")
        )

        #expect(message == "Memory save failed. Diagnostic: remember_failed:embeddingUnavailable.")
        #expect(!message.contains("No embedding model is currently loaded"))
    }

    @Test func saveMessageDistinguishesDuplicateAndEmptySkips() {
        let duplicate = MemoryTools.saveMessage(
            from: MemoryStore.RememberResult(mode: "skipped", diagnostic: "duplicate_memory")
        )
        let empty = MemoryTools.saveMessage(
            from: MemoryStore.RememberResult(mode: "skipped", diagnostic: "empty_content")
        )

        #expect(duplicate == "Memory already saved.")
        #expect(empty == "Need content.")
    }

    @Test func ragIndexFilesMessagePreservesFailureDiagnostic() {
        let result = RAGStore.IndexResult(indexedCount: 0, mode: .failed, diagnostic: "cleanup_persist_failed:test")
        let message = MemoryTools.ragIndexFilesMessage(from: result)
        #expect(message.contains("RAG indexing failed"))
        #expect(message.contains("cleanup_persist_failed:test"))
        #expect(!message.contains("no chunks were indexed"))
    }

    @Test func ragIndexPhotosMessageDistinguishesPermissionDeniedFromEmptyLibrary() {
        let denied = RAGStore.IndexResult(indexedCount: 0, mode: .failed, diagnostic: "photos_permission_denied:denied")
        let empty = RAGStore.IndexResult(indexedCount: 0, mode: .cleared, diagnostic: "empty_photo_library")

        #expect(MemoryTools.ragIndexPhotosMessage(from: denied).contains("photos_permission_denied:denied"))
        #expect(MemoryTools.ragIndexPhotosMessage(from: empty).contains("empty_photo_library"))
        #expect(MemoryTools.ragIndexPhotosMessage(from: empty).contains("cleared"))
        #expect(!MemoryTools.ragIndexPhotosMessage(from: empty).contains("skipped"))
    }

    @Test func ragIndexExecutionUsesTypedStatusInsteadOfStringHeuristics() {
        let indexed = RAGStore.IndexResult(indexedCount: 3, mode: .indexed, diagnostic: nil)
        let empty = RAGStore.IndexResult(indexedCount: 0, mode: .cleared, diagnostic: "no_imported_files")
        let deferred = RAGStore.IndexResult(indexedCount: 0, mode: .skipped, diagnostic: "cleanup_deferred:disk_write_budget_denied")
        let partial = RAGStore.IndexResult(indexedCount: 2, mode: .partial, diagnostic: "persist_failed")

        let indexedExecution = MemoryTools.ragIndexExecution(from: indexed, text: "indexed")
        let emptyExecution = MemoryTools.ragIndexExecution(from: empty, text: "empty")
        let deferredExecution = MemoryTools.ragIndexExecution(from: deferred, text: "deferred")
        let partialExecution = MemoryTools.ragIndexExecution(from: partial, text: "partial")

        #expect(indexedExecution.status == .success)
        #expect(indexedExecution.mode == .indexed)
        #expect(emptyExecution.status == .success)
        #expect(emptyExecution.mode == .cleared)
        #expect(deferredExecution.status == .unavailable)
        #expect(deferredExecution.mode == .skipped)
        #expect(partialExecution.status == .failed)
        #expect(partialExecution.mode == .partial)
    }
}
