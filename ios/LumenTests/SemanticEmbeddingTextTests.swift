import XCTest
@testable import Lumen

final class SemanticEmbeddingTextTests: XCTestCase {
    func testQueryUsesRetrievalTaskPrefix() {
        let text = SemanticEmbeddingText.query("  What did I write about Swift?\n")

        XCTAssertEqual(text, "search_query: What did I write about Swift?")
    }

    func testDocumentIncludesContextualMetadata() {
        let text = SemanticEmbeddingText.document(
            content: "Swift actors isolate mutable state.\nCode blocks keep shape.",
            sourceName: "Runtime Notes",
            sourceType: "note",
            chunkIndex: 2
        )

        XCTAssertTrue(text.hasPrefix("search_document:"))
        XCTAssertTrue(text.contains("Source type: note"))
        XCTAssertTrue(text.contains("Source name: Runtime Notes"))
        XCTAssertTrue(text.contains("Chunk index: 2"))
        XCTAssertTrue(text.contains("Content:\nSwift actors isolate mutable state.\nCode blocks keep shape."))
    }

    func testMemoryDocumentIncludesMemoryContext() {
        let text = SemanticEmbeddingText.memoryDocument(
            content: "I prefer concise bullet points.",
            kind: .preference,
            source: "manual",
            topic: "response style"
        )

        XCTAssertTrue(text.hasPrefix("search_document:"))
        XCTAssertTrue(text.contains("Memory kind: preference"))
        XCTAssertTrue(text.contains("Memory source: manual"))
        XCTAssertTrue(text.contains("Topic: response style"))
        XCTAssertTrue(text.contains("Content:\nI prefer concise bullet points."))
    }
}
