import XCTest
@testable import Lumen

final class RAGContextBuilderTests: XCTestCase {
    func testBudgetedSelection() {
        let src = RAGSource(id: "s", type: "file", title: "t", ref: nil)
        let list = (0..<5).map { RAGRetrievalResult(chunkID: UUID(), source: src, excerpt: String(repeating: "x", count: 100), score: Double(5-$0), retrievalMode: "lexical", offsetStart: nil, offsetEnd: nil) }
        let out = RAGContextBuilder.build(results: list, budgetChars: 220)
        XCTAssertLessThanOrEqual(out.totalChars, 220)
    }

    func testBudgetedSelectionReportsTokenAndConfidenceDiagnostics() {
        let src = RAGSource(id: "s", type: "file", title: "manual.md", ref: nil)
        let list = [
            RAGRetrievalResult(chunkID: UUID(), source: src, excerpt: String(repeating: "a", count: 80), score: 0.9, retrievalMode: "semantic", offsetStart: nil, offsetEnd: nil),
            RAGRetrievalResult(chunkID: UUID(), source: src, excerpt: String(repeating: "b", count: 80), score: 0.6, retrievalMode: "semantic", offsetStart: nil, offsetEnd: nil),
            RAGRetrievalResult(chunkID: UUID(), source: src, excerpt: String(repeating: "c", count: 80), score: 0.2, retrievalMode: "semantic", offsetStart: nil, offsetEnd: nil)
        ]

        let out = RAGContextBuilder.build(results: list, budgetTokens: 40)

        XCTAssertLessThanOrEqual(out.totalTokens, 40)
        XCTAssertEqual(out.candidateCount, 3)
        XCTAssertGreaterThan(out.confidence, 0.6)
    }

    func testDedupUsesSourceAndChunkIdentity() {
        let chunkID = UUID()
        let src = RAGSource(id: "s", type: "file", title: "manual.md", ref: nil)
        let duplicate = RAGRetrievalResult(chunkID: chunkID, source: src, excerpt: "same", score: 0.8, retrievalMode: "semantic", offsetStart: nil, offsetEnd: nil)

        let out = RAGContextBuilder.build(results: [duplicate, duplicate], budgetChars: 100)

        XCTAssertEqual(out.selected.count, 1)
        XCTAssertEqual(out.candidateCount, 2)
    }
}
