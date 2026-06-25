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

    func testSelectionDiversifiesSourcesBeforeThirdChunkFromSameSource() {
        let sourceA = RAGSource(id: "source-a", type: "file", title: "a.md", ref: nil)
        let sourceB = RAGSource(id: "source-b", type: "file", title: "b.md", ref: nil)
        let results = [
            RAGRetrievalResult(chunkID: UUID(), source: sourceA, excerpt: String(repeating: "a", count: 100), score: 0.99, retrievalMode: "semantic", offsetStart: nil, offsetEnd: nil),
            RAGRetrievalResult(chunkID: UUID(), source: sourceA, excerpt: String(repeating: "b", count: 100), score: 0.98, retrievalMode: "semantic", offsetStart: nil, offsetEnd: nil),
            RAGRetrievalResult(chunkID: UUID(), source: sourceA, excerpt: String(repeating: "c", count: 100), score: 0.97, retrievalMode: "semantic", offsetStart: nil, offsetEnd: nil),
            RAGRetrievalResult(chunkID: UUID(), source: sourceB, excerpt: String(repeating: "d", count: 100), score: 0.50, retrievalMode: "semantic", offsetStart: nil, offsetEnd: nil)
        ]

        let out = RAGContextBuilder.build(results: results, budgetChars: 300)

        XCTAssertEqual(out.selected.map(\.source.id), ["source-a", "source-a", "source-b"])
        XCTAssertEqual(out.selectedSourceCount, 2)
        XCTAssertTrue(out.diversityPassApplied)
    }

    func testSingleSourceCanUseRemainingBudgetAfterDiversityPass() {
        let source = RAGSource(id: "source-a", type: "file", title: "a.md", ref: nil)
        let results = (0..<4).map { index in
            RAGRetrievalResult(chunkID: UUID(), source: source, excerpt: String(repeating: "\(index)", count: 100), score: 1.0 - (Double(index) * 0.01), retrievalMode: "semantic", offsetStart: nil, offsetEnd: nil)
        }

        let out = RAGContextBuilder.build(results: results, budgetChars: 400)

        XCTAssertEqual(out.selected.count, 4)
        XCTAssertEqual(out.selectedSourceCount, 1)
        XCTAssertFalse(out.diversityPassApplied)
    }
}
