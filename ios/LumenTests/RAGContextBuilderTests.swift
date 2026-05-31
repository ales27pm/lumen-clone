import XCTest
@testable import Lumen

final class RAGContextBuilderTests: XCTestCase {
    func testBudgetedSelection() {
        let src = RAGSource(id: "s", type: "file", title: "t", ref: nil)
        let list = (0..<5).map { RAGRetrievalResult(chunkID: UUID(), source: src, excerpt: String(repeating: "x", count: 100), score: Double(5-$0), retrievalMode: "lexical", offsetStart: nil, offsetEnd: nil) }
        let out = RAGContextBuilder.build(results: list, budgetChars: 220)
        XCTAssertLessThanOrEqual(out.totalChars, 220)
    }
}
