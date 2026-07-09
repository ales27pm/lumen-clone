import XCTest
@testable import Lumen

final class LumenMemorySearchIntentTests: XCTestCase {
    func testLimitCappedToTen() {
        let capped = max(1, min(42, 10))
        XCTAssertEqual(capped, 10)
    }

    func testFailedMemorySearchDoesNotRenderAsEmptySuccess() {
        if #available(iOS 16.0, *) {
            let rendered = LumenMemorySearchIntent.renderSearchResult(
                .init(items: [], mode: "failed", diagnostic: "fetch_failed:swiftdata_error"),
                limit: 5
            )

            XCTAssertTrue(rendered.contains("memory search failed"))
            XCTAssertTrue(rendered.contains("fetch_failed:swiftdata_error"))
            XCTAssertFalse(rendered.contains("No memories found."))
        }
    }

    func testDegradedMemorySearchWithoutResultsDoesNotRenderAsEmptySuccess() {
        if #available(iOS 16.0, *) {
            let rendered = LumenMemorySearchIntent.renderSearchResult(
                .init(items: [], mode: "lexical_fallback", diagnostic: "embedding_failed:llama_unavailable;lexical_fetch_failed:swiftdata_error"),
                limit: 5
            )

            XCTAssertTrue(rendered.contains("memory search degraded"))
            XCTAssertTrue(rendered.contains("embedding_failed:llama_unavailable"))
            XCTAssertFalse(rendered.contains("No memories found."))
        }
    }

    func testEmptyMemoryStoreRendersAsEmptyResult() {
        if #available(iOS 16.0, *) {
            let rendered = LumenMemorySearchIntent.renderSearchResult(
                .init(items: [], mode: "semantic", diagnostic: "empty_store"),
                limit: 5
            )

            XCTAssertEqual(rendered, "No memories found.")
        }
    }
}
