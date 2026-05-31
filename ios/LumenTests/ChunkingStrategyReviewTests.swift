import XCTest
@testable import Lumen

final class ChunkingStrategyReviewTests: XCTestCase {
    func testInvalidConfigReturnsEmpty() {
        XCTAssertTrue(ChunkingStrategy.chunk("abc", type: .plain, config: .init(maxChars: 0, overlap: 0)).isEmpty)
        XCTAssertTrue(ChunkingStrategy.chunk("abc", type: .plain, config: .init(maxChars: 3, overlap: 3)).isEmpty)
    }

    func testOffsetsMatchOriginalSource() {
        let text = "  abcdef"
        let chunks = ChunkingStrategy.chunk(text, type: .plain, config: .init(maxChars: 3, overlap: 0))
        XCTAssertEqual(chunks.first?.start, 2)
        XCTAssertEqual(chunks.first?.text, "abc")
    }
}
