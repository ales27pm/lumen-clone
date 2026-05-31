import XCTest
@testable import Lumen

final class ChunkingStrategyTests: XCTestCase {
    func testPlainChunking() {
        let chunks = ChunkingStrategy.chunk(String(repeating: "a", count: 1500), type: .plain, config: .init(maxChars: 500, overlap: 50))
        XCTAssertGreaterThan(chunks.count, 2)
    }
}
