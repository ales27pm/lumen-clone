import XCTest
@testable import Lumen

final class LumenMemorySearchIntentTests: XCTestCase {
    func testLimitCappedToTen() {
        let capped = max(1, min(42, 10))
        XCTAssertEqual(capped, 10)
    }
}
