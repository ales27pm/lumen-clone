import XCTest
@testable import Lumen

final class RolePipelineGroundingReuseTests: XCTestCase {
    func testCacheTTLOptionExists() {
        let cache = LegacyGroundingCache(ttl: 120)
        XCTAssertNotNil(cache)
    }
}
