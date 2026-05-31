import XCTest
@testable import Lumen

final class LegacyGroundingCacheTests: XCTestCase {
    func testCacheMissExpired() async {
        let cache = LegacyGroundingCache(ttl: 0.01)
        let key1 = LegacyGroundingCache.Key(conversationID: UUID(), turnID: UUID(), userDigest: LegacyGroundingCache.digest("hello"), background: false, lowPowerMode: false, thermalState: .nominal)
        let result = await cache.get(key1, now: Date().addingTimeInterval(1))
        XCTAssertNil(result)
    }

    func testDigestDeterministicAndPolicyDimensionsDiffer() {
        let digest = LegacyGroundingCache.digest("hello")
        XCTAssertEqual(digest, LegacyGroundingCache.digest("hello"))
        let base = LegacyGroundingCache.Key(conversationID: nil, turnID: nil, userDigest: digest, background: false, lowPowerMode: false, thermalState: .nominal)
        let lowPower = LegacyGroundingCache.Key(conversationID: nil, turnID: nil, userDigest: digest, background: false, lowPowerMode: true, thermalState: .nominal)
        XCTAssertNotEqual(base, lowPower)
    }
}
