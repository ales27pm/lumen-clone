import XCTest
@testable import Lumen

final class BackgroundExecutionLeaseTests: XCTestCase {
    func testAcquireReleaseAndExpiry() async {
        let lease = BackgroundExecutionLease()
        let now = Date()
        XCTAssertTrue(await lease.acquire(category: "a", reason: "r", ttl: 1, now: now))
        XCTAssertFalse(await lease.acquire(category: "a", reason: "r2", ttl: 1, now: now))
        XCTAssertNotNil(await lease.activeLease(category: "a", now: now))
        XCTAssertNil(await lease.activeLease(category: "a", now: now.addingTimeInterval(2)))
    }
}
