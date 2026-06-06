import XCTest
@testable import Lumen

final class BackgroundExecutionLeaseTests: XCTestCase {
    func testAcquireReleaseAndExpiry() async {
        let lease = BackgroundExecutionLease()
        let now = Date()
        let firstAcquire = await lease.acquire(category: "a", reason: "r", ttl: 1, now: now)
        let secondAcquire = await lease.acquire(category: "a", reason: "r2", ttl: 1, now: now)
        let activeLease = await lease.activeLease(category: "a", now: now)
        let expiredLease = await lease.activeLease(category: "a", now: now.addingTimeInterval(2))

        XCTAssertTrue(firstAcquire)
        XCTAssertFalse(secondAcquire)
        XCTAssertNotNil(activeLease)
        XCTAssertNil(expiredLease)
    }
}
