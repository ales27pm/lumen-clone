import XCTest
@testable import Lumen

@MainActor
final class BackgroundDiagnosticsTests: XCTestCase {
    func testCachedSnapshotContainsEntitlementWarningsField() {
        let snap = DiagnosticsProvider().cachedSnapshot()
        XCTAssertNotNil(snap.background.entitlementWarnings)
    }
}
