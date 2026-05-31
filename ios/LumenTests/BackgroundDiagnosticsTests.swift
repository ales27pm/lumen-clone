import XCTest
@testable import Lumen

@MainActor
final class BackgroundDiagnosticsTests: XCTestCase {
    func testContainsEntitlementWarningsField() async {
        let snap = await DiagnosticsProvider().collect()
        XCTAssertNotNil(snap.background.entitlementWarnings)
    }
}
