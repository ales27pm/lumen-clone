import XCTest
@testable import Lumen

final class PermissionDiagnosticsTests: XCTestCase {
    func testEntitlementWarningsSurfaced() async {
        let diag = await PermissionDiagnostics.collect(infoDictionary: [:])
        XCTAssertFalse(diag.entitlementWarnings.isEmpty)
    }
}
