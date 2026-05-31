import XCTest
@testable import Lumen

@MainActor
final class DiagnosticsProviderTests: XCTestCase {
    func testCollectReturnsStructuredSnapshots() async {
        let snap = await DiagnosticsProvider().collect()
        XCTAssertFalse(snap.permissions.domains.isEmpty)
    }
}
