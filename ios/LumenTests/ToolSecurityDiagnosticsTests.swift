import XCTest
@testable import Lumen

@MainActor
final class ToolSecurityDiagnosticsTests: XCTestCase {
    func testToolRowsHaveCategories() {
        let snap = DiagnosticsProvider().cachedSnapshot()
        XCTAssertFalse(snap.tools.tools.isEmpty)
        XCTAssertFalse(snap.tools.tools.contains { $0.category.isEmpty })
    }
}
