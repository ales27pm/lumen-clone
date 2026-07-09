import XCTest
@testable import Lumen

@MainActor
final class RuntimeDiagnosticsTests: XCTestCase {
    func testMetricSummariesBounded() {
        let snap = DiagnosticsProvider().cachedSnapshot()
        XCTAssertLessThanOrEqual(snap.runtime.recentMetricSummaries.count, 5)
    }
}
