import XCTest
@testable import Lumen

@MainActor
final class GroundingDiagnosticsTests: XCTestCase {
    func testGroundingMetadataOnly() {
        let snap = DiagnosticsProvider().cachedSnapshot()
        XCTAssertNotNil(snap.grounding.contextSource)
    }
}
