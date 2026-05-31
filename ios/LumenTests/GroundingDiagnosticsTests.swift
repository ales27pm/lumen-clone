import XCTest
@testable import Lumen

@MainActor
final class GroundingDiagnosticsTests: XCTestCase {
    func testGroundingMetadataOnly() async {
        let snap = await DiagnosticsProvider().collect()
        XCTAssertNotNil(snap.grounding.contextSource)
    }
}
