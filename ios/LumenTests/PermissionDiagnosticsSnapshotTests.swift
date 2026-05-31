import XCTest
@testable import Lumen

@MainActor
final class PermissionDiagnosticsSnapshotTests: XCTestCase {
    func testIncludesKnownDomains() async {
        let snap = await DiagnosticsProvider().collect()
        XCTAssertTrue(snap.permissions.domains.contains { $0.domain == PermissionDomain.microphone.rawValue })
    }
}
