import XCTest
@testable import Lumen

@MainActor
final class PermissionDiagnosticsSnapshotTests: XCTestCase {
    func testIncludesKnownDomains() {
        let snap = DiagnosticsProvider().cachedSnapshot()
        XCTAssertTrue(snap.permissions.domains.contains { $0.domain == PermissionDomain.microphone.rawValue })
        XCTAssertTrue(snap.permissions.domains.contains { $0.domain == PermissionDomain.alarms.rawValue })
    }
}
