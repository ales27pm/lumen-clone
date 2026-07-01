import XCTest
@testable import Lumen

@MainActor
final class DiagnosticsProviderTests: XCTestCase {
    func testCollectReturnsStructuredSnapshots() async {
        let snap = await DiagnosticsProvider().collect()
        XCTAssertFalse(snap.permissions.domains.isEmpty)
        XCTAssertFalse(snap.build.bundleIdentifier.isEmpty)
        XCTAssertFalse(snap.build.bundleVersion.isEmpty)
    }

    func testBuildDiagnosticsReadsAlarmUsageDescriptionFromRuntimeInfoDictionary() {
        let snap = BuildDiagnosticsSnapshot.current(
            infoDictionary: [
                "CFBundleVersion": "42",
                "LumenBuildSourceIdentifier": "42",
                "LumenGitSHA": "abc123",
                "LumenBuildConfiguration": "Debug",
                "LumenBuildScheme": "Lumen",
                "NSAlarmKitUsageDescription": "Alarm scheduling"
            ],
            bundleIdentifier: "com.27pm.lumenclone"
        )

        XCTAssertEqual(snap.bundleIdentifier, "com.27pm.lumenclone")
        XCTAssertEqual(snap.bundleVersion, "42")
        XCTAssertEqual(snap.gitSHA, "abc123")
        XCTAssertEqual(snap.alarmKitUsageDescription, "Alarm scheduling")
    }
}
