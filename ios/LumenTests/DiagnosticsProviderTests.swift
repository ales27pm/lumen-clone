import XCTest
@testable import Lumen

@MainActor
final class DiagnosticsProviderTests: XCTestCase {
    func testCachedSnapshotReturnsStructuredSnapshots() {
        let snap = DiagnosticsProvider().cachedSnapshot()
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

    func testPersistentDiagnosticsSourceCommitUsesInjectedLumenGitSHA() {
        XCTAssertEqual(
            PersistentRuntimeDiagnosticsExporter.sourceCommit(
                infoDictionary: [
                    "LumenGitSHA": "current123",
                    "GitCommit": "legacy456"
                ]
            ),
            "current123"
        )
    }

    func testPersistentDiagnosticsSourceCommitFallsBackToLegacyKey() {
        XCTAssertEqual(
            PersistentRuntimeDiagnosticsExporter.sourceCommit(
                infoDictionary: ["GitCommit": "legacy456"]
            ),
            "legacy456"
        )
    }

    func testPersistentDiagnosticsSourceCommitRejectsUnresolvedValues() {
        XCTAssertNil(
            PersistentRuntimeDiagnosticsExporter.sourceCommit(
                infoDictionary: [
                    "LumenGitSHA": "$(LUMEN_GIT_SHA)",
                    "GitCommit": "unknown"
                ]
            )
        )
    }
}
