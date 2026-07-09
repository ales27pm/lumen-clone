import XCTest

final class LegacyPromptPathAuditTests: XCTestCase {
    func testAuditDocExists() {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        XCTAssertTrue(FileManager.default.fileExists(atPath: repoRoot.appendingPathComponent("docs/LEGACY_PROMPT_PATH_AUDIT.md").path))
    }
}
