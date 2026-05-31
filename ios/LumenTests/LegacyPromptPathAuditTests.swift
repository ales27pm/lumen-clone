import XCTest

final class LegacyPromptPathAuditTests: XCTestCase {
    func testAuditDocExists() {
        XCTAssertTrue(FileManager.default.fileExists(atPath: "docs/LEGACY_PROMPT_PATH_AUDIT.md"))
    }
}
