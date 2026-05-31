import XCTest
@testable import Lumen

final class AssistantOutputSanitizerTests: XCTestCase {
    func testSanitizeRemovesLumenWebPayloadAndRawBlob() {
        let input = """
        Here you go.

        <lumen_web_payload>{\"kind\":\"searchResults\",\"query\":\"x\",\"results\":[]}</lumen_web_payload>
        """

        let output = AssistantOutputSanitizer.sanitize(input)
        XCTAssertFalse(output.contains("<lumen_web_payload>"))
        XCTAssertFalse(output.contains("\"kind\":\"searchResults\""))
        XCTAssertEqual(output, "Here you go.")
    }

    func testSanitizeRemovesToolTraceJSONWhenNotInDebugMode() {
        let input = """
        Summary

        ```json
        {"tool":"web.search","args":{"q":"weather"},"trace":"raw"}
        ```
        """

        let output = AssistantOutputSanitizer.sanitize(input)
        XCTAssertEqual(output, "Summary")
    }
}
