import XCTest
@testable import Lumen

final class LegacySecureToolExecutorTests: XCTestCase {
    @MainActor func testWebFetchUsesSafeExecutorPath() async {
        let out = await LegacySecureToolExecutor.execute(toolID: "web.fetch", arguments: AgentJSONArguments(stringDictionary: ["url":"https://example.com"]))
        let lower = out.lowercased()
        XCTAssertFalse(lower.contains("denied by legacy secure policy"))
        XCTAssertFalse(lower.contains("pending secure migration"))
    }
}
