import XCTest
@testable import Lumen

final class LegacyToolAuditPolicyTests: XCTestCase {
    @MainActor func testAllowlistedReadOnlyCanPass() async {
        let out = await LegacySecureToolExecutor.execute(toolID: "trigger.list", arguments: AgentJSONArguments(stringDictionary: [:]))
        XCTAssertFalse(out.isEmpty)
    }
}
