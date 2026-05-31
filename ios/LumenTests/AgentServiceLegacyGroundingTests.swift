import XCTest
@testable import Lumen

final class AgentServiceLegacyGroundingTests: XCTestCase {
    func testLegacyToolBridgeProducesTools() {
        let defs = LegacyToolSchemaBridge.toLegacyToolDefinitions([SecureToolDefinition(id: "memory.search", displayName: "Memory", description: "", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "", resultPrivacyLevel: .low, maxOutputCharacters: 100)])
        XCTAssertEqual(defs.first?.id, "memory.search")
    }
}
