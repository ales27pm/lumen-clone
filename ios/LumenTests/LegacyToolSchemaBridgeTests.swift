import XCTest
@testable import Lumen

final class LegacyToolSchemaBridgeTests: XCTestCase {
    @MainActor func testMapping() {
        let defs = LegacyToolSchemaBridge.toLegacyToolDefinitions([SecureToolDefinition(id: "device.status", displayName: "Device", description: "x", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "{}", resultPrivacyLevel: .low, maxOutputCharacters: 100)])
        XCTAssertEqual(defs.first?.id, "device.status")
    }
}
