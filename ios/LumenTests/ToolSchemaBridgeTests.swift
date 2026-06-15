import XCTest
@testable import Lumen

final class ToolSchemaBridgeTests: XCTestCase {
    @MainActor func testMapping() {
        let defs = ToolSchemaBridge.toCatalogToolDefinitions([SecureToolDefinition(id: "device.status", displayName: "Device", description: "x", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "{}", resultPrivacyLevel: .low, maxOutputCharacters: 100)])
        XCTAssertEqual(defs.first?.id, "device.status")
    }

    func testPromptGroundingRendererCanonicalizesSecureToolAliases() {
        let tools = [
            SecureToolDefinition(id: "contacts.lookup", displayName: "Contacts", description: "Lookup contacts by name", category: .permissionRead, requiredPermissions: [.contacts], supportsBackgroundExecution: false, requiresUserApproval: false, argumentSchemaDescription: "{}", resultPrivacyLevel: .sensitive, maxOutputCharacters: 100),
            SecureToolDefinition(id: "memory.search", displayName: "Memory", description: "Search local memory items", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "{}", resultPrivacyLevel: .moderate, maxOutputCharacters: 100)
        ]

        let sections = PromptGroundingRenderer.render(
            memories: MemoryContextResult(selected: [], totalChars: 0, reasons: [:], sourceIDs: []),
            rag: RAGContextResult(selected: [], totalChars: 0),
            tools: tools,
            lowPower: false,
            thermal: .nominal
        )

        let toolSection = sections.first { $0.title == "Available tools" }
        XCTAssertNotNil(toolSection)
        XCTAssertTrue(toolSection?.content.contains("contacts.search") ?? false)
        XCTAssertTrue(toolSection?.content.contains("memory.recall") ?? false)
        XCTAssertFalse(toolSection?.content.contains("contacts.lookup") ?? true)
        XCTAssertFalse(toolSection?.content.contains("memory.search") ?? true)
        XCTAssertEqual(toolSection?.sourceIDs, ["contacts.search", "memory.recall"])
    }
}
