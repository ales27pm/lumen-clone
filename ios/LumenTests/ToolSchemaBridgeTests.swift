import XCTest
@testable import Lumen

final class ToolSchemaBridgeTests: XCTestCase {
    @MainActor func testMapping() {
        let defs = ToolSchemaBridge.toCatalogToolDefinitions([SecureToolDefinition(id: "device.status", displayName: "Device", description: "x", category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true, requiresUserApproval: false, argumentSchemaDescription: "{}", resultPrivacyLevel: .low, maxOutputCharacters: 100)])
        XCTAssertEqual(defs.first?.id, "device.status")
    }

    func testStructuredToolCallValidatorRejectsUnknownTool() {
        let action = AgentAction(tool: "system.delete_everything", args: [:])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.failure, .unknownTool("system.delete_everything"))
    }

    func testStructuredToolCallValidatorRejectsToolNotInAvailableManifest() {
        let action = AgentAction(tool: "weather", args: ["location": .string("Montreal")])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: [])

        XCTAssertEqual(result.failure, .toolNotAvailable("weather"))
    }

    func testStructuredToolCallValidatorRejectsMissingRequiredArgument() {
        let action = AgentAction(tool: "web.search", args: [:])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.failure, .missingRequiredArgument(tool: "web.search", argument: "query"))
    }

    func testStructuredToolCallValidatorRejectsWrongArgumentType() {
        let action = AgentAction(tool: "rag.search", args: ["query": .string("swift"), "limit": .string("3")])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.failure, .invalidArgumentType(tool: "rag.search", argument: "limit", expected: .number))
    }

    func testStructuredToolCallValidatorRejectsExtraDangerousArguments() {
        let action = AgentAction(tool: "web.search", args: ["query": .string("swift"), "deleteAfter": .bool(true)])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.failure, .extraArguments(tool: "web.search", arguments: ["deleteAfter"]))
    }

    func testStructuredToolCallValidatorAcceptsValidPayloadAndNormalizesAlias() {
        let action = AgentAction(tool: "web.search", args: ["q": .string("swift concurrency")])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.success?.canonicalToolID, "web.search")
        XCTAssertEqual(result.success?.arguments["query"], "swift concurrency")
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

private extension Result where Success == ValidatedStructuredToolCall, Failure == StructuredToolCallValidationError {
    var success: ValidatedStructuredToolCall? {
        guard case .success(let value) = self else { return nil }
        return value
    }

    var failure: StructuredToolCallValidationError? {
        guard case .failure(let error) = self else { return nil }
        return error
    }
}
