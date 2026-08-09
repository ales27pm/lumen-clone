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

    func testStructuredToolCallValidatorRejectsNestedObjectForStringAlias() {
        let action = AgentAction(tool: "web.search", args: ["q": .object(["term": .string("swift")])])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.failure, .invalidArgumentType(tool: "web.search", argument: "query", expected: .string))
    }

    func testStructuredToolCallValidatorRejectsArrayForNumericArgument() {
        let action = AgentAction(tool: "rag.search", args: ["query": .string("swift"), "limit": .array([.number(3)])])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.failure, .invalidArgumentType(tool: "rag.search", argument: "limit", expected: .number))
    }

    func testStructuredToolCallValidatorRejectsEnumValueOutsideContract() {
        let action = AgentAction(tool: "trigger.create", args: [
            "title": .string("Review"),
            "prompt": .string("Review reminders"),
            "schedule": .string("whenever")
        ])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.failure, .invalidEnumValue(tool: "trigger.create", argument: "schedule", allowed: ["absolute", "interval", "relative"]))
    }

    func testStructuredToolCallValidatorRejectsNumericValuesOutsideToolDomainsBeforeApproval() {
        let maximumScheduleMinutes = ToolArgumentValueDomains.maximumScheduleDelayMinutes
        let invalidAlarmMinutes: [AgentJSONValue] = [
            .number(-1),
            .number(0),
            .number(Double(maximumScheduleMinutes + 1)),
            .number(Double.greatestFiniteMagnitude),
            .number(.infinity),
            .number(1.5)
        ]

        for value in invalidAlarmMinutes {
            let result = StructuredToolCallValidator.validate(
                action: AgentAction(tool: "alarm.schedule", args: [
                    "title": .string("Wake up"),
                    "inMinutes": value
                ]),
                availableTools: ToolRegistry.all
            )
            XCTAssertEqual(
                result.failure,
                .invalidArgumentValue(tool: "alarm.schedule", argument: "inMinutes")
            )
        }

        let maximum = StructuredToolCallValidator.validate(
            action: AgentAction(tool: "alarm.schedule", args: [
                "title": .string("Wake up"),
                "inMinutes": .number(Double(maximumScheduleMinutes))
            ]),
            availableTools: ToolRegistry.all
        )
        XCTAssertEqual(maximum.success?.arguments["inMinutes"], String(maximumScheduleMinutes))
    }

    func testStructuredToolCallValidatorRejectsInvalidTriggerClockAndSnoozeDomains() {
        for value in ["24:00", "23:60", "999999999999999999999999:00"] {
            let result = StructuredToolCallValidator.validate(
                action: AgentAction(tool: "trigger.create", args: [
                    "title": .string("Review"),
                    "prompt": .string("Review reminders"),
                    "schedule": .string("absolute"),
                    "atTime": .string(value)
                ]),
                availableTools: ToolRegistry.all
            )
            XCTAssertEqual(
                result.failure,
                .invalidArgumentValue(tool: "trigger.create", argument: "atTime")
            )
        }

        for value in [0, ToolArgumentValueDomains.maximumSnoozeMinutes + 1] {
            let invalidSnooze = StructuredToolCallValidator.validate(
                action: AgentAction(tool: "alarm.schedule", args: [
                    "title": .string("Wake up"),
                    "inMinutes": .number(10),
                    "snoozeMinutes": .number(Double(value))
                ]),
                availableTools: ToolRegistry.all
            )
            XCTAssertEqual(
                invalidSnooze.failure,
                .invalidArgumentValue(tool: "alarm.schedule", argument: "snoozeMinutes")
            )
        }

        let invalidInterval = StructuredToolCallValidator.validate(
            action: AgentAction(tool: "trigger.create", args: [
                "title": .string("Review"),
                "prompt": .string("Review reminders"),
                "schedule": .string("interval"),
                "intervalSeconds": .number(Double(ToolArgumentValueDomains.maximumTriggerIntervalSeconds + 1))
            ]),
            availableTools: ToolRegistry.all
        )
        XCTAssertEqual(
            invalidInterval.failure,
            .invalidArgumentValue(tool: "trigger.create", argument: "intervalSeconds")
        )

        let invalidCountdown = StructuredToolCallValidator.validate(
            action: AgentAction(tool: "alarm.countdown", args: [
                "title": .string("Timer"),
                "durationSeconds": .number(Double(ToolArgumentValueDomains.maximumCountdownDurationSeconds + 1))
            ]),
            availableTools: ToolRegistry.all
        )
        XCTAssertEqual(
            invalidCountdown.failure,
            .invalidArgumentValue(tool: "alarm.countdown", argument: "durationSeconds")
        )
    }

    func testToolCatalogCarriesNumericDomainsWithoutChangingRuntimeArgumentShape() throws {
        let alarm = try XCTUnwrap(ToolRegistry.find(id: "alarm.schedule"))
        let alarmArguments = Dictionary(uniqueKeysWithValues: alarm.capabilityContract.arguments.map { ($0.name, $0) })
        XCTAssertEqual(alarmArguments["inMinutes"]?.valueDomain, ToolArgumentValueDomains.alarmScheduleDelayMinutes)
        XCTAssertEqual(alarmArguments["snoozeMinutes"]?.valueDomain, ToolArgumentValueDomains.alarmSnoozeMinutes)
        XCTAssertEqual(
            alarm.capabilityContract.runtimeArguments.first(where: { $0.name == "inMinutes" })?.type,
            "number"
        )

        let trigger = try XCTUnwrap(ToolRegistry.find(id: "trigger.create"))
        let triggerArguments = Dictionary(uniqueKeysWithValues: trigger.capabilityContract.arguments.map { ($0.name, $0) })
        XCTAssertEqual(triggerArguments["inMinutes"]?.valueDomain, ToolArgumentValueDomains.triggerDelayMinutes)
        XCTAssertEqual(triggerArguments["atTime"]?.valueDomain, ToolArgumentValueDomains.clockTime24Hour)
        XCTAssertEqual(triggerArguments["intervalSeconds"]?.valueDomain, ToolArgumentValueDomains.triggerIntervalSeconds)
        XCTAssertEqual(triggerArguments["beforeMinutes"]?.valueDomain, ToolArgumentValueDomains.triggerBeforeEventMinutes)
    }

    func testStructuredToolCallValidatorEnforcesRAGSourceScopeEnum() {
        let valid = StructuredToolCallValidator.validate(
            action: AgentAction(tool: "rag.search", args: ["query": .string("architecture"), "sourceScope": .string("documents")]),
            availableTools: ToolRegistry.all
        )
        XCTAssertEqual(valid.success?.arguments["sourceScope"], "documents")

        let invalid = StructuredToolCallValidator.validate(
            action: AgentAction(tool: "rag.search", args: ["query": .string("architecture"), "sourceScope": .string("internet")]),
            availableTools: ToolRegistry.all
        )
        XCTAssertEqual(invalid.failure, .invalidEnumValue(tool: "rag.search", argument: "sourceScope", allowed: ["all", "documents", "notes", "photos"]))
    }

    func testStructuredToolCallValidatorRejectsExtraDangerousArguments() {
        let action = AgentAction(tool: "web.search", args: ["query": .string("swift"), "deleteAfter": .bool(true)])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.failure, .extraArguments(tool: "web.search", arguments: ["deleteAfter"]))
    }

    func testStructuredToolCallValidatorSortsExtraArgumentPayload() {
        let action = AgentAction(tool: "web.search", args: [
            "query": .string("swift"),
            "zExtra": .string("z"),
            "aExtra": .string("a")
        ])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.failure, .extraArguments(tool: "web.search", arguments: ["aExtra", "zExtra"]))
    }

    func testStructuredToolCallValidatorAcceptsValidPayloadAndNormalizesAlias() {
        let action = AgentAction(tool: "web.search", args: ["q": .string("swift concurrency")])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.success?.canonicalToolID, "web.search")
        XCTAssertEqual(result.success?.arguments["query"], "swift concurrency")
    }

    func testStructuredToolCallValidatorAcceptsWeatherCityAlias() {
        let action = AgentAction(tool: "weather", args: ["city": .string("Montreal")])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.success?.canonicalToolID, "weather")
        XCTAssertEqual(result.success?.arguments["location"], "Montreal")
    }

    func testStructuredToolCallValidatorAcceptsMessageDraftAliases() {
        let action = AgentAction(tool: "messages.draft", args: [
            "recipient": .string("Alex"),
            "message": .string("Running late")
        ])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.success?.canonicalToolID, "messages.draft")
        XCTAssertEqual(result.success?.arguments["to"], "Alex")
        XCTAssertEqual(result.success?.arguments["body"], "Running late")
    }

    func testStructuredToolCallValidatorAcceptsMailDraftAliasesWithoutSubject() {
        let action = AgentAction(tool: "mail.draft", args: [
            "recipient": .string("alex@example.com"),
            "text": .string("Status update")
        ])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.success?.canonicalToolID, "mail.draft")
        XCTAssertEqual(result.success?.arguments["to"], "alex@example.com")
        XCTAssertEqual(result.success?.arguments["body"], "Status update")
    }

    func testStructuredToolCallValidatorAcceptsOutlookForwardAliases() {
        let action = AgentAction(tool: "outlook.message.forward", args: [
            "id": .string("msg-1"),
            "recipient": .string("alex@example.com"),
            "comment": .string("FYI")
        ])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.success?.canonicalToolID, "outlook.message.forward")
        XCTAssertEqual(result.success?.arguments["messageId"], "msg-1")
        XCTAssertEqual(result.success?.arguments["to"], "alex@example.com")
        XCTAssertEqual(result.success?.arguments["body"], "FYI")
    }

    func testStructuredToolCallValidatorAcceptsTriggerScheduleEnum() {
        let action = AgentAction(tool: "trigger.create", args: [
            "title": .string("Review"),
            "prompt": .string("Review reminders"),
            "schedule": .string("relative")
        ])
        let result = StructuredToolCallValidator.validate(action: action, availableTools: ToolRegistry.all)

        XCTAssertEqual(result.success?.canonicalToolID, "trigger.create")
        XCTAssertEqual(result.success?.arguments["schedule"], "relative")
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
