import Foundation

@MainActor
enum ToolSchemaBridge {
    nonisolated static func toCatalogToolDefinitions(_ secure: [SecureToolDefinition]) -> [ToolDefinition] {
        secure.map {
            if let canonical = canonicalCatalogToolDefinition(forSecureToolID: $0.id) {
                return canonical
            }
            return ToolDefinition(
                id: $0.id,
                name: $0.displayName,
                category: mapCategory($0.category),
                description: $0.description,
                icon: "wrench.and.screwdriver",
                tint: "blue",
                requiresApproval: $0.requiresUserApproval,
                permissionKey: mapPermission($0.requiredPermissions.first)
            )
        }
    }

    nonisolated static func canonicalCatalogToolDefinition(forSecureToolID id: String) -> ToolDefinition? {
        switch id {
        case "calendar.read":
            return ToolRegistry.find(id: "calendar.list")
        case "contacts.lookup":
            return ToolRegistry.find(id: "contacts.search")
        case "location.snapshot":
            return ToolRegistry.find(id: "location.current")
        case "memory.search":
            return ToolRegistry.find(id: "memory.recall")
        case "rag.search.secure":
            return ToolRegistry.find(id: "rag.search")
        default:
            return nil
        }
    }

    nonisolated static func toInvocation(toolID: String, arguments: [String: String], source: ToolInvocationSource, conversationID: UUID?, turnID: UUID?) -> ToolInvocation {
        ToolInvocation(id: UUID(), toolID: toolID, arguments: arguments, source: source, conversationID: conversationID, turnID: turnID, createdAt: Date())
    }

    private nonisolated static func mapCategory(_ c: SecureToolCategory) -> ToolCategory {
        switch c {
        case .readOnly, .permissionRead: return .knowledge
        case .userVisibleAction: return .productivity
        case .sensitiveAction, .destructiveAction: return .communication
        case .externalNetwork: return .knowledge
        }
    }

    private nonisolated static func mapPermission(_ p: PermissionDomain?) -> String? {
        switch p {
        case .calendars: return "NSCalendarsFullAccessUsageDescription"
        case .contacts: return "NSContactsUsageDescription"
        case .locationWhenInUse: return "NSLocationWhenInUseUsageDescription"
        case .notifications: return "NSUserNotificationUsageDescription"
        case .photoLibrary: return "NSPhotoLibraryUsageDescription"
        case .camera: return "NSCameraUsageDescription"
        case .microphone: return "NSMicrophoneUsageDescription"
        case .speechRecognition: return "NSSpeechRecognitionUsageDescription"
        case .alarms: return "NSAlarmKitUsageDescription"
        default: return nil
        }
    }
}

nonisolated enum StructuredToolCallValidationError: Error, Equatable, Sendable {
    case unknownTool(String)
    case toolNotAvailable(String)
    case missingRequiredArgument(tool: String, argument: String)
    case invalidArgumentType(tool: String, argument: String, expected: ToolArgumentValueType)
    case invalidArgumentValue(tool: String, argument: String)
    case invalidEnumValue(tool: String, argument: String, allowed: [String])
    case extraArguments(tool: String, arguments: [String])

    var diagnostic: String {
        switch self {
        case .unknownTool(let tool):
            return "unknown_tool:\(tool)"
        case .toolNotAvailable(let tool):
            return "tool_not_available:\(tool)"
        case .missingRequiredArgument(let tool, let argument):
            return "missing_required_argument:\(tool).\(argument)"
        case .invalidArgumentType(let tool, let argument, let expected):
            return "invalid_argument_type:\(tool).\(argument):expected_\(expected.rawValue)"
        case .invalidArgumentValue(let tool, let argument):
            return "invalid_argument_value:\(tool).\(argument)"
        case .invalidEnumValue(let tool, let argument, let allowed):
            return "invalid_enum_value:\(tool).\(argument):allowed_\(allowed.sorted().joined(separator: "|"))"
        case .extraArguments(let tool, let arguments):
            return "extra_arguments:\(tool):\(arguments.sorted().joined(separator: ","))"
        }
    }
}

nonisolated struct ValidatedStructuredToolCall: Sendable, Equatable {
    let canonicalToolID: String
    let arguments: [String: String]
}

nonisolated enum StructuredToolCallValidator {
    static func validate(
        action: AgentAction,
        availableTools: [ToolDefinition]
    ) -> Result<ValidatedStructuredToolCall, StructuredToolCallValidationError> {
        let canonicalToolID = ToolRouteGuard.canonicalToolID(action.tool)
        guard ToolRegistry.find(id: canonicalToolID) != nil else {
            return .failure(.unknownTool(canonicalToolID))
        }

        let availableCanonicalIDs = Set(availableTools.map { ToolRouteGuard.canonicalToolID($0.id) })
        guard availableCanonicalIDs.contains(canonicalToolID) else {
            return .failure(.toolNotAvailable(canonicalToolID))
        }

        let normalized = ToolRouteGuard.normalizedArguments(
            for: canonicalToolID,
            rawToolID: action.tool,
            arguments: action.args.stringCoerced
        )
        let contract = ToolRegistry.find(id: canonicalToolID)?.capabilityContract.arguments ?? []
        let contractByName = Dictionary(uniqueKeysWithValues: contract.map { ($0.name, $0) })
        let allowedNames = Set(contractByName.keys)
        let allowedAliasNames = ToolRouteGuard.aliasesAllowedDuringNormalization(for: canonicalToolID)
        let normalizedValues = normalizedJSONArguments(
            canonicalToolID: canonicalToolID,
            normalized: normalized,
            rawArguments: action.args,
            allowedAliasNames: allowedAliasNames
        )

        let extras = Set(normalized.keys).subtracting(allowedNames).subtracting(allowedAliasNames)
        if !extras.isEmpty {
            return .failure(.extraArguments(tool: canonicalToolID, arguments: extras.sorted()))
        }

        for argument in contract where argument.required {
            let value = normalized[argument.name]?.trimmingCharacters(in: .whitespacesAndNewlines)
            if value?.isEmpty ?? true {
                return .failure(.missingRequiredArgument(tool: canonicalToolID, argument: argument.name))
            }
        }

        for (name, value) in normalizedValues {
            guard let definition = contractByName[name] else { continue }
            guard isValue(value, compatibleWith: definition.type) else {
                return .failure(.invalidArgumentType(tool: canonicalToolID, argument: name, expected: definition.type))
            }
            if let valueDomain = definition.valueDomain,
               !valueDomain.accepts(value) {
                return .failure(.invalidArgumentValue(tool: canonicalToolID, argument: name))
            }
            if definition.type == .enumeration,
               let allowedValues = definition.allowedValues,
               case .string(let rawValue) = value {
                let normalizedValue = rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                let normalizedAllowedValues = Set(allowedValues.map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() })
                guard normalizedAllowedValues.contains(normalizedValue) else {
                    return .failure(.invalidEnumValue(tool: canonicalToolID, argument: name, allowed: allowedValues.sorted()))
                }
            }
        }

        return .success(ValidatedStructuredToolCall(
            canonicalToolID: canonicalToolID,
            arguments: normalized.filter { allowedNames.contains($0.key) }
        ))
    }

    private static func isValue(_ value: AgentJSONValue, compatibleWith type: ToolArgumentValueType) -> Bool {
        switch (type, value) {
        case (.string, .string):
            return true
        case (.number, .number):
            return true
        case (.bool, .bool):
            return true
        case (.array, .array):
            return true
        case (.object, .object):
            return true
        case (.enumeration, .string):
            return true
        default:
            return false
        }
    }

    private static func normalizedJSONArguments(
        canonicalToolID: String,
        normalized: [String: String],
        rawArguments: AgentJSONArguments,
        allowedAliasNames: Set<String>
    ) -> AgentJSONArguments {
        normalized.reduce(into: AgentJSONArguments()) { partial, entry in
            if let raw = rawArguments[entry.key] {
                partial[entry.key] = raw
                return
            }
            if let alias = allowedAliasNames.sorted().first(where: { alias in
                guard rawArguments[alias]?.stringValue == entry.value else { return false }
                return true
            }) {
                partial[entry.key] = rawArguments[alias]
                return
            }
            partial[entry.key] = .string(entry.value)
        }
    }

}
