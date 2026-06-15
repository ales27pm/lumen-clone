import Foundation

struct KnowledgeLocalTool: LocalTool {
    static let nativeToolIDs: Set<String> = [
        "web.search",
        "web.fetch",
        "files.read",
        "memory.save",
        "memory.recall",
        "rag.search",
        "rag.index_files",
        "rag.index_photos"
    ]

    @MainActor static var all: [KnowledgeLocalTool] {
        ToolRegistry.all
            .filter { nativeToolIDs.contains(ToolRouteGuard.canonicalToolID($0.id)) }
            .map(KnowledgeLocalTool.init)
    }

    let definition: SecureToolDefinition
    private let toolID: String

    init(_ catalogTool: ToolDefinition) {
        let canonical = ToolRouteGuard.canonicalToolID(catalogTool.id)
        self.toolID = canonical
        self.definition = SecureToolDefinition(
            id: canonical,
            displayName: catalogTool.name,
            description: catalogTool.description,
            category: Self.secureCategory(for: canonical),
            requiredPermissions: [],
            supportsBackgroundExecution: true,
            requiresUserApproval: catalogTool.requiresApproval,
            argumentSchemaDescription: Self.argumentSchemaDescription(from: catalogTool.description),
            resultPrivacyLevel: canonical == "web.search" || canonical == "web.fetch" ? .low : .moderate,
            maxOutputCharacters: Self.maxOutputCharacters(for: canonical)
        )
    }

    func validateArguments(_ arguments: [String: String]) throws {}

    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        let args = ToolRouteGuard.normalizedArguments(for: toolID, rawToolID: toolID, arguments: invocation.arguments)
        if let permissionFailure = await ToolRouteGuard.ensurePermissionIfNeeded(for: toolID, arguments: args) {
            return result(invocation: invocation, text: permissionFailure, status: .denied, metricsSummary: "permission_denied")
        }

        let text: String
        switch toolID {
        case "web.search":
            text = await WebTools.webSearch(query: args["query"] ?? "")
        case "web.fetch":
            text = await WebTools.webFetch(url: args["url"] ?? "")
        case "files.read":
            text = await FilesTools.readImportedFile(name: args["name"] ?? "")
        case "memory.save":
            text = await MemoryTools.save(content: args["content"] ?? "", kind: args["kind"] ?? "fact")
        case "memory.recall":
            text = await MemoryTools.recall(query: args["query"] ?? "")
        case "rag.search":
            text = await MemoryTools.ragSearch(query: args["query"] ?? "", limit: Int(args["limit"] ?? "5") ?? 5)
        case "rag.index_files":
            text = await MemoryTools.ragIndexFiles()
        case "rag.index_photos":
            text = await MemoryTools.ragIndexPhotos(months: Int(args["months"] ?? "6") ?? 6)
        default:
            text = "Unsupported native knowledge tool: \(toolID)."
        }
        return result(invocation: invocation, text: text, status: ToolResultStatusClassifier.status(from: text), metricsSummary: "native_knowledge_tool")
    }

    private func result(invocation: ToolInvocation, text: String, status: ToolResultStatus, metricsSummary: String) -> ToolResult {
        ToolResult(invocationID: invocation.id, status: status, displayText: text, modelText: text, structuredPayload: ["toolID": toolID, "implementation": "KnowledgeLocalTool"], privacyLevel: definition.resultPrivacyLevel, metricsSummary: status == .success ? metricsSummary : "\(metricsSummary)_\(status.rawValue)", errorCode: status == .success ? nil : status.rawValue)
    }

    private static func secureCategory(for canonical: String) -> SecureToolCategory {
        switch canonical {
        case "web.search", "web.fetch": return .externalNetwork
        default: return .readOnly
        }
    }

    private static func maxOutputCharacters(for canonical: String) -> Int {
        switch canonical {
        case "web.search", "web.fetch": return 4_000
        case "rag.search": return 3_000
        default: return 2_400
        }
    }

    private static func argumentSchemaDescription(from description: String) -> String {
        guard let range = description.range(of: "Args:") else { return "{}" }
        return String(description[range.lowerBound...])
    }
}
