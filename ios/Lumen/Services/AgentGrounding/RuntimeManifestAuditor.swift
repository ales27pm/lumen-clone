import Foundation

public struct RuntimeAgentManifestAuditReport: Codable, Hashable {
    public let passed: Bool
    public let score: Double
    public let failures: [RuntimeManifestFailure]
    public let generatedAt: Date
    public let recommendedDatasetRepairs: [String]
}

public struct RuntimeManifestFailure: Codable, Hashable, Identifiable {
    public var id: String { [type, agent ?? "", actual ?? "", problem].joined(separator: "|") }
    public let type: String
    public let agent: String?
    public let expected: [String]
    public let actual: String?
    public let scenario: String?
    public let problem: String
}

public struct RuntimeManifestLoadResult: Hashable {
    public let manifest: AgentBehaviorManifest
    public let source: String
    public let usedRuntimeFallback: Bool
}

public final class RuntimeManifestAuditor {
    private let registryProvider: RuntimeToolRegistryProviding
    private let supportedTypes: Set<String> = ["string", "number", "bool", "array", "object", "null"]

    public init(registryProvider: RuntimeToolRegistryProviding) {
        self.registryProvider = registryProvider
    }

    public func loadBundledManifest(resourceName: String = "AgentBehaviorManifest", bundle: Bundle = .main) throws -> AgentBehaviorManifest {
        guard let url = bundle.url(forResource: resourceName, withExtension: "json") else {
            throw RuntimeManifestAuditError.missingBundledManifest(resourceName)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(AgentBehaviorManifest.self, from: data)
    }

    public func loadManifestFromStoreBundleOrRuntimeFallback(resourceName: String = "AgentBehaviorManifest", bundle: Bundle = .main) -> RuntimeManifestLoadResult {
        do {
            let syncLabel = try AgentManifestStore.synchronizeWithBundle(resourceName: resourceName, bundle: bundle)
            if let stored = try AgentManifestStore.load() {
                return RuntimeManifestLoadResult(
                    manifest: stored,
                    source: syncLabel,
                    usedRuntimeFallback: false
                )
            }
        } catch {
            // Fall through to bundled manifest, then runtime fallback.
        }

        if let seeded = try? loadBundledManifest(resourceName: resourceName, bundle: bundle) {
            return RuntimeManifestLoadResult(
                manifest: seeded,
                source: "bundled:\(AgentManifestStore.bundledRelativeDirectory)/\(resourceName).json",
                usedRuntimeFallback: false
            )
        }

        do {
            return RuntimeManifestLoadResult(
                manifest: try loadBundledManifest(resourceName: resourceName, bundle: bundle),
                source: "bundled:\(resourceName).json",
                usedRuntimeFallback: false
            )
        } catch {
            return RuntimeManifestLoadResult(
                manifest: syntheticManifestFromRuntimeRegistry(missingResourceName: resourceName),
                source: "bundled-missing-runtime-fallback",
                usedRuntimeFallback: true
            )
        }
    }

    public func loadBundledManifestOrRuntimeFallback(resourceName: String = "AgentBehaviorManifest", bundle: Bundle = .main) -> RuntimeManifestLoadResult {
        loadManifestFromStoreBundleOrRuntimeFallback(resourceName: resourceName, bundle: bundle)
    }

    public func syntheticManifestFromRuntimeRegistry(missingResourceName: String = "AgentBehaviorManifest") -> AgentBehaviorManifest {
        let tools = registryProvider.currentToolDefinitions()
        return AgentBehaviorManifest(
            schemaVersion: "runtime-fallback-v1",
            app: ManifestAppInfo(
                name: "Lumen",
                bundleIdentifier: Bundle.main.bundleIdentifier,
                buildVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String,
                generatedAt: ISO8601DateFormatter().string(from: Date())
            ),
            sourceIntegrity: ManifestSourceIntegrity(
                commit: "runtime-fallback-missing-\(missingResourceName).json",
                files: []
            ),
            fleet: ManifestFleet(
                contractVersion: "runtime-fallback",
                slots: []
            ),
            tools: tools,
            intents: [],
            routingMatrix: [],
            memory: nil,
            sentinels: ManifestSentinels(
                forbiddenInUserOutput: [
                    "<user_final_text>",
                    "<private_reasoning>",
                    "<tool_json>",
                    "<internal_state>",
                    "<scratchpad>",
                    "<hidden_reasoning>",
                ]
            )
        )
    }

    public func audit(manifest: AgentBehaviorManifest) -> RuntimeAgentManifestAuditReport {
        let liveTools = registryProvider.currentToolDefinitions()
        var failures: [RuntimeManifestFailure] = []

        let manifestByID = Self.uniqueToolMap(manifest.tools, origin: "manifest", failures: &failures)
        let liveByID = Self.groupToolsByID(liveTools, origin: "runtime", failures: &failures)

        for manifestTool in manifest.tools {
            guard let liveTool = liveByID[manifestTool.id]?.first else {
                failures.append(RuntimeManifestFailure(
                    type: "missing_live_tool",
                    agent: "runtime",
                    expected: [manifestTool.id],
                    actual: nil,
                    scenario: nil,
                    problem: "Tool exists in manifest but not in runtime registry."
                ))
                continue
            }
            compare(manifestTool: manifestTool, liveTool: liveTool, failures: &failures)
        }

        for liveTool in liveTools where manifestByID[liveTool.id] == nil {
            failures.append(RuntimeManifestFailure(
                type: "unmanifested_live_tool",
                agent: "runtime",
                expected: Array(manifestByID.keys).sorted(),
                actual: liveTool.id,
                scenario: nil,
                problem: "Runtime registry exposes a tool absent from AgentBehaviorManifest.json."
            ))
        }

        let score = Self.score(failureCount: failures.count, manifestToolCount: manifest.tools.count, liveToolCount: liveTools.count)
        return RuntimeAgentManifestAuditReport(
            passed: failures.isEmpty,
            score: score,
            failures: failures,
            generatedAt: Date(),
            recommendedDatasetRepairs: Self.repairHints(for: failures)
        )
    }

    private func compare(manifestTool: RuntimeToolDefinition, liveTool: RuntimeToolDefinition, failures: inout [RuntimeManifestFailure]) {
        if manifestTool.requiresApproval != liveTool.requiresApproval {
            failures.append(RuntimeManifestFailure(
                type: "approval_mismatch",
                agent: "runtime",
                expected: [String(manifestTool.requiresApproval)],
                actual: String(liveTool.requiresApproval),
                scenario: manifestTool.id,
                problem: "requiresApproval differs between manifest and runtime registry."
            ))
        }

        if manifestTool.permissionKey != liveTool.permissionKey {
            failures.append(RuntimeManifestFailure(
                type: "permission_key_mismatch",
                agent: "runtime",
                expected: [manifestTool.permissionKey ?? "nil"],
                actual: liveTool.permissionKey ?? "nil",
                scenario: manifestTool.id,
                problem: "permissionKey differs between manifest and runtime registry."
            ))
        }

        let manifestArgs = Self.uniqueArgumentMap(manifestTool.arguments, toolID: manifestTool.id, origin: "manifest", failures: &failures)
        let liveArgs = Self.uniqueArgumentMap(liveTool.arguments, toolID: liveTool.id, origin: "runtime", failures: &failures)

        for (name, manifestArg) in manifestArgs {
            guard let liveArg = liveArgs[name] else {
                failures.append(RuntimeManifestFailure(
                    type: "missing_live_argument",
                    agent: "runtime",
                    expected: [name],
                    actual: nil,
                    scenario: manifestTool.id,
                    problem: "Manifest argument is absent from runtime tool definition."
                ))
                continue
            }
            if normalizeType(manifestArg.type) != normalizeType(liveArg.type) || manifestArg.required != liveArg.required {
                failures.append(RuntimeManifestFailure(
                    type: "argument_mismatch",
                    agent: "runtime",
                    expected: ["\(name):\(manifestArg.type):required=\(manifestArg.required)"],
                    actual: "\(name):\(liveArg.type):required=\(liveArg.required)",
                    scenario: manifestTool.id,
                    problem: "Argument type or required flag differs between manifest and runtime."
                ))
            }
            if !supportedTypes.contains(normalizeType(liveArg.type)) {
                failures.append(RuntimeManifestFailure(
                    type: "unsupported_runtime_argument_type",
                    agent: "runtime",
                    expected: Array(supportedTypes).sorted(),
                    actual: liveArg.type,
                    scenario: manifestTool.id,
                    problem: "Runtime argument type is unsupported by AgentJSONValue contract."
                ))
            }
        }

        for liveArg in liveTool.arguments where manifestArgs[liveArg.name] == nil {
            failures.append(RuntimeManifestFailure(
                type: "unmanifested_live_argument",
                agent: "runtime",
                expected: Array(manifestArgs.keys).sorted(),
                actual: liveArg.name,
                scenario: manifestTool.id,
                problem: "Runtime exposes an argument absent from the manifest."
            ))
        }
    }

    private static func uniqueToolMap(_ tools: [RuntimeToolDefinition], origin: String, failures: inout [RuntimeManifestFailure]) -> [String: RuntimeToolDefinition] {
        let grouped = Dictionary(grouping: tools, by: \RuntimeToolDefinition.id)
        for (toolID, group) in grouped where group.count > 1 {
            failures.append(RuntimeManifestFailure(
                type: "duplicate_\(origin)_tool_id",
                agent: "runtime",
                expected: [toolID],
                actual: toolID,
                scenario: nil,
                problem: "\(origin.capitalized) registry exposes duplicate tool ID \(toolID). Keeping the first entry for comparison."
            ))
        }
        return grouped.compactMapValues { $0.first }
    }

    private static func groupToolsByID(_ tools: [RuntimeToolDefinition], origin: String, failures: inout [RuntimeManifestFailure]) -> [String: [RuntimeToolDefinition]] {
        let grouped = Dictionary(grouping: tools, by: \RuntimeToolDefinition.id)
        for (toolID, group) in grouped where group.count > 1 {
            failures.append(RuntimeManifestFailure(
                type: "duplicate_\(origin)_tool_id",
                agent: "runtime",
                expected: [toolID],
                actual: toolID,
                scenario: nil,
                problem: "\(origin.capitalized) registry exposes duplicate tool ID \(toolID). Keeping the first entry for comparison."
            ))
        }
        return grouped
    }

    private static func uniqueArgumentMap(_ arguments: [RuntimeToolArgument], toolID: String, origin: String, failures: inout [RuntimeManifestFailure]) -> [String: RuntimeToolArgument] {
        let grouped = Dictionary(grouping: arguments, by: \RuntimeToolArgument.name)
        for (argumentName, group) in grouped where group.count > 1 {
            failures.append(RuntimeManifestFailure(
                type: "duplicate_\(origin)_argument_name",
                agent: "runtime",
                expected: [argumentName],
                actual: argumentName,
                scenario: toolID,
                problem: "\(origin.capitalized) tool \(toolID) exposes duplicate argument name \(argumentName). Keeping the first entry for comparison."
            ))
        }
        return grouped.compactMapValues { $0.first }
    }

    private func normalizeType(_ value: String) -> String {
        switch value.lowercased() {
        case "str", "text": return "string"
        case "double", "float", "int", "integer": return "number"
        case "boolean": return "bool"
        case "list": return "array"
        case "dictionary", "dict": return "object"
        case "nil", "none": return "null"
        default: return value.lowercased()
        }
    }

    private static func score(failureCount: Int, manifestToolCount: Int, liveToolCount: Int) -> Double {
        let denominator = max(1, manifestToolCount + liveToolCount)
        let penalty = Double(failureCount) / Double(denominator)
        return max(0.0, min(1.0, 1.0 - penalty))
    }

    private static func repairHints(for failures: [RuntimeManifestFailure]) -> [String] {
        Array(Set(failures.map { failure in
            switch failure.type {
            case "unmanifested_live_tool", "missing_live_tool", "duplicate_manifest_tool_id", "duplicate_runtime_tool_id":
                return "Regenerate AgentBehaviorManifest.json from the current Swift source and resolve duplicate tool IDs."
            case "approval_mismatch":
                return "Regenerate approval-boundary dataset samples for changed tools."
            case "argument_mismatch", "missing_live_argument", "unmanifested_live_argument", "duplicate_manifest_argument_name", "duplicate_runtime_argument_name":
                return "Regenerate Tool Executor schema samples and resolve duplicate argument names."
            default:
                return "Review runtime manifest drift and add a REM repair sample."
            }
        })).sorted()
    }
}

public enum RuntimeManifestAuditError: Error, LocalizedError {
    case missingBundledManifest(String)

    public var errorDescription: String? {
        switch self {
        case .missingBundledManifest(let name):
            return "Missing bundled \(name).json resource."
        }
    }
}
