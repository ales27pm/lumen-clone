import Foundation

public struct RuntimeScenario: Codable, Hashable, Identifiable {
    public let id: String
    public let intent: String
    public let expectedToolID: String
    public let requiresApproval: Bool
    public let prompt: String
}

public struct RuntimeScenarioResult: Codable, Hashable, Identifiable {
    public let id: String
    public let scenario: RuntimeScenario
    public let passed: Bool
    public let failures: [RuntimeManifestFailure]
}

public final class RuntimeScenarioRunner {
    public init() {}

    public func scenarios(from manifest: AgentBehaviorManifest) -> [RuntimeScenario] {
        var output: [RuntimeScenario] = []
        let approvalByTool = Dictionary(manifest.tools.map { ($0.id, $0.requiresApproval) }, uniquingKeysWith: { first, _ in first })
        for entry in manifest.routingMatrix {
            for toolID in entry.allowedTools {
                output.append(RuntimeScenario(
                    id: "\(entry.intent)::\(toolID)",
                    intent: entry.intent,
                    expectedToolID: toolID,
                    requiresApproval: approvalByTool[toolID] ?? false,
                    prompt: prompt(intent: entry.intent, toolID: toolID)
                ))
            }
        }
        return output
    }

    public func validateStaticScenarios(manifest: AgentBehaviorManifest) -> [RuntimeScenarioResult] {
        let knownTools = Set(manifest.tools.map(\.id))
        let forbiddenSentinels = manifest.sentinels.forbiddenInUserOutput
        return scenarios(from: manifest).map { scenario in
            var failures: [RuntimeManifestFailure] = []
            if !knownTools.contains(scenario.expectedToolID) {
                failures.append(RuntimeManifestFailure(
                    type: "scenario_unknown_tool",
                    agent: "cortex",
                    expected: Array(knownTools).sorted(),
                    actual: scenario.expectedToolID,
                    scenario: scenario.prompt,
                    problem: "Scenario expects a tool absent from the bundled manifest."
                ))
            }
            for sentinel in forbiddenSentinels where scenario.prompt.contains(sentinel) {
                failures.append(RuntimeManifestFailure(
                    type: "scenario_sentinel_leak",
                    agent: "mouth",
                    expected: [],
                    actual: sentinel,
                    scenario: scenario.prompt,
                    problem: "Scenario prompt contains an internal sentinel."
                ))
            }
            return RuntimeScenarioResult(
                id: scenario.id,
                scenario: scenario,
                passed: failures.isEmpty,
                failures: failures
            )
        }
    }

    private func prompt(intent: String, toolID: String) -> String {
        if toolID.contains("calendar") { return "Create a calendar event for a meeting in 10 minutes." }
        if toolID.contains("map") { return "Find a hardware store nearby." }
        if toolID.contains("web") { return "Search for current SwiftData migration details." }
        if toolID.contains("mail") || toolID.contains("email") { return "Draft an email update." }
        return "Handle intent \(intent)."
    }
}
