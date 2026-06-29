import Foundation

nonisolated enum SelfModelContextProvider {
    static let sectionTitle = "Self model"
    private static let foregroundMaxChars = 1_600
    private static let backgroundMaxChars = 900

    static func section(for snapshot: SelfModelSnapshot, budget: ContextBudgetPlan) -> PromptGroundingSection {
        let maxChars = maxChars(for: snapshot, budget: budget)
        let content = render(snapshot, maxChars: maxChars)
        return PromptGroundingSection(
            title: sectionTitle,
            content: content,
            estimatedChars: content.count,
            sourceIDs: sourceIDs(for: snapshot),
            privacyLevel: .low
        )
    }

    static func render(_ snapshot: SelfModelSnapshot, maxChars: Int) -> String {
        let boundedMax = max(0, maxChars)
        guard boundedMax > 0 else { return "" }

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]

        if let data = try? encoder.encode(snapshot),
           let json = String(data: data, encoding: .utf8),
           json.count <= boundedMax {
            return json
        }

        let compact = compactLines(for: snapshot).joined(separator: "\n")
        guard compact.count > boundedMax else { return compact }
        return String(compact.prefix(boundedMax))
    }

    private static func maxChars(for snapshot: SelfModelSnapshot, budget: ContextBudgetPlan) -> Int {
        let cap = snapshot.app.mode == "background" ? backgroundMaxChars : foregroundMaxChars
        return min(max(0, budget.charSections.runtime), cap)
    }

    private static func sourceIDs(for snapshot: SelfModelSnapshot) -> [String] {
        [
            "selfModelSnapshot/\(snapshot.schemaVersion)",
            "slot/\(snapshot.agent.activeSlot)",
            "runtime/\(snapshot.runtime.selectedRuntime)",
            "evidence/\(snapshot.evidence.exportPolicy.sourceLayer)"
        ]
    }

    private static func compactLines(for snapshot: SelfModelSnapshot) -> [String] {
        [
            "schemaVersion=\(snapshot.schemaVersion)",
            "mode=\(snapshot.app.mode)",
            "activeSlot=\(snapshot.agent.activeSlot)",
            "availableSlots=\(snapshot.agent.availableSlots.joined(separator: ","))",
            "fleetContractVersion=\(snapshot.agent.fleetContractVersion)",
            "selectedRuntime=\(snapshot.runtime.selectedRuntime)",
            "selectedRuntimePathKind=\(snapshot.runtime.selectedRuntimePathKind)",
            "contextProfile=\(snapshot.contextBudget.profile)",
            "maxInputTokens=\(snapshot.contextBudget.maxInputTokens)",
            "availableTools=\(snapshot.tools.available.prefix(24).joined(separator: ","))",
            "requiresApproval=\(snapshot.tools.requiresApproval.prefix(24).joined(separator: ","))",
            "backgroundSafe=\(snapshot.tools.backgroundSafe.prefix(24).joined(separator: ","))",
            "sourceLayer=\(snapshot.evidence.exportPolicy.sourceLayer)",
            "ownsLiveE2EScenarios=\(snapshot.evidence.exportPolicy.ownsLiveE2EScenarios)",
            "runtimeAuditPresent=\(snapshot.evidence.runtimeAuditPresent)",
            "policy=mustNotInventToolIDs,mustNotBypassApproval,mustCiteRuntimeSourceWhenClaimingRuntimeState"
        ]
    }
}
