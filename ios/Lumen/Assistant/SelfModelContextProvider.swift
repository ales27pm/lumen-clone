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

        // Facts are atomic: dropping a whole field is safer than turning a truncated
        // tool ID, policy, revision, or JSON value into an apparent complete claim.
        let required = requiredLines(for: snapshot)
        let optional = optionalLines(for: snapshot)
        let omissionReserve = "\nomittedFields=999".count
        var lines = required
        var used = required.joined(separator: "\n").count
        guard used + omissionReserve <= boundedMax else { return "" }
        var omitted = 0
        for line in optional {
            if used + 1 + line.count + omissionReserve <= boundedMax {
                lines.append(line)
                used += 1 + line.count
            } else {
                omitted += 1
            }
        }
        lines.append("omittedFields=\(omitted)")
        return lines.joined(separator: "\n")
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

    private static func requiredLines(for snapshot: SelfModelSnapshot) -> [String] {
        [
            "schemaVersion=\(snapshot.schemaVersion)",
            "identity=\(snapshot.agent.logicalIdentity)",
            "mode=\(snapshot.app.mode)",
            "activeSlot=\(snapshot.agent.activeSlot)",
            "slotAvailability=\(snapshot.agent.slotAvailability.rawValue)",
            "selectedRuntime=\(snapshot.runtime.selectedRuntime)",
            "runtimeReadiness=\(snapshot.runtime.readiness.rawValue)",
            "mustNotBypassApproval=\(snapshot.policy.mustNotBypassApproval)",
            "availableToolsCount=\(snapshot.tools.available.count)",
            "requiresApprovalCount=\(snapshot.tools.requiresApproval.count)",
            "embeddingReadiness=\(snapshot.runtime.embeddingReadiness.rawValue)"
        ]
    }

    private static func optionalLines(for snapshot: SelfModelSnapshot) -> [String] {
        [
            "availableTools=\(snapshot.tools.available.joined(separator: ","))",
            "requiresApproval=\(snapshot.tools.requiresApproval.joined(separator: ","))",
            "selectedRuntimePathKind=\(snapshot.runtime.selectedRuntimePathKind)",
            "toolAvailabilityScope=\(snapshot.tools.availabilityScope)",
            "mustNotInventToolIDs=\(snapshot.policy.mustNotInventToolIDs)",
            "mustCiteRuntimeSourceWhenClaimingRuntimeState=\(snapshot.policy.mustCiteRuntimeSourceWhenClaimingRuntimeState)",
            "declaredSlots=\(snapshot.agent.declaredSlots.joined(separator: ","))",
            "availableSlots=\(snapshot.agent.availableSlots.joined(separator: ","))",
            "sourceRevision=\(snapshot.app.sourceRevision ?? "unknown")",
            "manifestCommit=\(snapshot.agent.manifestCommit)",
            "manifestFreshness=\(snapshot.evidence.manifestFreshness)",
            "generatedAt=\(ISO8601DateFormatter().string(from: snapshot.generatedAt))",
            "selectionReason=\(snapshot.runtime.selectionReason)",
            "availableBackendKinds=\(snapshot.runtime.availableBackendKinds.joined(separator: ","))",
            "thermalState=\(snapshot.runtime.thermalState)",
            "powerState=\(snapshot.runtime.powerState)",
            "networkState=\(snapshot.runtime.networkState)",
            "backgroundSafe=\(snapshot.tools.backgroundSafe.joined(separator: ","))",
            "contextProfile=\(snapshot.contextBudget.profile)",
            "maxInputTokens=\(snapshot.contextBudget.maxInputTokens)",
            "fleetContractVersion=\(snapshot.agent.fleetContractVersion)",
            "sourceLayer=\(snapshot.evidence.exportPolicy.sourceLayer)",
            "ownsLiveE2EScenarios=\(snapshot.evidence.exportPolicy.ownsLiveE2EScenarios)",
            "includesDeterministicStaticScenarios=\(snapshot.evidence.exportPolicy.includesDeterministicStaticScenarios)",
            "runtimeAuditPresent=\(snapshot.evidence.runtimeAuditPresent)"
        ]
    }
}
