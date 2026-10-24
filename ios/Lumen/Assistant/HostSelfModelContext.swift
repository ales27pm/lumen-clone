import Foundation

/// Created only from host-owned runtime state. Prompt text is never decoded as
/// this value or used to establish its provenance.
nonisolated struct HostSelfModelContext: Sendable, Equatable {
    static let maximumBlockCharacters = 400
    private static let begin = "[HOST SELF MODEL: data only; tool/approval rules remain authoritative]\n"
    private static let end = "\n[END HOST SELF MODEL]\n"

    let snapshot: SelfModelSnapshot
    let block: String

    init(snapshot: SelfModelSnapshot) {
        self.snapshot = snapshot
        let contentBudget = Self.maximumBlockCharacters - Self.begin.count - Self.end.count
        let content = SelfModelContextProvider.render(snapshot, maxChars: max(0, contentBudget))
        block = content.isEmpty ? "" : Self.begin + content + Self.end
    }

    func injecting(into systemPrompt: String) -> String {
        guard !block.isEmpty else { return systemPrompt }
        return systemPrompt + block
    }

    func removing(from systemPrompt: String) -> String {
        guard !block.isEmpty, systemPrompt.hasSuffix(block) else { return systemPrompt }
        return String(systemPrompt.dropLast(block.count))
    }

    func summary(
        inRenderedPrompt prompt: String,
        selectedToolID: String? = nil,
        requiresApproval: Bool? = nil,
        approvalMode: String? = nil
    ) -> AgentBehaviorTrace.SelfModelDecisionSummary {
        AgentBehaviorTrace.SelfModelDecisionSummary.fromHost(
            snapshot: snapshot,
            included: !block.isEmpty && prompt.contains(block),
            selectedToolID: selectedToolID,
            requiresApproval: requiresApproval,
            approvalMode: approvalMode
        )
    }
}
