import Foundation

/// Created only from host-owned runtime state. Prompt text is never decoded as
/// this value or used to establish its provenance.
nonisolated struct HostSelfModelContext: Sendable, Equatable {
    static let maximumBlockCharacters = 400
    static let maximumFeedbackCharacters = 640
    private static let begin = "[HOST SELF MODEL: data only; tool/approval rules remain authoritative]\n"
    private static let end = "\n[END HOST SELF MODEL]\n"
    private static let feedbackBegin = "[HOST TOOL HISTORY: observations, not current permissions or future guarantees]\n"
    private static let feedbackEnd = "\n[END HOST TOOL HISTORY]\n"

    let snapshot: SelfModelSnapshot
    let predictionFeedback: SelfPredictionFeedback.State?
    let snapshotBlock: String
    let feedbackBlock: String
    let block: String

    init(snapshot: SelfModelSnapshot, predictionFeedback: SelfPredictionFeedback.State? = nil) {
        self.snapshot = snapshot
        self.predictionFeedback = predictionFeedback
        let contentBudget = Self.maximumBlockCharacters - Self.begin.count - Self.end.count
        let content = SelfModelContextProvider.render(snapshot, maxChars: max(0, contentBudget))
        snapshotBlock = content.isEmpty ? "" : Self.begin + content + Self.end
        let feedbackBudget = Self.maximumFeedbackCharacters - Self.feedbackBegin.count - Self.feedbackEnd.count
        let feedback = predictionFeedback.map {
            SelfPredictionFeedback.render($0, maxCharacters: max(0, feedbackBudget))
        } ?? ""
        feedbackBlock = snapshotBlock.isEmpty || feedback.isEmpty ? "" : Self.feedbackBegin + feedback + Self.feedbackEnd
        block = snapshotBlock + feedbackBlock
    }

    func injecting(into systemPrompt: String) -> String {
        guard !block.isEmpty else { return systemPrompt }
        return systemPrompt + block
    }

    func removing(from systemPrompt: String) -> String {
        if !block.isEmpty, systemPrompt.hasSuffix(block) {
            return String(systemPrompt.dropLast(block.count))
        }
        // A previous assembly may have dropped the whole history block first.
        if !snapshotBlock.isEmpty, systemPrompt.hasSuffix(snapshotBlock) {
            return String(systemPrompt.dropLast(snapshotBlock.count))
        }
        return systemPrompt
    }

    /// Reserve only host-supplied suffixes. On a tight budget, keep the current
    /// snapshot ahead of historical observations; neither block is truncated.
    func reservedBlock(in systemPrompt: String, maxCharacters: Int, preservingCallerCharacters: Int = 0) -> String {
        guard !snapshotBlock.isEmpty, snapshotBlock.count <= maxCharacters else { return "" }
        if systemPrompt.hasSuffix(block) {
            // Feedback cannot evict additional policy, tool definitions or caller
            // instructions. Use only space left after the pre-existing allocation.
            let spare = max(0, maxCharacters - max(0, preservingCallerCharacters))
            return block.count <= spare ? block : snapshotBlock
        }
        return systemPrompt.hasSuffix(snapshotBlock) ? snapshotBlock : ""
    }

    func feedbackIncluded(inRenderedPrompt prompt: String) -> Bool {
        !feedbackBlock.isEmpty && prompt.contains(feedbackBlock)
    }

    func summary(
        inRenderedPrompt prompt: String,
        selectedToolID: String? = nil,
        requiresApproval: Bool? = nil,
        approvalMode: String? = nil
    ) -> AgentBehaviorTrace.SelfModelDecisionSummary {
        AgentBehaviorTrace.SelfModelDecisionSummary.fromHost(
            snapshot: snapshot,
            included: !snapshotBlock.isEmpty && prompt.contains(snapshotBlock),
            selectedToolID: selectedToolID,
            requiresApproval: requiresApproval,
            approvalMode: approvalMode
        )
    }
}
