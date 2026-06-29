import Foundation
import SwiftData

struct LegacyGroundingBundle: Sendable {
    let grounding: AssistantGroundingContext
    let budgetPlan: ContextBudgetPlan
    let sections: [PromptGroundingSection]
    let renderedPromptContext: String
    let secureTools: [SecureToolDefinition]
    let metricsSummary: String
}

/// Implementation detail used by `LegacyTurnGroundingCoordinator` to build legacy grounding bundles.
/// Legacy callers should use the coordinator as the single entrypoint so cache keys, role metadata,
/// idempotency policy, and degraded behavior stay centralized.
@MainActor
final class LegacyGroundingBridge {
    private let memoryEngine = MemoryEngine()
    private let ragEngine = RAGEngine()
    private let toolRegistry: SecureToolRegistry
    private let metricsStore: RuntimeMetricsStore
    private let runtimeRouter: AssistantRuntimeRouter

    init(toolRegistry: SecureToolRegistry? = nil, metricsStore: RuntimeMetricsStore? = nil, runtimeRouter: AssistantRuntimeRouter? = nil) {
        self.toolRegistry = toolRegistry ?? .shared
        self.metricsStore = metricsStore ?? .shared
        self.runtimeRouter = runtimeRouter ?? AssistantRuntimeRouter()
    }

    /// Constructs a legacy grounding bundle with rendered prompt context for an assistant turn.
    /// - Returns: A bundle containing selected memory items, RAG results, available tools, and the final rendered prompt context.
    /// - Throws: An error if the operation is cancelled or if an underlying operation fails.
    func build(userMessage: String, conversationID: UUID?, turnID: UUID?, history: [(role: MessageRole, content: String)], modelContext: ModelContext, turn: AssistantTurnContext, cancellationToken: AgentGroundingCancellationToken? = nil) async throws -> LegacyGroundingBundle {
        try cancellationToken?.checkCancellation()
        let budget = ContextBudgetAllocator.allocate(for: turn, maxInputTokens: 800)
        let contextQuery = ContextQueryRewriter.rewrite(userInput: userMessage, history: history, relevantMemories: turn.relevantMemories)
        let mem = memoryEngine.buildContext(query: contextQuery.query, budget: budget.charSections.memories, context: modelContext)
        try cancellationToken?.checkCancellation()
        let rag = await ragEngine.buildContext(query: contextQuery.query, budget: budget.charSections.rag, context: modelContext)
        try cancellationToken?.checkCancellation()
        let tctx = ToolExecutionContext(isForeground: turn.isForeground, appState: nil, modelContext: modelContext, permissionRegistry: .shared, metricsStore: metricsStore)
        let tools = await toolRegistry.availableDefinitions(context: tctx, source: turn.isForeground ? .modelProposed : .backgroundTrigger)
        try cancellationToken?.checkCancellation()
        let lowPower = turn.lowPowerMode
        let thermal = DeviceThermalState.from(processThermalState: turn.thermalState)
        let maxChars = budget.charSections.memories + budget.charSections.rag + budget.charSections.tools + budget.charSections.runtime
        let runtimeSelection = runtimeRouter.selection(for: turn)
        let selfModelSnapshot = SelfModelSnapshotBuilder.build(
            turn: turn,
            budget: budget,
            selectedRuntime: runtimeSelection,
            tools: tools
        )
        let selfModelSection = SelfModelContextProvider.section(for: selfModelSnapshot, budget: budget)
        let sections = PromptGroundingRenderer.render(memories: mem, rag: rag, tools: tools, lowPower: lowPower, thermal: thermal, selfModel: selfModelSection)
        try cancellationToken?.checkCancellation()
        let rendered = await Task.detached(priority: .userInitiated) {
            PromptGroundingRenderer.renderForPrompt(sections, maxChars: maxChars)
        }.value
        try cancellationToken?.checkCancellation()
        let grounding = AssistantGroundingContext(
            memoryCount: mem.selected.count,
            ragCount: rag.selected.count,
            toolCount: tools.count,
            estimatedChars: rendered.count,
            estimatedTokens: ContextBudgetAllocator.estimateTokens(forCharacterCount: rendered.count),
            contextProfile: budget.profile.rawValue,
            maxInputTokens: budget.maxInputTokens,
            ragConfidence: rag.confidence,
            memoryTierCounts: mem.tierCounts,
            contextQueryExpanded: contextQuery.expansionApplied,
            selfModelIncluded: true,
            selfModelSchemaVersion: selfModelSnapshot.schemaVersion,
            selfModelEstimatedChars: selfModelSection.estimatedChars,
            selfModelSourceIDs: selfModelSection.sourceIDs,
            selfModelMode: selfModelSnapshot.app.mode
        )
        try? await metricsStore.appendMetric(.init(timestamp: Date(), runtimeName: "grounding", taskKind: "\(turn.task)", modelIDHash: nil, policySummary: "profile=\(budget.profile.rawValue),m=\(mem.selected.count),r=\(rag.selected.count),t=\(tools.count),selfModel=\(selfModelSnapshot.schemaVersion),memTiers=\(Self.tierSummary(mem.tierCounts)),queryExpanded=\(contextQuery.expansionApplied),queryTerms=\(contextQuery.addedTerms.count)", latencyMs: nil, success: true, errorCode: nil, thermalState: .from(processThermalState: turn.thermalState), lowPowerMode: turn.lowPowerMode, memoryWarningCount: 0))
        return .init(grounding: grounding, budgetPlan: budget, sections: sections, renderedPromptContext: rendered, secureTools: tools, metricsSummary: "ok")
    }

    private static func tierSummary(_ counts: [String: Int]) -> String {
        counts
            .filter { $0.value > 0 }
            .sorted { $0.key < $1.key }
            .map { "\($0.key):\($0.value)" }
            .joined(separator: "|")
    }
}
