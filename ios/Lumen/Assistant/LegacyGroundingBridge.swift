import Foundation
import SwiftData

struct LegacyGroundingBundle: Sendable {
    let grounding: AssistantGroundingContext
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

    init(toolRegistry: SecureToolRegistry? = nil, metricsStore: RuntimeMetricsStore? = nil) {
        self.toolRegistry = toolRegistry ?? .shared
        self.metricsStore = metricsStore ?? .shared
    }

    /// Constructs a legacy grounding bundle with rendered prompt context for an assistant turn.
    /// - Returns: A bundle containing selected memory items, RAG results, available tools, and the final rendered prompt context.
    /// - Throws: An error if the operation is cancelled or if an underlying operation fails.
    func build(userMessage: String, conversationID: UUID?, turnID: UUID?, history: [(role: MessageRole, content: String)], modelContext: ModelContext, turn: AssistantTurnContext, cancellationToken: AgentGroundingCancellationToken? = nil) async throws -> LegacyGroundingBundle {
        try cancellationToken?.checkCancellation()
        let budget = ContextBudgetAllocator.allocate(maxChars: 3200)
        let mem = memoryEngine.buildContext(query: userMessage, budget: budget.memories, context: modelContext)
        try cancellationToken?.checkCancellation()
        let rag = await ragEngine.buildContext(query: userMessage, budget: budget.rag, context: modelContext)
        try cancellationToken?.checkCancellation()
        let tctx = ToolExecutionContext(isForeground: turn.isForeground, appState: nil, modelContext: modelContext, permissionRegistry: .shared, metricsStore: metricsStore)
        let tools = await toolRegistry.availableDefinitions(context: tctx, source: turn.isForeground ? .modelProposed : .backgroundTrigger)
        try cancellationToken?.checkCancellation()
        let lowPower = turn.lowPowerMode
        let thermal = DeviceThermalState.from(processThermalState: turn.thermalState)
        let maxChars = budget.memories + budget.rag + budget.tools + budget.runtime
        let sections = PromptGroundingRenderer.render(memories: mem, rag: rag, tools: tools, lowPower: lowPower, thermal: thermal)
        try cancellationToken?.checkCancellation()
        let rendered = await Task.detached(priority: .userInitiated) {
            PromptGroundingRenderer.renderForPrompt(sections, maxChars: maxChars)
        }.value
        try cancellationToken?.checkCancellation()
        let grounding = AssistantGroundingContext(memoryCount: mem.selected.count, ragCount: rag.selected.count, toolCount: tools.count, estimatedChars: rendered.count)
        try? await metricsStore.appendMetric(.init(timestamp: Date(), runtimeName: "grounding", taskKind: "\(turn.task)", modelIDHash: nil, policySummary: "m=\(mem.selected.count),r=\(rag.selected.count),t=\(tools.count)", latencyMs: nil, success: true, errorCode: nil, thermalState: .from(processThermalState: turn.thermalState), lowPowerMode: turn.lowPowerMode, memoryWarningCount: 0))
        return .init(grounding: grounding, sections: sections, renderedPromptContext: rendered, secureTools: tools, metricsSummary: "ok")
    }
}
