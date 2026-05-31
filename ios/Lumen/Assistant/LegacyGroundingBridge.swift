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

    init(toolRegistry: SecureToolRegistry = .shared, metricsStore: RuntimeMetricsStore = .shared) {
        self.toolRegistry = toolRegistry
        self.metricsStore = metricsStore
    }

    func build(userMessage: String, conversationID: UUID?, turnID: UUID?, history: [(role: MessageRole, content: String)], modelContext: ModelContext, turn: AssistantTurnContext) async -> LegacyGroundingBundle {
        let budget = ContextBudgetAllocator.allocate(maxChars: 3200)
        let mem = memoryEngine.buildContext(query: userMessage, budget: budget.memories, context: modelContext)
        let rag = await ragEngine.buildContext(query: userMessage, budget: budget.rag, context: modelContext)
        let tctx = ToolExecutionContext(isForeground: turn.isForeground, appState: nil, modelContext: modelContext, permissionRegistry: .shared, metricsStore: metricsStore)
        let tools = await toolRegistry.availableDefinitions(context: tctx, source: turn.isForeground ? .modelProposed : .backgroundTrigger)
        let sections = PromptGroundingRenderer.render(memories: mem, rag: rag, tools: tools, lowPower: turn.lowPowerMode, thermal: .from(processThermalState: turn.thermalState))
        let rendered = PromptGroundingRenderer.renderForPrompt(sections, maxChars: budget.memories + budget.rag + budget.tools + budget.runtime)
        let grounding = AssistantGroundingContext(memoryCount: mem.selected.count, ragCount: rag.selected.count, toolCount: tools.count, estimatedChars: rendered.count)
        try? await metricsStore.appendMetric(.init(timestamp: Date(), runtimeName: "grounding", taskKind: "\(turn.task)", modelIDHash: nil, policySummary: "m=\(mem.selected.count),r=\(rag.selected.count),t=\(tools.count)", latencyMs: nil, success: true, errorCode: nil, thermalState: .from(processThermalState: turn.thermalState), lowPowerMode: turn.lowPowerMode, memoryWarningCount: 0))
        return .init(grounding: grounding, sections: sections, renderedPromptContext: rendered, secureTools: tools, metricsSummary: "ok")
    }
}
