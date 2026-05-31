import Foundation
import SwiftData

@MainActor
final class AssistantKernel {
    enum KernelError: Error, Sendable, Equatable {
        case unsupportedTaskForTextTurn(AssistantTaskKind)
        case unsupportedRuntimeForTextTurn(AssistantRuntimeKind)
    }

    private let router: AssistantRuntimeRouter
    private let metricsStore: RuntimeMetricsStore
    private let toolRegistry: SecureToolRegistry
    private let memoryEngine = MemoryEngine()
    private let ragEngine = RAGEngine()

    init(router: AssistantRuntimeRouter = .init(), metricsStore: RuntimeMetricsStore = .shared, toolRegistry: SecureToolRegistry = .shared) {
        self.router = router
        self.metricsStore = metricsStore
        self.toolRegistry = toolRegistry
    }

    func selectRuntime(for context: AssistantTurnContext) -> AssistantRuntimeKind {
        router.runtime(for: context)
    }

    func buildGroundingContext(turn: AssistantTurnContext, modelContext: ModelContext?) async -> AssistantGroundingContext {
        guard let modelContext else { return .empty }
        let budget = ContextBudgetAllocator.allocate(maxChars: 4000)
        let mem = memoryEngine.buildContext(query: turn.input, budget: budget.memories, context: modelContext)
        let rag = await ragEngine.buildContext(query: turn.input, budget: budget.rag, context: modelContext)
        let tctx = ToolExecutionContext(isForeground: turn.isForeground, appState: nil, modelContext: modelContext, permissionRegistry: .shared, metricsStore: metricsStore)
        let defs = await toolRegistry.availableDefinitions(context: tctx, source: turn.isForeground ? .modelProposed : .backgroundTrigger)
        return .init(memoryCount: mem.selected.count, ragCount: rag.selected.count, toolCount: defs.count, estimatedChars: mem.totalChars + rag.totalChars)
    }

    func runTextTurn(_ context: AssistantTurnContext) async throws -> String {
        switch context.task {
        case .embedding, .safetyClassification:
            throw KernelError.unsupportedTaskForTextTurn(context.task)
        default:
            break
        }
        let selection = router.selection(for: context)
        guard selection.runtime != .coreML else { throw KernelError.unsupportedRuntimeForTextTurn(.coreML) }
        let decision = ComputePolicy.decide(for: context)
        let request = TextGenerationRequest(prompt: context.input, systemPrompt: "", maxTokens: decision.maxTokens)
        let start = Date()

        do {
            let output: String
            switch selection.runtime {
            case .foundationModels:
                output = try await router.foundation.generate(request: request)
            case .llama:
                output = try await router.llama.generate(request: request)
            case .deterministicFallback, .coreML:
                output = try await router.fallback.generate(request: request)
            }
            let latency = Int(Date().timeIntervalSince(start) * 1000)
            try? await metricsStore.appendMetric(RuntimeMetric(timestamp: Date(), runtimeName: selection.runtime.rawValue, taskKind: "\(context.task)", modelIDHash: nil, policySummary: selection.reason, latencyMs: latency, success: true, errorCode: nil, thermalState: .from(processThermalState: context.thermalState), lowPowerMode: context.lowPowerMode, memoryWarningCount: 0))
            return output
        } catch {
            let latency = Int(Date().timeIntervalSince(start) * 1000)
            try? await metricsStore.appendMetric(RuntimeMetric(timestamp: Date(), runtimeName: selection.runtime.rawValue, taskKind: "\(context.task)", modelIDHash: nil, policySummary: selection.reason, latencyMs: latency, success: false, errorCode: RuntimeMetricErrorSanitizer.code(for: error), thermalState: .from(processThermalState: context.thermalState), lowPowerMode: context.lowPowerMode, memoryWarningCount: 0))
            throw error
        }
    }
}


extension AssistantKernel {
    func executeTool(_ invocation: ToolInvocation, modelContext: ModelContext? = nil) async -> ToolResult {
        let ctx = ToolExecutionContext(isForeground: invocation.source != .backgroundTrigger, appState: nil, modelContext: modelContext, permissionRegistry: .shared, metricsStore: metricsStore)
        return await toolRegistry.execute(invocation, context: ctx)
    }
}
