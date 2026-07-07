import Foundation
import SwiftData

@MainActor
final class AssistantKernel {
    static let shared = AssistantKernel()

    enum KernelError: Error, Sendable, Equatable {
        case unsupportedTaskForTextTurn(AssistantTaskKind)
        case unsupportedRuntimeForTextTurn(AssistantRuntimeKind)
    }

    private let router: AssistantRuntimeRouter
    let metricsStore: RuntimeMetricsStore
    let toolRegistry: SecureToolRegistry
    private let memoryEngine = MemoryEngine()
    private let ragEngine = RAGEngine()

    init(
        router: AssistantRuntimeRouter? = nil,
        metricsStore: RuntimeMetricsStore? = nil,
        toolRegistry: SecureToolRegistry? = nil
    ) {
        self.router = router ?? AssistantRuntimeRouter()
        self.metricsStore = metricsStore ?? .shared
        self.toolRegistry = toolRegistry ?? .shared
    }

    func selectRuntime(for context: AssistantTurnContext) -> AssistantRuntimeKind {
        selectRuntimeSelection(for: context).runtime
    }

    func selectRuntimeSelection(for context: AssistantTurnContext) -> AssistantRuntimeRouter.Selection {
        router.selection(for: context)
    }

    func buildGroundingContext(turn: AssistantTurnContext, modelContext: ModelContext?) async -> AssistantGroundingContext {
        guard let modelContext else { return .empty }
        let budget = ContextBudgetAllocator.allocate(for: turn)
        let contextQuery = ContextQueryRewriter.rewrite(userInput: turn.input, history: turn.history, relevantMemories: turn.relevantMemories)
        let mem = memoryEngine.buildContext(query: contextQuery.query, budget: budget.charSections.memories, context: modelContext)
        let rag = await ragEngine.buildContext(query: contextQuery.query, budget: budget.charSections.rag, context: modelContext)
        let tctx = ToolExecutionContext(isForeground: turn.isForeground, appState: nil, modelContext: modelContext, permissionRegistry: .shared, metricsStore: metricsStore)
        let defs = await toolRegistry.availableDefinitions(context: tctx, source: turn.isForeground ? .modelProposed : .backgroundTrigger)
        let runtimeSelection = router.selection(for: turn)
        let selfModelSnapshot = SelfModelSnapshotBuilder.build(
            turn: turn,
            budget: budget,
            selectedRuntime: runtimeSelection,
            tools: defs
        )
        let selfModelSection = SelfModelContextProvider.section(for: selfModelSnapshot, budget: budget)
        return .init(
            memoryCount: mem.selected.count,
            ragCount: rag.selected.count,
            toolCount: defs.count,
            estimatedChars: mem.totalChars + rag.totalChars + selfModelSection.estimatedChars,
            estimatedTokens: ContextBudgetAllocator.estimateTokens(forCharacterCount: mem.totalChars + selfModelSection.estimatedChars) + rag.totalTokens,
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
        let request = TextGenerationRequest(
            prompt: context.input,
            systemPrompt: context.systemPrompt,
            history: context.history,
            temperature: context.temperature,
            topP: context.topP,
            repetitionPenalty: context.repetitionPenalty,
            maxTokens: min(context.maxTokens, decision.maxTokens),
            relevantMemories: context.relevantMemories,
            attachments: context.attachments
        )
        let start = Date()
        if selection.runtime == .deterministicFallback {
            RuntimeFallbackLogger.record(
                source: "assistant-kernel-runtime-selection",
                primaryBehavior: "run preferred on-device model runtime",
                fallbackBehavior: "run debug diagnostic deterministic runtime",
                reason: selection.reason,
                consequence: "debug diagnostic path selected; Release routing excludes this runtime",
                values: [
                    "task": String(describing: context.task),
                    "promptSHA256": RuntimeFallbackLogger.promptHash(context.input),
                    "promptChars": String(context.input.count),
                    "isForeground": String(context.isForeground),
                    "allowHeavyRuntime": String(decision.allowHeavyRuntime),
                    "budgetPolicy": decision.budgetPolicy.rawValue,
                    "budgetDenialReason": decision.denialReason ?? "none"
                ]
            )
        }

        do {
            let output = try await generateText(request: request, runtime: selection.runtime)
            let latency = Int(Date().timeIntervalSince(start) * 1000)
            try? await metricsStore.appendMetric(RuntimeMetric(timestamp: Date(), runtimeName: selection.runtime.rawValue, taskKind: "\(context.task)", modelIDHash: nil, policySummary: selection.reason, latencyMs: latency, success: true, errorCode: nil, thermalState: .from(processThermalState: context.thermalState), lowPowerMode: context.lowPowerMode, memoryWarningCount: 0))
            return output
        } catch {
            let latency = Int(Date().timeIntervalSince(start) * 1000)
            try? await metricsStore.appendMetric(RuntimeMetric(timestamp: Date(), runtimeName: selection.runtime.rawValue, taskKind: "\(context.task)", modelIDHash: nil, policySummary: selection.reason, latencyMs: latency, success: false, errorCode: RuntimeMetricErrorSanitizer.code(for: error), thermalState: .from(processThermalState: context.thermalState), lowPowerMode: context.lowPowerMode, memoryWarningCount: 0))
            throw error
        }
    }

    private func generateText(request: TextGenerationRequest, runtime: AssistantRuntimeKind) async throws -> String {
        switch runtime {
        case .foundationModels:
            return try await router.foundation.generate(request: request)
        case .llama:
            return try await router.llama.generate(request: request)
        case .deterministicFallback:
            return try await router.fallback.generate(request: request)
        case .coreML:
            throw KernelError.unsupportedRuntimeForTextTurn(.coreML)
        }
    }
}


extension AssistantKernel {
    func executeTool(_ invocation: ToolInvocation, modelContext: ModelContext? = nil) async -> ToolResult {
        let ctx = ToolExecutionContext(isForeground: invocation.source != .backgroundTrigger, appState: nil, modelContext: modelContext, permissionRegistry: .shared, metricsStore: metricsStore)
        return await toolRegistry.execute(invocation, context: ctx)
    }
}
