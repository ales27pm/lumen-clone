import Foundation

struct AssistantRuntimeRouter {
    let foundation: FoundationModelsRuntimeAdapter
    let llama: LlamaRuntimeAdapter
    let fallback: DeterministicFallbackRuntime
    let coreML: CoreMLRuntimeAdapter
    let allowDiagnosticFallbackSelection: Bool

    init(
        foundation: FoundationModelsRuntimeAdapter = .init(),
        llamaService: (any LlamaRuntimeStreamingService)? = nil,
        llamaSlot: LumenModelSlot = .mouth,
        llama: LlamaRuntimeAdapter? = nil,
        fallback: DeterministicFallbackRuntime = .init(),
        coreML: CoreMLRuntimeAdapter = .init(modelURL: nil),
        allowDiagnosticFallbackSelection: Bool = Self.defaultAllowDiagnosticFallbackSelection
    ) {
        self.foundation = foundation
        self.llama = llama ?? .live(service: llamaService ?? AppLlamaService.shared, slot: llamaSlot)
        self.fallback = fallback
        self.coreML = coreML
        self.allowDiagnosticFallbackSelection = allowDiagnosticFallbackSelection
    }

    private static var defaultAllowDiagnosticFallbackSelection: Bool {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }

    struct Selection: Sendable, Equatable {
        let runtime: AssistantRuntimeKind
        let reason: String
    }

    func selection(for context: AssistantTurnContext) -> Selection {
        let decision = ComputePolicy.decide(for: context)
        switch context.task {
        case .embedding, .safetyClassification:
            if decision.allowHeavyRuntime, llama.hasKnownSelectableEmbeddingRuntime {
                return .init(runtime: .llama, reason: "llama embedding available")
            }
            if coreML.supportsEmbeddings, coreML.isAvailable {
                return .init(runtime: .coreML, reason: "embedding uses CoreML")
            }
            if allowDiagnosticFallbackSelection, fallback.isAvailable {
                return .init(runtime: .deterministicFallback, reason: coreML.unavailableReason ?? "CoreML embedding unavailable")
            }
            return .init(runtime: .coreML, reason: coreML.unavailableReason ?? "CoreML embedding runtime disabled")
        case .backgroundTrigger, .remConsolidation:
            if decision.allowHeavyRuntime, llama.isAvailable {
                return .init(runtime: .llama, reason: "background heavy runtime allowed")
            }
            if allowDiagnosticFallbackSelection, fallback.isAvailable {
                return .init(runtime: .deterministicFallback, reason: decision.allowHeavyRuntime ? (llama.unavailableReason ?? "llama unavailable") : (decision.denialReason ?? "heavy runtime disallowed"))
            }
            return .init(runtime: .unavailable, reason: decision.allowHeavyRuntime ? (llama.unavailableReason ?? "llama unavailable") : (decision.denialReason ?? "heavy runtime disallowed"))
        case .chat, .agentPlan, .toolDecision, .summarization, .memoryExtraction, .speechCommandParsing:
            if context.prefersFoundationModels, foundation.supportsGeneration, foundation.isAvailable, decision.allowHeavyRuntime, context.isForeground, !DiskWriteBudget.shared.isGenerationActive() {
                return .init(runtime: .foundationModels, reason: "preferred on-device foundation runtime")
            }
            if decision.allowHeavyRuntime, llama.isAvailable {
                return .init(runtime: .llama, reason: "llama available")
            }
            if allowDiagnosticFallbackSelection, fallback.isAvailable {
                return .init(runtime: .deterministicFallback, reason: decision.allowHeavyRuntime ? "no capable runtime" : (decision.denialReason ?? "heavy runtime disallowed"))
            }
            return .init(runtime: .unavailable, reason: decision.allowHeavyRuntime ? (llama.unavailableReason ?? "llama unavailable") : (decision.denialReason ?? "heavy runtime disallowed"))
        }
    }

    func selectionIncludingRuntimeState(for context: AssistantTurnContext) async -> Selection {
        let decision = ComputePolicy.decide(for: context)
        switch context.task {
        case .embedding, .safetyClassification:
            if decision.allowHeavyRuntime, await llama.isEmbeddingSelectable() {
                return .init(runtime: .llama, reason: "llama embedding available")
            }
            if coreML.supportsEmbeddings, coreML.isAvailable {
                return .init(runtime: .coreML, reason: "embedding uses CoreML")
            }
            if allowDiagnosticFallbackSelection, fallback.isAvailable {
                return .init(runtime: .deterministicFallback, reason: coreML.unavailableReason ?? "CoreML embedding unavailable")
            }
            return .init(runtime: .coreML, reason: coreML.unavailableReason ?? "CoreML embedding runtime disabled")
        case .backgroundTrigger, .remConsolidation, .chat, .agentPlan, .toolDecision, .summarization, .memoryExtraction, .speechCommandParsing:
            return selection(for: context)
        }
    }

    func runtime(for context: AssistantTurnContext) -> AssistantRuntimeKind {
        selection(for: context).runtime
    }
}
