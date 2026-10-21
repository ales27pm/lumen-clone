import Foundation

struct AssistantRuntimeRouter {
    let foundation: FoundationModelsRuntimeAdapter
    let llama: LlamaRuntimeAdapter
    let fallback: DeterministicFallbackRuntime
    let coreML: CoreMLRuntimeAdapter
    let coreAI: CoreAIGemmaRuntimeAdapter
    let allowDiagnosticFallbackSelection: Bool

    init(
        foundation: FoundationModelsRuntimeAdapter = .init(),
        llamaService: (any LlamaRuntimeStreamingService)? = nil,
        llamaSlot: LumenModelSlot = .mouth,
        llama: LlamaRuntimeAdapter? = nil,
        fallback: DeterministicFallbackRuntime = .init(),
        coreML: CoreMLRuntimeAdapter = .init(modelURL: nil),
        coreAI: CoreAIGemmaRuntimeAdapter = .init(),
        allowDiagnosticFallbackSelection: Bool = Self.defaultAllowDiagnosticFallbackSelection
    ) {
        self.foundation = foundation
        self.llama = llama ?? .live(service: llamaService ?? AppLlamaService.shared, slot: llamaSlot)
        self.fallback = fallback
        self.coreML = coreML
        self.coreAI = coreAI
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

    private func embeddingSelection(
        decision: ComputeDecision,
        llamaEmbeddingSelectable: Bool,
        allowDiagnosticFallback: Bool
    ) -> Selection {
        if decision.allowHeavyRuntime, llamaEmbeddingSelectable {
            return .init(runtime: .llama, reason: "llama embedding available")
        }
        if coreML.supportsEmbeddings, coreML.isAvailable {
            return .init(runtime: .coreML, reason: "embedding uses CoreML")
        }
        if allowDiagnosticFallback, fallback.isAvailable {
            return .init(runtime: .deterministicFallback, reason: coreML.unavailableReason ?? "CoreML embedding unavailable")
        }
        return .init(runtime: .coreML, reason: coreML.unavailableReason ?? "CoreML embedding runtime disabled")
    }

    func selection(for context: AssistantTurnContext) -> Selection {
        let decision = ComputePolicy.decide(for: context)
        switch context.task {
        case .embedding, .safetyClassification:
            return embeddingSelection(
                decision: decision,
                llamaEmbeddingSelectable: llama.hasKnownSelectableEmbeddingRuntime,
                allowDiagnosticFallback: allowDiagnosticFallbackSelection
            )
        case .backgroundTrigger, .remConsolidation:
            if coreAI.isPreferred {
                return .init(runtime: .unavailable, reason: "Gemma generation is foreground only; background maintenance does not select another model")
            }
            if decision.allowHeavyRuntime, llama.isAvailable {
                return .init(runtime: .llama, reason: "background heavy runtime allowed")
            }
            if allowDiagnosticFallbackSelection, fallback.isAvailable {
                return .init(runtime: .deterministicFallback, reason: decision.allowHeavyRuntime ? (llama.unavailableReason ?? "llama unavailable") : (decision.denialReason ?? "heavy runtime disallowed"))
            }
            return .init(runtime: .unavailable, reason: decision.allowHeavyRuntime ? (llama.unavailableReason ?? "llama unavailable") : (decision.denialReason ?? "heavy runtime disallowed"))
        case .chat, .agentPlan, .toolDecision, .summarization, .memoryExtraction, .speechCommandParsing:
            if coreAI.isPreferred {
                guard context.isForeground else {
                    return .init(runtime: .unavailable, reason: "Gemma generation requires the foreground app; no alternate runtime is selected")
                }
                guard decision.allowHeavyRuntime else {
                    return .init(runtime: .unavailable, reason: decision.denialReason ?? "heavy runtime disallowed")
                }
                guard coreAI.isAvailable else {
                    return .init(runtime: .unavailable, reason: coreAI.unavailableReason ?? "Core AI Gemma unavailable")
                }
                return .init(runtime: .coreAI, reason: "Core AI Gemma selected")
            }
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
            return embeddingSelection(
                decision: decision,
                llamaEmbeddingSelectable: await llama.isEmbeddingSelectable(),
                allowDiagnosticFallback: false
            )
        case .backgroundTrigger, .remConsolidation, .chat, .agentPlan, .toolDecision, .summarization, .memoryExtraction, .speechCommandParsing:
            return selection(for: context)
        }
    }

    func runtime(for context: AssistantTurnContext) -> AssistantRuntimeKind {
        selection(for: context).runtime
    }
}
