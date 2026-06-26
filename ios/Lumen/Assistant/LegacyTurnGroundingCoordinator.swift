import Foundation
import SwiftData

struct LegacyTurnGroundingOutput: Sendable {
    let grounding: AssistantGroundingContext
    let budgetPlan: ContextBudgetPlan
    let sections: [PromptGroundingSection]
    let legacyTools: [ToolDefinition]
    let promptInjection: String
    let metricsSummary: String
}

@MainActor
final class LegacyTurnGroundingCoordinator {
    static let shared = LegacyTurnGroundingCoordinator()
    private let bridge = LegacyGroundingBridge()
    private let cache = LegacyGroundingCache()

    private static func memorySection(_ memories: ArraySlice<MemoryContextItem>, sourceID: String? = nil) -> PromptGroundingSection {
        let items = Array(memories)
        let content = items.map { "- \($0.content)" }.joined(separator: "\n")
        return .init(title: "Relevant memories", content: content, estimatedChars: content.count, sourceIDs: sourceID.map { [$0] } ?? items.map { $0.id.uuidString }, privacyLevel: .moderate)
    }

    private static func toolsSection(_ tools: ArraySlice<ToolDefinition>) -> PromptGroundingSection {
        let items = Array(tools)
        let content = items.map { "- \($0.id): \($0.description)" }.joined(separator: "\n")
        return .init(title: "Available tools", content: content, estimatedChars: content.count, sourceIDs: items.map { $0.id }, privacyLevel: .low)
    }

    private static func runtimeSection(_ content: String) -> PromptGroundingSection {
        .init(title: "Runtime policy", content: content, estimatedChars: content.count, sourceIDs: [], privacyLevel: .low)
    }

    func build(userMessage: String, conversationID: UUID?, turnID: UUID?, history: [(role: MessageRole, content: String)], modelContext: ModelContext, isBackground: Bool, task: AssistantTaskKind, role: String? = nil, cancellationToken: AgentGroundingCancellationToken? = nil) async throws -> LegacyTurnGroundingOutput {
        try cancellationToken?.checkCancellation()
        AgentGroundingInstrumentation.mark("before LegacyGroundingBridge.build", metrics: .init(promptChars: userMessage.count))
        let bridgeStart = ProcessInfo.processInfo.systemUptime
        let lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
        let thermal = DeviceThermalState.from(processThermalState: ProcessInfo.processInfo.thermalState)
        let roleKey = role.map { "\nrole=\($0)" } ?? ""
        let key = LegacyGroundingCache.Key(conversationID: conversationID, turnID: turnID, userDigest: LegacyGroundingCache.digest(userMessage + roleKey), background: isBackground, lowPowerMode: lowPower, thermalState: thermal)
        if let cached = await cache.get(key) {
            AgentGroundingInstrumentation.mark("after LegacyGroundingBridge.build", metrics: .init(sectionCount: cached.sections.count, toolCount: cached.secureTools.count, memoryCount: cached.grounding.memoryCount, promptChars: cached.renderedPromptContext.count), elapsedMs: AgentGroundingInstrumentation.elapsedMs(since: bridgeStart))
            PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .groundingCost, values: ["elapsedMs": String(Int(AgentGroundingInstrumentation.elapsedMs(since: bridgeStart))), "sectionCount": String(cached.sections.count), "promptChars": String(cached.renderedPromptContext.count), "toolCount": String(cached.secureTools.count), "memoryCount": String(cached.grounding.memoryCount), "source": "cache"]))
            return .init(grounding: cached.grounding, budgetPlan: cached.budgetPlan, sections: cached.sections, legacyTools: ToolSchemaBridge.toCatalogToolDefinitions(cached.secureTools), promptInjection: cached.renderedPromptContext, metricsSummary: "cache")
        }
        let turn = AssistantTurnContext(task: task, input: userMessage, isForeground: !isBackground, lowPowerMode: lowPower, thermalState: ProcessInfo.processInfo.thermalState)
        let bundle = try await bridge.build(userMessage: userMessage, conversationID: conversationID, turnID: turnID, history: history, modelContext: modelContext, turn: turn, cancellationToken: cancellationToken)
        try cancellationToken?.checkCancellation()
        AgentGroundingInstrumentation.mark("after LegacyGroundingBridge.build", metrics: .init(sectionCount: bundle.sections.count, toolCount: bundle.secureTools.count, memoryCount: bundle.grounding.memoryCount, promptChars: bundle.renderedPromptContext.count), elapsedMs: AgentGroundingInstrumentation.elapsedMs(since: bridgeStart))
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .groundingCost, values: ["elapsedMs": String(Int(AgentGroundingInstrumentation.elapsedMs(since: bridgeStart))), "sectionCount": String(bundle.sections.count), "promptChars": String(bundle.renderedPromptContext.count), "toolCount": String(bundle.secureTools.count), "memoryCount": String(bundle.grounding.memoryCount), "source": "bridge"]))
        var roleAwareBundle = bundle
        if let role, !role.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            var sections = bundle.sections
            let content = String(role.trimmingCharacters(in: .whitespacesAndNewlines).prefix(180))
            sections.append(.init(title: "Role stage", content: content, estimatedChars: content.count, sourceIDs: ["roleOrSlot"], privacyLevel: .low))
            let rendered = await Task.detached(priority: .userInitiated) {
                PromptGroundingRenderer.renderForPrompt(sections, maxChars: 3200)
            }.value
            let grounding = AssistantGroundingContext(
                memoryCount: bundle.grounding.memoryCount,
                ragCount: bundle.grounding.ragCount,
                toolCount: bundle.grounding.toolCount,
                estimatedChars: rendered.count,
                estimatedTokens: ContextBudgetAllocator.estimateTokens(forCharacterCount: rendered.count),
                contextProfile: bundle.grounding.contextProfile,
                maxInputTokens: bundle.grounding.maxInputTokens,
                ragConfidence: bundle.grounding.ragConfidence,
                memoryTierCounts: bundle.grounding.memoryTierCounts,
                contextQueryExpanded: bundle.grounding.contextQueryExpanded
            )
            roleAwareBundle = .init(grounding: grounding, budgetPlan: bundle.budgetPlan, sections: sections, renderedPromptContext: rendered, secureTools: bundle.secureTools, metricsSummary: bundle.metricsSummary)
        }
        await cache.put(key, bundle: roleAwareBundle)
        return .init(grounding: roleAwareBundle.grounding, budgetPlan: roleAwareBundle.budgetPlan, sections: roleAwareBundle.sections, legacyTools: ToolSchemaBridge.toCatalogToolDefinitions(roleAwareBundle.secureTools), promptInjection: roleAwareBundle.renderedPromptContext, metricsSummary: roleAwareBundle.metricsSummary)
    }

    func prepareGroundedRequest(_ request: LegacyGroundingRequest) async -> LegacyGroundingResult {
        await prepareGroundedRequest(request, provider: LegacyGroundingContextProvider())
    }

    func prepareGroundedRequest(_ request: LegacyGroundingRequest, provider: LegacyGroundingContextProvider, cancellationToken: AgentGroundingCancellationToken? = nil) async -> LegacyGroundingResult {
        do { try cancellationToken?.checkCancellation() } catch { return Self.cancelledResult(for: request) }
        AgentGroundingInstrumentation.mark("before LegacyTurnGroundingCoordinator.prepareGroundedRequest", metrics: .init(toolCount: request.externalAvailableTools.count, memoryCount: request.externalRelevantMemories.count, promptChars: request.userMessage.count))
        let requestStart = ProcessInfo.processInfo.systemUptime
        let context = provider.resolveContext()
        var degraded: [String] = []
        guard let modelContext = context else {
            if let reason = provider.degradedReason { degraded.append(reason) }
            let fallbackSections: [PromptGroundingSection] = [
                Self.memorySection(request.externalRelevantMemories.prefix(8)),
                Self.toolsSection(request.externalAvailableTools.prefix(24)),
                Self.runtimeSection("degraded-legacy-grounding")
            ].filter { !$0.content.isEmpty }
            AgentGroundingInstrumentation.mark("before LegacyPromptAssembler.assemble", metrics: .init(sectionCount: fallbackSections.count, toolCount: request.externalAvailableTools.count, memoryCount: request.externalRelevantMemories.count, promptChars: request.userMessage.count))
            let assembleStart = ProcessInfo.processInfo.systemUptime
            let budgetPlan = Self.budgetPlan(for: request)
            let assembled = await Task.detached(priority: .userInitiated) {
                LegacyPromptAssembler.assemble(baseSystemPrompt: request.baseSystemPrompt, baseUserMessage: request.userMessage, sections: fallbackSections, policy: request.policy, roleMetadata: request.roleOrSlot, preventDoubleGrounding: request.preventDoubleGrounding, budgetPlan: budgetPlan)
            }.value
            AgentGroundingInstrumentation.mark("after LegacyPromptAssembler.assemble", metrics: .init(sectionCount: fallbackSections.count, toolCount: request.externalAvailableTools.count, memoryCount: request.externalRelevantMemories.count, promptChars: assembled.estimatedChars), elapsedMs: AgentGroundingInstrumentation.elapsedMs(since: assembleStart))
            AgentGroundingInstrumentation.mark("after LegacyTurnGroundingCoordinator.prepareGroundedRequest", metrics: .init(sectionCount: fallbackSections.count, toolCount: request.externalAvailableTools.count, memoryCount: request.externalRelevantMemories.count, promptChars: assembled.estimatedChars), elapsedMs: AgentGroundingInstrumentation.elapsedMs(since: requestStart))
            PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .groundingCost, values: ["elapsedMs": String(Int(AgentGroundingInstrumentation.elapsedMs(since: requestStart))), "sectionCount": String(fallbackSections.count), "promptChars": String(assembled.estimatedChars), "toolCount": String(request.externalAvailableTools.count), "memoryCount": String(request.externalRelevantMemories.count), "source": "degraded"]))
            RuntimeFallbackLogger.record(
                source: "legacy-grounding-coordinator",
                primaryBehavior: "build grounding from live ModelContext",
                fallbackBehavior: "assemble degraded external memory/tool sections",
                reason: degraded.joined(separator: ",").isEmpty ? "missing-model-context" : degraded.joined(separator: ","),
                consequence: "grounding may omit app-state and persisted context needed by the primary behavior",
                values: [
                    "turnID": request.turnID?.uuidString ?? "none",
                    "conversationID": request.conversationID?.uuidString ?? "none",
                    "promptSHA256": RuntimeFallbackLogger.promptHash(request.userMessage),
                    "promptChars": String(request.userMessage.count),
                    "contextProfile": budgetPlan.profile.rawValue,
                    "maxInputTokens": String(budgetPlan.maxInputTokens),
                    "sectionCount": String(fallbackSections.count),
                    "toolCount": String(request.externalAvailableTools.count),
                    "memoryCount": String(request.externalRelevantMemories.count)
                ]
            )
            return .init(systemPrompt: assembled.systemPrompt, userMessage: assembled.userMessage, grounding: nil, sections: fallbackSections, bridgedTools: request.externalAvailableTools, degradedReasons: degraded, metricsSummary: "degraded", truncationOccurred: assembled.truncationOccurred)
        }

        let output: LegacyTurnGroundingOutput
        do {
            output = try await build(userMessage: request.userMessage, conversationID: request.conversationID, turnID: request.turnID, history: request.history, modelContext: modelContext, isBackground: request.mode != .foreground, task: request.task, role: request.roleOrSlot, cancellationToken: cancellationToken)
        } catch {
            return Self.cancelledResult(for: request)
        }
        do { try cancellationToken?.checkCancellation() } catch { return Self.cancelledResult(for: request) }
        var sections = output.sections
        if !request.externalRelevantMemories.isEmpty {
            sections.append(Self.memorySection(request.externalRelevantMemories.prefix(6), sourceID: "legacyCallerMemory"))
        }
        AgentGroundingInstrumentation.mark("before LegacyPromptAssembler.assemble", metrics: .init(sectionCount: sections.count, toolCount: output.legacyTools.count, memoryCount: request.externalRelevantMemories.count, promptChars: request.userMessage.count))
        let assembleStart = ProcessInfo.processInfo.systemUptime
        let budgetPlan = output.budgetPlan
        let assembled = await Task.detached(priority: .userInitiated) {
            LegacyPromptAssembler.assemble(baseSystemPrompt: request.baseSystemPrompt, baseUserMessage: request.userMessage, sections: sections, policy: request.policy, roleMetadata: request.roleOrSlot, preventDoubleGrounding: request.preventDoubleGrounding, budgetPlan: budgetPlan)
        }.value
        AgentGroundingInstrumentation.mark("after LegacyPromptAssembler.assemble", metrics: .init(sectionCount: sections.count, toolCount: output.legacyTools.count, memoryCount: request.externalRelevantMemories.count, promptChars: assembled.estimatedChars), elapsedMs: AgentGroundingInstrumentation.elapsedMs(since: assembleStart))
        let tools = output.legacyTools.isEmpty ? request.externalAvailableTools : output.legacyTools
        AgentGroundingInstrumentation.mark("after LegacyTurnGroundingCoordinator.prepareGroundedRequest", metrics: .init(sectionCount: sections.count, toolCount: tools.count, memoryCount: output.grounding.memoryCount, promptChars: assembled.estimatedChars), elapsedMs: AgentGroundingInstrumentation.elapsedMs(since: requestStart))
        PersistentRuntimeDiagnosticsObserver.shared.emit(.init(kind: .groundingCost, values: ["elapsedMs": String(Int(AgentGroundingInstrumentation.elapsedMs(since: requestStart))), "sectionCount": String(sections.count), "promptChars": String(assembled.estimatedChars), "toolCount": String(tools.count), "memoryCount": String(output.grounding.memoryCount), "source": "coordinator"]))
        return .init(systemPrompt: assembled.systemPrompt, userMessage: assembled.userMessage, grounding: output.grounding, sections: sections, bridgedTools: tools, degradedReasons: degraded, metricsSummary: output.metricsSummary, truncationOccurred: assembled.truncationOccurred)
    }

    private static func budgetPlan(for request: LegacyGroundingRequest) -> ContextBudgetPlan {
        let turn = AssistantTurnContext(
            task: request.task,
            input: request.userMessage,
            isForeground: request.mode == .foreground,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            thermalState: ProcessInfo.processInfo.thermalState
        )
        return ContextBudgetAllocator.allocate(for: turn, maxInputTokens: 800)
    }

    private static func cancelledResult(for request: LegacyGroundingRequest) -> LegacyGroundingResult {
        .init(systemPrompt: request.baseSystemPrompt, userMessage: request.userMessage, grounding: nil, sections: [], bridgedTools: [], degradedReasons: ["cancelled"], metricsSummary: "cancelled", truncationOccurred: false)
    }
}
