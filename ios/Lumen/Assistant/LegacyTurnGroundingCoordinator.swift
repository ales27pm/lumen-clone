import Foundation
import SwiftData

struct LegacyTurnGroundingOutput: Sendable {
    let grounding: AssistantGroundingContext
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

    func build(userMessage: String, conversationID: UUID?, turnID: UUID?, history: [(role: MessageRole, content: String)], modelContext: ModelContext, isBackground: Bool, task: AssistantTaskKind, role: String? = nil) async -> LegacyTurnGroundingOutput {
        let lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
        let thermal = DeviceThermalState.from(processThermalState: ProcessInfo.processInfo.thermalState)
        let roleKey = role.map { "\nrole=\($0)" } ?? ""
        let key = LegacyGroundingCache.Key(conversationID: conversationID, turnID: turnID, userDigest: LegacyGroundingCache.digest(userMessage + roleKey), background: isBackground, lowPowerMode: lowPower, thermalState: thermal)
        if let cached = await cache.get(key) {
            return .init(grounding: cached.grounding, sections: cached.sections, legacyTools: LegacyToolSchemaBridge.toLegacyToolDefinitions(cached.secureTools), promptInjection: cached.renderedPromptContext, metricsSummary: "cache")
        }
        let turn = AssistantTurnContext(task: task, input: userMessage, isForeground: !isBackground, lowPowerMode: lowPower, thermalState: ProcessInfo.processInfo.thermalState)
        let bundle = await bridge.build(userMessage: userMessage, conversationID: conversationID, turnID: turnID, history: history, modelContext: modelContext, turn: turn)
        var roleAwareBundle = bundle
        if let role, !role.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            var sections = bundle.sections
            let content = String(role.trimmingCharacters(in: .whitespacesAndNewlines).prefix(180))
            sections.append(.init(title: "Role stage", content: content, estimatedChars: content.count, sourceIDs: ["roleOrSlot"], privacyLevel: .low))
            let rendered = PromptGroundingRenderer.renderForPrompt(sections, maxChars: 3200)
            let grounding = AssistantGroundingContext(memoryCount: bundle.grounding.memoryCount, ragCount: bundle.grounding.ragCount, toolCount: bundle.grounding.toolCount, estimatedChars: rendered.count)
            roleAwareBundle = .init(grounding: grounding, sections: sections, renderedPromptContext: rendered, secureTools: bundle.secureTools, metricsSummary: bundle.metricsSummary)
        }
        await cache.put(key, bundle: roleAwareBundle)
        return .init(grounding: roleAwareBundle.grounding, sections: roleAwareBundle.sections, legacyTools: LegacyToolSchemaBridge.toLegacyToolDefinitions(roleAwareBundle.secureTools), promptInjection: roleAwareBundle.renderedPromptContext, metricsSummary: roleAwareBundle.metricsSummary)
    }

    func prepareGroundedRequest(_ request: LegacyGroundingRequest) async -> LegacyGroundingResult {
        await prepareGroundedRequest(request, provider: LegacyGroundingContextProvider())
    }

    func prepareGroundedRequest(_ request: LegacyGroundingRequest, provider: LegacyGroundingContextProvider) async -> LegacyGroundingResult {
        let context = provider.resolveContext()
        var degraded: [String] = []
        guard let modelContext = context else {
            if let reason = provider.degradedReason { degraded.append(reason) }
            let fallbackSections: [PromptGroundingSection] = [
                Self.memorySection(request.externalRelevantMemories.prefix(8)),
                Self.toolsSection(request.externalAvailableTools.prefix(24)),
                Self.runtimeSection("degraded-legacy-grounding")
            ].filter { !$0.content.isEmpty }
            let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: request.baseSystemPrompt, baseUserMessage: request.userMessage, sections: fallbackSections, policy: request.policy, roleMetadata: request.roleOrSlot, preventDoubleGrounding: request.preventDoubleGrounding)
            return .init(systemPrompt: assembled.systemPrompt, userMessage: assembled.userMessage, grounding: nil, sections: fallbackSections, bridgedTools: request.externalAvailableTools, degradedReasons: degraded, metricsSummary: "degraded", truncationOccurred: assembled.truncationOccurred)
        }

        let output = await build(userMessage: request.userMessage, conversationID: request.conversationID, turnID: request.turnID, history: request.history, modelContext: modelContext, isBackground: request.mode != .foreground, task: request.task, role: request.roleOrSlot)
        var sections = output.sections
        if !request.externalRelevantMemories.isEmpty {
            sections.append(Self.memorySection(request.externalRelevantMemories.prefix(6), sourceID: "legacyCallerMemory"))
        }
        let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: request.baseSystemPrompt, baseUserMessage: request.userMessage, sections: sections, policy: request.policy, roleMetadata: request.roleOrSlot, preventDoubleGrounding: request.preventDoubleGrounding)
        let tools = output.legacyTools.isEmpty ? request.externalAvailableTools : output.legacyTools
        return .init(systemPrompt: assembled.systemPrompt, userMessage: assembled.userMessage, grounding: output.grounding, sections: sections, bridgedTools: tools, degradedReasons: degraded, metricsSummary: output.metricsSummary, truncationOccurred: assembled.truncationOccurred)
    }
}
