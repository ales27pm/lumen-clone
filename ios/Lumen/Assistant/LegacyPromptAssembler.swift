import Foundation

struct LegacyPromptAssembled: Sendable {
    let systemPrompt: String
    let userMessage: String
    let groundingAppendix: String
    let estimatedChars: Int
    let estimatedTokens: Int
    let contextProfile: String?
    let maxInputTokens: Int?
    let truncationOccurred: Bool
    let memorySectionChars: Int
    let ragSectionChars: Int
    let toolSectionChars: Int
    let runtimeSectionChars: Int
}

enum LegacyPromptAssembler {
    static func assemble(
        baseSystemPrompt: String,
        baseUserMessage: String,
        sections: [PromptGroundingSection],
        policy: LegacyPromptInjectionPolicy,
        roleMetadata: String? = nil,
        preventDoubleGrounding: Bool = true,
        budgetPlan: ContextBudgetPlan? = nil
    ) -> LegacyPromptAssembled {
        func titled(_ name: String, _ body: String) -> String { body.isEmpty ? "" : "[\(name)]\n\(body)\n" }
        let mem = sections.first(where: { isMemorySection($0.title) && (policy.allowSensitiveSections || $0.privacyLevel != .sensitive) })?.content ?? ""
        let rag = sections.first(where: { isSourceSection($0.title) && (policy.allowSensitiveSections || $0.privacyLevel != .sensitive) })?.content ?? ""
        let tool = sections.first(where: { isToolSection($0.title) })?.content ?? ""
        let runtime = sections.first(where: { isRuntimeSection($0.title) })?.content ?? ""

        let caps = sectionCaps(policy: policy, budgetPlan: budgetPlan)
        let memC = String(mem.prefix(caps.memories))
        let ragC = String(rag.prefix(caps.rag))
        let toolC = String(tool.prefix(caps.tools))
        let runC = String(runtime.prefix(caps.runtime))
        let roleC = roleMetadata.map { String($0.prefix(180)) } ?? ""
        var appendix = PromptGroundingIdempotencyGuard.marker + "\n"
        appendix += titled("LOCAL MEMORY", memC)
        appendix += titled("LOCAL SOURCES", ragC)
        appendix += titled("AVAILABLE LOCAL TOOLS", toolC)
        appendix += titled("RUNTIME POLICY", runC)
        if !roleC.isEmpty { appendix += titled("ROLE STAGE", roleC) }
        let normalizedBase: String
        if preventDoubleGrounding {
            let stripped = PromptGroundingIdempotencyGuard.stripExistingGrounding(from: baseUserMessage)
            normalizedBase = stripped.text
            // A single user-authored grounding-like header is ambiguous, not generated grounding.
            // Preserve the user text and still inject the bounded generated appendix.
        } else {
            normalizedBase = baseUserMessage
        }
        let finalUser = normalizedBase + (appendix.isEmpty ? "" : "\n\n" + appendix)
        let trunc = mem.count > memC.count || rag.count > ragC.count || tool.count > toolC.count || runtime.count > runC.count || (roleMetadata?.count ?? 0) > roleC.count
        let estimatedChars = finalUser.count + baseSystemPrompt.count
        return .init(
            systemPrompt: baseSystemPrompt,
            userMessage: finalUser,
            groundingAppendix: appendix,
            estimatedChars: estimatedChars,
            estimatedTokens: ContextBudgetAllocator.estimateTokens(forCharacterCount: estimatedChars),
            contextProfile: budgetPlan?.profile.rawValue,
            maxInputTokens: budgetPlan?.maxInputTokens,
            truncationOccurred: trunc,
            memorySectionChars: memC.count,
            ragSectionChars: ragC.count,
            toolSectionChars: toolC.count,
            runtimeSectionChars: runC.count
        )
    }

    private static func sectionCaps(policy: LegacyPromptInjectionPolicy, budgetPlan: ContextBudgetPlan?) -> (memories: Int, rag: Int, tools: Int, runtime: Int) {
        guard let budgetPlan else {
            return (policy.memoryMax, policy.ragMax, policy.toolMax, policy.runtimeMax)
        }
        return (
            max(0, budgetPlan.charSections.memories),
            max(0, budgetPlan.charSections.rag),
            max(0, budgetPlan.charSections.tools),
            max(0, budgetPlan.charSections.runtime)
        )
    }

    private static func isMemorySection(_ title: String) -> Bool {
        let normalized = title.lowercased()
        return normalized.contains("memory") || normalized.contains("memories")
    }

    private static func isSourceSection(_ title: String) -> Bool {
        let normalized = title.lowercased()
        return normalized.contains("source") || normalized.contains("rag") || normalized.contains("retrieved")
    }

    private static func isToolSection(_ title: String) -> Bool {
        title.lowercased().contains("tool")
    }

    private static func isRuntimeSection(_ title: String) -> Bool {
        title.lowercased().contains("runtime")
    }
}
