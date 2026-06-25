import Foundation

struct ContextBudgetTokenSections: Sendable, Equatable {
    let system: Int
    let history: Int
    let memories: Int
    let rag: Int
    let tools: Int
    let runtime: Int

    var total: Int {
        system + history + memories + rag + tools + runtime
    }
}

struct ContextBudgetSections: Sendable, Equatable {
    let system: Int
    let history: Int
    let memories: Int
    let rag: Int
    let tools: Int
    let runtime: Int

    var total: Int {
        system + history + memories + rag + tools + runtime
    }
}

enum ContextPolicyProfile: String, Codable, Sendable, Equatable {
    case chat
    case code
    case rag
    case tool
    case memory
    case background
    case diagnostics

    var defaultInputTokens: Int {
        switch self {
        case .chat: return 1_024
        case .code: return 2_048
        case .rag: return 2_048
        case .tool: return 1_536
        case .memory: return 1_536
        case .background: return 768
        case .diagnostics: return 1_024
        }
    }

    fileprivate var ratios: (system: Double, history: Double, memories: Double, rag: Double, tools: Double) {
        switch self {
        case .chat:
            return (0.18, 0.34, 0.18, 0.20, 0.06)
        case .code:
            return (0.14, 0.24, 0.08, 0.40, 0.06)
        case .rag:
            return (0.12, 0.18, 0.10, 0.50, 0.04)
        case .tool:
            return (0.16, 0.20, 0.10, 0.12, 0.34)
        case .memory:
            return (0.14, 0.18, 0.46, 0.10, 0.04)
        case .background:
            return (0.16, 0.16, 0.26, 0.18, 0.08)
        case .diagnostics:
            return (0.12, 0.10, 0.10, 0.20, 0.10)
        }
    }
}

struct ContextBudgetPlan: Sendable, Equatable {
    let profile: ContextPolicyProfile
    let maxInputTokens: Int
    let tokenSections: ContextBudgetTokenSections
    let charSections: ContextBudgetSections
    let charsPerToken: Int
    let estimatedInputTokens: Int
}

enum ContextBudgetAllocator {
    static let defaultCharsPerToken = 4

    static func allocate(maxChars: Int) -> ContextBudgetSections {
        let bounded = max(0, maxChars)
        let system = Int(Double(bounded) * 0.18)
        let history = Int(Double(bounded) * 0.34)
        let memories = Int(Double(bounded) * 0.18)
        let rag = Int(Double(bounded) * 0.20)
        let tools = Int(Double(bounded) * 0.06)
        let runtime = max(0, bounded - (system + history + memories + rag + tools))
        return ContextBudgetSections(system: system, history: history, memories: memories, rag: rag, tools: tools, runtime: runtime)
    }

    static func allocate(
        profile: ContextPolicyProfile,
        maxInputTokens: Int? = nil,
        input: String = "",
        charsPerToken: Int = defaultCharsPerToken
    ) -> ContextBudgetPlan {
        let boundedTokens = max(1, maxInputTokens ?? profile.defaultInputTokens)
        let ratios = profile.ratios
        let system = Int(Double(boundedTokens) * ratios.system)
        let history = Int(Double(boundedTokens) * ratios.history)
        let memories = Int(Double(boundedTokens) * ratios.memories)
        let rag = Int(Double(boundedTokens) * ratios.rag)
        let tools = Int(Double(boundedTokens) * ratios.tools)
        let runtime = max(0, boundedTokens - (system + history + memories + rag + tools))
        let tokenSections = ContextBudgetTokenSections(
            system: system,
            history: history,
            memories: memories,
            rag: rag,
            tools: tools,
            runtime: runtime
        )
        let charSections = ContextBudgetSections(
            system: system * charsPerToken,
            history: history * charsPerToken,
            memories: memories * charsPerToken,
            rag: rag * charsPerToken,
            tools: tools * charsPerToken,
            runtime: runtime * charsPerToken
        )
        return ContextBudgetPlan(
            profile: profile,
            maxInputTokens: boundedTokens,
            tokenSections: tokenSections,
            charSections: charSections,
            charsPerToken: charsPerToken,
            estimatedInputTokens: estimateTokens(for: input)
        )
    }

    static func allocate(for turn: AssistantTurnContext, maxInputTokens: Int? = nil) -> ContextBudgetPlan {
        allocate(profile: profile(for: turn), maxInputTokens: maxInputTokens, input: turn.input)
    }

    static func profile(for turn: AssistantTurnContext) -> ContextPolicyProfile {
        if !turn.isForeground { return .background }
        switch turn.task {
        case .backgroundTrigger:
            return .background
        case .agentPlan, .toolDecision, .speechCommandParsing:
            return .tool
        case .summarization, .memoryExtraction, .remConsolidation:
            return .memory
        case .embedding, .safetyClassification:
            return .diagnostics
        case .chat:
            let lower = turn.input.lowercased()
            if containsAny(lower, ["rag", "document", "file", "source", "citation", "research", "summarize"]) {
                return .rag
            }
            if containsAny(lower, ["code", "swift", "xcode", "build", "compile", "stack trace", "diff"]) {
                return .code
            }
            if containsAny(lower, ["remember", "preference", "what do you know about me", "recall"]) {
                return .memory
            }
            if containsAny(lower, ["call ", "email", "message", "calendar", "reminder", "trigger", "weather"]) {
                return .tool
            }
            return .chat
        }
    }

    static func estimateTokens(for text: String) -> Int {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return 0 }
        return estimateTokens(forCharacterCount: trimmed.count)
    }

    static func estimateTokens(forCharacterCount characterCount: Int) -> Int {
        guard characterCount > 0 else { return 0 }
        return max(1, Int(ceil(Double(characterCount) / Double(defaultCharsPerToken))))
    }

    private static func containsAny(_ text: String, _ needles: [String]) -> Bool {
        needles.contains { text.contains($0) }
    }
}
