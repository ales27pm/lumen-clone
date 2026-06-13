import Foundation

nonisolated enum AssistantTaskKind: Sendable, Equatable {
    case chat
    case agentPlan
    case toolDecision
    case embedding
    case summarization
    case memoryExtraction
    case safetyClassification
    case speechCommandParsing
    case backgroundTrigger
    case remConsolidation
}

struct AssistantTurnContext: Sendable, Equatable {
    let task: AssistantTaskKind
    let input: String
    let systemPrompt: String
    let history: [(role: MessageRole, content: String)]
    let relevantMemories: [MemoryContextItem]
    let attachments: [ChatAttachment]
    let isForeground: Bool
    let lowPowerMode: Bool
    let thermalState: ProcessInfo.ThermalState
    let prefersFoundationModels: Bool
    let allowHeavyRuntime: Bool
    let temperature: Double
    let topP: Double
    let repetitionPenalty: Double
    let maxTokens: Int

    init(
        task: AssistantTaskKind,
        input: String,
        systemPrompt: String = "",
        history: [(role: MessageRole, content: String)] = [],
        relevantMemories: [MemoryContextItem] = [],
        attachments: [ChatAttachment] = [],
        isForeground: Bool,
        lowPowerMode: Bool,
        thermalState: ProcessInfo.ThermalState,
        prefersFoundationModels: Bool = true,
        allowHeavyRuntime: Bool = true,
        temperature: Double = 0.7,
        topP: Double = 0.9,
        repetitionPenalty: Double = 1.1,
        maxTokens: Int = 1024
    ) {
        self.task = task
        self.input = input
        self.systemPrompt = systemPrompt
        self.history = history
        self.relevantMemories = relevantMemories
        self.attachments = attachments
        self.isForeground = isForeground
        self.lowPowerMode = lowPowerMode
        self.thermalState = thermalState
        self.prefersFoundationModels = prefersFoundationModels
        self.allowHeavyRuntime = allowHeavyRuntime
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = max(1, maxTokens)
    }

    static func == (lhs: AssistantTurnContext, rhs: AssistantTurnContext) -> Bool {
        lhs.task == rhs.task &&
        lhs.input == rhs.input &&
        lhs.systemPrompt == rhs.systemPrompt &&
        lhs.history.count == rhs.history.count &&
        zip(lhs.history, rhs.history).allSatisfy { left, right in
            left.role == right.role && left.content == right.content
        } &&
        lhs.relevantMemories == rhs.relevantMemories &&
        lhs.attachments == rhs.attachments &&
        lhs.isForeground == rhs.isForeground &&
        lhs.lowPowerMode == rhs.lowPowerMode &&
        lhs.thermalState == rhs.thermalState &&
        lhs.prefersFoundationModels == rhs.prefersFoundationModels &&
        lhs.allowHeavyRuntime == rhs.allowHeavyRuntime &&
        lhs.temperature == rhs.temperature &&
        lhs.topP == rhs.topP &&
        lhs.repetitionPenalty == rhs.repetitionPenalty &&
        lhs.maxTokens == rhs.maxTokens
    }
}
