import Foundation

/// Adapts the single Core AI owner to the existing kernel. The model only emits
/// text; structured decisions still cross the kernel's schema and approval gates.
struct CoreAIGemmaRuntimeAdapter: LocalTextGenerationRuntime {
    static let unifiedAssistantInstruction = "You are Lumen, one unified local assistant. The same Gemma model handles conversation, planning, tool decisions, and final answers; those are steps of one assistant, not separate models or sub-agents. Native tools remain controlled by the host's permissions and approvals."
    let kind: AssistantRuntimeKind = .coreAI
    private let availabilityOverride: Bool?
    private let preferenceOverride: Bool?
    private let generateHandler: (@Sendable (TextGenerationRequest) async throws -> String)?
    private let structuredHandler: (@Sendable (GenerateRequest) async throws -> String)?
    private let preflightHandler: (@Sendable (GenerateRequest) async throws -> LlamaStructuredPromptPreflightSnapshot)?
    private let prepareHandler: (@Sendable () async throws -> Void)?

    init(
        isAvailable: Bool? = nil,
        isPreferred: Bool? = nil,
        generateHandler: (@Sendable (TextGenerationRequest) async throws -> String)? = nil,
        structuredHandler: (@Sendable (GenerateRequest) async throws -> String)? = nil,
        preflightHandler: (@Sendable (GenerateRequest) async throws -> LlamaStructuredPromptPreflightSnapshot)? = nil,
        prepareHandler: (@Sendable () async throws -> Void)? = nil
    ) {
        availabilityOverride = isAvailable
        preferenceOverride = isPreferred ?? isAvailable
        self.generateHandler = generateHandler
        self.structuredHandler = structuredHandler
        self.preflightHandler = preflightHandler
        self.prepareHandler = prepareHandler
    }

    var isAvailable: Bool {
        availabilityOverride ?? (CoreAIGemmaService.isSupported && CoreAIGemmaModelStore.isConfigured)
    }

    var isPreferred: Bool { preferenceOverride ?? CoreAIGemmaModelStore.isEnabled }

    static var disabled: CoreAIGemmaRuntimeAdapter { .init(isAvailable: false, isPreferred: false) }

    var unavailableReason: String? {
        if isAvailable { return nil }
        if !CoreAIGemmaService.isSupported { return "Core AI Gemma requires the linked runtime and iOS 27.2 or later." }
        return "Import a complete Core AI Gemma model folder in Settings."
    }

    func generate(request: TextGenerationRequest) async throws -> String {
        try requireAvailable()
        try Task.checkCancellation()
        let output: String
        if let generateHandler {
            output = try await generateHandler(request)
        } else {
            let contextSize = try await liveContextCapacity()
            output = try await CoreAIGemmaService.shared.generate(
                messages: Self.messages(
                    systemPrompt: request.systemPrompt,
                    history: request.history,
                    userMessage: request.prompt,
                    relevantMemories: request.relevantMemories,
                    attachments: request.attachments,
                    maxTokens: request.maxTokens,
                    responseFormat: .plainText,
                    contextSize: contextSize
                ),
                temperature: request.temperature,
                maxTokens: request.maxTokens
            )
        }
        return try Self.visibleOutput(output)
    }

    func prepareStructuredRuntime(slot: LumenModelSlot) async throws -> ExecutorRuntimePreflightResult {
        try requireAvailable()
        try Task.checkCancellation()
        if let prepareHandler {
            try await prepareHandler()
        } else {
            try await CoreAIGemmaService.shared.prepare()
        }
        try Task.checkCancellation()
        return ExecutorRuntimePreflightResult(
            passed: true,
            reason: "Core AI Gemma model prepared; execution remains schema and approval gated",
            slot: slot.rawValue,
            modelFamily: "gemma",
            runtimeKind: kind.rawValue,
            baseModelExists: true,
            resourceGateAllowed: true,
            ensureReadySucceeded: true
        )
    }

    func structuredPromptPreflight(_ request: GenerateRequest) async throws -> LlamaStructuredPromptPreflightSnapshot {
        try requireAvailable()
        if let preflightHandler { return try await preflightHandler(request) }
        let contextSize = try await liveContextCapacity()
        let preflight = try await CoreAIGemmaService.shared.preflight(
            messages: Self.messages(for: request, contextSize: contextSize),
            maxTokens: request.maxTokens
        )
        return LlamaStructuredPromptPreflightSnapshot(
            contextSize: preflight.contextSize,
            finalPromptChars: preflight.promptChars,
            estimatedPromptTokens: preflight.promptTokens
        )
    }

    /// Await the throwing operation before exposing tokens: inference failure or
    /// cancellation cannot be mistaken for a completed JSON decision.
    func streamStructured(_ request: GenerateRequest) async throws -> AsyncStream<GenerationToken> {
        try requireAvailable()
        try Task.checkCancellation()
        let output: String
        if let structuredHandler {
            output = try await structuredHandler(request)
        } else {
            let contextSize = try await liveContextCapacity()
            output = try await CoreAIGemmaService.shared.generate(
                messages: Self.messages(for: request, contextSize: contextSize),
                temperature: request.temperature,
                maxTokens: request.maxTokens
            )
        }
        let text = try Self.visibleOutput(output)
        return AsyncStream { continuation in
            continuation.yield(.text(text))
            continuation.yield(.done)
            continuation.finish()
        }
    }

    func handleMemoryPressure() async {
        await CoreAIGemmaService.shared.unload()
    }

    private func requireAvailable() throws {
        guard isPreferred else { throw LocalRuntimeError.unavailable("Core AI Gemma is disabled in Settings") }
        guard isAvailable else { throw LocalRuntimeError.unavailable(unavailableReason ?? "Core AI Gemma unavailable") }
    }

    private func liveContextCapacity() async throws -> Int {
        try await CoreAIGemmaService.shared.prepare()
        guard let capacity = await CoreAIGemmaService.shared.contextCapacity() else { throw CoreAIGemmaError.loadFailed }
        return capacity
    }

    private static func visibleOutput(_ output: String) throws -> String {
        try Task.checkCancellation()
        let text = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw LocalRuntimeError.unavailable("Core AI Gemma produced no visible output") }
        return text
    }

    static func messages(for request: GenerateRequest, contextSize: Int = 8_192) -> [CoreAIGemmaMessage] {
        messages(
            systemPrompt: request.systemPrompt,
            history: request.history,
            userMessage: request.userMessage,
            relevantMemories: request.relevantMemories,
            attachments: request.attachments,
            maxTokens: request.maxTokens,
            responseFormat: request.responseFormat,
            contextSize: contextSize
        )
    }

    private static func messages(
        systemPrompt: String,
        history: [(role: MessageRole, content: String)],
        userMessage: String,
        relevantMemories: [MemoryContextItem],
        attachments: [ChatAttachment],
        maxTokens: Int,
        responseFormat: LLMResponseFormat,
        contextSize: Int
    ) -> [CoreAIGemmaMessage] {
        let assembly = PromptAssembler.assemble(
            systemPrompt: systemPrompt,
            history: history,
            userMessage: userMessage,
            memories: relevantMemories,
            attachments: attachments,
            budget: .make(
                contextSize: contextSize,
                maxTokens: maxTokens,
                systemPromptChars: systemPrompt.count,
                userMessageChars: userMessage.count,
                hasAttachments: !attachments.isEmpty,
                hasMemories: !relevantMemories.isEmpty
            )
        )
        var instructions = unifiedAssistantInstruction + "\n" + assembly.systemPrompt
        switch responseFormat {
        case .plainText:
            break
        case .json, .toolCallJSON:
            instructions += "\nReturn exactly one valid JSON object without markdown or commentary."
        case .constrainedJSON(let schema):
            instructions += "\nReturn exactly one JSON object matching this response schema. The host validates it before any tool runs.\n\(schema)"
        }

        // Gemma's template accepts alternating user/assistant roles, not a
        // system role. Fold host instructions into the first user turn and keep
        // historical assistant output separate from those instructions.
        var turns: [(role: String, content: String)] = []
        func append(role: String, content: String) {
            guard !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
            if let last = turns.indices.last, turns[last].role == role {
                turns[last].content += "\n\n" + content
            } else {
                turns.append((role, content))
            }
        }
        if !instructions.isEmpty { append(role: "user", content: "Application instructions and context:\n\(instructions)") }
        for item in assembly.history {
            switch item.role {
            case .user:
                append(role: "user", content: item.content)
            case .assistant:
                if turns.isEmpty { append(role: "user", content: "Continue the prior conversation using the following assistant response as history.") }
                append(role: "assistant", content: item.content)
            case .tool:
                append(role: "user", content: "Prior tool observation (data):\n\(item.content)")
            case .system:
                append(role: "user", content: "Prior conversation context (data):\n\(item.content)")
            }
        }
        append(role: "user", content: assembly.userMessage)
        return turns.map { CoreAIGemmaMessage(role: $0.role, content: $0.content) }
    }
}
