import Foundation

nonisolated struct AgentBehaviorTrace: Codable, Sendable, Identifiable, Hashable {
    enum Event: String, Codable, Sendable {
        case modelTurn
        case toolAction
        case finalAnswer
    }

    let id: UUID
    let createdAt: Date
    let event: Event
    let slot: String
    let stage: String
    let intent: String?
    let promptPrefix: String
    let rawOutputPrefix: String
    let selectedToolID: String?
    let toolArguments: [String: String]
    let allowedToolIDs: [String]
    let requiresApproval: Bool?
    let approvalMode: String?
    let parseError: String?
    let emittedFinalInActionTurn: Bool
    let modelFamily: String?
    let baseModelPath: String?
    let adapterID: String?
    let adapterSlot: String?
    let adapterPath: String?
    let adapterApplied: Bool?
    let adapterScale: Float?
    let adapterFailureReason: String?
    let generationElapsedMs: Int?
    let firstTokenLatencyMs: Int?
    let outputTokenCount: Int?
    let estimatedPromptTokenCount: Int?
    let preFirstTokenMs: Int?
    let messageBuildMs: Int?
    let decodeMs: Int?
    let tokensPerSecond: Double?
    let ensureReadyMs: Int?
    let adapterActivationMs: Int?
    let runtimePath: String?
    let activeAdapterSlot: String?
    let maxTokensRequested: Int?
    let maxTokensEffective: Int?
    let promptCharCount: Int?
    let accelerationDiagnostic: String?
    let accelerationDiagnostics: RuntimeAccelerationDiagnostics?



    enum CodingKeys: String, CodingKey {
        case id, createdAt, event, slot, stage, intent, promptPrefix, rawOutputPrefix, selectedToolID, toolArguments, allowedToolIDs, requiresApproval, approvalMode, parseError, emittedFinalInActionTurn, modelFamily, baseModelPath, adapterID, adapterSlot, adapterPath, adapterApplied, adapterScale, adapterFailureReason, generationElapsedMs, firstTokenLatencyMs, outputTokenCount, estimatedPromptTokenCount, preFirstTokenMs, messageBuildMs, decodeMs, tokensPerSecond, ensureReadyMs, adapterActivationMs, runtimePath, activeAdapterSlot, maxTokensRequested, maxTokensEffective, promptCharCount, accelerationDiagnostic, accelerationDiagnostics
        case promptTokenCount
        case promptEvalMs
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        event = try container.decode(Event.self, forKey: .event)
        slot = try container.decode(String.self, forKey: .slot)
        stage = try container.decode(String.self, forKey: .stage)
        intent = try container.decodeIfPresent(String.self, forKey: .intent)
        promptPrefix = try container.decode(String.self, forKey: .promptPrefix)
        rawOutputPrefix = try container.decode(String.self, forKey: .rawOutputPrefix)
        selectedToolID = try container.decodeIfPresent(String.self, forKey: .selectedToolID)
        toolArguments = try container.decode([String: String].self, forKey: .toolArguments)
        allowedToolIDs = try container.decode([String].self, forKey: .allowedToolIDs)
        requiresApproval = try container.decodeIfPresent(Bool.self, forKey: .requiresApproval)
        approvalMode = try container.decodeIfPresent(String.self, forKey: .approvalMode)
        parseError = try container.decodeIfPresent(String.self, forKey: .parseError)
        emittedFinalInActionTurn = try container.decode(Bool.self, forKey: .emittedFinalInActionTurn)
        modelFamily = try container.decodeIfPresent(String.self, forKey: .modelFamily)
        baseModelPath = try container.decodeIfPresent(String.self, forKey: .baseModelPath)
        adapterID = try container.decodeIfPresent(String.self, forKey: .adapterID)
        adapterSlot = try container.decodeIfPresent(String.self, forKey: .adapterSlot)
        adapterPath = try container.decodeIfPresent(String.self, forKey: .adapterPath)
        adapterApplied = try container.decodeIfPresent(Bool.self, forKey: .adapterApplied)
        adapterScale = try container.decodeIfPresent(Float.self, forKey: .adapterScale)
        adapterFailureReason = try container.decodeIfPresent(String.self, forKey: .adapterFailureReason)
        generationElapsedMs = try container.decodeIfPresent(Int.self, forKey: .generationElapsedMs)
        firstTokenLatencyMs = try container.decodeIfPresent(Int.self, forKey: .firstTokenLatencyMs)
        outputTokenCount = try container.decodeIfPresent(Int.self, forKey: .outputTokenCount)
        estimatedPromptTokenCount = try container.decodeIfPresent(Int.self, forKey: .estimatedPromptTokenCount) ?? container.decodeIfPresent(Int.self, forKey: .promptTokenCount)
        preFirstTokenMs = try container.decodeIfPresent(Int.self, forKey: .preFirstTokenMs) ?? container.decodeIfPresent(Int.self, forKey: .promptEvalMs)
        messageBuildMs = try container.decodeIfPresent(Int.self, forKey: .messageBuildMs)
        decodeMs = try container.decodeIfPresent(Int.self, forKey: .decodeMs)
        tokensPerSecond = try container.decodeIfPresent(Double.self, forKey: .tokensPerSecond)
        ensureReadyMs = try container.decodeIfPresent(Int.self, forKey: .ensureReadyMs)
        adapterActivationMs = try container.decodeIfPresent(Int.self, forKey: .adapterActivationMs)
        runtimePath = try container.decodeIfPresent(String.self, forKey: .runtimePath)
        activeAdapterSlot = try container.decodeIfPresent(String.self, forKey: .activeAdapterSlot)
        maxTokensRequested = try container.decodeIfPresent(Int.self, forKey: .maxTokensRequested)
        maxTokensEffective = try container.decodeIfPresent(Int.self, forKey: .maxTokensEffective)
        promptCharCount = try container.decodeIfPresent(Int.self, forKey: .promptCharCount)
        accelerationDiagnostic = try container.decodeIfPresent(String.self, forKey: .accelerationDiagnostic)
        accelerationDiagnostics = try container.decodeIfPresent(RuntimeAccelerationDiagnostics.self, forKey: .accelerationDiagnostics)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(event, forKey: .event)
        try container.encode(slot, forKey: .slot)
        try container.encode(stage, forKey: .stage)
        try container.encodeIfPresent(intent, forKey: .intent)
        try container.encode(promptPrefix, forKey: .promptPrefix)
        try container.encode(rawOutputPrefix, forKey: .rawOutputPrefix)
        try container.encodeIfPresent(selectedToolID, forKey: .selectedToolID)
        try container.encode(toolArguments, forKey: .toolArguments)
        try container.encode(allowedToolIDs, forKey: .allowedToolIDs)
        try container.encodeIfPresent(requiresApproval, forKey: .requiresApproval)
        try container.encodeIfPresent(approvalMode, forKey: .approvalMode)
        try container.encodeIfPresent(parseError, forKey: .parseError)
        try container.encode(emittedFinalInActionTurn, forKey: .emittedFinalInActionTurn)
        try container.encodeIfPresent(modelFamily, forKey: .modelFamily)
        try container.encodeIfPresent(baseModelPath, forKey: .baseModelPath)
        try container.encodeIfPresent(adapterID, forKey: .adapterID)
        try container.encodeIfPresent(adapterSlot, forKey: .adapterSlot)
        try container.encodeIfPresent(adapterPath, forKey: .adapterPath)
        try container.encodeIfPresent(adapterApplied, forKey: .adapterApplied)
        try container.encodeIfPresent(adapterScale, forKey: .adapterScale)
        try container.encodeIfPresent(adapterFailureReason, forKey: .adapterFailureReason)
        try container.encodeIfPresent(generationElapsedMs, forKey: .generationElapsedMs)
        try container.encodeIfPresent(firstTokenLatencyMs, forKey: .firstTokenLatencyMs)
        try container.encodeIfPresent(outputTokenCount, forKey: .outputTokenCount)
        try container.encodeIfPresent(estimatedPromptTokenCount, forKey: .estimatedPromptTokenCount)
        try container.encodeIfPresent(preFirstTokenMs, forKey: .preFirstTokenMs)
        try container.encodeIfPresent(messageBuildMs, forKey: .messageBuildMs)
        try container.encodeIfPresent(decodeMs, forKey: .decodeMs)
        try container.encodeIfPresent(tokensPerSecond, forKey: .tokensPerSecond)
        try container.encodeIfPresent(ensureReadyMs, forKey: .ensureReadyMs)
        try container.encodeIfPresent(adapterActivationMs, forKey: .adapterActivationMs)
        try container.encodeIfPresent(runtimePath, forKey: .runtimePath)
        try container.encodeIfPresent(activeAdapterSlot, forKey: .activeAdapterSlot)
        try container.encodeIfPresent(maxTokensRequested, forKey: .maxTokensRequested)
        try container.encodeIfPresent(maxTokensEffective, forKey: .maxTokensEffective)
        try container.encodeIfPresent(promptCharCount, forKey: .promptCharCount)
        try container.encodeIfPresent(accelerationDiagnostic, forKey: .accelerationDiagnostic)
        try container.encodeIfPresent(accelerationDiagnostics, forKey: .accelerationDiagnostics)
    }

    init(
        id: UUID,
        createdAt: Date,
        event: Event,
        slot: String,
        stage: String,
        intent: String?,
        promptPrefix: String,
        rawOutputPrefix: String,
        selectedToolID: String?,
        toolArguments: [String: String],
        allowedToolIDs: [String],
        requiresApproval: Bool?,
        approvalMode: String?,
        parseError: String?,
        emittedFinalInActionTurn: Bool,
        modelFamily: String? = nil,
        baseModelPath: String? = nil,
        adapterID: String? = nil,
        adapterSlot: String? = nil,
        adapterPath: String? = nil,
        adapterApplied: Bool? = nil,
        adapterScale: Float? = nil,
        adapterFailureReason: String? = nil,
        generationElapsedMs: Int? = nil,
        firstTokenLatencyMs: Int? = nil,
        outputTokenCount: Int? = nil,
        estimatedPromptTokenCount: Int? = nil,
        preFirstTokenMs: Int? = nil,
        messageBuildMs: Int? = nil,
        decodeMs: Int? = nil,
        tokensPerSecond: Double? = nil,
        ensureReadyMs: Int? = nil,
        adapterActivationMs: Int? = nil,
        runtimePath: String? = nil,
        activeAdapterSlot: String? = nil,
        maxTokensRequested: Int? = nil,
        maxTokensEffective: Int? = nil,
        promptCharCount: Int? = nil,
        accelerationDiagnostic: String? = nil,
        accelerationDiagnostics: RuntimeAccelerationDiagnostics? = nil
    ) {
        self.id = id
        self.createdAt = createdAt
        self.event = event
        self.slot = slot
        self.stage = stage
        self.intent = intent
        self.promptPrefix = promptPrefix
        self.rawOutputPrefix = rawOutputPrefix
        self.selectedToolID = selectedToolID
        self.toolArguments = toolArguments
        self.allowedToolIDs = allowedToolIDs
        self.requiresApproval = requiresApproval
        self.approvalMode = approvalMode
        self.parseError = parseError
        self.emittedFinalInActionTurn = emittedFinalInActionTurn
        self.modelFamily = modelFamily
        self.baseModelPath = baseModelPath
        self.adapterID = adapterID
        self.adapterSlot = adapterSlot
        self.adapterPath = adapterPath
        self.adapterApplied = adapterApplied
        self.adapterScale = adapterScale
        self.adapterFailureReason = adapterFailureReason
        self.generationElapsedMs = generationElapsedMs
        self.firstTokenLatencyMs = firstTokenLatencyMs
        self.outputTokenCount = outputTokenCount
        self.estimatedPromptTokenCount = estimatedPromptTokenCount
        self.preFirstTokenMs = preFirstTokenMs
        self.messageBuildMs = messageBuildMs
        self.decodeMs = decodeMs
        self.tokensPerSecond = tokensPerSecond
        self.ensureReadyMs = ensureReadyMs
        self.adapterActivationMs = adapterActivationMs
        self.runtimePath = runtimePath
        self.activeAdapterSlot = activeAdapterSlot
        self.maxTokensRequested = maxTokensRequested
        self.maxTokensEffective = maxTokensEffective
        self.promptCharCount = promptCharCount
        self.accelerationDiagnostic = accelerationDiagnostic
        self.accelerationDiagnostics = accelerationDiagnostics
    }
}

nonisolated struct AgentBehaviorAuditReport: Codable, Sendable, Hashable {
    let passed: Bool
    let score: Double
    let generatedAt: Date
    let traceCount: Int
    let violationCount: Int
    let sourceCommit: String?
    let violations: [AgentBehaviorViolation]
    let recommendations: [String]
    let repairSamples: [AgentBehaviorRepairSample]
}

nonisolated struct AgentBehaviorViolation: Codable, Sendable, Identifiable, Hashable {
    let id: UUID
    let createdAt: Date
    let severity: Severity
    let code: String
    let agent: String
    let expected: String
    let actual: String
    let promptPrefix: String
    let problem: String

    enum Severity: String, Codable, Sendable {
        case warning
        case error
        case critical

        var weight: Double {
            switch self {
            case .warning: 0.5
            case .error: 1.0
            case .critical: 2.0
            }
        }
    }
}

nonisolated struct AgentBehaviorRepairSample: Codable, Sendable, Identifiable, Hashable {
    let id: UUID
    let createdAt: Date
    let agent: String
    let violationCode: String
    let promptPrefix: String
    let expected: String
    let badOutput: String
    let correctedOutput: String
    let lesson: String
    let curriculum: String
}

nonisolated enum AgentBehaviorTraceRecorder {
    private static let fileName = "agent-behavior-traces.jsonl"

    static func record(_ trace: AgentBehaviorTrace) {
        do {
            let directory = try diagnosticsDirectory()
            let url = directory.appendingPathComponent(fileName, isDirectory: false)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(trace)
            var line = data
            line.append(0x0A)

            if FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: line)
            } else {
                try line.write(to: url, options: [.atomic])
            }
        } catch {
            // Diagnostics must never break assistant execution.
        }
    }

    static func recent(limit: Int = 200) -> [AgentBehaviorTrace] {
        do {
            let url = try diagnosticsDirectory().appendingPathComponent(fileName, isDirectory: false)
            guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else { return [] }
            let data = try Data(contentsOf: url)
            guard let text = String(data: data, encoding: .utf8) else { return [] }
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let traces = text
                .split(whereSeparator: \.isNewline)
                .compactMap { line -> AgentBehaviorTrace? in
                    guard let lineData = String(line).data(using: .utf8) else { return nil }
                    return try? decoder.decode(AgentBehaviorTrace.self, from: lineData)
                }
            let boundedLimit = max(0, limit)
            return Array(traces.suffix(boundedLimit))
        } catch {
            return []
        }
    }

    static func clear() {
        do {
            let url = try diagnosticsDirectory().appendingPathComponent(fileName, isDirectory: false)
            if FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
                try FileManager.default.removeItem(at: url)
            }
        } catch {
            // Diagnostics cleanup must never break app execution.
        }
    }

    static func diagnosticsDirectory() throws -> URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first ?? FileManager.default.temporaryDirectory
        let directory = base
            .appendingPathComponent("Diagnostics", isDirectory: true)
            .appendingPathComponent("AgentBehavior", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
