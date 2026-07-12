import Foundation

nonisolated struct AgentBehaviorTrace: Codable, Sendable, Identifiable, Hashable {
    nonisolated struct SelfModelDecisionSummary: Codable, Sendable, Hashable {
        let included: Bool
        let schemaVersion: String?
        let mode: String?
        let activeSlot: String?
        let sourceIDs: [String]
        let runtimeEvidenceSourceLayer: String?
        let selectedToolID: String?
        let requiresApproval: Bool?
        let approvalMode: String?

        static func fromPrompt(
            _ prompt: String,
            selectedToolID: String?,
            requiresApproval: Bool?,
            approvalMode: String?
        ) -> SelfModelDecisionSummary? {
            guard let block = Self.selfModelBlock(in: prompt) else { return nil }
            let schemaVersion = Self.value(for: "schemaVersion", in: block)
            let mode = Self.value(for: "mode", in: block) ?? Self.jsonStringValue(for: "mode", in: block)
            let activeSlot = Self.value(for: "activeSlot", in: block) ?? Self.jsonStringValue(for: "activeSlot", in: block)
            let sourceLayer = Self.value(for: "sourceLayer", in: block) ?? Self.jsonStringValue(for: "sourceLayer", in: block)
            let sourceIDs = [
                schemaVersion.map { "selfModelSnapshot/\($0)" },
                activeSlot.map { "slot/\($0)" },
                sourceLayer.map { "evidence/\($0)" }
            ].compactMap(\.self)
            return .init(
                included: true,
                schemaVersion: schemaVersion,
                mode: mode,
                activeSlot: activeSlot,
                sourceIDs: sourceIDs,
                runtimeEvidenceSourceLayer: sourceLayer,
                selectedToolID: selectedToolID.map(ToolRouteGuard.canonicalToolID),
                requiresApproval: requiresApproval,
                approvalMode: approvalMode
            )
        }

        private static func selfModelBlock(in prompt: String) -> String? {
            guard let range = prompt.range(of: "[SELF MODEL]") else { return nil }
            let tail = prompt[range.upperBound...]
            if let nextHeader = tail.range(of: "\n[") {
                return String(tail[..<nextHeader.lowerBound])
            }
            return String(tail)
        }

        private static func value(for key: String, in block: String) -> String? {
            for line in block.split(whereSeparator: \.isNewline) {
                let prefix = "\(key)="
                guard line.hasPrefix(prefix) else { continue }
                let value = line.dropFirst(prefix.count).trimmingCharacters(in: .whitespacesAndNewlines)
                return value.isEmpty ? nil : value
            }
            return nil
        }

        private static func jsonStringValue(for key: String, in block: String) -> String? {
            let pattern = #""# + NSRegularExpression.escapedPattern(for: key) + #"":"([^"]+)""#
            guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
            let range = NSRange(block.startIndex..<block.endIndex, in: block)
            guard let match = regex.firstMatch(in: block, range: range), match.numberOfRanges > 1 else { return nil }
            guard let valueRange = Range(match.range(at: 1), in: block) else { return nil }
            return String(block[valueRange])
        }
    }

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
    let scenarioID: String?
    let e2eRunID: UUID?
    let agentRunID: UUID?
    let conversationID: UUID?
    let turnID: UUID?
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
    let emptyOutputReason: String?
    let streamStarted: Bool?
    let selectedRuntime: String?
    let selectedAdapter: String?
    let modelIdentifier: String?
    let modelLoaded: Bool?
    let stopSequences: [String]
    let temperature: Double?
    let topP: Double?
    let cancellationStateBeforeStream: String?
    let firstChunkReceived: Bool?
    let textChunkCount: Int?
    let finalChunkReceived: Bool?
    let streamTerminationReason: String?
    let successfulObservationCount: Int?
    let finalizerAccepted: Bool?
    let finalizerRejectionReason: String?
    let finalValidatorAcceptedCandidate: Bool?
    let finalValidatorReplacementSource: String?
    let finalValidatorRejectionReason: String?
    let selfModel: SelfModelDecisionSummary?

    enum CodingKeys: String, CodingKey {
        case id, createdAt, event, slot, stage, scenarioID, e2eRunID, agentRunID, conversationID, turnID, intent, promptPrefix, rawOutputPrefix, selectedToolID, toolArguments, allowedToolIDs, requiresApproval, approvalMode, parseError, emittedFinalInActionTurn, modelFamily, baseModelPath, adapterID, adapterSlot, adapterPath, adapterApplied, adapterScale, adapterFailureReason, generationElapsedMs, firstTokenLatencyMs, outputTokenCount, estimatedPromptTokenCount, preFirstTokenMs, messageBuildMs, decodeMs, tokensPerSecond, ensureReadyMs, adapterActivationMs, runtimePath, activeAdapterSlot, maxTokensRequested, maxTokensEffective, promptCharCount, accelerationDiagnostic, accelerationDiagnostics, emptyOutputReason, streamStarted, selectedRuntime, selectedAdapter, modelIdentifier, modelLoaded, stopSequences, temperature, topP, cancellationStateBeforeStream, firstChunkReceived, textChunkCount, finalChunkReceived, streamTerminationReason, successfulObservationCount, finalizerAccepted, finalizerRejectionReason, finalValidatorAcceptedCandidate, finalValidatorReplacementSource, finalValidatorRejectionReason, selfModel
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
        scenarioID = try container.decodeIfPresent(String.self, forKey: .scenarioID)
        e2eRunID = try container.decodeIfPresent(UUID.self, forKey: .e2eRunID)
        agentRunID = try container.decodeIfPresent(UUID.self, forKey: .agentRunID)
        conversationID = try container.decodeIfPresent(UUID.self, forKey: .conversationID)
        turnID = try container.decodeIfPresent(UUID.self, forKey: .turnID)
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
        emptyOutputReason = try container.decodeIfPresent(String.self, forKey: .emptyOutputReason)
        streamStarted = try container.decodeIfPresent(Bool.self, forKey: .streamStarted)
        selectedRuntime = try container.decodeIfPresent(String.self, forKey: .selectedRuntime)
        selectedAdapter = try container.decodeIfPresent(String.self, forKey: .selectedAdapter)
        modelIdentifier = try container.decodeIfPresent(String.self, forKey: .modelIdentifier)
        modelLoaded = try container.decodeIfPresent(Bool.self, forKey: .modelLoaded)
        stopSequences = try container.decodeIfPresent([String].self, forKey: .stopSequences) ?? []
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        cancellationStateBeforeStream = try container.decodeIfPresent(String.self, forKey: .cancellationStateBeforeStream)
        firstChunkReceived = try container.decodeIfPresent(Bool.self, forKey: .firstChunkReceived)
        textChunkCount = try container.decodeIfPresent(Int.self, forKey: .textChunkCount)
        finalChunkReceived = try container.decodeIfPresent(Bool.self, forKey: .finalChunkReceived)
        streamTerminationReason = try container.decodeIfPresent(String.self, forKey: .streamTerminationReason)
        successfulObservationCount = try container.decodeIfPresent(Int.self, forKey: .successfulObservationCount)
        finalizerAccepted = try container.decodeIfPresent(Bool.self, forKey: .finalizerAccepted)
        finalizerRejectionReason = try container.decodeIfPresent(String.self, forKey: .finalizerRejectionReason)
        finalValidatorAcceptedCandidate = try container.decodeIfPresent(Bool.self, forKey: .finalValidatorAcceptedCandidate)
        finalValidatorReplacementSource = try container.decodeIfPresent(String.self, forKey: .finalValidatorReplacementSource)
        finalValidatorRejectionReason = try container.decodeIfPresent(String.self, forKey: .finalValidatorRejectionReason)
        selfModel = try container.decodeIfPresent(SelfModelDecisionSummary.self, forKey: .selfModel)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(event, forKey: .event)
        try container.encode(slot, forKey: .slot)
        try container.encode(stage, forKey: .stage)
        try container.encodeIfPresent(scenarioID, forKey: .scenarioID)
        try container.encodeIfPresent(e2eRunID, forKey: .e2eRunID)
        try container.encodeIfPresent(agentRunID, forKey: .agentRunID)
        try container.encodeIfPresent(conversationID, forKey: .conversationID)
        try container.encodeIfPresent(turnID, forKey: .turnID)
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
        try container.encodeIfPresent(emptyOutputReason, forKey: .emptyOutputReason)
        try container.encodeIfPresent(streamStarted, forKey: .streamStarted)
        try container.encodeIfPresent(selectedRuntime, forKey: .selectedRuntime)
        try container.encodeIfPresent(selectedAdapter, forKey: .selectedAdapter)
        try container.encodeIfPresent(modelIdentifier, forKey: .modelIdentifier)
        try container.encodeIfPresent(modelLoaded, forKey: .modelLoaded)
        try container.encode(stopSequences, forKey: .stopSequences)
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(topP, forKey: .topP)
        try container.encodeIfPresent(cancellationStateBeforeStream, forKey: .cancellationStateBeforeStream)
        try container.encodeIfPresent(firstChunkReceived, forKey: .firstChunkReceived)
        try container.encodeIfPresent(textChunkCount, forKey: .textChunkCount)
        try container.encodeIfPresent(finalChunkReceived, forKey: .finalChunkReceived)
        try container.encodeIfPresent(streamTerminationReason, forKey: .streamTerminationReason)
        try container.encodeIfPresent(successfulObservationCount, forKey: .successfulObservationCount)
        try container.encodeIfPresent(finalizerAccepted, forKey: .finalizerAccepted)
        try container.encodeIfPresent(finalizerRejectionReason, forKey: .finalizerRejectionReason)
        try container.encodeIfPresent(finalValidatorAcceptedCandidate, forKey: .finalValidatorAcceptedCandidate)
        try container.encodeIfPresent(finalValidatorReplacementSource, forKey: .finalValidatorReplacementSource)
        try container.encodeIfPresent(finalValidatorRejectionReason, forKey: .finalValidatorRejectionReason)
        try container.encodeIfPresent(selfModel, forKey: .selfModel)
    }

    init(
        id: UUID,
        createdAt: Date,
        event: Event,
        slot: String,
        stage: String,
        scenarioID: String? = nil,
        e2eRunID: UUID? = nil,
        agentRunID: UUID? = nil,
        conversationID: UUID? = nil,
        turnID: UUID? = nil,
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
        accelerationDiagnostics: RuntimeAccelerationDiagnostics? = nil,
        emptyOutputReason: String? = nil,
        streamStarted: Bool? = nil,
        selectedRuntime: String? = nil,
        selectedAdapter: String? = nil,
        modelIdentifier: String? = nil,
        modelLoaded: Bool? = nil,
        stopSequences: [String] = [],
        temperature: Double? = nil,
        topP: Double? = nil,
        cancellationStateBeforeStream: String? = nil,
        firstChunkReceived: Bool? = nil,
        textChunkCount: Int? = nil,
        finalChunkReceived: Bool? = nil,
        streamTerminationReason: String? = nil,
        successfulObservationCount: Int? = nil,
        finalizerAccepted: Bool? = nil,
        finalizerRejectionReason: String? = nil,
        finalValidatorAcceptedCandidate: Bool? = nil,
        finalValidatorReplacementSource: String? = nil,
        finalValidatorRejectionReason: String? = nil,
        selfModel: SelfModelDecisionSummary? = nil
    ) {
        self.id = id
        self.createdAt = createdAt
        self.event = event
        self.slot = slot
        self.stage = stage
        self.scenarioID = scenarioID
        self.e2eRunID = e2eRunID
        self.agentRunID = agentRunID
        self.conversationID = conversationID
        self.turnID = turnID
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
        self.emptyOutputReason = emptyOutputReason
        self.streamStarted = streamStarted
        self.selectedRuntime = selectedRuntime
        self.selectedAdapter = selectedAdapter
        self.modelIdentifier = modelIdentifier
        self.modelLoaded = modelLoaded
        self.stopSequences = stopSequences
        self.temperature = temperature
        self.topP = topP
        self.cancellationStateBeforeStream = cancellationStateBeforeStream
        self.firstChunkReceived = firstChunkReceived
        self.textChunkCount = textChunkCount
        self.finalChunkReceived = finalChunkReceived
        self.streamTerminationReason = streamTerminationReason
        self.successfulObservationCount = successfulObservationCount
        self.finalizerAccepted = finalizerAccepted
        self.finalizerRejectionReason = finalizerRejectionReason
        self.finalValidatorAcceptedCandidate = finalValidatorAcceptedCandidate
        self.finalValidatorReplacementSource = finalValidatorReplacementSource
        self.finalValidatorRejectionReason = finalValidatorRejectionReason
        self.selfModel = selfModel
    }
}

nonisolated extension AgentBehaviorTrace {
    func redactedForPersistentDiagnostics() -> AgentBehaviorTrace {
        AgentBehaviorTrace(
            id: id,
            createdAt: createdAt,
            event: event,
            slot: slot,
            stage: stage,
            scenarioID: scenarioID,
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID,
            intent: intent,
            promptPrefix: AgentDiagnosticFileRedactor.summary(label: "prompt", text: promptPrefix),
            rawOutputPrefix: AgentDiagnosticFileRedactor.summary(label: "rawOutput", text: rawOutputPrefix),
            selectedToolID: selectedToolID,
            toolArguments: AgentDiagnosticFileRedactor.redactedMap(toolArguments),
            allowedToolIDs: allowedToolIDs,
            requiresApproval: requiresApproval,
            approvalMode: approvalMode,
            parseError: parseError,
            emittedFinalInActionTurn: emittedFinalInActionTurn,
            modelFamily: modelFamily,
            baseModelPath: baseModelPath.map { AgentDiagnosticFileRedactor.summary(label: "baseModelPath", text: $0) },
            adapterID: adapterID,
            adapterSlot: adapterSlot,
            adapterPath: adapterPath.map { AgentDiagnosticFileRedactor.summary(label: "adapterPath", text: $0) },
            adapterApplied: adapterApplied,
            adapterScale: adapterScale,
            adapterFailureReason: adapterFailureReason,
            generationElapsedMs: generationElapsedMs,
            firstTokenLatencyMs: firstTokenLatencyMs,
            outputTokenCount: outputTokenCount,
            estimatedPromptTokenCount: estimatedPromptTokenCount,
            preFirstTokenMs: preFirstTokenMs,
            messageBuildMs: messageBuildMs,
            decodeMs: decodeMs,
            tokensPerSecond: tokensPerSecond,
            ensureReadyMs: ensureReadyMs,
            adapterActivationMs: adapterActivationMs,
            runtimePath: runtimePath.map { AgentDiagnosticFileRedactor.summary(label: "runtimePath", text: $0) },
            activeAdapterSlot: activeAdapterSlot,
            maxTokensRequested: maxTokensRequested,
            maxTokensEffective: maxTokensEffective,
            promptCharCount: promptCharCount,
            accelerationDiagnostic: accelerationDiagnostic,
            accelerationDiagnostics: accelerationDiagnostics,
            emptyOutputReason: emptyOutputReason,
            streamStarted: streamStarted,
            selectedRuntime: selectedRuntime,
            selectedAdapter: selectedAdapter,
            modelIdentifier: modelIdentifier,
            modelLoaded: modelLoaded,
            stopSequences: stopSequences.map { AgentDiagnosticFileRedactor.summary(label: "stop", text: $0) },
            temperature: temperature,
            topP: topP,
            cancellationStateBeforeStream: cancellationStateBeforeStream,
            firstChunkReceived: firstChunkReceived,
            textChunkCount: textChunkCount,
            finalChunkReceived: finalChunkReceived,
            streamTerminationReason: streamTerminationReason,
            successfulObservationCount: successfulObservationCount,
            finalizerAccepted: finalizerAccepted,
            finalizerRejectionReason: finalizerRejectionReason,
            finalValidatorAcceptedCandidate: finalValidatorAcceptedCandidate,
            finalValidatorReplacementSource: finalValidatorReplacementSource,
            finalValidatorRejectionReason: finalValidatorRejectionReason,
            selfModel: selfModel
        )
    }
}

nonisolated enum AgentBehaviorTraceEmitter {
    static func recordModelTurn(
        correlation: AgentTraceCorrelation? = nil,
        slot: String,
        stage: String,
        intent: String?,
        prompt: String,
        rawOutput: String,
        selectedToolID: String? = nil,
        toolArguments: [String: String] = [:],
        allowedToolIDs: [String] = [],
        requiresApproval: Bool? = nil,
        approvalMode: String? = nil,
        parseError: String? = nil,
        emittedFinalInActionTurn: Bool = false,
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
        accelerationDiagnostics: RuntimeAccelerationDiagnostics? = nil,
        emptyOutputReason: String? = nil,
        streamStarted: Bool? = nil,
        selectedRuntime: String? = nil,
        selectedAdapter: String? = nil,
        modelIdentifier: String? = nil,
        modelLoaded: Bool? = nil,
        stopSequences: [String] = [],
        temperature: Double? = nil,
        topP: Double? = nil,
        cancellationStateBeforeStream: String? = nil,
        firstChunkReceived: Bool? = nil,
        textChunkCount: Int? = nil,
        finalChunkReceived: Bool? = nil,
        streamTerminationReason: String? = nil,
        successfulObservationCount: Int? = nil,
        finalizerAccepted: Bool? = nil,
        finalizerRejectionReason: String? = nil,
        finalValidatorAcceptedCandidate: Bool? = nil,
        finalValidatorReplacementSource: String? = nil,
        finalValidatorRejectionReason: String? = nil,
        selfModel: AgentBehaviorTrace.SelfModelDecisionSummary? = nil
    ) {
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: .modelTurn,
                slot: slot,
                stage: stage,
                scenarioID: correlation?.scenarioID,
                e2eRunID: correlation?.e2eRunID,
                agentRunID: correlation?.agentRunID,
                conversationID: correlation?.conversationID,
                turnID: correlation?.turnID,
                intent: intent,
                promptPrefix: ModelOutputSanitizer.boundedPrefix(prompt, limit: 1200),
                rawOutputPrefix: ModelOutputSanitizer.boundedPrefix(rawOutput, limit: 1600),
                selectedToolID: selectedToolID.map(ToolRouteGuard.canonicalToolID),
                toolArguments: toolArguments,
                allowedToolIDs: allowedToolIDs.map(ToolRouteGuard.canonicalToolID).sorted(),
                requiresApproval: requiresApproval,
                approvalMode: approvalMode,
                parseError: parseError,
                emittedFinalInActionTurn: emittedFinalInActionTurn,
                modelFamily: modelFamily,
                baseModelPath: baseModelPath,
                adapterID: adapterID,
                adapterSlot: adapterSlot,
                adapterPath: adapterPath,
                adapterApplied: adapterApplied,
                adapterScale: adapterScale,
                adapterFailureReason: adapterFailureReason,
                generationElapsedMs: generationElapsedMs,
                firstTokenLatencyMs: firstTokenLatencyMs,
                outputTokenCount: outputTokenCount,
                estimatedPromptTokenCount: estimatedPromptTokenCount,
                preFirstTokenMs: preFirstTokenMs,
                messageBuildMs: messageBuildMs,
                decodeMs: decodeMs,
                tokensPerSecond: tokensPerSecond,
                ensureReadyMs: ensureReadyMs,
                adapterActivationMs: adapterActivationMs,
                runtimePath: runtimePath,
                activeAdapterSlot: activeAdapterSlot,
                maxTokensRequested: maxTokensRequested,
                maxTokensEffective: maxTokensEffective,
                promptCharCount: promptCharCount,
                accelerationDiagnostic: accelerationDiagnostic,
                accelerationDiagnostics: accelerationDiagnostics,
                emptyOutputReason: emptyOutputReason,
                streamStarted: streamStarted,
                selectedRuntime: selectedRuntime,
                selectedAdapter: selectedAdapter,
                modelIdentifier: modelIdentifier,
                modelLoaded: modelLoaded,
                stopSequences: stopSequences,
                temperature: temperature,
                topP: topP,
                cancellationStateBeforeStream: cancellationStateBeforeStream,
                firstChunkReceived: firstChunkReceived,
                textChunkCount: textChunkCount,
                finalChunkReceived: finalChunkReceived,
                streamTerminationReason: streamTerminationReason,
                successfulObservationCount: successfulObservationCount,
                finalizerAccepted: finalizerAccepted,
                finalizerRejectionReason: finalizerRejectionReason,
                finalValidatorAcceptedCandidate: finalValidatorAcceptedCandidate,
                finalValidatorReplacementSource: finalValidatorReplacementSource,
                finalValidatorRejectionReason: finalValidatorRejectionReason,
                selfModel: selfModel
            )
        )
    }

    static func recordPolicyFirstToolAction(
        correlation: AgentTraceCorrelation?,
        prompt: String,
        intent: String?,
        stage: String = "compatibility-tool-action",
        selectedToolID: String,
        toolArguments: [String: String],
        allowedToolIDs: [String],
        requiresApproval: Bool,
        approvalMode: String? = nil,
        startedAt: Date
    ) {
        let canonicalToolID = ToolRouteGuard.canonicalToolID(selectedToolID)
        let rawOutput = "\(canonicalToolID)(validated)"
        recordPolicyFirst(
            correlation: correlation,
            event: .toolAction,
            prompt: prompt,
            rawOutput: rawOutput,
            intent: intent,
            stage: stage,
            selectedToolID: canonicalToolID,
            toolArguments: toolArguments,
            allowedToolIDs: allowedToolIDs,
            requiresApproval: requiresApproval,
            approvalMode: approvalMode,
            emittedFinalInActionTurn: false,
            startedAt: startedAt,
            streamTerminationReason: "validated-tool-action"
        )
    }

    static func recordPolicyFirstFinal(
        correlation: AgentTraceCorrelation?,
        prompt: String,
        intent: String?,
        stage: String = "compatibility-final",
        finalText: String,
        selectedToolID: String? = nil,
        toolArguments: [String: String] = [:],
        allowedToolIDs: [String],
        requiresApproval: Bool? = nil,
        approvalMode: String? = nil,
        startedAt: Date,
        streamTerminationReason: String = "stop",
        finalizerAccepted: Bool? = nil,
        finalizerRejectionReason: String? = nil,
        finalValidatorAcceptedCandidate: Bool? = nil,
        finalValidatorReplacementSource: String? = nil,
        finalValidatorRejectionReason: String? = nil
    ) {
        let trimmed = finalText.trimmingCharacters(in: .whitespacesAndNewlines)
        recordPolicyFirst(
            correlation: correlation,
            event: .finalAnswer,
            prompt: prompt,
            rawOutput: trimmed.isEmpty ? "deterministic compatibility final completed without text" : trimmed,
            intent: intent,
            stage: stage,
            selectedToolID: selectedToolID.map(ToolRouteGuard.canonicalToolID),
            toolArguments: toolArguments,
            allowedToolIDs: allowedToolIDs,
            requiresApproval: requiresApproval,
            approvalMode: approvalMode,
            emittedFinalInActionTurn: true,
            startedAt: startedAt,
            streamTerminationReason: streamTerminationReason,
            finalizerAccepted: finalizerAccepted,
            finalizerRejectionReason: finalizerRejectionReason,
            finalValidatorAcceptedCandidate: finalValidatorAcceptedCandidate,
            finalValidatorReplacementSource: finalValidatorReplacementSource,
            finalValidatorRejectionReason: finalValidatorRejectionReason
        )
    }

    private static func recordPolicyFirst(
        correlation: AgentTraceCorrelation?,
        event: AgentBehaviorTrace.Event,
        prompt: String,
        rawOutput: String,
        intent: String?,
        stage: String,
        selectedToolID: String?,
        toolArguments: [String: String],
        allowedToolIDs: [String],
        requiresApproval: Bool?,
        approvalMode: String?,
        emittedFinalInActionTurn: Bool,
        startedAt: Date,
        streamTerminationReason: String,
        finalizerAccepted: Bool? = nil,
        finalizerRejectionReason: String? = nil,
        finalValidatorAcceptedCandidate: Bool? = nil,
        finalValidatorReplacementSource: String? = nil,
        finalValidatorRejectionReason: String? = nil
    ) {
        let tokenCount = rawOutput.split(whereSeparator: \.isWhitespace).count
        AgentBehaviorTraceRecorder.record(
            AgentBehaviorTrace(
                id: UUID(),
                createdAt: Date(),
                event: event,
                slot: "executor",
                stage: stage,
                scenarioID: correlation?.scenarioID,
                e2eRunID: correlation?.e2eRunID,
                agentRunID: correlation?.agentRunID,
                conversationID: correlation?.conversationID,
                turnID: correlation?.turnID,
                intent: intent,
                promptPrefix: ModelOutputSanitizer.boundedPrefix(prompt, limit: 1200),
                rawOutputPrefix: ModelOutputSanitizer.boundedPrefix(rawOutput, limit: 1600),
                selectedToolID: selectedToolID,
                toolArguments: toolArguments,
                allowedToolIDs: allowedToolIDs.map(ToolRouteGuard.canonicalToolID).sorted(),
                requiresApproval: requiresApproval,
                approvalMode: approvalMode,
                parseError: nil,
                emittedFinalInActionTurn: emittedFinalInActionTurn,
                modelFamily: "policy-first",
                generationElapsedMs: Int(Date().timeIntervalSince(startedAt) * 1000),
                outputTokenCount: tokenCount,
                runtimePath: "deterministic-compatibility",
                promptCharCount: prompt.count,
                streamStarted: false,
                selectedRuntime: "deterministic-compatibility",
                modelLoaded: false,
                firstChunkReceived: false,
                textChunkCount: 0,
                finalChunkReceived: event == .finalAnswer,
                streamTerminationReason: streamTerminationReason,
                finalizerAccepted: finalizerAccepted,
                finalizerRejectionReason: finalizerRejectionReason,
                finalValidatorAcceptedCandidate: finalValidatorAcceptedCandidate,
                finalValidatorReplacementSource: finalValidatorReplacementSource,
                finalValidatorRejectionReason: finalValidatorRejectionReason
            )
        )
    }
}

nonisolated enum AgentDiagnosticFileRedactor {
    static func summary(label: String, text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        return "\(label)_chars=\(trimmed.count);sha256=\(String(RuntimeFallbackLogger.promptHash(trimmed).prefix(16)))"
    }

    static func redactedMap(_ values: [String: String]) -> [String: String] {
        Dictionary(uniqueKeysWithValues: values.map { key, value in
            (key, summary(label: key, text: value))
        })
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

private final class AgentBehaviorTraceMemoryCache: @unchecked Sendable {
    private let lock = NSLock()
    private var traces: [AgentBehaviorTrace] = []
    private let maxTraces = 512

    func remember(_ trace: AgentBehaviorTrace) {
        lock.lock()
        traces.append(trace)
        if traces.count > maxTraces {
            traces.removeFirst(traces.count - maxTraces)
        }
        lock.unlock()
    }

    func recent(limit: Int) -> [AgentBehaviorTrace] {
        lock.lock()
        defer { lock.unlock() }
        return Array(traces.suffix(max(0, limit)))
    }

    func clear() {
        lock.lock()
        traces.removeAll()
        lock.unlock()
    }
}

nonisolated enum AgentBehaviorTraceRecorder {
    private static let fileName = "agent-behavior-traces.jsonl"
    private static let maxRecentReadBytes = 1_048_576
    private static let memoryCache = AgentBehaviorTraceMemoryCache()

    static func record(_ trace: AgentBehaviorTrace) {
        memoryCache.remember(trace)
        do {
            let directory = try diagnosticsDirectory()
            let url = directory.appendingPathComponent(fileName, isDirectory: false)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(trace.redactedForPersistentDiagnostics())
            var line = data
            line.append(0x0A)
            guard DiskWriteBudget.shared.canWrite(bytes: line.count, category: .diagnostics) else { return }

            if FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: line)
            } else {
                try line.write(to: url, options: [.atomic])
            }
            DiskWriteBudget.shared.recordWrite(bytes: line.count, category: .diagnostics)
        } catch {
            // Diagnostics must never break assistant execution.
        }
    }

    static func recent(limit: Int = 200) -> [AgentBehaviorTrace] {
        let boundedLimit = max(0, limit)
        guard boundedLimit > 0, !Task.isCancelled else { return [] }
        let inMemory = memoryCache.recent(limit: boundedLimit)

        do {
            let url = try diagnosticsDirectory().appendingPathComponent(fileName, isDirectory: false)
            let path = url.path(percentEncoded: false)
            guard FileManager.default.fileExists(atPath: path) else { return inMemory }

            let attributes = try FileManager.default.attributesOfItem(atPath: path)
            let fileSize = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
            let readByteCount = min(UInt64(maxRecentReadBytes), fileSize)
            let didReadSuffix = readByteCount < fileSize
            let data: Data

            if didReadSuffix {
                let handle = try FileHandle(forReadingFrom: url)
                defer { try? handle.close() }
                let suffixStart = fileSize - readByteCount
                try handle.seek(toOffset: suffixStart - 1)
                let suffixData = try handle.read(upToCount: Int(readByteCount + 1)) ?? Data()
                data = completeLineData(fromSuffixIncludingPreviousByte: suffixData)
            } else {
                data = try Data(contentsOf: url)
            }

            guard !Task.isCancelled else { return [] }

            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let diskTraces = data.split(separator: 0x0A).compactMap { line -> AgentBehaviorTrace? in
                var lineData = Data(line)
                if lineData.last == 0x0D {
                    lineData.removeLast()
                }
                return try? decoder.decode(AgentBehaviorTrace.self, from: lineData)
            }
            return mergedRecentTraces(diskTraces, inMemory, limit: boundedLimit)
        } catch {
            return inMemory
        }
    }

    private static func mergedRecentTraces(_ groups: [AgentBehaviorTrace]..., limit: Int) -> [AgentBehaviorTrace] {
        var byID: [UUID: AgentBehaviorTrace] = [:]
        for trace in groups.flatMap({ $0 }) {
            byID[trace.id] = trace
        }
        let sorted = Array(byID.values)
            .sorted { lhs, rhs in
                if lhs.createdAt == rhs.createdAt { return lhs.id.uuidString < rhs.id.uuidString }
                return lhs.createdAt < rhs.createdAt
            }
        return Array(sorted.suffix(max(0, limit)))
    }

    private static func completeLineData(fromSuffixIncludingPreviousByte data: Data) -> Data {
        guard !data.isEmpty else { return Data() }
        guard data.first != 0x0A else { return Data(data.dropFirst()) }
        guard let newlineIndex = data.firstIndex(of: 0x0A) else { return Data() }

        let firstCompleteLineIndex = data.index(after: newlineIndex)
        guard firstCompleteLineIndex < data.endIndex else { return Data() }
        return Data(data[firstCompleteLineIndex...])
    }

    static func recentAsync(limit: Int = 200) async -> [AgentBehaviorTrace] {
        guard !Task.isCancelled else { return [] }

        let task = Task.detached(priority: .utility) {
            recent(limit: limit)
        }
        return await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            task.cancel()
        }
    }

    static func clear() {
        memoryCache.clear()
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
