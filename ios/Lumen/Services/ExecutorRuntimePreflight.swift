import Foundation

nonisolated struct ExecutorRuntimePreflightResult: Sendable, Equatable {
    let passed: Bool
    let reason: String
    let slot: String
    let modelFamily: String
    let runtimeKind: String
    let baseModelPath: String?
    let baseModelExists: Bool
    let adapterPath: String?
    let adapterExists: Bool
    let activeAdapterSlot: String?
    let resourceGateAllowed: Bool
    let budgetReason: String?
    let ensureReadySucceeded: Bool
    let smokeProbeSucceeded: Bool
    let failureKind: String?

    init(
        passed: Bool,
        reason: String,
        slot: String = LumenModelSlot.executor.rawValue,
        modelFamily: String = LumenModelFamily.persistedSelected.rawValue,
        runtimeKind: String = String(describing: LumenTrainedModelRuntimeRegistry.selected.mode),
        baseModelPath: String? = nil,
        baseModelExists: Bool = false,
        adapterPath: String? = nil,
        adapterExists: Bool = false,
        activeAdapterSlot: String? = nil,
        resourceGateAllowed: Bool = false,
        budgetReason: String? = nil,
        ensureReadySucceeded: Bool = false,
        smokeProbeSucceeded: Bool = false,
        failureKind: String? = nil
    ) {
        self.passed = passed
        self.reason = reason
        self.slot = slot
        self.modelFamily = modelFamily
        self.runtimeKind = runtimeKind
        self.baseModelPath = baseModelPath
        self.baseModelExists = baseModelExists
        self.adapterPath = adapterPath
        self.adapterExists = adapterExists
        self.activeAdapterSlot = activeAdapterSlot
        self.resourceGateAllowed = resourceGateAllowed
        self.budgetReason = budgetReason
        self.ensureReadySucceeded = ensureReadySucceeded
        self.smokeProbeSucceeded = smokeProbeSucceeded
        self.failureKind = failureKind
    }

    var diagnosticsMetadata: [String: String] {
        [
            "slot": slot,
            "modelFamily": modelFamily,
            "runtimeKind": runtimeKind,
            "baseModelPath": baseModelPath ?? "none",
            "baseModelExists": String(baseModelExists),
            "adapterPath": adapterPath ?? "none",
            "adapterExists": String(adapterExists),
            "activeAdapterSlot": activeAdapterSlot ?? "none",
            "resourceGateAllowed": String(resourceGateAllowed),
            "budgetReason": budgetReason ?? "none",
            "ensureReadySucceeded": String(ensureReadySucceeded),
            "smokeProbeSucceeded": String(smokeProbeSucceeded),
            "failureKind": failureKind ?? "none"
        ]
    }

    var diagnosticsSummary: String {
        diagnosticsMetadata
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: "; ")
    }

    func withSmokeProbeResult(_ probe: ExecutorRuntimePreflightResult) -> ExecutorRuntimePreflightResult {
        if probe.passed {
            return ExecutorRuntimePreflightResult(
                passed: true,
                reason: "\(reason); \(probe.reason)",
                slot: slot,
                modelFamily: modelFamily,
                runtimeKind: runtimeKind,
                baseModelPath: baseModelPath,
                baseModelExists: baseModelExists,
                adapterPath: adapterPath,
                adapterExists: adapterExists,
                activeAdapterSlot: activeAdapterSlot,
                resourceGateAllowed: resourceGateAllowed,
                budgetReason: budgetReason,
                ensureReadySucceeded: ensureReadySucceeded,
                smokeProbeSucceeded: true,
                failureKind: nil
            )
        }
        if probe.isNonFatalSmokeProbeFormattingFailure {
            return ExecutorRuntimePreflightResult(
                passed: true,
                reason: "\(reason); agent JSON smoke probe produced malformed structured output after runtime readiness passed; \(probe.reason)",
                slot: slot,
                modelFamily: modelFamily,
                runtimeKind: runtimeKind,
                baseModelPath: baseModelPath,
                baseModelExists: baseModelExists,
                adapterPath: adapterPath,
                adapterExists: adapterExists,
                activeAdapterSlot: activeAdapterSlot,
                resourceGateAllowed: resourceGateAllowed,
                budgetReason: budgetReason,
                ensureReadySucceeded: ensureReadySucceeded,
                smokeProbeSucceeded: false,
                failureKind: probe.failureKind
            )
        }
        return ExecutorRuntimePreflightResult(
            passed: false,
            reason: "executor preflight failed: agent JSON smoke probe failed; slot=.executor; \(probe.reason)",
            slot: slot,
            modelFamily: modelFamily,
            runtimeKind: runtimeKind,
            baseModelPath: baseModelPath,
            baseModelExists: baseModelExists,
            adapterPath: adapterPath,
            adapterExists: adapterExists,
            activeAdapterSlot: activeAdapterSlot,
            resourceGateAllowed: resourceGateAllowed,
            budgetReason: budgetReason,
            ensureReadySucceeded: ensureReadySucceeded,
            smokeProbeSucceeded: false,
            failureKind: probe.failureKind ?? "smokeProbeFailed"
        )
    }

    private var isNonFatalSmokeProbeFormattingFailure: Bool {
        failureKind == "smokeProbeParseError"
    }
}

nonisolated enum ExecutorRuntimePreflight {
    private static let budgetReason = "strict-live-training.executor-preflight"
    private static let smokeProbeAttemptCount = 2

    static func run() async -> ExecutorRuntimePreflightResult {
        let readiness = await checkReadiness(allowsLoadedMemoryPressureContinuation: true)
        guard readiness.passed else { return readiness }

        let probe = await smokeProbe(slot: .executor)
        return readiness.withSmokeProbeResult(probe)
    }

    static func checkReadiness(allowsLoadedMemoryPressureContinuation: Bool) async -> ExecutorRuntimePreflightResult {
        let slot = LumenModelSlot.executor
        let family = LumenModelFamily.persistedSelected
        let runtimeKind = String(describing: LumenTrainedModelRuntimeRegistry.selected.mode)
        let prefix = "executor preflight failed"
        let slotContract: LumenModelSlotContract
        do {
            slotContract = try LumenModelSlotContract.requiredContract(for: slot)
        } catch {
            return .init(passed: false, reason: "\(prefix): slot contract missing; slot=.executor; error=\(localizedRuntimeDescription(error))", runtimeKind: runtimeKind, failureKind: "slotContractMissing")
        }

        guard let assignment = await SlotModelRuntimeCoordinator.shared.assignment(for: slot) else {
            return .init(passed: false, reason: "\(prefix): slot=.executor assignment missing; modelFamily=\(family.rawValue); runtimeKind=\(runtimeKind)", modelFamily: family.rawValue, runtimeKind: runtimeKind, failureKind: "assignmentMissing")
        }
        let modelFamily = assignment.modelFamily?.rawValue ?? family.rawValue
        let baseModelPath = assignment.localPath
        let baseModelExists = FileManager.default.fileExists(atPath: baseModelPath)
        let adapterPath = assignment.adapterPath
        let adapterExists = adapterPath.map { FileManager.default.fileExists(atPath: $0) } ?? false
        guard slotContract.outputContract == .structuredJSON else {
            return result(
                passed: false,
                reason: "\(prefix): output contract mismatch; slot=.executor; expected=structuredJSON; actual=\(slotContract.outputContract.rawValue); modelFamily=\(modelFamily); runtimeKind=\(runtimeKind)",
                modelFamily: modelFamily,
                runtimeKind: runtimeKind,
                baseModelPath: baseModelPath,
                baseModelExists: baseModelExists,
                adapterPath: adapterPath,
                adapterExists: adapterExists,
                failureKind: "outputContractMismatch"
            )
        }
        guard baseModelExists else {
            return result(
                passed: false,
                reason: "\(prefix): base model missing; slot=.executor; modelFamily=\(modelFamily); runtimeKind=\(runtimeKind); baseModelPath=\(baseModelPath)",
                modelFamily: modelFamily,
                runtimeKind: runtimeKind,
                baseModelPath: baseModelPath,
                baseModelExists: false,
                adapterPath: adapterPath,
                adapterExists: adapterExists,
                failureKind: "baseModelMissing"
            )
        }

        let adapterRequired = assignment.requiresRoleAdapterForRuntime
        if adapterRequired {
            guard let adapterPath = assignment.adapterPath, !adapterPath.isEmpty else {
                return result(
                    passed: false,
                    reason: "\(prefix): adapter required but adapter path missing; slot=.executor; modelFamily=\(modelFamily); runtimeKind=\(runtimeKind); baseModelPath=\(baseModelPath); expectedAdapterRepo=\(assignment.expectedRoleAdapterRepoID ?? "unknown"); expectedAdapterFile=\(assignment.expectedRoleAdapterFileName ?? "unknown")",
                    modelFamily: modelFamily,
                    runtimeKind: runtimeKind,
                    baseModelPath: baseModelPath,
                    baseModelExists: baseModelExists,
                    adapterPath: nil,
                    adapterExists: false,
                    failureKind: "adapterPathMissing"
                )
            }
            guard FileManager.default.fileExists(atPath: adapterPath) else {
                return result(
                    passed: false,
                    reason: "\(prefix): adapter required but file missing; slot=.executor; modelFamily=\(modelFamily); runtimeKind=\(runtimeKind); baseModelPath=\(baseModelPath); adapterPath=\(adapterPath); expectedAdapterRepo=\(assignment.expectedRoleAdapterRepoID ?? "unknown"); expectedAdapterFile=\(assignment.expectedRoleAdapterFileName ?? "unknown")",
                    modelFamily: modelFamily,
                    runtimeKind: runtimeKind,
                    baseModelPath: baseModelPath,
                    baseModelExists: baseModelExists,
                    adapterPath: adapterPath,
                    adapterExists: false,
                    failureKind: "adapterFileMissing"
                )
            }
        } else if let adapterPath = assignment.adapterPath, !adapterPath.isEmpty, !FileManager.default.fileExists(atPath: adapterPath) {
            return result(
                passed: false,
                reason: "\(prefix): adapter configured but file missing; slot=.executor; modelFamily=\(modelFamily); runtimeKind=\(runtimeKind); baseModelPath=\(baseModelPath); adapterPath=\(adapterPath)",
                modelFamily: modelFamily,
                runtimeKind: runtimeKind,
                baseModelPath: baseModelPath,
                baseModelExists: baseModelExists,
                adapterPath: adapterPath,
                adapterExists: false,
                failureKind: "adapterFileMissing"
            )
        }

        let budgetSnapshot = await MainActor.run { ResourceBudgetGate.diagnosticSnapshot() }
        let budget = await MainActor.run {
            ResourceBudgetGate.budgetDenialReason(
                policy: slotContract.budgetPolicy,
                snapshot: budgetSnapshot,
                reason: budgetReason
            )
        }
        let deferredBudgetReason: String?
        if let budget {
            let shouldDeferToLoadedContinuation = await MainActor.run {
                shouldDeferResourceBudgetDenialToLoadedContinuation(
                    policy: slotContract.budgetPolicy,
                    snapshot: budgetSnapshot,
                    denialReason: budget,
                    allowsLoadedMemoryPressureContinuation: allowsLoadedMemoryPressureContinuation
                )
            }
            guard shouldDeferToLoadedContinuation else {
                return result(
                    passed: false,
                    reason: "\(prefix): resource-budget-denied-before-prompt-eval; slot=.executor; budgetReason=\(budget); modelFamily=\(modelFamily); runtimeKind=\(runtimeKind); baseModelPath=\(baseModelPath); adapterPath=\(adapterPath ?? "none")",
                    modelFamily: modelFamily,
                    runtimeKind: runtimeKind,
                    baseModelPath: baseModelPath,
                    baseModelExists: baseModelExists,
                    adapterPath: adapterPath,
                    adapterExists: adapterExists,
                    resourceGateAllowed: false,
                    budgetReason: budget,
                    failureKind: "resourceBudgetDenied"
                )
            }
            deferredBudgetReason = budget
        } else {
            deferredBudgetReason = nil
        }

        let readinessMetrics: RuntimeReadinessMetrics
        do {
            readinessMetrics = try await SlotModelRuntimeCoordinator.shared.ensureReadyWithMetrics(
                slot: slot,
                allowsLoadedMemoryPressureContinuation: allowsLoadedMemoryPressureContinuation
            )
        } catch {
            return result(
                passed: false,
                reason: "\(prefix): ensureReady failed; slot=.executor; modelFamily=\(modelFamily); runtimeKind=\(runtimeKind); baseModelPath=\(baseModelPath); adapterPath=\(adapterPath ?? "none"); error=\(localizedRuntimeDescription(error))",
                modelFamily: modelFamily,
                runtimeKind: runtimeKind,
                baseModelPath: baseModelPath,
                baseModelExists: baseModelExists,
                adapterPath: adapterPath,
                adapterExists: adapterExists,
                resourceGateAllowed: deferredBudgetReason == nil,
                budgetReason: deferredBudgetReason,
                ensureReadySucceeded: false,
                failureKind: "ensureReadyFailed"
            )
        }
        guard slotContract.acceptsRuntimePath(readinessMetrics.runtimePath) else {
            return result(
                passed: false,
                reason: "\(prefix): runtime policy rejected; slot=.executor; runtimePath=\(readinessMetrics.runtimePath); runtimePathKind=\(LumenModelSlotContract.runtimePathKind(for: readinessMetrics.runtimePath).rawValue); acceptedRuntimePathKinds=\(slotContract.acceptedRuntimePathKinds.map(\.rawValue).sorted().joined(separator: ",")); modelFamily=\(modelFamily); runtimeKind=\(runtimeKind); baseModelPath=\(baseModelPath); adapterPath=\(adapterPath ?? "none")",
                modelFamily: modelFamily,
                runtimeKind: runtimeKind,
                baseModelPath: baseModelPath,
                baseModelExists: baseModelExists,
                adapterPath: adapterPath,
                adapterExists: adapterExists,
                resourceGateAllowed: true,
                ensureReadySucceeded: true,
                failureKind: "runtimePolicyRejected"
            )
        }

        let activeAdapter = await AppLlamaService.shared.activeAdapterSlotValue?.rawValue
        if adapterRequired {
            guard activeAdapter == slot.rawValue else {
                return result(
                    passed: false,
                    reason: "\(prefix): adapterUnavailable; slot=.executor; activeAdapterSlot=\(activeAdapter ?? "none"); requiredAdapterSlot=.executor; adapterPath=\(adapterPath ?? "none"); expectedAdapterRepo=\(assignment.expectedRoleAdapterRepoID ?? "unknown"); expectedAdapterFile=\(assignment.expectedRoleAdapterFileName ?? "unknown")",
                    modelFamily: modelFamily,
                    runtimeKind: runtimeKind,
                    baseModelPath: baseModelPath,
                    baseModelExists: baseModelExists,
                    adapterPath: adapterPath,
                    adapterExists: adapterExists,
                    activeAdapterSlot: activeAdapter,
                    resourceGateAllowed: true,
                    ensureReadySucceeded: true,
                    failureKind: "adapterUnavailable"
                )
            }
        }
        return result(
            passed: true,
            reason: "executor readiness preflight passed; slot=.executor; modelFamily=\(modelFamily); runtimeKind=\(runtimeKind); baseModelPath=\(baseModelPath); adapterPath=\(adapterPath ?? "none")",
            modelFamily: modelFamily,
            runtimeKind: runtimeKind,
            baseModelPath: baseModelPath,
            baseModelExists: baseModelExists,
            adapterPath: adapterPath,
            adapterExists: adapterExists,
            activeAdapterSlot: activeAdapter,
            resourceGateAllowed: true,
            budgetReason: deferredBudgetReason,
            ensureReadySucceeded: true
        )
    }

    @MainActor
    private static func shouldDeferResourceBudgetDenialToLoadedContinuation(
        policy: LumenSlotBudgetPolicy,
        snapshot: ResourceBudgetGate.Snapshot,
        denialReason: String,
        allowsLoadedMemoryPressureContinuation: Bool
    ) -> Bool {
        guard allowsLoadedMemoryPressureContinuation,
              policy == .foregroundInteractive,
              denialReason.contains("recent-memory-warning") else {
            return false
        }
        return ResourceBudgetGate.allowsLoadedForegroundContinuationAfterMemoryPressure(
            snapshot: snapshot,
            reason: budgetReason
        )
    }

    #if DEBUG
    @MainActor
    static func shouldDeferResourceBudgetDenialToLoadedContinuationForTests(
        policy: LumenSlotBudgetPolicy,
        snapshot: ResourceBudgetGate.Snapshot,
        denialReason: String,
        allowsLoadedMemoryPressureContinuation: Bool
    ) -> Bool {
        shouldDeferResourceBudgetDenialToLoadedContinuation(
            policy: policy,
            snapshot: snapshot,
            denialReason: denialReason,
            allowsLoadedMemoryPressureContinuation: allowsLoadedMemoryPressureContinuation
        )
    }
    #endif

    private static func smokeProbe(slot: LumenModelSlot) async -> ExecutorRuntimePreflightResult {
        var lastFailure: ExecutorRuntimePreflightResult?
        for attempt in 0..<smokeProbeAttemptCount {
            let request = smokeProbeRequest(attempt: attempt)
            let result = await runSmokeProbeAttempt(request: request, slot: slot, attempt: attempt)
            if result.passed { return result }
            lastFailure = result
            if !shouldRetrySmokeProbe(result) { break }
        }
        return lastFailure ?? .init(
            passed: false,
            reason: "smoke probe did not run",
            failureKind: "smokeProbeNotRun"
        )
    }

    private static func smokeProbeRequest(attempt: Int) -> GenerateRequest {
        GenerateRequest(
            systemPrompt: smokeProbeSystemPrompt(attempt: attempt),
            history: [],
            userMessage: smokeProbeUserMessage(attempt: attempt),
            temperature: 0,
            topP: 0.1,
            repetitionPenalty: 1,
            maxTokens: attempt == 0 ? 64 : 96,
            modelName: "agent-json",
            relevantMemories: [],
            attachments: [],
            responseFormat: .constrainedJSON(schema: AgentService.structuredAgentResponseSchema),
            seed: UInt32(17 + attempt),
            allowsMemoryPressureContinuation: true
        )
    }

    private static func smokeProbeSystemPrompt(attempt: Int) -> String {
        var lines = [
            "You are validating Lumen's executor structured JSON channel.",
            "Output exactly one JSON object that matches the production agent turn schema.",
            #"Valid final response: {"final":"ok"}"#,
            "Do not output an empty object, schema, markdown, comments, or extra text."
        ]
        if attempt > 0 {
            lines.append("This is a retry after malformed structured output. Emit the valid final response now.")
        }
        return lines.joined(separator: "\n")
    }

    private static func smokeProbeUserMessage(attempt: Int) -> String {
        if attempt > 0 {
            return #"Retry executor JSON channel validation. Emit exactly {"final":"ok"} and nothing else. /no_think"#
        }
        return #"Emit exactly {"final":"ok"} and nothing else. /no_think"#
    }

    private static func shouldRetrySmokeProbe(_ result: ExecutorRuntimePreflightResult) -> Bool {
        result.failureKind == "smokeProbeEmptyOutput" || result.failureKind == "smokeProbeParseError"
    }

    private static func runSmokeProbeAttempt(
        request: GenerateRequest,
        slot: LumenModelSlot,
        attempt: Int
    ) async -> ExecutorRuntimePreflightResult {
        var raw = ""
        var streamStarted = false
        var firstChunkReceived = false
        streamLoop: for await token in await AppLlamaService.shared.stream(request, slot: slot) {
            streamStarted = true
            switch token {
            case .text(let text):
                if !text.isEmpty { firstChunkReceived = true }
                raw += text
            case .done:
                break streamLoop
            }
        }
        let payload = await AppLlamaService.shared.takeCompletedTracePayload(requestID: request.id)
        return evaluateSmokeProbeText(
            raw,
            payload: payload,
            streamStarted: streamStarted,
            firstChunkReceived: firstChunkReceived,
            attempt: attempt
        )
    }

    private static func evaluateSmokeProbeText(
        _ raw: String,
        payload: CompletedGenerationTracePayload?,
        streamStarted: Bool,
        firstChunkReceived: Bool,
        attempt: Int
    ) -> ExecutorRuntimePreflightResult {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let attemptSummary = "attempt=\(attempt + 1)/\(smokeProbeAttemptCount)"
        guard !text.isEmpty else {
            let reason = payload?.emptyOutputReason ?? "empty"
            return .init(passed: false, reason: "\(attemptSummary); emptyOutputReason=\(reason); outputTokens=0; streamStarted=\(payload?.streamStarted ?? streamStarted); firstChunkReceived=\(payload?.firstChunkReceived ?? firstChunkReceived)", failureKind: "smokeProbeEmptyOutput")
        }
        let parsed = AgentTurnParser.parse(text)
        guard parsed.parseError == nil else {
            return .init(passed: false, reason: "\(attemptSummary); parseError=\(parsed.parseError?.rawValue ?? "unknown"); outputPrefix=\(ModelOutputSanitizer.boundedPrefix(text, limit: 160))", failureKind: "smokeProbeParseError")
        }
        return .init(passed: true, reason: "agent JSON smoke probe passed; \(attemptSummary)", smokeProbeSucceeded: true)
    }

    #if DEBUG
    static func smokeProbeRequestForTests(attempt: Int) -> GenerateRequest {
        smokeProbeRequest(attempt: attempt)
    }

    static func evaluateSmokeProbeTextForTests(_ raw: String, attempt: Int = 0) -> ExecutorRuntimePreflightResult {
        evaluateSmokeProbeText(
            raw,
            payload: nil,
            streamStarted: true,
            firstChunkReceived: !raw.isEmpty,
            attempt: attempt
        )
    }
    #endif

    private static func result(
        passed: Bool,
        reason: String,
        modelFamily: String,
        runtimeKind: String,
        baseModelPath: String?,
        baseModelExists: Bool,
        adapterPath: String?,
        adapterExists: Bool,
        activeAdapterSlot: String? = nil,
        resourceGateAllowed: Bool = false,
        budgetReason: String? = nil,
        ensureReadySucceeded: Bool = false,
        failureKind: String? = nil
    ) -> ExecutorRuntimePreflightResult {
        ExecutorRuntimePreflightResult(
            passed: passed,
            reason: reason,
            slot: LumenModelSlot.executor.rawValue,
            modelFamily: modelFamily,
            runtimeKind: runtimeKind,
            baseModelPath: baseModelPath,
            baseModelExists: baseModelExists,
            adapterPath: adapterPath,
            adapterExists: adapterExists,
            activeAdapterSlot: activeAdapterSlot,
            resourceGateAllowed: resourceGateAllowed,
            budgetReason: budgetReason,
            ensureReadySucceeded: ensureReadySucceeded,
            smokeProbeSucceeded: false,
            failureKind: failureKind
        )
    }

    private static func localizedRuntimeDescription(_ error: Error) -> String {
        if let localized = error as? LocalizedError, let description = localized.errorDescription, !description.isEmpty {
            return description
        }
        return String(describing: error)
    }
}
