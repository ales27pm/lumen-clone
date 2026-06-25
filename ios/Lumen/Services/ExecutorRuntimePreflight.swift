import Foundation

nonisolated struct ExecutorRuntimePreflightResult: Sendable, Equatable {
    let passed: Bool
    let reason: String
}

nonisolated enum ExecutorRuntimePreflight {
    static func run() async -> ExecutorRuntimePreflightResult {
        let readiness = await checkReadiness(allowsLoadedMemoryPressureContinuation: true)
        guard readiness.passed else { return readiness }

        let probe = await smokeProbe(slot: .executor)
        guard probe.passed else {
            return .init(passed: false, reason: "executor preflight failed: tiny JSON smoke probe failed; slot=.executor; \(probe.reason)")
        }
        return .init(passed: true, reason: "\(readiness.reason); tiny JSON smoke probe passed")
    }

    static func checkReadiness(allowsLoadedMemoryPressureContinuation: Bool) async -> ExecutorRuntimePreflightResult {
        let slot = LumenModelSlot.executor
        let family = LumenModelFamily.persistedSelected
        let runtimeKind = LumenTrainedModelRuntimeRegistry.selected.mode
        let prefix = "executor preflight failed"
        let slotContract: LumenModelSlotContract
        do {
            slotContract = try LumenModelSlotContract.requiredContract(for: slot)
        } catch {
            return .init(passed: false, reason: "\(prefix): slot contract missing; slot=.executor; error=\(localizedRuntimeDescription(error))")
        }

        guard let assignment = await SlotModelRuntimeCoordinator.shared.assignment(for: slot) else {
            return .init(passed: false, reason: "\(prefix): slot=.executor assignment missing; modelFamily=\(family.rawValue); runtimeKind=\(runtimeKind)")
        }
        guard slotContract.outputContract == .structuredJSON else {
            return .init(passed: false, reason: "\(prefix): output contract mismatch; slot=.executor; expected=structuredJSON; actual=\(slotContract.outputContract.rawValue); modelFamily=\(assignment.modelFamily?.rawValue ?? family.rawValue); runtimeKind=\(runtimeKind)")
        }
        guard FileManager.default.fileExists(atPath: assignment.localPath) else {
            return .init(passed: false, reason: "\(prefix): base model missing; slot=.executor; modelFamily=\(assignment.modelFamily?.rawValue ?? family.rawValue); runtimeKind=\(runtimeKind); baseModelPath=\(assignment.localPath)")
        }

        let adapterRequired = assignment.usesRoleAdapter
            || (assignment.modelFamily == .qwen3 && LumenTrainedModelRuntimeRegistry.contract(for: .qwen3).adapterRole(for: slot) != nil)
        if adapterRequired {
            guard let adapterPath = assignment.adapterPath, !adapterPath.isEmpty else {
                return .init(passed: false, reason: "\(prefix): adapter required but adapter path missing; slot=.executor; modelFamily=\(assignment.modelFamily?.rawValue ?? family.rawValue); runtimeKind=\(runtimeKind); baseModelPath=\(assignment.localPath)")
            }
            guard FileManager.default.fileExists(atPath: adapterPath) else {
                return .init(passed: false, reason: "\(prefix): adapter required but file missing; slot=.executor; modelFamily=\(assignment.modelFamily?.rawValue ?? family.rawValue); runtimeKind=\(runtimeKind); baseModelPath=\(assignment.localPath); adapterPath=\(adapterPath)")
            }
        }

        let budget = await MainActor.run {
            ResourceBudgetGate.budgetDenialReason(
                policy: slotContract.budgetPolicy,
                reason: "strict-live-training.executor-preflight"
            )
        }
        if let budget {
            return .init(passed: false, reason: "\(prefix): resource-budget-denied-before-prompt-eval; slot=.executor; budgetReason=\(budget); modelFamily=\(assignment.modelFamily?.rawValue ?? family.rawValue); runtimeKind=\(runtimeKind); baseModelPath=\(assignment.localPath); adapterPath=\(assignment.adapterPath ?? "none")")
        }

        let readinessMetrics: RuntimeReadinessMetrics
        do {
            readinessMetrics = try await SlotModelRuntimeCoordinator.shared.ensureReadyWithMetrics(
                slot: slot,
                allowsLoadedMemoryPressureContinuation: allowsLoadedMemoryPressureContinuation
            )
        } catch {
            return .init(passed: false, reason: "\(prefix): ensureReady failed; slot=.executor; modelFamily=\(assignment.modelFamily?.rawValue ?? family.rawValue); runtimeKind=\(runtimeKind); baseModelPath=\(assignment.localPath); adapterPath=\(assignment.adapterPath ?? "none"); error=\(localizedRuntimeDescription(error))")
        }
        guard slotContract.acceptsRuntimePath(readinessMetrics.runtimePath) else {
            return .init(passed: false, reason: "\(prefix): runtime policy rejected; slot=.executor; runtimePath=\(readinessMetrics.runtimePath); runtimePathKind=\(LumenModelSlotContract.runtimePathKind(for: readinessMetrics.runtimePath).rawValue); acceptedRuntimePathKinds=\(slotContract.acceptedRuntimePathKinds.map(\.rawValue).sorted().joined(separator: ",")); modelFamily=\(assignment.modelFamily?.rawValue ?? family.rawValue); runtimeKind=\(runtimeKind); baseModelPath=\(assignment.localPath); adapterPath=\(assignment.adapterPath ?? "none")")
        }

        if adapterRequired {
            let activeAdapter = await AppLlamaService.shared.activeAdapterSlotValue?.rawValue
            guard activeAdapter == slot.rawValue else {
                return .init(passed: false, reason: "\(prefix): adapterUnavailable; slot=.executor; activeAdapterSlot=\(activeAdapter ?? "none"); requiredAdapterSlot=.executor; adapterPath=\(assignment.adapterPath ?? "none")")
            }
        }
        return .init(passed: true, reason: "executor readiness preflight passed; slot=.executor; modelFamily=\(assignment.modelFamily?.rawValue ?? family.rawValue); runtimeKind=\(runtimeKind); baseModelPath=\(assignment.localPath); adapterPath=\(assignment.adapterPath ?? "none")")
    }

    private static func smokeProbe(slot: LumenModelSlot) async -> ExecutorRuntimePreflightResult {
        let request = GenerateRequest(
            systemPrompt: "Return only valid JSON.",
            history: [],
            userMessage: #"Return exactly {"final":"ok"}"#,
            temperature: 0,
            topP: 0.1,
            repetitionPenalty: 1,
            maxTokens: 24,
            modelName: "executor-preflight-json",
            relevantMemories: [],
            attachments: [],
            responseFormat: .constrainedJSON(schema: #"{"type":"object","required":["final"],"properties":{"final":{"const":"ok"}}}"#),
            allowsMemoryPressureContinuation: true
        )
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
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            let reason = payload?.emptyOutputReason ?? "empty"
            return .init(passed: false, reason: "emptyOutputReason=\(reason); outputTokens=0; streamStarted=\(payload?.streamStarted ?? streamStarted); firstChunkReceived=\(payload?.firstChunkReceived ?? firstChunkReceived)")
        }
        let parsed = AgentTurnParser.parse(text)
        guard parsed.parseError == nil else {
            return .init(passed: false, reason: "parseError=\(parsed.parseError?.rawValue ?? "unknown"); outputPrefix=\(ModelOutputSanitizer.boundedPrefix(text, limit: 160))")
        }
        return .init(passed: true, reason: "tiny JSON smoke probe passed")
    }

    private static func localizedRuntimeDescription(_ error: Error) -> String {
        if let localized = error as? LocalizedError, let description = localized.errorDescription, !description.isEmpty {
            return description
        }
        return String(describing: error)
    }
}
