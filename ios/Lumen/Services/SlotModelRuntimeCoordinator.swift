import Foundation
import OSLog

nonisolated struct RuntimeReadinessMetrics: Sendable {
    let ensureReadyMs: Int
    let adapterActivationMs: Int
    let runtimePath: String
    let activeAdapterSlot: String?
    let accelerationDiagnostic: String
    let accelerationDiagnostics: RuntimeAccelerationDiagnostics
}

actor SlotModelRuntimeCoordinator {
    static let shared = SlotModelRuntimeCoordinator()

    private let logger = Logger(subsystem: "ai.lumen.app", category: "slot-runtime")
    private var assignments: [LumenModelSlot: LumenModelAssignment] = [:]
    private var contextSize: Int = 2048
    private var preferExclusiveChatRuntime = true

    private init() {}

    func configure(
        assignments: [LumenModelSlot: LumenModelAssignment],
        contextSize: Int,
        preferExclusiveChatRuntime: Bool
    ) {
        self.assignments = assignments.filter { slot, assignment in
            slot != .embedding && FileManager.default.fileExists(atPath: assignment.localPath)
        }
        self.contextSize = max(512, contextSize)
        self.preferExclusiveChatRuntime = preferExclusiveChatRuntime
    }

    func assignment(for slot: LumenModelSlot) -> LumenModelAssignment? {
        assignments[slot]
    }

    var configuredAssignments: [LumenModelSlot: LumenModelAssignment] {
        assignments
    }

    @discardableResult
    func ensureChatModel(
        appState: AppState?,
        candidates: [StoredModelLoadItem],
        preferredID: String?
    ) async -> Bool {
        guard let selectedID = await ensureChatModelSelection(candidates: candidates, preferredID: preferredID) else { return false }
        await MainActor.run { appState?.activeChatModelID = selectedID }
        return true
    }

    func ensureChatModelSelection(
        candidates: [StoredModelLoadItem],
        preferredID: String?
    ) async -> String? {
        guard await MainActor.run(body: { ResourceBudgetGate.allowsWork(policy: .foregroundInteractive, reason: ModelLoadIntent.userChat.rawValue) }), !Task.isCancelled else { return nil }
        let orderedCandidates = orderedCandidates(candidates: candidates, preferredID: preferredID)
        for (index, candidate) in orderedCandidates.enumerated() {
            await Task.yield()
            let path = candidate.resolvedPath
            logger.info("transition event=attempt role=chat index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public)")
            guard FileManager.default.fileExists(atPath: path) else {
                logger.info("transition event=skip_missing role=chat index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public)")
                continue
            }

            do {
                await AppLlamaService.shared.unloadAllChat()
                try await AppLlamaService.shared.loadChatModel(path: path, contextSize: contextSize)
                logger.info("transition event=\(self.selectionEvent(index: index, candidateID: candidate.id.uuidString, preferredID: preferredID), privacy: .public) role=chat model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public)")
                return candidate.id.uuidString
            } catch {
                if isContextInitFailed(error) {
                    do {
                        logger.info("transition event=retry_context_2048 role=chat model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public)")
                        await AppLlamaService.shared.unloadAllChat()
                        try await AppLlamaService.shared.loadChatModel(path: path, contextSize: 2048)
                        logger.info("transition event=\(self.selectionEvent(index: index, candidateID: candidate.id.uuidString, preferredID: preferredID), privacy: .public) role=chat model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public) context=2048")
                        return candidate.id.uuidString
                    } catch {
                        logger.error("transition event=retry_context_2048_failed role=chat index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public) error=\(String(describing: error), privacy: .public)")
                        logger.error("transition event=failed_candidate role=chat index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public) error=\(String(describing: error), privacy: .public)")
                        continue
                    }
                }
                logger.error("transition event=failed_candidate role=chat index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public) error=\(String(describing: error), privacy: .public)")
                continue
            }
        }
        logger.error("transition event=failed_all role=chat")
        return nil
    }

    /// Loads an embedding model from available candidates, prioritizing a preferred model if specified.
    /// - Parameters:
    ///   - appState: Updated with the ID of the successfully loaded embedding model.
    ///   - preferredID: Optional UUID string to prioritize among candidates.
    /// - Returns: `true` if an embedding model is successfully loaded, `false` otherwise.
    @discardableResult
    func ensureEmbeddingModel(
        appState: AppState,
        candidates: [StoredModel],
        preferredID: String?
    ) async -> Bool {
        guard await MainActor.run(body: { ResourceBudgetGate.allowsWork(policy: .embedding, reason: ModelLoadIntent.userChat.rawValue) }), !Task.isCancelled else { return false }
        let orderedCandidates = orderedCandidates(candidates: candidates, preferredID: preferredID)
        for (index, candidate) in orderedCandidates.enumerated() {
            await Task.yield()
            let path = ModelStorage.resolvedModelURL(from: candidate.localPath, fileName: candidate.fileName).path
            logger.info("transition event=attempt role=embedding index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public)")
            guard FileManager.default.fileExists(atPath: path) else {
                logger.info("transition event=skip_missing role=embedding index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public)")
                continue
            }

            do {
                try await AppLlamaService.shared.loadEmbeddingModel(path: path)
                let candidateID = candidate.id.uuidString
                await MainActor.run { appState.activeEmbeddingModelID = candidateID }
                logger.info("transition event=\(self.selectionEvent(index: index, candidateID: candidateID, preferredID: preferredID), privacy: .public) role=embedding model_id=\(candidateID, privacy: .public) path=\(path, privacy: .public)")
                return true
            } catch {
                logger.error("transition event=failed_candidate role=embedding index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public) error=\(String(describing: error), privacy: .public)")
                continue
            }
        }
        logger.error("transition event=failed_all role=embedding")
        return false
    }

    func ensureReady(slot: LumenModelSlot) async throws {
        _ = try await ensureReadyWithMetrics(slot: slot)
    }

    func ensureReadyWithMetrics(
        slot: LumenModelSlot,
        allowsLoadedMemoryPressureContinuation: Bool = false
    ) async throws -> RuntimeReadinessMetrics {
        let started = Date()
        guard !Task.isCancelled else {
            throw LocalRuntimeError.unavailable("resource budget denied model load")
        }
        let slotContract = LumenModelSlotContract.contract(for: slot)
        let budgetPolicy = slotContract?.budgetPolicy ?? (slot == .embedding ? .embedding : .foregroundInteractive)
        let budgetReason = "slot-runtime.\(slot.rawValue)"
        let budgetSnapshot = await MainActor.run { ResourceBudgetGate.diagnosticSnapshot() }
        let budgetDenial = await MainActor.run {
            ResourceBudgetGate.budgetDenialReason(policy: budgetPolicy, snapshot: budgetSnapshot, reason: budgetReason)
        }
        if let budgetDenial {
            if budgetPolicy == .foregroundInteractive,
               allowsLoadedMemoryPressureContinuation,
               await MainActor.run(body: {
                   ResourceBudgetGate.allowsLoadedForegroundContinuationAfterMemoryPressure(snapshot: budgetSnapshot, reason: budgetReason)
               }),
               let loadedMetrics = await loadedContinuationMetricsIfReady(slot: slot, started: started) {
                return loadedMetrics
            }
            throw LocalRuntimeError.unavailable("resource budget denied model load: \(budgetDenial)")
        }
        guard slot != .embedding else {
            return RuntimeReadinessMetrics(
                ensureReadyMs: 0,
                adapterActivationMs: 0,
                runtimePath: "embedding",
                activeAdapterSlot: nil,
                accelerationDiagnostic: "GPU/offload introspection unavailable in wrapper; see runtime init logs.",
                accelerationDiagnostics: RuntimeAccelerationDiagnostics.forCurrentRuntime(requestedBackend: "unknown", requestedGpuLayers: nil, requestedKQVOffload: nil)
            )
        }
        let assignment = resolvedAssignment(for: slot)
        guard let assignment else {
            if await AppLlamaService.shared.isChatLoaded {
                let elapsed = Int(Date().timeIntervalSince(started) * 1000)
                return RuntimeReadinessMetrics(
                    ensureReadyMs: elapsed,
                    adapterActivationMs: 0,
                    runtimePath: "loadedChatFallback",
                    activeAdapterSlot: nil,
                    accelerationDiagnostic: "Using already-loaded standalone chat runtime without a fleet slot assignment.",
                    accelerationDiagnostics: await AppLlamaService.shared.currentAccelerationDiagnostics()
                )
            }
            throw LlamaError.slotModelNotLoaded("\(slot.rawValue): no assigned model")
        }
        guard FileManager.default.fileExists(atPath: assignment.localPath) else {
            throw LlamaError.modelFileNotFound(assignment.localPath)
        }

        if assignment.usesRoleAdapter || assignment.modelFamily == .qwen3 {
            let activationMs = try await ensureAdapterRuntimeReady(slot: slot, assignment: assignment)
            let activeAdapterSlot = await AppLlamaService.shared.activeAdapterSlotValue?.rawValue
            let elapsed = Int(Date().timeIntervalSince(started) * 1000)
            return RuntimeReadinessMetrics(
                ensureReadyMs: elapsed,
                adapterActivationMs: activationMs,
                runtimePath: "sharedAdapter",
                activeAdapterSlot: activeAdapterSlot,
                accelerationDiagnostic: "See accelerationDiagnostics for parsed llama.cpp Metal/offload evidence.",
                accelerationDiagnostics: await AppLlamaService.shared.currentAccelerationDiagnostics()
            )
        }

        try await ensureLegacyRuntimeReady(slot: slot, assignment: assignment)
        let elapsed = Int(Date().timeIntervalSince(started) * 1000)
        return RuntimeReadinessMetrics(
            ensureReadyMs: elapsed,
            adapterActivationMs: 0,
            runtimePath: "legacySlot",
            activeAdapterSlot: nil,
            accelerationDiagnostic: "See accelerationDiagnostics for parsed SwiftLlama/llama.cpp Metal/offload evidence.",
            accelerationDiagnostics: await AppLlamaService.shared.currentAccelerationDiagnostics()
        )
    }


    func hasLoadedRuntimeReadyForContinuation(slot: LumenModelSlot) async -> Bool {
        await loadedContinuationMetricsIfReady(slot: slot, started: Date()) != nil
    }

    private func loadedContinuationMetricsIfReady(slot: LumenModelSlot, started: Date) async -> RuntimeReadinessMetrics? {
        guard slot != .embedding else { return nil }

        guard let assignment = resolvedAssignment(for: slot) else {
            guard await AppLlamaService.shared.isChatLoaded else { return nil }
            let elapsed = Int(Date().timeIntervalSince(started) * 1000)
            return RuntimeReadinessMetrics(
                ensureReadyMs: elapsed,
                adapterActivationMs: 0,
                runtimePath: "loadedChatFallback",
                activeAdapterSlot: nil,
                accelerationDiagnostic: "Using already-loaded standalone chat runtime without a fleet slot assignment.",
                accelerationDiagnostics: await AppLlamaService.shared.currentAccelerationDiagnostics()
            )
        }

        guard FileManager.default.fileExists(atPath: assignment.localPath) else { return nil }

        if assignment.usesRoleAdapter || assignment.modelFamily == .qwen3 {
            let requiresRoleAdapter = requiresRoleAdapter(assignment: assignment)
            guard await AppLlamaService.shared.loadedChatPath == assignment.localPath else { return nil }
            let activeAdapterSlot = await AppLlamaService.shared.activeAdapterSlotValue
            if requiresRoleAdapter {
                guard let adapterPath = assignment.adapterPath,
                      FileManager.default.fileExists(atPath: adapterPath),
                      activeAdapterSlot == slot else {
                    return nil
                }
            } else if let adapterPath = assignment.adapterPath,
                      FileManager.default.fileExists(atPath: adapterPath),
                      activeAdapterSlot != slot {
                return nil
            }
            let elapsed = Int(Date().timeIntervalSince(started) * 1000)
            return RuntimeReadinessMetrics(
                ensureReadyMs: elapsed,
                adapterActivationMs: 0,
                runtimePath: "sharedAdapterLoadedContinuation",
                activeAdapterSlot: activeAdapterSlot?.rawValue,
                accelerationDiagnostic: "Continuing with already-loaded shared runtime after memory-pressure gate.",
                accelerationDiagnostics: await AppLlamaService.shared.currentAccelerationDiagnostics()
            )
        }

        let slotLoaded = await AppLlamaService.shared.loadedChatPath(for: slot) == assignment.localPath
        let aliasLoaded = await AppLlamaService.shared.slotLoaded(withPath: assignment.localPath) != nil
        guard slotLoaded || aliasLoaded else { return nil }
        let elapsed = Int(Date().timeIntervalSince(started) * 1000)
        return RuntimeReadinessMetrics(
            ensureReadyMs: elapsed,
            adapterActivationMs: 0,
            runtimePath: "legacySlotLoadedContinuation",
            activeAdapterSlot: nil,
            accelerationDiagnostic: "Continuing with already-loaded slot runtime after memory-pressure gate.",
            accelerationDiagnostics: await AppLlamaService.shared.currentAccelerationDiagnostics()
        )
    }

    private func resolvedAssignment(for slot: LumenModelSlot) -> LumenModelAssignment? {
        if let direct = assignments[slot] {
            return direct
        }

        // Speech mode and simple chat can route through Mouth even when only a
        // Cortex/base chat artifact is installed. Fall back to Cortex to avoid
        // hard failures when the Mouth slot has no explicit assignment. At load
        // time we alias the Mouth slot to any already-loaded runtime for this
        // same model path so we do not force an unnecessary unload/reload cycle.
        if slot == .mouth {
            return assignments[.cortex]
        }

        return nil
    }

    private func ensureAdapterRuntimeReady(slot: LumenModelSlot, assignment: LumenModelAssignment) async throws -> Int {
        let activationStart = Date()
        let requiresRoleAdapter = requiresRoleAdapter(assignment: assignment)
        if await AppLlamaService.shared.loadedChatPath != assignment.localPath {
            do {
                try await AppLlamaService.shared.loadSharedChatModel(path: assignment.localPath, contextSize: contextSize)
            } catch {
                logger.error("shared_base_load_failed slot=\(slot.rawValue, privacy: .public) path=\(assignment.localPath, privacy: .public) context=\(self.contextSize, privacy: .public) error=\(String(describing: error), privacy: .public)")
                if contextSize > 2048 {
                    try await AppLlamaService.shared.loadSharedChatModel(path: assignment.localPath, contextSize: 2048)
                } else {
                    throw error
                }
            }
        }

        guard let adapterPath = assignment.adapterPath else {
            await AppLlamaService.shared.clearActiveRoleAdapter()
            if requiresRoleAdapter {
                let expectedRepo = assignment.expectedRoleAdapterRepoID ?? "unknown"
                let expectedFile = assignment.expectedRoleAdapterFileName ?? "unknown"
                throw LocalRuntimeError.unavailable("role adapter missing for \(slot.rawValue): expectedAdapterRepo=\(expectedRepo); expectedAdapterFile=\(expectedFile)")
            }
            return Int(Date().timeIntervalSince(activationStart) * 1000)
        }
        guard FileManager.default.fileExists(atPath: adapterPath) else {
            logger.error("role_adapter_missing slot=\(slot.rawValue, privacy: .public) path=\(adapterPath, privacy: .public)")
            await AppLlamaService.shared.clearActiveRoleAdapter()
            throw LlamaError.modelFileNotFound(adapterPath)
        }

        do {
            let loadedNow = try await AppLlamaService.shared.loadRoleAdapterIfNeeded(slot: slot, path: adapterPath, scale: assignment.adapterScale)
            let activatedNow = try await AppLlamaService.shared.activateRoleAdapterIfNeeded(slot: slot)
            if loadedNow || activatedNow {
                logger.info("role_adapter_activation_performed slot=\(slot.rawValue, privacy: .public) path=\(adapterPath, privacy: .public)")
            }
        } catch {
            logger.error("role_adapter_activation_failed slot=\(slot.rawValue, privacy: .public) path=\(adapterPath, privacy: .public) error=\(String(describing: error), privacy: .public)")
            await AppLlamaService.shared.unloadRoleAdapter(slot: slot)
            throw error
        }
        return Int(Date().timeIntervalSince(activationStart) * 1000)
    }

    private func requiresRoleAdapter(assignment: LumenModelAssignment) -> Bool {
        assignment.requiresRoleAdapterForRuntime
    }

    private func ensureLegacyRuntimeReady(slot: LumenModelSlot, assignment: LumenModelAssignment) async throws {
        if await AppLlamaService.shared.loadedChatPath(for: slot) == assignment.localPath {
            return
        }
        if let loadedSlot = await AppLlamaService.shared.slotLoaded(withPath: assignment.localPath) {
            await AppLlamaService.shared.aliasChatRuntime(from: loadedSlot, to: slot)
            return
        }

        if preferExclusiveChatRuntime {
            await AppLlamaService.shared.unloadAllChat()
        } else {
            await AppLlamaService.shared.unloadChat(for: slot)
        }

        do {
            try await AppLlamaService.shared.loadChatModel(
                path: assignment.localPath,
                for: slot,
                contextSize: contextSize
            )
        } catch {
            logger.error("slot_model_load_failed slot=\(slot.rawValue, privacy: .public) path=\(assignment.localPath, privacy: .public) context=\(self.contextSize, privacy: .public) error=\(String(describing: error), privacy: .public)")
            if contextSize > 2048 {
                await AppLlamaService.shared.unloadAllChat()
                try await AppLlamaService.shared.loadChatModel(
                    path: assignment.localPath,
                    for: slot,
                    contextSize: 2048
                )
            } else {
                throw error
            }
        }
    }

    func ensurePrimaryReady(preferredSlots: [LumenModelSlot] = [.mouth, .cortex]) async -> Bool {
        for slot in preferredSlots {
            guard resolvedAssignment(for: slot) != nil else { continue }
            do {
                try await ensureReady(slot: slot)
                return true
            } catch {
                logger.error("primary_slot_ready_failed slot=\(slot.rawValue, privacy: .public) error=\(String(describing: error), privacy: .public)")
                continue
            }
        }
        return false
    }

    private func orderedCandidates(candidates: [StoredModelLoadItem], preferredID: String?) -> [StoredModelLoadItem] {
        let pool = candidates.filter { FileManager.default.fileExists(atPath: $0.resolvedPath) }
        var ordered: [StoredModelLoadItem] = []
        if let preferredID, let preferred = pool.first(where: { $0.id.uuidString == preferredID }) {
            ordered.append(preferred)
        }
        for candidate in pool where !ordered.contains(where: { $0.id == candidate.id }) {
            ordered.append(candidate)
        }
        return ordered
    }

    private func orderedCandidates(candidates: [StoredModel], preferredID: String?) -> [StoredModel] {
        let pool = candidates.filter { ModelFileIntegrity.validateInstalledFile($0) }
        var ordered: [StoredModel] = []
        if let preferredID, let preferred = pool.first(where: { $0.id.uuidString == preferredID }) {
            ordered.append(preferred)
        }
        for candidate in pool where !ordered.contains(where: { $0.id == candidate.id }) {
            ordered.append(candidate)
        }
        return ordered
    }

    private func isContextInitFailed(_ error: Error) -> Bool {
        guard case LlamaError.failedToInitializeContext = error else {
            return false
        }
        return true
    }

    nonisolated func selectionEvent(index: Int, candidateID: String, preferredID: String?) -> String {
        if index > 0 { return "fallback_selected" }
        if let preferredID, candidateID != preferredID { return "fallback_selected" }
        return "selected"
    }
}
