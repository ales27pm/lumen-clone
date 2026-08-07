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
    private var configurationGeneration: UInt64 = 0

    private init() {}

    func configure(
        assignments: [LumenModelSlot: LumenModelAssignment],
        contextSize: Int,
        preferExclusiveChatRuntime: Bool
    ) {
        configurationGeneration &+= 1
        self.assignments = assignments.filter { slot, assignment in
            slot != .embedding && FileManager.default.fileExists(atPath: assignment.localPath)
        }
        self.contextSize = max(512, contextSize)
        self.preferExclusiveChatRuntime = preferExclusiveChatRuntime
    }

    func assignment(for slot: LumenModelSlot) -> LumenModelAssignment? {
        assignments[slot]
    }

    #if DEBUG
    func resolvedAssignmentForTesting(for slot: LumenModelSlot) -> LumenModelAssignment? {
        resolvedAssignment(for: slot)
    }
    #endif

    var configuredAssignments: [LumenModelSlot: LumenModelAssignment] {
        assignments
    }

    @discardableResult
    func ensureChatModel(
        appState: AppState?,
        candidates: [StoredModelLoadItem],
        preferredID: String?
    ) async -> Bool {
        guard appState?.activeChatModelID == preferredID,
              let selectedID = await ensureChatModelSelection(candidates: candidates, preferredID: preferredID),
              selectedID == preferredID
        else { return false }
        return appState?.activeChatModelID == preferredID
    }

    func ensureChatModelSelection(
        candidates: [StoredModelLoadItem],
        preferredID: String?
    ) async -> String? {
        guard await MainActor.run(body: { ResourceBudgetGate.allowsWork(policy: .foregroundInteractive, reason: ModelLoadIntent.userChat.rawValue) }), !Task.isCancelled else { return nil }
        let orderedCandidates = await orderedCandidates(candidates: candidates, preferredID: preferredID)
        for (index, candidate) in orderedCandidates.enumerated() {
            await Task.yield()
            guard !Task.isCancelled else { return nil }
            let path = candidate.resolvedPath
            logger.info("transition event=attempt role=chat index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .private)")
            guard FileManager.default.fileExists(atPath: path) else {
                logger.info("transition event=skip_missing role=chat index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .private)")
                continue
            }

            do {
                await AppLlamaService.shared.unloadAllChat()
                try await AppLlamaService.shared.loadChatModel(path: path, contextSize: contextSize)
                guard !Task.isCancelled else {
                    await AppLlamaService.shared.unloadAllChat()
                    return nil
                }
                logger.info("transition event=\(self.selectionEvent(index: index, candidateID: candidate.id.uuidString, preferredID: preferredID), privacy: .public) role=chat model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .private)")
                return candidate.id.uuidString
            } catch {
                if error is CancellationError { return nil }
                if isContextInitFailed(error) {
                    do {
                        logger.info("transition event=retry_context_2048 role=chat model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .private)")
                        await AppLlamaService.shared.unloadAllChat()
                        try await AppLlamaService.shared.loadChatModel(path: path, contextSize: 2048)
                        guard !Task.isCancelled else {
                            await AppLlamaService.shared.unloadAllChat()
                            return nil
                        }
                        logger.info("transition event=\(self.selectionEvent(index: index, candidateID: candidate.id.uuidString, preferredID: preferredID), privacy: .public) role=chat model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .private) context=2048")
                        return candidate.id.uuidString
                    } catch {
                        if error is CancellationError { return nil }
                        logger.error("transition event=retry_context_2048_failed role=chat index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .private) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
                        logger.error("transition event=failed_candidate role=chat index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .private) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
                        continue
                    }
                }
                logger.error("transition event=failed_candidate role=chat index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .private) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
                continue
            }
        }
        logger.error("transition event=failed_all role=chat")
        return nil
    }

    /// Loads an embedding model from available candidates, prioritizing a preferred model if specified.
    /// Returns the loaded candidate ID. The caller owns guarded publication to app state so a
    /// stale load cannot replace a model selection made while the actor was suspended.
    func ensureEmbeddingModelSelection(
        candidates: [StoredModelLoadItem],
        preferredID: String?
    ) async -> String? {
        guard await MainActor.run(body: { ResourceBudgetGate.allowsWork(policy: .embedding, reason: ModelLoadIntent.userChat.rawValue) }), !Task.isCancelled else { return nil }
        let orderedCandidates = await orderedCandidates(candidates: candidates, preferredID: preferredID)
        for (index, candidate) in orderedCandidates.enumerated() {
            await Task.yield()
            guard !Task.isCancelled else { return nil }
            let path = candidate.resolvedPath
            logger.info("transition event=attempt role=embedding index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .private)")
            guard FileManager.default.fileExists(atPath: path) else {
                logger.info("transition event=skip_missing role=embedding index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .private)")
                continue
            }

            do {
                try await AppLlamaService.shared.loadEmbeddingModel(path: path)
                guard !Task.isCancelled else {
                    await AppLlamaService.shared.unloadEmbed()
                    return nil
                }
                let candidateID = candidate.id.uuidString
                logger.info("transition event=\(self.selectionEvent(index: index, candidateID: candidateID, preferredID: preferredID), privacy: .public) role=embedding model_id=\(candidateID, privacy: .public) path=\(path, privacy: .private)")
                return candidateID
            } catch {
                if error is CancellationError { return nil }
                logger.error("transition event=failed_candidate role=embedding index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .private) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
                continue
            }
        }
        logger.error("transition event=failed_all role=embedding")
        return nil
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
        let assignmentGeneration = configurationGeneration
        let assignment = resolvedAssignment(for: slot)
        guard let assignment else {
            throw LlamaError.slotModelNotLoaded("\(slot.rawValue): no assigned model")
        }
        guard FileManager.default.fileExists(atPath: assignment.localPath) else {
            throw LlamaError.modelFileNotFound(assignment.localPath)
        }

        if assignment.usesRoleAdapter || assignment.modelFamily == .qwen3 {
            let activationMs = try await ensureAdapterRuntimeReady(
                slot: slot,
                assignment: assignment,
                generation: assignmentGeneration
            )
            guard assignmentRemainsOwned(assignment, slot: slot, generation: assignmentGeneration) else {
                throw CancellationError()
            }
            let activeAdapterSlot = await AppLlamaService.shared.activeAdapterSlotValue?.rawValue
            guard assignmentRemainsOwned(assignment, slot: slot, generation: assignmentGeneration) else {
                throw CancellationError()
            }
            let elapsed = Int(Date().timeIntervalSince(started) * 1000)
            let accelerationDiagnostics = await AppLlamaService.shared.currentAccelerationDiagnostics()
            guard assignmentRemainsOwned(assignment, slot: slot, generation: assignmentGeneration) else {
                throw CancellationError()
            }
            return RuntimeReadinessMetrics(
                ensureReadyMs: elapsed,
                adapterActivationMs: activationMs,
                runtimePath: "sharedAdapter",
                activeAdapterSlot: activeAdapterSlot,
                accelerationDiagnostic: "See accelerationDiagnostics for parsed llama.cpp Metal/offload evidence.",
                accelerationDiagnostics: accelerationDiagnostics
            )
        }

        try await ensureLegacyRuntimeReady(
            slot: slot,
            assignment: assignment,
            generation: assignmentGeneration
        )
        guard assignmentRemainsOwned(assignment, slot: slot, generation: assignmentGeneration) else {
            throw CancellationError()
        }
        let elapsed = Int(Date().timeIntervalSince(started) * 1000)
        let accelerationDiagnostics = await AppLlamaService.shared.currentAccelerationDiagnostics()
        guard assignmentRemainsOwned(assignment, slot: slot, generation: assignmentGeneration) else {
            throw CancellationError()
        }
        return RuntimeReadinessMetrics(
            ensureReadyMs: elapsed,
            adapterActivationMs: 0,
            runtimePath: "legacySlot",
            activeAdapterSlot: nil,
            accelerationDiagnostic: "See accelerationDiagnostics for parsed SwiftLlama/llama.cpp Metal/offload evidence.",
            accelerationDiagnostics: accelerationDiagnostics
        )
    }


    func hasLoadedRuntimeReadyForContinuation(slot: LumenModelSlot) async -> Bool {
        await loadedContinuationMetricsIfReady(slot: slot, started: Date()) != nil
    }

    private func loadedContinuationMetricsIfReady(slot: LumenModelSlot, started: Date) async -> RuntimeReadinessMetrics? {
        guard slot != .embedding else { return nil }

        let assignmentGeneration = configurationGeneration
        guard let assignment = resolvedAssignment(for: slot) else { return nil }

        guard FileManager.default.fileExists(atPath: assignment.localPath) else { return nil }

        if assignment.usesRoleAdapter || assignment.modelFamily == .qwen3 {
            guard assignment.configuredSharedBaseMatchesContract else { return nil }
            do {
                try await validateArtifact(
                    localPath: assignment.localPath,
                    fileName: assignment.fileName,
                    expectedSizeBytes: assignment.sizeBytes,
                    expectedSHA256: assignment.expectedSHA256,
                    role: "chat"
                )
            } catch {
                return nil
            }
            guard assignmentRemainsOwned(assignment, slot: slot, generation: assignmentGeneration) else { return nil }
            let requiresRoleAdapter = requiresRoleAdapter(assignment: assignment)
            guard await AppLlamaService.shared.loadedChatPath == assignment.localPath else { return nil }
            guard assignmentRemainsOwned(assignment, slot: slot, generation: assignmentGeneration) else { return nil }
            let activeAdapterSlot = await AppLlamaService.shared.activeAdapterSlotValue
            guard assignmentRemainsOwned(assignment, slot: slot, generation: assignmentGeneration) else { return nil }
            if requiresRoleAdapter {
                guard let adapterPath = assignment.adapterPath,
                      FileManager.default.fileExists(atPath: adapterPath),
                      assignment.configuredRoleAdapterMatchesContract,
                      activeAdapterSlot == slot else {
                    return nil
                }
                do {
                    try await validateArtifact(
                        localPath: adapterPath,
                        fileName: assignment.adapterFileName ?? URL(fileURLWithPath: adapterPath).lastPathComponent,
                        expectedSizeBytes: assignment.adapterSizeBytes ?? 0,
                        expectedSHA256: assignment.adapterExpectedSHA256,
                        role: "roleAdapter"
                    )
                } catch {
                    return nil
                }
                guard assignmentRemainsOwned(assignment, slot: slot, generation: assignmentGeneration) else { return nil }
            } else if let adapterPath = assignment.adapterPath,
                      FileManager.default.fileExists(atPath: adapterPath),
                      activeAdapterSlot != slot {
                return nil
            }
            let elapsed = Int(Date().timeIntervalSince(started) * 1000)
            let accelerationDiagnostics = await AppLlamaService.shared.currentAccelerationDiagnostics()
            guard assignmentRemainsOwned(assignment, slot: slot, generation: assignmentGeneration) else { return nil }
            return RuntimeReadinessMetrics(
                ensureReadyMs: elapsed,
                adapterActivationMs: 0,
                runtimePath: "sharedAdapterLoadedContinuation",
                activeAdapterSlot: activeAdapterSlot?.rawValue,
                accelerationDiagnostic: "Continuing with already-loaded shared runtime after memory-pressure gate.",
                accelerationDiagnostics: accelerationDiagnostics
            )
        }

        let slotLoaded = await AppLlamaService.shared.loadedChatPath(for: slot) == assignment.localPath
        guard assignmentRemainsOwned(assignment, slot: slot, generation: assignmentGeneration) else { return nil }
        let aliasLoaded = await AppLlamaService.shared.slotLoaded(withPath: assignment.localPath) != nil
        guard assignmentRemainsOwned(assignment, slot: slot, generation: assignmentGeneration) else { return nil }
        guard slotLoaded || aliasLoaded else { return nil }
        let elapsed = Int(Date().timeIntervalSince(started) * 1000)
        let accelerationDiagnostics = await AppLlamaService.shared.currentAccelerationDiagnostics()
        guard assignmentRemainsOwned(assignment, slot: slot, generation: assignmentGeneration) else { return nil }
        return RuntimeReadinessMetrics(
            ensureReadyMs: elapsed,
            adapterActivationMs: 0,
            runtimePath: "legacySlotLoadedContinuation",
            activeAdapterSlot: nil,
            accelerationDiagnostic: "Continuing with already-loaded slot runtime after memory-pressure gate.",
            accelerationDiagnostics: accelerationDiagnostics
        )
    }

    private func resolvedAssignment(for slot: LumenModelSlot) -> LumenModelAssignment? {
        assignments[slot]
    }

    private func assignmentRemainsOwned(
        _ assignment: LumenModelAssignment,
        slot: LumenModelSlot,
        generation: UInt64
    ) -> Bool {
        configurationGeneration == generation && assignments[slot] == assignment
    }

    private func ensureAdapterRuntimeReady(
        slot: LumenModelSlot,
        assignment: LumenModelAssignment,
        generation: UInt64
    ) async throws -> Int {
        let activationStart = Date()
        let requiresRoleAdapter = requiresRoleAdapter(assignment: assignment)
        guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else {
            throw CancellationError()
        }
        guard assignment.configuredSharedBaseMatchesContract else {
            logger.error("shared_base_identity_mismatch slot=\(slot.rawValue, privacy: .public)")
            throw LocalRuntimeError.unavailable("shared chat base identity mismatch for \(slot.rawValue)")
        }
        do {
            try await validateArtifact(
                localPath: assignment.localPath,
                fileName: assignment.fileName,
                expectedSizeBytes: assignment.sizeBytes,
                expectedSHA256: assignment.expectedSHA256,
                role: "chat"
            )
        } catch {
            throw error
        }
        guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else {
            throw CancellationError()
        }
        let loadedChatPath = await AppLlamaService.shared.loadedChatPath
        guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else {
            throw CancellationError()
        }
        if loadedChatPath != assignment.localPath {
            do {
                try await AppLlamaService.shared.loadSharedChatModel(path: assignment.localPath, contextSize: contextSize)
            } catch {
                if error is CancellationError { throw error }
                logger.error("shared_base_load_failed slot=\(slot.rawValue, privacy: .public) path=\(assignment.localPath, privacy: .private) context=\(self.contextSize, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
                guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else {
                    throw CancellationError()
                }
                if contextSize > 2048 {
                    try await AppLlamaService.shared.loadSharedChatModel(path: assignment.localPath, contextSize: 2048)
                } else {
                    throw error
                }
            }
            guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else {
                throw CancellationError()
            }
        }
        try Task.checkCancellation()

        guard let adapterPath = assignment.adapterPath else {
            if requiresRoleAdapter {
                let expectedRepo = assignment.expectedRoleAdapterRepoID ?? "unknown"
                let expectedFile = assignment.expectedRoleAdapterFileName ?? "unknown"
                throw LocalRuntimeError.unavailable("role adapter missing for \(slot.rawValue): expectedAdapterRepo=\(expectedRepo); expectedAdapterFile=\(expectedFile)")
            }
            return Int(Date().timeIntervalSince(activationStart) * 1000)
        }
        guard FileManager.default.fileExists(atPath: adapterPath) else {
            logger.error("role_adapter_missing slot=\(slot.rawValue, privacy: .public) path=\(adapterPath, privacy: .private)")
            throw LlamaError.modelFileNotFound(adapterPath)
        }
        if assignment.expectedRoleAdapterContract != nil,
           !assignment.configuredRoleAdapterMatchesContract {
            logger.error("role_adapter_identity_mismatch slot=\(slot.rawValue, privacy: .public)")
            throw LocalRuntimeError.unavailable("role adapter identity mismatch for \(slot.rawValue)")
        }
        try await validateArtifact(
            localPath: adapterPath,
            fileName: assignment.adapterFileName ?? URL(fileURLWithPath: adapterPath).lastPathComponent,
            expectedSizeBytes: assignment.adapterSizeBytes ?? 0,
            expectedSHA256: assignment.adapterExpectedSHA256,
            role: "roleAdapter"
        )
        guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else {
            throw CancellationError()
        }

        do {
            let loadedNow = try await AppLlamaService.shared.loadRoleAdapterIfNeeded(slot: slot, path: adapterPath, scale: assignment.adapterScale)
            guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else {
                throw CancellationError()
            }
            let activatedNow = try await AppLlamaService.shared.activateRoleAdapterIfNeeded(slot: slot)
            guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else {
                throw CancellationError()
            }
            if loadedNow || activatedNow {
                logger.info("role_adapter_activation_performed slot=\(slot.rawValue, privacy: .public) path=\(adapterPath, privacy: .private)")
            }
        } catch {
            logger.error("role_adapter_activation_failed slot=\(slot.rawValue, privacy: .public) path=\(adapterPath, privacy: .private) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else {
                throw CancellationError()
            }
            await AppLlamaService.shared.unloadRoleAdapter(slot: slot, ifPathEquals: adapterPath)
            guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else {
                throw CancellationError()
            }
            throw error
        }
        return Int(Date().timeIntervalSince(activationStart) * 1000)
    }

    private func requiresRoleAdapter(assignment: LumenModelAssignment) -> Bool {
        assignment.requiresRoleAdapterForRuntime
    }

    private func ensureLegacyRuntimeReady(
        slot: LumenModelSlot,
        assignment: LumenModelAssignment,
        generation: UInt64
    ) async throws {
        guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else { throw CancellationError() }
        let loadedPath = await AppLlamaService.shared.loadedChatPath(for: slot)
        guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else { throw CancellationError() }
        if loadedPath == assignment.localPath {
            return
        }
        if let loadedSlot = await AppLlamaService.shared.slotLoaded(withPath: assignment.localPath) {
            guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else { throw CancellationError() }
            await AppLlamaService.shared.aliasChatRuntime(from: loadedSlot, to: slot)
            guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else { throw CancellationError() }
            return
        }
        guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else { throw CancellationError() }

        try await validateArtifact(
            localPath: assignment.localPath,
            fileName: assignment.fileName,
            expectedSizeBytes: assignment.sizeBytes,
            expectedSHA256: assignment.expectedSHA256,
            role: "chat"
        )
        guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else { throw CancellationError() }

        if preferExclusiveChatRuntime {
            await AppLlamaService.shared.unloadAllChat()
        } else {
            await AppLlamaService.shared.unloadChat(for: slot)
        }
        guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else { throw CancellationError() }

        do {
            try await AppLlamaService.shared.loadChatModel(
                path: assignment.localPath,
                for: slot,
                contextSize: contextSize
            )
            guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else { throw CancellationError() }
        } catch {
            if error is CancellationError { throw error }
            logger.error("slot_model_load_failed slot=\(slot.rawValue, privacy: .public) path=\(assignment.localPath, privacy: .private) context=\(self.contextSize, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
            guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else { throw CancellationError() }
            if contextSize > 2048 {
                await AppLlamaService.shared.unloadAllChat()
                guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else { throw CancellationError() }
                try await AppLlamaService.shared.loadChatModel(
                    path: assignment.localPath,
                    for: slot,
                    contextSize: 2048
                )
                guard assignmentRemainsOwned(assignment, slot: slot, generation: generation) else { throw CancellationError() }
            } else {
                throw error
            }
        }
        try Task.checkCancellation()
    }

    func ensurePrimaryReady(preferredSlots: [LumenModelSlot] = [.mouth, .cortex]) async -> Bool {
        for slot in preferredSlots {
            guard resolvedAssignment(for: slot) != nil else { continue }
            do {
                try await ensureReady(slot: slot)
                guard !Task.isCancelled else {
                    return false
                }
                return true
            } catch {
                if error is CancellationError {
                    return false
                }
                logger.error("primary_slot_ready_failed slot=\(slot.rawValue, privacy: .public) error_code=\(RuntimeMetricErrorSanitizer.code(for: error), privacy: .public)")
                continue
            }
        }
        return false
    }

    private func orderedCandidates(candidates: [StoredModelLoadItem], preferredID: String?) async -> [StoredModelLoadItem] {
        var pool: [StoredModelLoadItem] = []
        for candidate in candidates {
            switch await ModelFileIntegrity.validateInstalledFileWithDiagnosticsAsync(candidate) {
            case .success:
                pool.append(candidate)
            case .failure(let failure):
                logger.error("transition event=skip_invalid_artifact role=\(candidate.role, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) diagnostic=\(failure.diagnosticCode, privacy: .public)")
            }
        }
        var ordered: [StoredModelLoadItem] = []
        if let preferredID, let preferred = pool.first(where: { $0.id.uuidString == preferredID }) {
            ordered.append(preferred)
        }
        for candidate in pool where !ordered.contains(where: { $0.id == candidate.id }) {
            ordered.append(candidate)
        }
        return ordered
    }

    private func validateArtifact(
        localPath: String,
        fileName: String,
        expectedSizeBytes: Int64,
        expectedSHA256: String?,
        role: String
    ) async throws {
        switch await ModelFileIntegrity.validateInstalledFileWithDiagnosticsAsync(
            localPath: localPath,
            fileName: fileName,
            expectedSizeBytes: expectedSizeBytes,
            expectedSHA256: expectedSHA256
        ) {
        case .success:
            return
        case .failure(.cancelled):
            throw CancellationError()
        case .failure(let failure):
            logger.error("transition event=integrity_failed role=\(role, privacy: .public) diagnostic=\(failure.diagnosticCode, privacy: .public)")
            throw LocalRuntimeError.unavailable("model integrity validation failed: \(failure.diagnosticCode)")
        }
    }

    private func orderedCandidates(candidates: [StoredModel], preferredID: String?) -> [StoredModel] {
        var pool: [StoredModel] = []
        for candidate in candidates {
            switch ModelFileIntegrity.validateInstalledFileWithDiagnostics(candidate) {
            case .success:
                pool.append(candidate)
            case .failure(let failure):
                logger.error("transition event=skip_invalid_artifact role=embedding model_id=\(candidate.id.uuidString, privacy: .public) diagnostic=\(failure.diagnosticCode, privacy: .public)")
            }
        }
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
