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

@MainActor
final class SlotModelRuntimeCoordinator {
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
        appState: AppState,
        candidates: [StoredModel],
        preferredID: String?
    ) async -> Bool {
        let orderedCandidates = orderedCandidates(candidates: candidates, preferredID: preferredID)
        for (index, candidate) in orderedCandidates.enumerated() {
            let path = ModelStorage.resolvedModelURL(from: candidate.localPath, fileName: candidate.fileName).path
            logger.info("transition event=attempt role=chat index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public)")
            guard FileManager.default.fileExists(atPath: path) else {
                logger.info("transition event=skip_missing role=chat index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public)")
                continue
            }

            do {
                try await AppLlamaService.shared.unloadAllChat()
                try await AppLlamaService.shared.loadChatModel(path: path, contextSize: contextSize)
                appState.activeChatModelID = candidate.id.uuidString
                logger.info("transition event=\(self.selectionEvent(index: index, candidateID: candidate.id.uuidString, preferredID: preferredID), privacy: .public) role=chat model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public)")
                return true
            } catch {
                if isContextInitFailed(error) {
                    do {
                        logger.info("transition event=retry_context_2048 role=chat model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public)")
                        try await AppLlamaService.shared.unloadAllChat()
                        try await AppLlamaService.shared.loadChatModel(path: path, contextSize: 2048)
                        appState.activeChatModelID = candidate.id.uuidString
                        logger.info("transition event=\(self.selectionEvent(index: index, candidateID: candidate.id.uuidString, preferredID: preferredID), privacy: .public) role=chat model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public) context=2048")
                        return true
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
        return false
    }

    @discardableResult
    func ensureEmbeddingModel(
        appState: AppState,
        candidates: [StoredModel],
        preferredID: String?
    ) async -> Bool {
        let orderedCandidates = orderedCandidates(candidates: candidates, preferredID: preferredID)
        for (index, candidate) in orderedCandidates.enumerated() {
            let path = ModelStorage.resolvedModelURL(from: candidate.localPath, fileName: candidate.fileName).path
            logger.info("transition event=attempt role=embedding index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public)")
            guard FileManager.default.fileExists(atPath: path) else {
                logger.info("transition event=skip_missing role=embedding index=\(index, privacy: .public) model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public)")
                continue
            }

            do {
                try await AppLlamaService.shared.loadEmbeddingModel(path: path)
                appState.activeEmbeddingModelID = candidate.id.uuidString
                logger.info("transition event=\(self.selectionEvent(index: index, candidateID: candidate.id.uuidString, preferredID: preferredID), privacy: .public) role=embedding model_id=\(candidate.id.uuidString, privacy: .public) path=\(path, privacy: .public)")
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

    func ensureReadyWithMetrics(slot: LumenModelSlot) async throws -> RuntimeReadinessMetrics {
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
        let started = Date()

        let assignment = resolvedAssignment(for: slot)
        guard let assignment else {
            throw LlamaError.slotModelNotLoaded("\(slot.rawValue): no assigned model")
        }
        guard FileManager.default.fileExists(atPath: assignment.localPath) else {
            throw LlamaError.modelFileNotFound(assignment.localPath)
        }

        if assignment.usesRoleAdapter || assignment.modelFamily == .qwen3 {
            let activationMs = try await ensureAdapterRuntimeReady(slot: slot, assignment: assignment)
            let elapsed = Int(Date().timeIntervalSince(started) * 1000)
            return RuntimeReadinessMetrics(
                ensureReadyMs: elapsed,
                adapterActivationMs: activationMs,
                runtimePath: "sharedAdapter",
                activeAdapterSlot: slot.rawValue,
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
            return Int(Date().timeIntervalSince(activationStart) * 1000)
        }
        guard FileManager.default.fileExists(atPath: adapterPath) else {
            logger.error("role_adapter_missing slot=\(slot.rawValue, privacy: .public) path=\(adapterPath, privacy: .public)")
            await AppLlamaService.shared.clearActiveRoleAdapter()
            return Int(Date().timeIntervalSince(activationStart) * 1000)
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
        }
        return Int(Date().timeIntervalSince(activationStart) * 1000)
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

    func selectionEvent(index: Int, candidateID: String, preferredID: String?) -> String {
        if index > 0 { return "fallback_selected" }
        if let preferredID, candidateID != preferredID { return "fallback_selected" }
        return "selected"
    }
}
