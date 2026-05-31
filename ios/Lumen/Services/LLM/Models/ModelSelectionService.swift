import Foundation

actor ModelSelectionService {
    private let storage: LLMModelStorage
    private let policy: DeviceModelPolicy

    init(
        storage: LLMModelStorage,
        policy: DeviceModelPolicy
    ) {
        self.storage = storage
        self.policy = policy
    }

    func installedUsableModels(appIsForeground: Bool) async throws -> [InstalledModelRecord] {
        let records = try await storage.listInstalledModels().filter(\.isUsable)
        var usableRecords: [InstalledModelRecord] = []

        for record in records {
            let decision = await evaluateInstalledModel(record, appIsForeground: appIsForeground)
            if decision.isAllowed {
                usableRecords.append(record)
            }
        }

        return usableRecords
    }

    func bestModel(
        for recommendedUse: ModelRecommendedUse,
        appIsForeground: Bool
    ) async throws -> InstalledModelRecord? {
        let records = try await storage.listInstalledModels().filter(\.isUsable)
        var candidates: [(record: InstalledModelRecord, use: ModelRecommendedUse)] = []

        for record in records {
            let decision = await evaluateInstalledModel(record, appIsForeground: appIsForeground)
            guard decision.isAllowed else { continue }
            candidates.append((record, useForRecord(record)))
        }

        return candidates.sorted { lhs, rhs in
            let lhsRank = selectionRank(record: lhs.record, use: lhs.use, requestedUse: recommendedUse)
            let rhsRank = selectionRank(record: rhs.record, use: rhs.use, requestedUse: recommendedUse)
            if lhsRank != rhsRank {
                return lhsRank > rhsRank
            }

            let lhsParameters = lhs.record.model.parameterCountBillion ?? 0
            let rhsParameters = rhs.record.model.parameterCountBillion ?? 0
            if lhsParameters != rhsParameters {
                return lhsParameters > rhsParameters
            }

            return lhs.record.id < rhs.record.id
        }.first?.record
    }

    func evaluateInstalledModel(
        _ record: InstalledModelRecord,
        appIsForeground: Bool
    ) async -> ModelFitDecision {
        let use = useForRecord(record)
        let request = requestConfiguration(for: use, backend: record.model.backend)

        return await policy.evaluate(
            model: record.model,
            requestedProfile: request.profile,
            requestedBudget: request.budget,
            appIsForeground: appIsForeground
        )
    }

    private func useForRecord(_ record: InstalledModelRecord) -> ModelRecommendedUse {
        if let catalogID = record.catalogID, let entry = BuiltInModelCatalog.entry(id: catalogID) {
            return entry.recommendedUse
        }

        switch record.model.backend {
        case .tinyIntent:
            return .tinyIntent
        case .mock:
            return .testing
        case .remote, .gguf, .coreML:
            return .standardChat
        }
    }

    private func requestConfiguration(
        for use: ModelRecommendedUse,
        backend: LLMBackendKind
    ) -> (profile: InferenceProfile, budget: InferenceBudget) {
        if backend == .tinyIntent {
            return (.simulatorSafe, .fast)
        }

        switch use {
        case .tinyIntent:
            return (.simulatorSafe, .fast)
        case .fastChat:
            return (.iphoneBalanced, .fast)
        case .standardChat:
            return (.iphoneBalanced, .standard)
        case .deepThink:
            return (.iphoneDeepThink, .deepThink)
        case .embedding:
            return (
                .iphoneBalanced,
                InferenceBudget(
                    maxPromptTokens: 2_048,
                    maxCompletionTokens: 1,
                    maxWallClockSeconds: 30,
                    maxMemoryMB: 1_536,
                    allowGPU: true,
                    allowBackgroundExecution: false
                )
            )
        case .reranking, .vision, .testing:
            return (.simulatorSafe, .fast)
        }
    }

    private func selectionRank(
        record: InstalledModelRecord,
        use: ModelRecommendedUse,
        requestedUse: ModelRecommendedUse
    ) -> Int {
        if use == requestedUse {
            return 2
        }
        if record.model.backend == .tinyIntent {
            return 1
        }
        return 0
    }
}
