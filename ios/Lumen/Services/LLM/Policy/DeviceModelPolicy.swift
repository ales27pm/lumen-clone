import Foundation

actor DeviceModelPolicy {
    private let provider: any DeviceCapabilityProviding

    init(provider: any DeviceCapabilityProviding = SystemDeviceCapabilityProvider()) {
        self.provider = provider
    }

    func evaluate(
        model: LocalLLMModel,
        requestedProfile: InferenceProfile,
        requestedBudget: InferenceBudget,
        appIsForeground: Bool
    ) async -> ModelFitDecision {
        let snapshot = await provider.currentSnapshot(appIsForeground: appIsForeground)
        var selectedProfile = requestedProfile
        var selectedBudget = requestedBudget
        var reasons: [ModelFitReason] = []

        switch model.backend {
        case .tinyIntent:
            add(.tinyIntentAlwaysAllowed, to: &reasons)
            let estimate = ModelMemoryEstimator.estimate(model: model, profile: selectedProfile, budget: selectedBudget)
            return .allowed(report(
                model: model,
                snapshot: snapshot,
                requestedProfile: requestedProfile,
                selectedProfile: selectedProfile,
                requestedBudget: requestedBudget,
                selectedBudget: selectedBudget,
                memoryEstimate: estimate,
                reasons: reasons
            ))
        case .mock:
            add(.modelFitsMemoryBudget, to: &reasons)
            let estimate = ModelMemoryEstimator.estimate(model: model, profile: selectedProfile, budget: selectedBudget)
            return .allowed(report(
                model: model,
                snapshot: snapshot,
                requestedProfile: requestedProfile,
                selectedProfile: selectedProfile,
                requestedBudget: requestedBudget,
                selectedBudget: selectedBudget,
                memoryEstimate: estimate,
                reasons: reasons
            ))
        case .remote:
            add(.remoteDoesNotNeedLocalModelMemory, to: &reasons)
            let estimate = ModelMemoryEstimator.estimate(model: model, profile: selectedProfile, budget: selectedBudget)
            return .allowed(report(
                model: model,
                snapshot: snapshot,
                requestedProfile: requestedProfile,
                selectedProfile: selectedProfile,
                requestedBudget: requestedBudget,
                selectedBudget: selectedBudget,
                memoryEstimate: estimate,
                reasons: reasons
            ))
        case .gguf, .coreML:
            break
        }

        if snapshot.isSimulator {
            applySimulatorDowngrade(profile: &selectedProfile, budget: &selectedBudget, reasons: &reasons)
        }

        if !snapshot.hasMetalSupport {
            applyMetalUnavailableDowngrade(profile: &selectedProfile, budget: &selectedBudget, reasons: &reasons)
        }

        if !appIsForeground && !selectedBudget.allowBackgroundExecution && requiresHeavyGPU(profile: selectedProfile, budget: selectedBudget) {
            add(.backgroundExecutionBlocked, to: &reasons)
            let estimate = ModelMemoryEstimator.estimate(model: model, profile: selectedProfile, budget: selectedBudget)
            return .rejected(report(
                model: model,
                snapshot: snapshot,
                requestedProfile: requestedProfile,
                selectedProfile: selectedProfile,
                requestedBudget: requestedBudget,
                selectedBudget: selectedBudget,
                memoryEstimate: estimate,
                reasons: reasons
            ))
        }

        if snapshot.powerState.thermalPressure == .critical {
            add(.thermalPressureTooHigh, to: &reasons)
            let estimate = ModelMemoryEstimator.estimate(model: model, profile: selectedProfile, budget: selectedBudget)
            return .rejected(report(
                model: model,
                snapshot: snapshot,
                requestedProfile: requestedProfile,
                selectedProfile: selectedProfile,
                requestedBudget: requestedBudget,
                selectedBudget: selectedBudget,
                memoryEstimate: estimate,
                reasons: reasons
            ))
        }

        if snapshot.powerState.thermalPressure == .serious {
            applySeriousThermalDowngrade(profile: &selectedProfile, budget: &selectedBudget, reasons: &reasons)
        }

        if snapshot.powerState.isLowPowerModeEnabled {
            applyLowPowerDowngrade(profile: &selectedProfile, budget: &selectedBudget, reasons: &reasons)
        }

        if let parameters = model.parameterCountBillion, parameters > snapshot.performanceTier.defaultMaximumModelParametersBillion {
            add(.modelExceedsDeviceTier, to: &reasons)
            let tierLimit = snapshot.performanceTier.defaultMaximumModelParametersBillion
            if parameters > tierLimit * 1.25 {
                let estimate = ModelMemoryEstimator.estimate(model: model, profile: selectedProfile, budget: selectedBudget)
                return .rejected(report(
                    model: model,
                    snapshot: snapshot,
                    requestedProfile: requestedProfile,
                    selectedProfile: selectedProfile,
                    requestedBudget: requestedBudget,
                    selectedBudget: selectedBudget,
                    memoryEstimate: estimate,
                    reasons: reasons
                ))
            }
            applyTierOverageDowngrade(profile: &selectedProfile, budget: &selectedBudget, reasons: &reasons)
        }

        var estimate = ModelMemoryEstimator.estimate(model: model, profile: selectedProfile, budget: selectedBudget)
        if estimate.confidence == .low {
            add(.unknownModelSize, to: &reasons)
        }

        let memoryCeiling = effectiveMemoryCeiling(snapshot: snapshot, budget: selectedBudget)
        if estimate.estimatedTotalMB > memoryCeiling {
            add(.modelExceedsMemoryBudget, to: &reasons)
            if estimate.estimatedTotalMB > Int(Double(memoryCeiling) * 1.25) {
                return .rejected(report(
                    model: model,
                    snapshot: snapshot,
                    requestedProfile: requestedProfile,
                    selectedProfile: selectedProfile,
                    requestedBudget: requestedBudget,
                    selectedBudget: selectedBudget,
                    memoryEstimate: estimate,
                    reasons: reasons
                ))
            }

            applyMemoryDowngrade(profile: &selectedProfile, budget: &selectedBudget, reasons: &reasons)
            estimate = ModelMemoryEstimator.estimate(model: model, profile: selectedProfile, budget: selectedBudget)

            if estimate.estimatedTotalMB > effectiveMemoryCeiling(snapshot: snapshot, budget: selectedBudget) {
                return .rejected(report(
                    model: model,
                    snapshot: snapshot,
                    requestedProfile: requestedProfile,
                    selectedProfile: selectedProfile,
                    requestedBudget: requestedBudget,
                    selectedBudget: selectedBudget,
                    memoryEstimate: estimate,
                    reasons: reasons
                ))
            }
        } else {
            add(.modelFitsMemoryBudget, to: &reasons)
        }

        let finalReport = report(
            model: model,
            snapshot: snapshot,
            requestedProfile: requestedProfile,
            selectedProfile: selectedProfile,
            requestedBudget: requestedBudget,
            selectedBudget: selectedBudget,
            memoryEstimate: estimate,
            reasons: reasons
        )

        if selectedProfile != requestedProfile || selectedBudget != requestedBudget || reasons.containsDowngradeReason {
            return .downgraded(finalReport)
        }
        return .allowed(finalReport)
    }

    private func applySimulatorDowngrade(
        profile: inout InferenceProfile,
        budget: inout InferenceBudget,
        reasons: inout [ModelFitReason]
    ) {
        let newContext = min(profile.contextTokens, 2_048)
        let newPrompt = min(budget.maxPromptTokens, 2_048)
        let newCompletion = min(budget.maxCompletionTokens, 512)

        if profile.useMetal || profile.gpuLayerCount > 0 || budget.allowGPU {
            add(.simulatorDisablesMetal, to: &reasons)
            add(.gpuDisabled, to: &reasons)
        }
        if newContext < profile.contextTokens || newPrompt < budget.maxPromptTokens {
            add(.contextReduced, to: &reasons)
        }
        if newCompletion < budget.maxCompletionTokens {
            add(.completionBudgetReduced, to: &reasons)
        }

        profile = profile.adjusted(
            contextTokens: newContext,
            batchSize: min(profile.batchSize, 128),
            threadCount: min(profile.threadCount, 2),
            gpuLayerCount: 0,
            useMetal: false,
            lowPowerMode: true
        )
        budget = budget.adjusted(
            maxPromptTokens: newPrompt,
            maxCompletionTokens: newCompletion,
            allowGPU: false,
            allowBackgroundExecution: false
        )
    }

    private func applyMetalUnavailableDowngrade(
        profile: inout InferenceProfile,
        budget: inout InferenceBudget,
        reasons: inout [ModelFitReason]
    ) {
        guard profile.useMetal || profile.gpuLayerCount > 0 || budget.allowGPU else { return }
        add(.metalUnavailable, to: &reasons)
        add(.gpuDisabled, to: &reasons)
        profile = profile.adjusted(gpuLayerCount: 0, useMetal: false)
        budget = budget.adjusted(allowGPU: false)
    }

    private func applySeriousThermalDowngrade(
        profile: inout InferenceProfile,
        budget: inout InferenceBudget,
        reasons: inout [ModelFitReason]
    ) {
        add(.thermalPressureTooHigh, to: &reasons)
        let newContext = min(profile.contextTokens, 2_048)
        let newPrompt = min(budget.maxPromptTokens, 2_048)
        let newCompletion = min(budget.maxCompletionTokens, 512)
        if newContext < profile.contextTokens || newPrompt < budget.maxPromptTokens {
            add(.contextReduced, to: &reasons)
        }
        if newCompletion < budget.maxCompletionTokens {
            add(.completionBudgetReduced, to: &reasons)
        }
        profile = profile.adjusted(
            contextTokens: newContext,
            batchSize: min(profile.batchSize, 128),
            threadCount: min(profile.threadCount, 4),
            lowPowerMode: true
        )
        budget = budget.adjusted(maxPromptTokens: newPrompt, maxCompletionTokens: newCompletion)
    }

    private func applyLowPowerDowngrade(
        profile: inout InferenceProfile,
        budget: inout InferenceBudget,
        reasons: inout [ModelFitReason]
    ) {
        add(.lowPowerModeDowngrade, to: &reasons)
        let newContext = min(profile.contextTokens, 4_096)
        let newPrompt = min(budget.maxPromptTokens, 4_096)
        let newCompletion = min(budget.maxCompletionTokens, 512)
        if newContext < profile.contextTokens || newPrompt < budget.maxPromptTokens {
            add(.contextReduced, to: &reasons)
        }
        if newCompletion < budget.maxCompletionTokens {
            add(.completionBudgetReduced, to: &reasons)
        }
        profile = profile.adjusted(
            contextTokens: newContext,
            batchSize: min(profile.batchSize, 256),
            threadCount: min(profile.threadCount, 4),
            lowPowerMode: true
        )
        budget = budget.adjusted(maxPromptTokens: newPrompt, maxCompletionTokens: newCompletion)
    }

    private func applyTierOverageDowngrade(
        profile: inout InferenceProfile,
        budget: inout InferenceBudget,
        reasons: inout [ModelFitReason]
    ) {
        let newContext = min(profile.contextTokens, 4_096)
        let newCompletion = min(budget.maxCompletionTokens, 768)
        if newContext < profile.contextTokens {
            add(.contextReduced, to: &reasons)
        }
        if newCompletion < budget.maxCompletionTokens {
            add(.completionBudgetReduced, to: &reasons)
        }
        profile = profile.adjusted(contextTokens: newContext, batchSize: min(profile.batchSize, 256))
        budget = budget.adjusted(maxPromptTokens: min(budget.maxPromptTokens, 4_096), maxCompletionTokens: newCompletion)
    }

    private func applyMemoryDowngrade(
        profile: inout InferenceProfile,
        budget: inout InferenceBudget,
        reasons: inout [ModelFitReason]
    ) {
        let newContext = min(profile.contextTokens, 2_048)
        let newCompletion = min(budget.maxCompletionTokens, 512)
        if newContext < profile.contextTokens || budget.maxPromptTokens > 2_048 {
            add(.contextReduced, to: &reasons)
        }
        if newCompletion < budget.maxCompletionTokens {
            add(.completionBudgetReduced, to: &reasons)
        }
        if profile.useMetal || profile.gpuLayerCount > 0 || budget.allowGPU {
            add(.gpuDisabled, to: &reasons)
        }
        profile = profile.adjusted(
            contextTokens: newContext,
            batchSize: min(profile.batchSize, 128),
            gpuLayerCount: 0,
            useMetal: false,
            lowPowerMode: true
        )
        budget = budget.adjusted(
            maxPromptTokens: min(budget.maxPromptTokens, 2_048),
            maxCompletionTokens: newCompletion,
            allowGPU: false
        )
    }

    private func requiresHeavyGPU(profile: InferenceProfile, budget: InferenceBudget) -> Bool {
        profile.useMetal || profile.gpuLayerCount > 0 || budget.allowGPU
    }

    private func effectiveMemoryCeiling(snapshot: DeviceCapabilitySnapshot, budget: InferenceBudget) -> Int {
        min(snapshot.recommendedLLMMemoryCeilingMB, budget.maxMemoryMB ?? snapshot.recommendedLLMMemoryCeilingMB)
    }

    private func report(
        model: LocalLLMModel,
        snapshot: DeviceCapabilitySnapshot,
        requestedProfile: InferenceProfile,
        selectedProfile: InferenceProfile,
        requestedBudget: InferenceBudget,
        selectedBudget: InferenceBudget,
        memoryEstimate: ModelMemoryEstimate,
        reasons: [ModelFitReason]
    ) -> ModelFitReport {
        ModelFitReport(
            model: model,
            snapshot: snapshot,
            requestedProfile: requestedProfile,
            selectedProfile: selectedProfile,
            requestedBudget: requestedBudget,
            selectedBudget: selectedBudget,
            memoryEstimate: memoryEstimate,
            reasons: reasons
        )
    }

    private func add(_ reason: ModelFitReason, to reasons: inout [ModelFitReason]) {
        guard !reasons.contains(reason) else { return }
        reasons.append(reason)
    }
}

private extension Array where Element == ModelFitReason {
    var containsDowngradeReason: Bool {
        contains(.simulatorDisablesMetal)
            || contains(.metalUnavailable)
            || contains(.thermalPressureTooHigh)
            || contains(.lowPowerModeDowngrade)
            || contains(.contextReduced)
            || contains(.completionBudgetReduced)
            || contains(.gpuDisabled)
            || contains(.modelExceedsDeviceTier)
            || contains(.modelExceedsMemoryBudget)
    }
}
