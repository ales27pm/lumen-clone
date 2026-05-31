import Foundation
import Testing
@testable import Lumen

struct LLMDevicePolicyTests {
    @Test func tinyIntentIsAllowedUnderCriticalThermalAndBackground() async {
        let provider = TestDeviceCapabilityProvider(
            performanceTier: .constrained,
            physicalMemoryBytes: 2 * Self.gibibyte,
            hasMetalSupport: false,
            powerState: RuntimePowerState(
                isLowPowerModeEnabled: true,
                thermalPressure: .critical,
                appIsForeground: false
            )
        )
        let policy = DeviceModelPolicy(provider: provider)

        let decision = await policy.evaluate(
            model: tinyIntentModel(),
            requestedProfile: InferenceProfile.iphoneDeepThink,
            requestedBudget: InferenceBudget.maximumForeground,
            appIsForeground: false
        )

        #expect(decision.isAllowed)
        #expect(decision.isRejected == false)
        #expect(decision.report.reasons.contains(ModelFitReason.tinyIntentAlwaysAllowed))
    }

    @Test func simulatorDisablesMetalAndDowngradesGGUFProfile() async {
        let provider = TestDeviceCapabilityProvider(
            formFactor: .simulator,
            performanceTier: .simulator,
            physicalMemoryBytes: 16 * Self.gibibyte,
            hasMetalSupport: true,
            isSimulator: true
        )
        let policy = DeviceModelPolicy(provider: provider)

        let decision = await policy.evaluate(
            model: ggufModel(id: "gguf.750m.q4.simulator", parameters: 0.75, quantization: "Q4_K_M"),
            requestedProfile: InferenceProfile.ipadProMaximum,
            requestedBudget: InferenceBudget.maximumForeground,
            appIsForeground: true
        )

        #expect(decision.isAllowed)
        #expect(decision.report.selectedProfile.useMetal == false)
        #expect(decision.report.selectedProfile.gpuLayerCount == 0)
        #expect(decision.report.selectedProfile.contextTokens <= 2_048)
        #expect(decision.report.selectedBudget.allowGPU == false)
        #expect(decision.report.reasons.contains(ModelFitReason.simulatorDisablesMetal))
    }

    @Test func criticalThermalRejectsHeavyGGUFInference() async {
        let provider = TestDeviceCapabilityProvider(
            performanceTier: .high,
            physicalMemoryBytes: 8 * Self.gibibyte,
            powerState: RuntimePowerState(
                isLowPowerModeEnabled: false,
                thermalPressure: .critical,
                appIsForeground: true
            )
        )
        let policy = DeviceModelPolicy(provider: provider)

        let decision = await policy.evaluate(
            model: ggufModel(id: "gguf.3b.q4", parameters: 3, quantization: "Q4_K_M"),
            requestedProfile: InferenceProfile.iphoneBalanced,
            requestedBudget: InferenceBudget.standard,
            appIsForeground: true
        )

        #expect(decision.isRejected)
        #expect(decision.report.reasons.contains(ModelFitReason.thermalPressureTooHigh))
    }

    @Test func lowPowerModeDowngradesContextAndCompletionBudget() async {
        let provider = TestDeviceCapabilityProvider(
            performanceTier: .high,
            physicalMemoryBytes: 8 * Self.gibibyte,
            powerState: RuntimePowerState(
                isLowPowerModeEnabled: true,
                thermalPressure: .nominal,
                appIsForeground: true
            )
        )
        let policy = DeviceModelPolicy(provider: provider)

        let decision = await policy.evaluate(
            model: ggufModel(id: "gguf.3b.q4", parameters: 3, quantization: "Q4_K_M"),
            requestedProfile: InferenceProfile.iphoneDeepThink,
            requestedBudget: InferenceBudget.deepThink,
            appIsForeground: true
        )

        #expect(decision.isAllowed)
        #expect(decision.report.selectedProfile.contextTokens <= 4_096)
        #expect(decision.report.selectedProfile.lowPowerMode)
        #expect(decision.report.selectedBudget.maxCompletionTokens <= 512)
        #expect(decision.report.reasons.contains(ModelFitReason.lowPowerModeDowngrade))
    }

    @Test func memoryEstimateUsesFileSizeWhenAvailable() {
        let model = ggufModel(
            id: "gguf.file-size",
            parameters: nil,
            quantization: nil,
            fileSizeBytes: Int64(1_500 * Self.mebibyte)
        )

        let estimate = ModelMemoryEstimator.estimate(
            model: model,
            profile: InferenceProfile.simulatorSafe,
            budget: InferenceBudget.fast
        )

        #expect(estimate.estimatedModelMemoryMB == 1_500)
        #expect(estimate.confidence == EstimateConfidence.high)
        #expect(estimate.notes.contains { $0.contains("file size") })
    }

    @Test func memoryEstimateUsesParameterCountAndQuantization() {
        let estimate = ModelMemoryEstimator.estimate(
            model: ggufModel(id: "gguf.3b.q4.estimate", parameters: 3, quantization: "Q4_K_M"),
            profile: InferenceProfile.simulatorSafe,
            budget: InferenceBudget.fast
        )

        #expect((1_600...1_620).contains(estimate.estimatedModelMemoryMB))
        #expect(estimate.confidence == EstimateConfidence.medium)
        #expect(estimate.notes.contains { $0.contains("parameter count") })
    }

    @Test func q4KMQuantizationIsTreatedNearFourPointFiveBits() {
        let estimate = ModelMemoryEstimator.estimate(
            model: ggufModel(id: "gguf.1b.q4km", parameters: 1, quantization: "Q4_K_M"),
            profile: InferenceProfile.simulatorSafe,
            budget: InferenceBudget.fast
        )

        #expect((530...545).contains(estimate.estimatedModelMemoryMB))
    }

    @Test func kvCacheEstimateUsesTightestContextLimit() {
        let model = ggufModel(id: "gguf.2b.q4.context", parameters: 2, quantization: "Q4_K_M")
        let profile = InferenceProfile(
            name: "Estimator Context Test",
            contextTokens: 4_096,
            batchSize: 128,
            threadCount: 2,
            gpuLayerCount: 0,
            useMetal: false,
            useMemoryMapping: true,
            lowPowerMode: false
        )
        let largeContextBudget = InferenceBudget(
            maxPromptTokens: 4_096,
            maxCompletionTokens: 256,
            maxWallClockSeconds: 30,
            allowGPU: false,
            allowBackgroundExecution: false
        )
        let tightContextBudget = InferenceBudget(
            maxPromptTokens: 1_024,
            maxCompletionTokens: 256,
            maxWallClockSeconds: 30,
            allowGPU: false,
            allowBackgroundExecution: false
        )

        let largeContextEstimate = ModelMemoryEstimator.estimate(
            model: model,
            profile: profile,
            budget: largeContextBudget
        )
        let tightContextEstimate = ModelMemoryEstimator.estimate(
            model: model,
            profile: profile,
            budget: tightContextBudget
        )

        #expect(tightContextEstimate.estimatedKVCacheMB < largeContextEstimate.estimatedKVCacheMB)
        #expect(tightContextEstimate.estimatedKVCacheMB == 128)
    }

    @Test func oversizedSevenBModelIsRejectedOnConstrainedTier() async {
        let provider = TestDeviceCapabilityProvider(
            performanceTier: .constrained,
            physicalMemoryBytes: 4 * Self.gibibyte,
            processorCount: 4,
            activeProcessorCount: 4
        )
        let policy = DeviceModelPolicy(provider: provider)

        let decision = await policy.evaluate(
            model: ggufModel(id: "gguf.7b.q4", parameters: 7, quantization: "Q4_K_M"),
            requestedProfile: InferenceProfile.iphoneBalanced,
            requestedBudget: InferenceBudget.standard,
            appIsForeground: true
        )

        #expect(decision.isRejected)
        #expect(decision.report.reasons.contains(ModelFitReason.modelExceedsDeviceTier))
    }

    @Test func threeBQ4ModelIsAllowedOnBalancedTierWhenMemoryFits() async {
        let provider = TestDeviceCapabilityProvider(
            performanceTier: .balanced,
            physicalMemoryBytes: 6 * Self.gibibyte,
            processorCount: 6,
            activeProcessorCount: 6
        )
        let policy = DeviceModelPolicy(provider: provider)

        let decision = await policy.evaluate(
            model: ggufModel(id: "gguf.3b.q4.balanced", parameters: 3, quantization: "Q4_K_M"),
            requestedProfile: InferenceProfile.iphoneBalanced,
            requestedBudget: InferenceBudget.maximumForeground,
            appIsForeground: true
        )

        #expect(decision.isAllowed)
        #expect(decision.isRejected == false)
        #expect(decision.report.memoryEstimate.estimatedTotalMB <= decision.report.snapshot.recommendedLLMMemoryCeilingMB)
        #expect(decision.report.reasons.contains(ModelFitReason.modelExceedsDeviceTier) == false)
    }

    @Test func backgroundHeavyGGUFIsRejectedWhenBackgroundExecutionIsNotAllowed() async {
        let provider = TestDeviceCapabilityProvider(
            performanceTier: .high,
            physicalMemoryBytes: 8 * Self.gibibyte,
            powerState: RuntimePowerState(
                isLowPowerModeEnabled: false,
                thermalPressure: .nominal,
                appIsForeground: false
            )
        )
        let policy = DeviceModelPolicy(provider: provider)

        let decision = await policy.evaluate(
            model: ggufModel(id: "gguf.background", parameters: 3, quantization: "Q4_K_M"),
            requestedProfile: InferenceProfile.iphoneBalanced,
            requestedBudget: InferenceBudget.standard,
            appIsForeground: false
        )

        #expect(decision.isRejected)
        #expect(decision.report.selectedProfile.useMetal)
        #expect(decision.report.selectedBudget.allowGPU)
        #expect(decision.report.reasons.contains(ModelFitReason.backgroundExecutionBlocked))
    }

    @Test func backgroundGGUFWithGPUDisabledByPolicyDoesNotUseStaleRequestForRejection() async {
        let provider = TestDeviceCapabilityProvider(
            performanceTier: .high,
            physicalMemoryBytes: 8 * Self.gibibyte,
            hasMetalSupport: false,
            powerState: RuntimePowerState(
                isLowPowerModeEnabled: false,
                thermalPressure: .nominal,
                appIsForeground: false
            )
        )
        let policy = DeviceModelPolicy(provider: provider)

        let decision = await policy.evaluate(
            model: ggufModel(id: "gguf.background.cpu", parameters: 3, quantization: "Q4_K_M"),
            requestedProfile: InferenceProfile.iphoneBalanced,
            requestedBudget: InferenceBudget.maximumForeground,
            appIsForeground: false
        )

        #expect(decision.isAllowed)
        #expect(decision.report.selectedProfile.useMetal == false)
        #expect(decision.report.selectedProfile.gpuLayerCount == 0)
        #expect(decision.report.selectedBudget.allowGPU == false)
        #expect(decision.report.reasons.contains(ModelFitReason.metalUnavailable))
        #expect(decision.report.reasons.contains(ModelFitReason.backgroundExecutionBlocked) == false)
    }

    @Test func selectedProfileReflectsSimulatorAndThermalDowngrades() async {
        let simulatorPolicy = DeviceModelPolicy(provider: TestDeviceCapabilityProvider(
            formFactor: .simulator,
            performanceTier: .simulator,
            physicalMemoryBytes: 16 * Self.gibibyte,
            isSimulator: true
        ))
        let simulatorDecision = await simulatorPolicy.evaluate(
            model: ggufModel(id: "gguf.simulator.fields", parameters: 1, quantization: "Q4_K_M"),
            requestedProfile: InferenceProfile.ipadProMaximum,
            requestedBudget: InferenceBudget.maximumForeground,
            appIsForeground: true
        )

        #expect(simulatorDecision.report.selectedProfile.useMetal == false)
        #expect(simulatorDecision.report.selectedProfile.gpuLayerCount == 0)

        let thermalPolicy = DeviceModelPolicy(provider: TestDeviceCapabilityProvider(
            performanceTier: .high,
            physicalMemoryBytes: 8 * Self.gibibyte,
            powerState: RuntimePowerState(
                isLowPowerModeEnabled: false,
                thermalPressure: .serious,
                appIsForeground: true
            )
        ))
        let thermalDecision = await thermalPolicy.evaluate(
            model: ggufModel(id: "gguf.thermal.fields", parameters: 3, quantization: "Q4_K_M"),
            requestedProfile: InferenceProfile.iphoneDeepThink,
            requestedBudget: InferenceBudget.deepThink,
            appIsForeground: true
        )

        #expect(thermalDecision.report.selectedProfile.contextTokens <= 2_048)
        #expect(thermalDecision.report.selectedBudget.maxCompletionTokens <= 512)
    }

    private static let mebibyte: Int = 1_048_576
    private static let gibibyte: UInt64 = 1_073_741_824

    private func tinyIntentModel() -> LocalLLMModel {
        LocalLLMModel(
            id: "tiny.intent.policy",
            displayName: "Tiny Intent",
            backend: .tinyIntent,
            contextLength: 512
        )
    }

    private func ggufModel(
        id: String,
        parameters: Double?,
        quantization: String?,
        fileSizeBytes: Int64? = nil
    ) -> LocalLLMModel {
        LocalLLMModel(
            id: id,
            displayName: "Policy Test GGUF",
            backend: .gguf,
            parameterCountBillion: parameters,
            quantization: quantization,
            contextLength: 8_192,
            fileSizeBytes: fileSizeBytes
        )
    }
}
