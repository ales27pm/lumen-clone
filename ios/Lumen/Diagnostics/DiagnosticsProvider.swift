import Foundation

@MainActor
final class DiagnosticsProvider {
    private(set) var explicitCollectionCount = 0

    private struct ContinuedProcessingDiagnostics {
        let backgroundGPUSupported: Bool
        let status: String
        let registrationIdentifier: String
        let lastSubmittedIdentifier: String?
        let registrationErrorDomain: String?
        let registrationErrorCode: Int?
        let submitErrorDomain: String?
        let submitErrorCode: Int?
        let registeredBeforeLaunchCompletion: Bool?
        let provisioningEntitlementPresent: Bool?
        let expectedEntitlementValue: String?
    }

    private func continuedProcessingDiagnostics() -> ContinuedProcessingDiagnostics {
        let coordinator = BackgroundContinuedProcessingCoordinator.shared
        return ContinuedProcessingDiagnostics(
            backgroundGPUSupported: coordinator.gpuSupported,
            status: coordinator.lastSubmissionStatus,
            registrationIdentifier: coordinator.lastRegistrationIdentifier,
            lastSubmittedIdentifier: coordinator.lastSubmittedIdentifier,
            registrationErrorDomain: coordinator.lastRegistrationErrorDomain,
            registrationErrorCode: coordinator.lastRegistrationErrorCode,
            submitErrorDomain: coordinator.lastSubmitErrorDomain,
            submitErrorCode: coordinator.lastSubmitErrorCode,
            registeredBeforeLaunchCompletion: coordinator.lastRegistrationBeforeAppLaunchCompletion,
            provisioningEntitlementPresent: BackgroundDiagnosticsEntitlements.provisioningProfileContainsContinuedProcessingEntitlement(),
            expectedEntitlementValue: BackgroundDiagnosticsEntitlements.expectedContinuedProcessingEntitlementValue()
        )
    }

    func cachedSnapshot() -> DiagnosticsSnapshot {
        let build = BuildDiagnosticsSnapshot.current()
        let profiler = DeviceCapabilityProfiler().captureSnapshot()
        let capabilityMatrix = AssistantRuntimeCapabilityMatrix.current()
        let continuedProcessing = continuedProcessingDiagnostics()
        let runtime = RuntimeDiagnosticsSnapshot(
            foundationModelsAvailable: profiler.foundationModelsAvailable,
            foundationModelsStatus: profiler.foundationModelsStatus,
            coreMLAvailable: profiler.coreMLAvailable,
            coreMLStatus: profiler.coreMLStatus,
            runtimeCapabilityRows: capabilityMatrix.rows,
            metalAvailable: profiler.metalAvailable,
            lowPowerModeEnabled: profiler.lowPowerModeEnabled,
            thermalState: profiler.thermalState.rawValue,
            memoryWarningCount: MemoryPressureMonitor.shared.recentWarningCount(),
            recentMetricSummaries: []
        )
        let permissions = PermissionDiagnosticsSnapshot(domains: PermissionDomain.allCases.map { ($0.rawValue, "cached") })
        let tools = ToolSecuritySnapshot(tools: SecureToolRegistry.shared.definitions().map { def in
            ToolSecuritySnapshot.ToolRow(id: def.id, category: def.category.rawValue, requiredPermissions: def.requiredPermissions.map(\.rawValue), supportsBackground: def.supportsBackgroundExecution, requiresApproval: def.requiresUserApproval)
        })
        let background = BackgroundDiagnosticsSnapshot(
            permittedIdentifiers: [],
            entitlementWarnings: [],
            entitlementStates: ExpectedEntitlementManifest.currentStates(),
            backgroundGPUSupported: continuedProcessing.backgroundGPUSupported,
            continuedProcessingStatus: continuedProcessing.status,
            continuedProcessingRegistrationIdentifier: continuedProcessing.registrationIdentifier,
            continuedProcessingLastSubmittedIdentifier: continuedProcessing.lastSubmittedIdentifier,
            continuedProcessingRegistrationErrorDomain: continuedProcessing.registrationErrorDomain,
            continuedProcessingRegistrationErrorCode: continuedProcessing.registrationErrorCode,
            continuedProcessingSubmitErrorDomain: continuedProcessing.submitErrorDomain,
            continuedProcessingSubmitErrorCode: continuedProcessing.submitErrorCode,
            continuedProcessingRegisteredBeforeLaunchCompletion: continuedProcessing.registeredBeforeLaunchCompletion,
            continuedProcessingProvisioningEntitlementPresent: continuedProcessing.provisioningEntitlementPresent,
            continuedProcessingExpectedEntitlementValue: continuedProcessing.expectedEntitlementValue,
            availableMemoryBytes: profiler.availableMemoryBytes,
            energyKit: EnergyKitCapabilitySnapshot(frameworkAvailable: false, expectedEntitlementConfigured: ExpectedEntitlementManifest.expectedBoolValue(for: ExpectedEntitlementKey.energyKit), status: "cached", venueCount: nil),
            storeKit: StoreKitCapabilitySnapshot(frameworkAvailable: true, status: "cached", environment: "cached")
        )
        let grounding = GroundingDiagnosticsSnapshot(contextSource: "cached", degradedReasons: [], sectionCounts: [:], doubleGroundingNormalized: true)
        let privacy = PrivacyReportSnapshot(networkToolsEnabled: nil, networkAccessState: "cached", recentToolCategories: Array(Set(tools.tools.map(\.category))).sorted(), appIntentLimitations: ["Expensive diagnostics require explicit refresh"])
        return DiagnosticsSnapshot(build: build, runtime: runtime, permissions: permissions, tools: tools, background: background, grounding: grounding, privacy: privacy)
    }

    func collect() async -> DiagnosticsSnapshot {
        explicitCollectionCount += 1
        let cpuToken = CPUWatchdogGuard.shared.begin(category: .diagnostics)
        defer { CPUWatchdogGuard.shared.end(token: cpuToken) }
        guard !CPUWatchdogGuard.shared.shouldDegrade(category: .diagnostics), ResourceBudgetGate.allowsMaintenance(reason: "diagnostics.collect") else {
            return cachedSnapshot()
        }
        let info = Bundle.main.infoDictionary ?? [:]
        let build = BuildDiagnosticsSnapshot.current(infoDictionary: info)
        let profiler = DeviceCapabilityProfiler().captureSnapshot()
        let metrics = (try? await RuntimeMetricsStore.shared.recentMetrics(limit: 10)) ?? []
        let capabilityMatrix = await AssistantRuntimeCapabilityMatrix.currentIncludingRuntimeState()
        let runtime = RuntimeDiagnosticsSnapshot(
            foundationModelsAvailable: profiler.foundationModelsAvailable,
            foundationModelsStatus: profiler.foundationModelsStatus,
            coreMLAvailable: profiler.coreMLAvailable,
            coreMLStatus: profiler.coreMLStatus,
            runtimeCapabilityRows: capabilityMatrix.rows,
            metalAvailable: profiler.metalAvailable,
            lowPowerModeEnabled: profiler.lowPowerModeEnabled,
            thermalState: profiler.thermalState.rawValue,
            memoryWarningCount: metrics.last?.memoryWarningCount ?? 0,
            recentMetricSummaries: metrics.suffix(5).map { "\($0.runtimeName):\($0.taskKind):\($0.success ? "ok" : "fail")" }
        )

        let permStates = await PermissionRegistry.shared.diagnostics()
        let permissions = PermissionDiagnosticsSnapshot(domains: PermissionDomain.allCases.map { d in (d.rawValue, (permStates[d] ?? .unknown).rawValue) })

        let toolRows = SecureToolRegistry.shared.definitions().map { def in
            ToolSecuritySnapshot.ToolRow(id: def.id, category: def.category.rawValue, requiredPermissions: def.requiredPermissions.map(\.rawValue), supportsBackground: def.supportsBackgroundExecution, requiresApproval: def.requiresUserApproval)
        }
        let tools = ToolSecuritySnapshot(tools: toolRows)

        let warnings = BackgroundEntitlementValidator.validate(infoDictionary: info)
        let permitted: [String]
        if let values = info["BGTaskSchedulerPermittedIdentifiers"] as? [String] { permitted = values }
        else if let value = info["BGTaskSchedulerPermittedIdentifiers"] as? String { permitted = value.split { $0 == " " || $0 == ";" || $0 == "," }.map(String.init) }
        else { permitted = [] }
        async let energyKit = EnergyKitCapabilityService.snapshot()
        async let storeKit = StoreKitCapabilityService.snapshot()
        let continuedProcessing = continuedProcessingDiagnostics()
        let background = await BackgroundDiagnosticsSnapshot(
            permittedIdentifiers: permitted,
            entitlementWarnings: warnings.map(\.message),
            entitlementStates: ExpectedEntitlementManifest.currentStates(),
            backgroundGPUSupported: continuedProcessing.backgroundGPUSupported,
            continuedProcessingStatus: continuedProcessing.status,
            continuedProcessingRegistrationIdentifier: continuedProcessing.registrationIdentifier,
            continuedProcessingLastSubmittedIdentifier: continuedProcessing.lastSubmittedIdentifier,
            continuedProcessingRegistrationErrorDomain: continuedProcessing.registrationErrorDomain,
            continuedProcessingRegistrationErrorCode: continuedProcessing.registrationErrorCode,
            continuedProcessingSubmitErrorDomain: continuedProcessing.submitErrorDomain,
            continuedProcessingSubmitErrorCode: continuedProcessing.submitErrorCode,
            continuedProcessingRegisteredBeforeLaunchCompletion: continuedProcessing.registeredBeforeLaunchCompletion,
            continuedProcessingProvisioningEntitlementPresent: continuedProcessing.provisioningEntitlementPresent,
            continuedProcessingExpectedEntitlementValue: continuedProcessing.expectedEntitlementValue,
            availableMemoryBytes: profiler.availableMemoryBytes,
            energyKit: energyKit,
            storeKit: storeKit
        )

        let grounding = GroundingDiagnosticsSnapshot(contextSource: SharedContainer.shared == nil ? "unavailable" : "sharedContainer", degradedReasons: SharedContainer.shared == nil ? ["model_context_unavailable"] : [], sectionCounts: [:], doubleGroundingNormalized: true)

        let networkState = (permStates[.networkAccess] ?? .unknown).rawValue
        let privacy = PrivacyReportSnapshot(
            networkToolsEnabled: networkState == AssistantPermissionState.granted.rawValue,
            networkAccessState: networkState,
            recentToolCategories: Array(Set(toolRows.map(\.category))).sorted(),
            appIntentLimitations: [
                "Sensitive actions require open-app approval",
                "Network tools default to disabled; connected services and model downloads are separate user actions"
            ]
        )

        return DiagnosticsSnapshot(build: build, runtime: runtime, permissions: permissions, tools: tools, background: background, grounding: grounding, privacy: privacy)
    }
}
