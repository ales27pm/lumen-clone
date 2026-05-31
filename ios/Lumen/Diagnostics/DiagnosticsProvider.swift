import Foundation

@MainActor
final class DiagnosticsProvider {
    func collect() async -> DiagnosticsSnapshot {
        let profiler = DeviceCapabilityProfiler().captureSnapshot()
        let metrics = (try? await RuntimeMetricsStore.shared.recentMetrics(limit: 10)) ?? []
        let runtime = RuntimeDiagnosticsSnapshot(
            foundationModelsAvailable: profiler.foundationModelsAvailable,
            coreMLAvailable: profiler.coreMLAvailable,
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

        let info = Bundle.main.infoDictionary ?? [:]
        let warnings = BackgroundEntitlementValidator.validate(infoDictionary: info)
        let permitted: [String]
        if let values = info["BGTaskSchedulerPermittedIdentifiers"] as? [String] { permitted = values }
        else if let value = info["BGTaskSchedulerPermittedIdentifiers"] as? String { permitted = value.split { $0 == " " || $0 == ";" || $0 == "," }.map(String.init) }
        else { permitted = [] }
        let background = BackgroundDiagnosticsSnapshot(permittedIdentifiers: permitted, entitlementWarnings: warnings.map(\.message))

        let grounding = GroundingDiagnosticsSnapshot(contextSource: SharedContainer.shared == nil ? "unavailable" : "sharedContainer", degradedReasons: SharedContainer.shared == nil ? ["model_context_unavailable"] : [], sectionCounts: [:], doubleGroundingNormalized: true)

        let networkState = (permStates[.networkAccess] ?? .unknown).rawValue
        let privacy = PrivacyReportSnapshot(localOnlyMode: networkState != AssistantPermissionState.granted.rawValue, networkAccessState: networkState, recentToolCategories: Array(Set(toolRows.map(\.category))).sorted(), appIntentLimitations: ["Sensitive actions require open-app approval", "No external network by default"])

        return DiagnosticsSnapshot(runtime: runtime, permissions: permissions, tools: tools, background: background, grounding: grounding, privacy: privacy)
    }
}
