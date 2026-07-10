import Foundation
import SwiftData
import BackgroundTasks
import SwiftUI

@MainActor
final class BackgroundOrchestrator {
    static let shared = BackgroundOrchestrator()

    private let lease: BackgroundExecutionLease
    private let metrics: RuntimeMetricsStore
    private let modelHousekeeping: @MainActor () async -> FleetRuntimeCleanupResult

    init(
        metricsStore: RuntimeMetricsStore = .shared,
        lease: BackgroundExecutionLease = BackgroundExecutionLease(),
        modelHousekeeping: @escaping @MainActor () async -> FleetRuntimeCleanupResult = {
            await FleetRuntimeCleanup.unloadOptionalChatSlotsNow()
        }
    ) {
        self.metrics = metricsStore
        self.lease = lease
        self.modelHousekeeping = modelHousekeeping
    }

    func register() {
        TriggerScheduler.shared.registerTasks()
    }

    func schedule() {
        TriggerScheduler.shared.scheduleBackgroundRefresh()
    }

    @discardableResult
    func requestPermission() async -> Bool {
        await TriggerScheduler.shared.requestPermission()
    }

    func handleAppRefresh(task: BGAppRefreshTask) async {
        await handleBackgroundTask(task, runProcessingWork: false)
    }

    func handleProcessing(task: BGProcessingTask) async {
        await handleBackgroundTask(task, runProcessingWork: true)
    }

    func handleAppRefresh() async {
        guard let container = SharedContainer.shared else {
            await appendSharedContainerUnavailable(taskKind: .triggerScan)
            return
        }
        let context = ModelContext(container)
        do {
            try await runTriggerScan(context: context)
        } catch is CancellationError {
            await appendBackgroundCancellationMetric(
                taskKind: .triggerScan,
                reason: "cancelled",
                policySummary: "app refresh cancelled"
            )
        } catch {
            await appendMetric(
                taskKind: .triggerScan,
                policySummary: "failed: \(RuntimeMetricErrorSanitizer.code(for: error))",
                success: false,
                errorCode: RuntimeMetricErrorSanitizer.code(for: error)
            )
        }
    }

    func handleProcessing() async {
        guard let container = SharedContainer.shared else {
            await appendSharedContainerUnavailable(taskKind: .triggerScan)
            return
        }
        let deadline = Date().addingTimeInterval(4.5)
        let context = ModelContext(container)
        do {
            try await runTriggerScan(context: context)
            try await runProcessingMaintenance(until: deadline)
        } catch is CancellationError {
            await appendBackgroundCancellationMetric(
                taskKind: .triggerScan,
                reason: "cancelled",
                policySummary: "background processing cancelled"
            )
        } catch {
            await appendMetric(
                taskKind: .triggerScan,
                policySummary: "failed: \(RuntimeMetricErrorSanitizer.code(for: error))",
                success: false,
                errorCode: RuntimeMetricErrorSanitizer.code(for: error)
            )
        }
    }

    func runTriggerScan(context: ModelContext) async throws {
        try Task.checkCancellation()
        let decision = triggerScanDecision()
        guard decision.allow else {
            await appendMetric(
                taskKind: .triggerScan,
                policySummary: "skipped: \(decision.denyReason ?? "background policy denied")",
                success: true,
                errorCode: "background_policy_denied"
            )
            return
        }
        let startedAt = Date()
        let acquired = await lease.acquire(category: "triggerScan", reason: "background trigger scan")
        guard acquired else {
            await appendMetric(
                taskKind: .triggerScan,
                policySummary: "skipped: trigger scan lease already active",
                success: true,
                errorCode: "background_lease_active"
            )
            return
        }
        do {
            try Task.checkCancellation()
            await TriggerScheduler.shared.fireDueTriggers(context: context, settings: SettingsSnapshot.loadFromDisk())
            try Task.checkCancellation()
            await appendMetric(
                taskKind: .triggerScan,
                policySummary: "trigger scheduler fireDueTriggers; model loading denied",
                success: true,
                latencyMs: Int(Date().timeIntervalSince(startedAt) * 1000)
            )
            await lease.release(category: "triggerScan")
        } catch {
            await lease.release(category: "triggerScan")
            throw error
        }
    }

    func runMemoryConsolidationIfAllowed() async throws {
        try Task.checkCancellation()
        let decision = maintenanceDecision(taskKind: .memoryConsolidation, estimatedCost: 2)
        guard decision.allow else {
            await appendMetric(
                taskKind: .memoryConsolidation,
                policySummary: "skipped: \(decision.denyReason ?? "background policy denied")",
                success: true,
                errorCode: "background_policy_denied"
            )
            return
        }
        guard let container = SharedContainer.shared else {
            await appendSharedContainerUnavailable(taskKind: .memoryConsolidation)
            return
        }
        try await Self.performMemoryConsolidation(
            container: container,
            metrics: metrics,
            promoteQueuedCaptures: decision.allowModelLoading
        )
        try Task.checkCancellation()
    }

    func runRAGMaintenanceIfAllowed() async throws {
        try Task.checkCancellation()
        let decision = maintenanceDecision(taskKind: .ragMaintenance, estimatedCost: 2)
        guard decision.allow else {
            await appendMetric(
                taskKind: .ragMaintenance,
                policySummary: "skipped: \(decision.denyReason ?? "background policy denied")",
                success: true,
                errorCode: "background_policy_denied"
            )
            return
        }
        guard let container = SharedContainer.shared else {
            await appendSharedContainerUnavailable(taskKind: .ragMaintenance)
            return
        }
        let result = try await Self.performRAGMaintenance(container: container)
        try Task.checkCancellation()
        await appendMetric(
            taskKind: .ragMaintenance,
            policySummary: result.metricSummary,
            success: result.success,
            errorCode: result.success ? nil : "maintenance_failed"
        )
    }

    func runModelHousekeepingIfAllowed() async throws {
        try Task.checkCancellation()
        let decision = maintenanceDecision(taskKind: .modelHousekeeping, estimatedCost: 0)
        guard decision.allow else {
            await appendMetric(
                taskKind: .modelHousekeeping,
                policySummary: "skipped: \(decision.denyReason ?? "background policy denied")",
                success: true,
                errorCode: "background_policy_denied"
            )
            return
        }
        let startedAt = Date()
        let result = await modelHousekeeping()
        try Task.checkCancellation()
        await appendMetric(
            taskKind: .modelHousekeeping,
            policySummary: "optional chat slot cleanup; unloaded=\(result.unloadedSlotSummary)",
            success: true,
            latencyMs: Int(Date().timeIntervalSince(startedAt) * 1000)
        )
    }

    func runProcessingMaintenance(until deadline: Date) async throws {
        try Task.checkCancellation()
        guard await continueProcessing(before: .selfImprovement, deadline: deadline) else { return }
        try await runSelfImprovementIfAllowed(until: deadline)

        try Task.checkCancellation()
        guard await continueProcessing(before: .memoryConsolidation, deadline: deadline) else { return }
        try await runMemoryConsolidationIfAllowed()

        try Task.checkCancellation()
        guard await continueProcessing(before: .ragMaintenance, deadline: deadline) else { return }
        try await runRAGMaintenanceIfAllowed()

        try Task.checkCancellation()
        guard await continueProcessing(before: .modelHousekeeping, deadline: deadline) else { return }
        try await runModelHousekeepingIfAllowed()
    }

    func runSelfImprovementIfAllowed(until deadline: Date) async throws {
        try Task.checkCancellation()
        let decision = maintenanceDecision(taskKind: .selfImprovement, estimatedCost: 2)
        guard decision.allow else {
            await appendMetric(
                taskKind: .selfImprovement,
                policySummary: "skipped: \(decision.denyReason ?? "background policy denied")",
                success: true,
                errorCode: "background_policy_denied"
            )
            return
        }
        guard let container = SharedContainer.shared else {
            await appendSharedContainerUnavailable(taskKind: .selfImprovement)
            return
        }
        let startedAt = Date()
        let outcome = await SelfImprovementLoop.shared.run(
            trigger: .backgroundProcessing,
            container: container,
            deadline: deadline,
            maintenanceMode: .snapshotOnly
        )
        try Task.checkCancellation()
        await appendSelfImprovementOutcomeMetric(outcome, latencyMs: Int(Date().timeIntervalSince(startedAt) * 1000))
    }

    private func handleBackgroundTask(_ task: BGTask, runProcessingWork: Bool) async {
        schedule()
        let completion = BackgroundTaskCompletion(task: task)
        let work = Task { () -> Bool in
            guard let container = SharedContainer.shared else {
                await appendSharedContainerUnavailable(taskKind: .triggerScan)
                return true
            }
            let deadline = Date().addingTimeInterval(4.5)
            let context = ModelContext(container)
            do {
                try Task.checkCancellation()
                try await runTriggerScan(context: context)
                try Task.checkCancellation()
                if runProcessingWork, Date() < deadline {
                    try await runProcessingMaintenance(until: deadline)
                }
                try Task.checkCancellation()
                return true
            } catch is CancellationError {
                if completion.claimCancellationMetric() {
                    await appendBackgroundCancellationMetric(
                        taskKind: .triggerScan,
                        reason: "cancelled",
                        policySummary: "background task cancelled; processingWork=\(runProcessingWork)"
                    )
                }
                return false
            } catch {
                return false
            }
        }
        task.expirationHandler = {
            work.cancel()
            Task { @MainActor in
                if completion.claimCancellationMetric() {
                    await self.appendBackgroundCancellationMetric(
                        taskKind: .triggerScan,
                        reason: "cancelled",
                        policySummary: "background task expiration requested cancellation; processingWork=\(runProcessingWork)"
                    )
                }
            }
        }
        let success = await work.value
        completion.complete(success: success)
    }

    private func triggerScanDecision() -> BackgroundTaskDecision {
        let snapshot = ResourceBudgetGate.diagnosticSnapshot()
        return BackgroundTaskPolicy.decide(.init(
            taskKind: .triggerScan,
            lowPowerMode: snapshot.lowPowerModeEnabled ?? ProcessInfo.processInfo.isLowPowerModeEnabled,
            thermalState: snapshot.thermalState ?? .unknown,
            isForeground: snapshot.scenePhase == .active,
            backgroundAgentsEnabled: true,
            requiresNetwork: false,
            estimatedCost: 1
        ))
    }

    private func maintenanceDecision(
        taskKind: BackgroundTaskKind,
        requiresNetwork: Bool = false,
        estimatedCost: Int
    ) -> BackgroundTaskDecision {
        guard ResourceBudgetGate.allowsMaintenance(reason: ModelLoadIntent.background.rawValue) else {
            return .init(
                allow: false,
                denyReason: "resource budget denied maintenance",
                maxSteps: 0,
                maxTokens: 0,
                allowModelLoading: false,
                allowNetwork: false
            )
        }
        let snapshot = ResourceBudgetGate.diagnosticSnapshot()
        let backgroundAgentsEnabled = taskKind == .selfImprovement
            ? SettingsSnapshot.loadFromDisk().agentModeEnabled
            : true
        return BackgroundTaskPolicy.decide(.init(
            taskKind: taskKind,
            lowPowerMode: snapshot.lowPowerModeEnabled ?? ProcessInfo.processInfo.isLowPowerModeEnabled,
            thermalState: snapshot.thermalState ?? .unknown,
            isForeground: snapshot.scenePhase == .active,
            backgroundAgentsEnabled: backgroundAgentsEnabled,
            requiresNetwork: requiresNetwork,
            estimatedCost: estimatedCost
        ))
    }

    private func appendMetric(
        taskKind: BackgroundTaskKind,
        policySummary: String,
        success: Bool,
        errorCode: String? = nil,
        latencyMs: Int? = nil
    ) async {
        try? await metrics.appendMetric(RuntimeMetric(
            timestamp: Date(),
            runtimeName: "background",
            taskKind: taskKind.rawValue,
            modelIDHash: nil,
            policySummary: PersistentRuntimeDiagnosticsRedactor.redact(policySummary),
            latencyMs: latencyMs,
            success: success,
            errorCode: errorCode.map(PersistentRuntimeDiagnosticsRedactor.safeCode),
            thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState),
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            memoryWarningCount: MemoryPressureMonitor.shared.warningCount
        ))
    }

    private static func performMemoryConsolidation(
        container: ModelContainer,
        metrics: RuntimeMetricsStore,
        promoteQueuedCaptures: Bool
    ) async throws {
        try Task.checkCancellation()
        let context = ModelContext(container)
        await MemoryConsolidator.consolidate(
            context: context,
            metricsStore: metrics,
            promoteQueuedCaptures: promoteQueuedCaptures
        )
        try Task.checkCancellation()
    }

    private static func performRAGMaintenance(container: ModelContainer) async throws -> RAGMaintenanceResult {
        try Task.checkCancellation()
        let context = ModelContext(container)
        let result = await RAGEngine().maintenance(context: context)
        try Task.checkCancellation()
        return result
    }

    private func appendSelfImprovementOutcomeMetric(_ outcome: SelfImprovementOutcome, latencyMs: Int) async {
        switch outcome {
        case .applied(let summary):
            await appendMetric(
                taskKind: .selfImprovement,
                policySummary: "self-improvement applied: \(summary)",
                success: true,
                latencyMs: latencyMs
            )
        case .skipped(let reason):
            await appendMetric(
                taskKind: .selfImprovement,
                policySummary: "skipped: \(reason)",
                success: true,
                errorCode: reason,
                latencyMs: latencyMs
            )
        case .cancelled:
            await appendMetric(
                taskKind: .selfImprovement,
                policySummary: "cancelled",
                success: false,
                errorCode: "cancelled",
                latencyMs: latencyMs
            )
        case .failed(let code):
            await appendMetric(
                taskKind: .selfImprovement,
                policySummary: "failed: \(code)",
                success: false,
                errorCode: code,
                latencyMs: latencyMs
            )
        }
    }

    private func appendSharedContainerUnavailable(taskKind: BackgroundTaskKind) async {
        await appendMetric(
            taskKind: taskKind,
            policySummary: "skipped: shared container unavailable",
            success: false,
            errorCode: "shared_container_unavailable"
        )
    }

    private func appendBackgroundCancellationMetric(
        taskKind: BackgroundTaskKind,
        reason: String,
        policySummary: String
    ) async {
        await appendMetric(
            taskKind: taskKind,
            policySummary: policySummary,
            success: false,
            errorCode: reason
        )
    }

    private func continueProcessing(before taskKind: BackgroundTaskKind, deadline: Date) async -> Bool {
        guard Date() < deadline else {
            await appendMetric(
                taskKind: taskKind,
                policySummary: "skipped: background processing deadline expired",
                success: true,
                errorCode: "deadline_exceeded"
            )
            return false
        }
        return true
    }
}

@MainActor
private final class BackgroundTaskCompletion {
    private let task: BGTask
    private var didComplete = false
    private var didClaimCancellationMetric = false

    init(task: BGTask) {
        self.task = task
    }

    func claimCancellationMetric() -> Bool {
        guard !didClaimCancellationMetric else { return false }
        didClaimCancellationMetric = true
        return true
    }

    func complete(success: Bool) {
        guard !didComplete else { return }
        didComplete = true
        task.setTaskCompleted(success: success)
    }
}
