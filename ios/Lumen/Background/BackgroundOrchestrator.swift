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
        await runTriggerScan(context: context)
    }

    func handleProcessing() async {
        guard let container = SharedContainer.shared else {
            await appendSharedContainerUnavailable(taskKind: .triggerScan)
            return
        }
        let deadline = Date().addingTimeInterval(4.5)
        let context = ModelContext(container)
        await runTriggerScan(context: context)
        await runProcessingMaintenance(until: deadline)
    }

    func runTriggerScan(context: ModelContext) async {
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
        await TriggerScheduler.shared.fireDueTriggers(context: context, settings: SettingsSnapshot.loadFromDisk())
        await appendMetric(
            taskKind: .triggerScan,
            policySummary: "trigger scheduler fireDueTriggers; model loading denied",
            success: true,
            latencyMs: Int(Date().timeIntervalSince(startedAt) * 1000)
        )
        await lease.release(category: "triggerScan")
    }

    func runMemoryConsolidationIfAllowed() async {
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
        await Self.performMemoryConsolidation(
            container: container,
            metrics: metrics,
            promoteQueuedCaptures: decision.allowModelLoading
        )
    }

    func runRAGMaintenanceIfAllowed() async {
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
        let result = await Self.performRAGMaintenance(container: container)
        await appendMetric(
            taskKind: .ragMaintenance,
            policySummary: result.metricSummary,
            success: result.success,
            errorCode: result.success ? nil : "maintenance_failed"
        )
    }

    func runModelHousekeepingIfAllowed() async {
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
        await appendMetric(
            taskKind: .modelHousekeeping,
            policySummary: "optional chat slot cleanup; unloaded=\(result.unloadedSlotSummary)",
            success: true,
            latencyMs: Int(Date().timeIntervalSince(startedAt) * 1000)
        )
    }

    func runProcessingMaintenance(until deadline: Date) async {
        guard !Task.isCancelled else { return }
        guard await continueProcessing(before: .selfImprovement, deadline: deadline) else { return }
        await runSelfImprovementIfAllowed(until: deadline)

        guard !Task.isCancelled else { return }
        guard await continueProcessing(before: .memoryConsolidation, deadline: deadline) else { return }
        await runMemoryConsolidationIfAllowed()

        guard !Task.isCancelled else { return }
        guard await continueProcessing(before: .ragMaintenance, deadline: deadline) else { return }
        await runRAGMaintenanceIfAllowed()

        guard !Task.isCancelled else { return }
        guard await continueProcessing(before: .modelHousekeeping, deadline: deadline) else { return }
        await runModelHousekeepingIfAllowed()
    }

    func runSelfImprovementIfAllowed(until deadline: Date) async {
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
                await runTriggerScan(context: context)
                try Task.checkCancellation()
                if runProcessingWork, Date() < deadline {
                    await runProcessingMaintenance(until: deadline)
                }
                try Task.checkCancellation()
                return true
            } catch is CancellationError {
                return false
            } catch {
                return false
            }
        }
        task.expirationHandler = {
            work.cancel()
            Task { @MainActor in
                completion.complete(success: false)
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
            backgroundAgentsEnabled: SettingsSnapshot.loadFromDisk().agentModeEnabled,
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
        return BackgroundTaskPolicy.decide(.init(
            taskKind: taskKind,
            lowPowerMode: snapshot.lowPowerModeEnabled ?? ProcessInfo.processInfo.isLowPowerModeEnabled,
            thermalState: snapshot.thermalState ?? .unknown,
            isForeground: snapshot.scenePhase == .active,
            backgroundAgentsEnabled: SettingsSnapshot.loadFromDisk().agentModeEnabled,
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

    nonisolated private static func performMemoryConsolidation(
        container: ModelContainer,
        metrics: RuntimeMetricsStore,
        promoteQueuedCaptures: Bool
    ) async {
        let context = ModelContext(container)
        await MemoryConsolidator.consolidate(
            context: context,
            metricsStore: metrics,
            promoteQueuedCaptures: promoteQueuedCaptures
        )
    }

    nonisolated private static func performRAGMaintenance(container: ModelContainer) async -> RAGMaintenanceResult {
        let context = ModelContext(container)
        return await RAGEngine().maintenance(context: context)
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

    init(task: BGTask) {
        self.task = task
    }

    func complete(success: Bool) {
        guard !didComplete else { return }
        didComplete = true
        task.setTaskCompleted(success: success)
    }
}
