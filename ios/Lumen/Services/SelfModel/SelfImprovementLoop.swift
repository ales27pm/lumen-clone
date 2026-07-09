import Foundation
import SwiftData

enum SelfImprovementTrigger: String, Sendable, Equatable {
    case appLaunch
    case backgroundProcessing
    case manualDebug
    case test

    var isForeground: Bool {
        switch self {
        case .appLaunch, .manualDebug, .test:
            return true
        case .backgroundProcessing:
            return false
        }
    }

    var toolSource: ToolInvocationSource {
        isForeground ? .system : .backgroundTrigger
    }
}

enum SelfImprovementOutcome: Sendable, Equatable {
    case applied(String)
    case skipped(String)
    case failed(String)
    case cancelled
}

enum SelfImprovementLoopError: Error, Equatable {
    case deadlineExceeded
    case maintenanceFailed(String)
}

struct SelfImprovementConfig: Sendable, Equatable {
    var cooldownSeconds: TimeInterval = 600
    var maxRunDurationSeconds: TimeInterval = 8
    var failureThreshold: Int = 3
    var circuitOpenSeconds: TimeInterval = 1_800
    var metricCompactionMaxEntries: Int = 500

    static let `default` = SelfImprovementConfig()
}

struct SelfImprovementMaintenanceResult: Sendable, Equatable {
    let applied: Bool
    let summary: String

    static func applied(_ summary: String) -> SelfImprovementMaintenanceResult {
        .init(applied: true, summary: summary)
    }

    static func skipped(_ reason: String) -> SelfImprovementMaintenanceResult {
        .init(applied: false, summary: reason)
    }
}

actor SelfImprovementCoordinator {
    enum Phase: Sendable, Equatable {
        case idle
        case running(UUID)
        case cooldown(until: Date)
        case circuitOpen(until: Date)
    }

    struct Snapshot: Sendable, Equatable {
        var phase: Phase = .idle
        var consecutiveFailures: Int = 0
        var lastStartedAt: Date?
        var lastFinishedAt: Date?
        var lastSuccessAt: Date?
        var lastReason: String?
    }

    private var snapshot = Snapshot()

    func state() -> Snapshot {
        snapshot
    }

    func begin(now: Date, force: Bool, cooldown: TimeInterval) -> UUID? {
        resetExpiredWindow(now: now)
        switch snapshot.phase {
        case .running:
            snapshot.lastReason = "already_running"
            return nil
        case .cooldown(let until) where !force && now < until:
            snapshot.lastReason = "cooldown_active"
            return nil
        case .circuitOpen(let until) where !force && now < until:
            snapshot.lastReason = "circuit_open"
            return nil
        default:
            let id = UUID()
            snapshot.phase = .running(id)
            snapshot.lastStartedAt = now
            snapshot.lastReason = nil
            return id
        }
    }

    func success(runID: UUID, now: Date, cooldown: TimeInterval, reason: String) {
        guard case .running(let active) = snapshot.phase, active == runID else { return }
        snapshot.phase = .cooldown(until: now.addingTimeInterval(max(0, cooldown)))
        snapshot.consecutiveFailures = 0
        snapshot.lastFinishedAt = now
        snapshot.lastSuccessAt = now
        snapshot.lastReason = reason
    }

    func skip(reason: String) {
        snapshot.lastReason = reason
    }

    func failure(runID: UUID, now: Date, threshold: Int, openDuration: TimeInterval, reason: String) {
        guard case .running(let active) = snapshot.phase, active == runID else { return }
        snapshot.consecutiveFailures += 1
        snapshot.lastFinishedAt = now
        snapshot.lastReason = reason
        if snapshot.consecutiveFailures >= max(1, threshold) {
            snapshot.phase = .circuitOpen(until: now.addingTimeInterval(max(0, openDuration)))
        } else {
            snapshot.phase = .idle
        }
    }

    func cancel(runID: UUID, now: Date) {
        guard case .running(let active) = snapshot.phase, active == runID else { return }
        snapshot.phase = .idle
        snapshot.lastFinishedAt = now
        snapshot.lastReason = "cancelled"
    }

    func resetIfWindowExpired(now: Date) {
        resetExpiredWindow(now: now)
    }

    private func resetExpiredWindow(now: Date) {
        switch snapshot.phase {
        case .cooldown(let until) where now >= until:
            snapshot.phase = .idle
        case .circuitOpen(let until) where now >= until:
            snapshot.phase = .idle
            snapshot.consecutiveFailures = 0
        default:
            break
        }
    }
}

@MainActor
final class SelfImprovementLoop {
    typealias NowProvider = @MainActor () -> Date
    typealias Maintenance = @MainActor (SelfImprovementTrigger, ModelContext?, Date?) async throws -> SelfImprovementMaintenanceResult

    static let shared = SelfImprovementLoop()

    private let coordinator: SelfImprovementCoordinator
    private let metricsStore: RuntimeMetricsStore
    private let config: SelfImprovementConfig
    private let now: NowProvider
    private let maintenance: Maintenance

    init(
        coordinator: SelfImprovementCoordinator = SelfImprovementCoordinator(),
        metricsStore: RuntimeMetricsStore = .shared,
        config: SelfImprovementConfig = .default,
        now: @escaping NowProvider = { Date() },
        maintenance: Maintenance? = nil
    ) {
        self.coordinator = coordinator
        self.metricsStore = metricsStore
        self.config = config
        self.now = now
        self.maintenance = maintenance ?? { trigger, context, deadline in
            try await Self.defaultMaintenance(
                trigger: trigger,
                context: context,
                deadline: deadline,
                metricsStore: metricsStore,
                metricCompactionMaxEntries: config.metricCompactionMaxEntries
            )
        }
    }

    @discardableResult
    func run(
        trigger: SelfImprovementTrigger,
        context: ModelContext?,
        deadline: Date? = nil,
        force: Bool = false
    ) async -> SelfImprovementOutcome {
        let startedAt = now()
        guard deadline.map({ startedAt < $0 }) ?? true else {
            await coordinator.skip(reason: "deadline_expired")
            await appendMetric(
                summary: "skipped: deadline_expired trigger=\(trigger.rawValue)",
                success: true,
                errorCode: "deadline_expired",
                latencyMs: 0
            )
            return .skipped("deadline_expired")
        }

        guard policyAllows(trigger: trigger) else {
            await coordinator.skip(reason: "policy_denied")
            await appendMetric(
                summary: "skipped: policy_denied trigger=\(trigger.rawValue)",
                success: true,
                errorCode: "policy_denied",
                latencyMs: 0
            )
            return .skipped("policy_denied")
        }

        await coordinator.resetIfWindowExpired(now: startedAt)
        guard let runID = await coordinator.begin(now: startedAt, force: force, cooldown: config.cooldownSeconds) else {
            let reason = await coordinator.state().lastReason ?? "already_running_or_window_active"
            await appendMetric(
                summary: "skipped: \(reason) trigger=\(trigger.rawValue)",
                success: true,
                errorCode: reason,
                latencyMs: 0
            )
            return .skipped(reason)
        }

        do {
            try Task.checkCancellation()
            try checkDeadline(startedAt: startedAt, deadline: deadline)
            let result = try await maintenance(trigger, context, deadline)
            try Task.checkCancellation()
            try checkDeadline(startedAt: startedAt, deadline: deadline)

            let finishedAt = now()
            await coordinator.success(
                runID: runID,
                now: finishedAt,
                cooldown: config.cooldownSeconds,
                reason: result.applied ? "applied" : result.summary
            )
            await appendMetric(
                summary: "trigger=\(trigger.rawValue); \(result.summary)",
                success: true,
                errorCode: nil,
                latencyMs: Int(finishedAt.timeIntervalSince(startedAt) * 1000)
            )
            return result.applied ? .applied(result.summary) : .skipped(result.summary)
        } catch is CancellationError {
            await coordinator.cancel(runID: runID, now: now())
            await appendMetric(
                summary: "cancelled trigger=\(trigger.rawValue)",
                success: false,
                errorCode: "cancelled",
                latencyMs: Int(now().timeIntervalSince(startedAt) * 1000)
            )
            return .cancelled
        } catch SelfImprovementLoopError.deadlineExceeded {
            let finishedAt = now()
            await coordinator.success(
                runID: runID,
                now: finishedAt,
                cooldown: config.cooldownSeconds,
                reason: "deadline_expired"
            )
            await appendMetric(
                summary: "skipped: deadline_expired trigger=\(trigger.rawValue)",
                success: true,
                errorCode: "deadline_expired",
                latencyMs: Int(finishedAt.timeIntervalSince(startedAt) * 1000)
            )
            return .skipped("deadline_expired")
        } catch {
            let code = Self.errorCode(for: error)
            await coordinator.failure(
                runID: runID,
                now: now(),
                threshold: config.failureThreshold,
                openDuration: config.circuitOpenSeconds,
                reason: code
            )
            await appendMetric(
                summary: "failed trigger=\(trigger.rawValue) error=\(code)",
                success: false,
                errorCode: code,
                latencyMs: Int(now().timeIntervalSince(startedAt) * 1000)
            )
            return .failed(code)
        }
    }

    private func policyAllows(trigger: SelfImprovementTrigger) -> Bool {
        let snapshot = ResourceBudgetGate.diagnosticSnapshot()
        let decision = BackgroundTaskPolicy.decide(.init(
            taskKind: .selfImprovement,
            lowPowerMode: snapshot.lowPowerModeEnabled ?? ProcessInfo.processInfo.isLowPowerModeEnabled,
            thermalState: snapshot.thermalState ?? .unknown,
            isForeground: trigger.isForeground,
            backgroundAgentsEnabled: true,
            requiresNetwork: false,
            estimatedCost: 2
        ))
        return decision.allow
    }

    private func checkDeadline(startedAt: Date, deadline: Date?) throws {
        let configuredDeadline = startedAt.addingTimeInterval(max(0, config.maxRunDurationSeconds))
        if now() >= configuredDeadline {
            throw SelfImprovementLoopError.deadlineExceeded
        }
        if let deadline, now() >= deadline {
            throw SelfImprovementLoopError.deadlineExceeded
        }
    }

    private func appendMetric(
        summary: String,
        success: Bool,
        errorCode: String?,
        latencyMs: Int?
    ) async {
        try? await metricsStore.appendMetric(RuntimeMetric(
            timestamp: now(),
            runtimeName: "selfImprovement",
            taskKind: BackgroundTaskKind.selfImprovement.rawValue,
            modelIDHash: nil,
            policySummary: summary,
            latencyMs: latencyMs,
            success: success,
            errorCode: errorCode,
            thermalState: .from(processThermalState: ProcessInfo.processInfo.thermalState),
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            memoryWarningCount: MemoryPressureMonitor.shared.warningCount
        ))
    }

    private static func defaultMaintenance(
        trigger: SelfImprovementTrigger,
        context: ModelContext?,
        deadline: Date?,
        metricsStore: RuntimeMetricsStore,
        metricCompactionMaxEntries: Int
    ) async throws -> SelfImprovementMaintenanceResult {
        guard let context else {
            return .skipped("shared_container_unavailable")
        }
        try checkDeadline(deadline)

        let snapshot = await buildSnapshot(trigger: trigger, context: context)
        try checkDeadline(deadline)
        let rendered = SelfModelContextProvider.render(snapshot, maxChars: trigger.isForeground ? 1_200 : 700)
        try Task.checkCancellation()
        try checkDeadline(deadline)

        let memorySummary: String
        let ragSummary: String
        if trigger == .backgroundProcessing {
            memorySummary = "already_run"
            ragSummary = "already_run"
        } else {
            await MemoryConsolidator.consolidate(context: context, metricsStore: metricsStore, promoteQueuedCaptures: false)
            try Task.checkCancellation()
            try checkDeadline(deadline)

            let rag = await RAGEngine().maintenance(context: context)
            guard rag.success else {
                throw SelfImprovementLoopError.maintenanceFailed("rag_maintenance_failed")
            }
            try Task.checkCancellation()
            try checkDeadline(deadline)
            ragSummary = rag.metricSummary
            memorySummary = "dedupe"
        }

        try checkDeadline(deadline)
        try? await metricsStore.compact(maxEntries: metricCompactionMaxEntries)
        try checkDeadline(deadline)

        let summary = [
            "snapshot=\(snapshot.schemaVersion)",
            "mode=\(snapshot.app.mode)",
            "selfModelChars=\(rendered.count)",
            "memory=\(memorySummary)",
            "rag=\(ragSummary)",
            "metrics=compact"
        ].joined(separator: "; ")
        return .applied(summary)
    }

    private static func checkDeadline(_ deadline: Date?) throws {
        if let deadline, Date() >= deadline {
            throw SelfImprovementLoopError.deadlineExceeded
        }
    }

    private static func buildSnapshot(trigger: SelfImprovementTrigger, context: ModelContext) async -> SelfModelSnapshot {
        let turn = AssistantTurnContext(
            task: .backgroundTrigger,
            input: "",
            isForeground: trigger.isForeground,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            thermalState: ProcessInfo.processInfo.thermalState,
            prefersFoundationModels: false,
            allowHeavyRuntime: false,
            maxTokens: 128
        )
        let budget = ContextBudgetAllocator.allocate(for: turn, maxInputTokens: 512)
        let toolContext = ToolExecutionContext(
            isForeground: trigger.isForeground,
            appState: nil,
            modelContext: context,
            permissionRegistry: .shared,
            metricsStore: .shared
        )
        let tools = await SecureToolRegistry.shared.availableDefinitions(context: toolContext, source: trigger.toolSource)
        return SelfModelSnapshotBuilder.build(
            turn: turn,
            budget: budget,
            selectedRuntime: .init(runtime: .unavailable, reason: "self-improvement runtime maintenance does not load models"),
            tools: tools,
            availableBackendKinds: [],
            activeSlot: .rem,
            now: Date()
        )
    }

    private static func errorCode(for error: Error) -> String {
        if let error = error as? SelfImprovementLoopError {
            switch error {
            case .deadlineExceeded:
                return "deadline_exceeded"
            case .maintenanceFailed(let code):
                return code
            }
        }
        return RuntimeMetricErrorSanitizer.code(for: error)
    }
}
