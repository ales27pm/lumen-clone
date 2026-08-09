import BackgroundTasks
import SwiftData
import SwiftUI
import XCTest
@testable import Lumen

@MainActor
final class BackgroundTriggerOutcomeTests: XCTestCase {
    private final class Registrar: BackgroundTaskRegistering {
        func register(identifier: String, handler: @escaping (BGTask) -> Void) -> Bool { true }
    }

    private final class Submitter: BackgroundTaskSubmitting {
        func submit(_ request: BGTaskRequest) throws {}
        func cancelAllTaskRequests() {}
    }

    private final class SafetyStore: TriggerExecutionSafetyStoring {
        var autonomousExecutionSuspensionTokens: Set<String> = []
    }

    override func tearDown() async throws {
        ResourceBudgetGate.testSnapshotOverride = nil
        try await super.tearDown()
    }

    func testDeferredHeadlessRunDoesNotConsumeOneTimeTrigger() async throws {
        let (context, trigger) = try dueOneTimeTrigger()
        let originalNextFireAt = trigger.nextFireAt
        let issue = TriggerExecutionIssue(code: "background_model_not_loaded", message: "Model not loaded")
        let scheduler = makeScheduler { _, _, _, _ in
            .deferred(text: issue.message, code: issue.code)
        }

        let outcome = await scheduler.runTriggerWithPersistenceOutcome(
            trigger,
            context: context,
            settings: .loadFromDisk(),
            notify: false
        )

        XCTAssertEqual(outcome, .deferred(issue))
        XCTAssertFalse(trigger.isPaused)
        XCTAssertNil(trigger.lastRunAt)
        XCTAssertNil(trigger.lastResult)
        XCTAssertEqual(trigger.nextFireAt, originalNextFireAt)
    }

    func testFailedHeadlessRunDoesNotAdvanceRecurringTrigger() async throws {
        let container = try triggerContainer()
        let context = ModelContext(container)
        let trigger = Trigger(
            title: "Recurring",
            prompt: "Synthetic prompt",
            scheduleType: .interval,
            intervalSeconds: 900
        )
        let originalNextFireAt = Date(timeIntervalSince1970: 9_000)
        trigger.nextFireAt = originalNextFireAt
        context.insert(trigger)
        try context.save()
        let issue = TriggerExecutionIssue(
            category: .executionFailure,
            code: "agent_kernel_failed",
            message: "Kernel failed"
        )
        let scheduler = makeScheduler { _, _, _, _ in
            .failed(text: issue.message, code: issue.code)
        }

        let outcome = await scheduler.runTriggerWithPersistenceOutcome(
            trigger,
            context: context,
            settings: .loadFromDisk(),
            notify: false
        )

        XCTAssertEqual(outcome, .failed(issue))
        XCTAssertNil(trigger.lastRunAt)
        XCTAssertNil(trigger.lastResult)
        XCTAssertEqual(trigger.nextFireAt, originalNextFireAt)
    }

    func testCompletedHeadlessRunStillConsumesOneTimeTrigger() async throws {
        let (context, trigger) = try dueOneTimeTrigger()
        let scheduler = makeScheduler { _, _, _, _ in
            .completed(text: "Synthetic result", steps: [])
        }

        let outcome = await scheduler.runTriggerWithPersistenceOutcome(
            trigger,
            context: context,
            settings: .loadFromDisk(),
            notify: false
        )

        XCTAssertEqual(outcome, .completed("Synthetic result"))
        XCTAssertTrue(trigger.isPaused)
        XCTAssertNotNil(trigger.lastRunAt)
        XCTAssertEqual(trigger.lastResult, "Synthetic result")
        XCTAssertNil(trigger.nextFireAt)
    }

    func testDueTriggerDeferralMakesScanAndBackgroundCompletionFailClosed() async throws {
        let (context, trigger) = try dueOneTimeTrigger()
        let originalNextFireAt = trigger.nextFireAt
        let issue = TriggerExecutionIssue(code: "background_model_not_loaded", message: "Model not loaded")
        let scheduler = makeScheduler { _, _, _, _ in
            .deferred(text: issue.message, code: issue.code)
        }

        let outcome = await scheduler.fireDueTriggers(context: context, settings: .loadFromDisk())

        XCTAssertEqual(outcome, .deferred(issue))
        XCTAssertFalse(outcome.backgroundTaskSucceeded)
        XCTAssertFalse(trigger.isPaused)
        XCTAssertNil(trigger.lastRunAt)
        XCTAssertEqual(trigger.nextFireAt, originalNextFireAt)
    }

    func testScanAggregatesIssuesAndDoesNotStarveIndependentDueTriggers() async throws {
        let container = try triggerContainer()
        let context = ModelContext(container)
        let legacy = Trigger(
            title: "Legacy calendar",
            prompt: "legacy",
            scheduleType: .beforeNextEvent,
            beforeNextEventMinutes: 15
        )
        let modelDependent = dueOneTimeTrigger(in: context, title: "Model", prompt: "model")
        let failing = dueOneTimeTrigger(in: context, title: "Failure", prompt: "failure")
        let backgroundSafe = dueOneTimeTrigger(in: context, title: "Safe", prompt: "safe")
        context.insert(legacy)
        try context.save()

        var attemptedPrompts: [String] = []
        let scheduler = makeScheduler { prompt, _, _, _ in
            attemptedPrompts.append(prompt)
            switch prompt {
            case "model":
                return .deferred(text: "Model not loaded", code: "background_model_not_loaded")
            case "failure":
                return .failed(text: "Kernel failed", code: "agent_kernel_failed")
            case "safe":
                return .completed(text: "Safe result", steps: [])
            default:
                return .failed(text: "Unexpected prompt", code: "unexpected_prompt")
            }
        }

        let outcome = await scheduler.fireDueTriggers(context: context, settings: .loadFromDisk())

        XCTAssertEqual(Set(attemptedPrompts), Set(["model", "failure", "safe"]))
        XCTAssertEqual(outcome.severity, .failed)
        XCTAssertEqual(outcome.completedCount, 1)
        XCTAssertEqual(outcome.deferredIssues.map(\.code), ["background_model_not_loaded"])
        XCTAssertEqual(
            Set(outcome.failedIssues.map(\.category)),
            Set([.executionFailure, .unsupportedSchedule])
        )
        XCTAssertFalse(outcome.backgroundTaskSucceeded)

        XCTAssertTrue(legacy.isPaused)
        XCTAssertNil(legacy.nextFireAt)
        XCTAssertFalse(modelDependent.isPaused)
        XCTAssertNil(modelDependent.lastRunAt)
        XCTAssertFalse(failing.isPaused)
        XCTAssertNil(failing.lastRunAt)
        XCTAssertTrue(backgroundSafe.isPaused)
        XCTAssertEqual(backgroundSafe.lastResult, "Safe result")
    }

    func testPermanentInteractiveBlockPausesTriggerWithoutReplayOrCompletion() async throws {
        let (context, trigger) = try dueOneTimeTrigger()
        var invocationCount = 0
        let scheduler = makeScheduler { _, _, _, _ in
            invocationCount += 1
            return .blocked(
                text: "User approval required",
                code: "background_tool_requires_user_interaction"
            )
        }

        let first = await scheduler.fireDueTriggers(context: context, settings: .loadFromDisk())

        XCTAssertEqual(first.severity, .failed)
        XCTAssertEqual(first.completedCount, 0)
        XCTAssertEqual(first.failedIssues.first?.category, .userInteractionRequired)
        XCTAssertEqual(invocationCount, 1)
        XCTAssertTrue(trigger.isPaused)
        XCTAssertNil(trigger.nextFireAt)
        XCTAssertNil(trigger.lastRunAt)
        XCTAssertNil(trigger.lastResult)

        let second = await scheduler.fireDueTriggers(context: context, settings: .loadFromDisk())

        XCTAssertEqual(second, .completed)
        XCTAssertEqual(invocationCount, 1)
        XCTAssertNil(trigger.lastRunAt)
        XCTAssertNil(trigger.lastResult)
    }

    func testToolResultStatusesDistinguishPermanentInteractionFromTransientUnavailable() {
        let denied = HeadlessAgentKernelRunner.nonSuccessResult(for: toolResult(status: .denied))
        let approval = HeadlessAgentKernelRunner.nonSuccessResult(for: toolResult(status: .requiresApproval))
        let unavailable = HeadlessAgentKernelRunner.nonSuccessResult(for: toolResult(status: .unavailable))
        let failed = HeadlessAgentKernelRunner.nonSuccessResult(for: toolResult(status: .failed))

        XCTAssertEqual(denied?.status, .blocked)
        XCTAssertEqual(denied?.issueCategory, .userInteractionRequired)
        XCTAssertEqual(approval?.status, .blocked)
        XCTAssertEqual(approval?.issueCategory, .userInteractionRequired)
        XCTAssertEqual(unavailable?.status, .deferred)
        XCTAssertEqual(unavailable?.issueCategory, .transientUnavailable)
        XCTAssertEqual(failed?.status, .failed)
        XCTAssertEqual(failed?.issueCategory, .executionFailure)
        XCTAssertNil(HeadlessAgentKernelRunner.nonSuccessResult(for: toolResult(status: .success)))
    }

    func testBackgroundAssessmentStatusesMapToTypedNonCompletion() {
        XCTAssertNil(HeadlessAgentKernelRunner.nonRunnableBackgroundAssessmentResult(for: assessment(status: .runnable)))
        XCTAssertNil(HeadlessAgentKernelRunner.nonRunnableBackgroundAssessmentResult(for: assessment(status: .notToolBacked)))

        let clarification = HeadlessAgentKernelRunner.nonRunnableBackgroundAssessmentResult(
            for: assessment(status: .clarificationRequired)
        )
        let unsafe = HeadlessAgentKernelRunner.nonRunnableBackgroundAssessmentResult(
            for: assessment(status: .noBackgroundSafeRoutedTools)
        )
        let policyBlocked = HeadlessAgentKernelRunner.nonRunnableBackgroundAssessmentResult(
            for: assessment(status: .blockedByCurrentPolicy)
        )
        let missingRoute = HeadlessAgentKernelRunner.nonRunnableBackgroundAssessmentResult(
            for: assessment(status: .noRoutedTools)
        )
        let missingMapping = HeadlessAgentKernelRunner.nonRunnableBackgroundAssessmentResult(
            for: assessment(status: .toolMappingUnavailable)
        )

        XCTAssertEqual(clarification?.status, .blocked)
        XCTAssertEqual(clarification?.code, "background_clarification_required")
        XCTAssertEqual(unsafe?.status, .blocked)
        XCTAssertEqual(unsafe?.code, "background_tool_not_safe")
        XCTAssertEqual(policyBlocked?.status, .blocked)
        XCTAssertEqual(policyBlocked?.code, "background_tool_blocked_by_policy")
        XCTAssertEqual(missingRoute?.status, .failed)
        XCTAssertEqual(missingRoute?.code, "background_tool_route_missing")
        XCTAssertEqual(missingMapping?.status, .failed)
        XCTAssertEqual(missingMapping?.code, "background_tool_mapping_unavailable")
    }

    func testBackgroundUnsafeAssessmentPausesRecurringTriggerWithoutRecordingCompletion() async throws {
        let container = try triggerContainer()
        let context = ModelContext(container)
        let trigger = Trigger(
            title: "Recurring unsafe tool",
            prompt: "search the web for background task docs",
            scheduleType: .interval,
            intervalSeconds: 900
        )
        trigger.nextFireAt = Date(timeIntervalSince1970: 9_000)
        context.insert(trigger)
        try context.save()
        let headlessResult = try XCTUnwrap(
            HeadlessAgentKernelRunner.nonRunnableBackgroundAssessmentResult(
                for: assessment(status: .noBackgroundSafeRoutedTools)
            )
        )
        let scheduler = makeScheduler { _, _, _, _ in headlessResult }

        let outcome = await scheduler.runTriggerWithPersistenceOutcome(
            trigger,
            context: context,
            settings: .loadFromDisk(),
            notify: false
        )

        XCTAssertEqual(
            outcome,
            .blocked(.init(
                category: .userInteractionRequired,
                code: "background_tool_not_safe",
                message: headlessResult.text
            ))
        )
        XCTAssertTrue(trigger.isPaused)
        XCTAssertNil(trigger.nextFireAt)
        XCTAssertNil(trigger.lastRunAt)
        XCTAssertNil(trigger.lastResult)
    }

    func testOrchestratorRecordsDeferredScanAsFailure() async throws {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: false,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        let issue = TriggerExecutionIssue(code: "background_model_not_loaded", message: "Model not loaded")
        let expected = TriggerScanOutcome.deferred(issue)
        let metricsURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("background-trigger-outcome-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: metricsURL) }
        let store = RuntimeMetricsStore(fileURL: metricsURL)
        let orchestrator = BackgroundOrchestrator(
            metricsStore: store,
            triggerScan: { _, _ in expected }
        )
        let context = ModelContext(try triggerContainer())

        let outcome = try await orchestrator.runTriggerScan(context: context)
        let metric = try await store.recentMetrics(limit: 1).last

        XCTAssertEqual(outcome, expected)
        XCTAssertFalse(outcome.backgroundTaskSucceeded)
        XCTAssertEqual(metric?.success, false)
        XCTAssertEqual(metric?.errorCode, issue.code)
        XCTAssertTrue(metric?.policySummary.contains("deferred") == true)
    }

    func testBeforeNextEventCreationIsTypedUnavailableAndHasNoSideEffect() async throws {
        XCTAssertEqual(
            TriggerScheduleType.beforeNextEvent.creationUnavailability,
            .beforeNextEventRequiresForegroundCalendarIntegration
        )
        XCTAssertFalse(TriggerScheduleType.creatableCases.contains(.beforeNextEvent))

        let parsed = TriggerTools.createArguments(from: [
            "title": "Calendar reminder",
            "prompt": "Synthetic prompt",
            "schedule": "before_next_event",
            "beforeMinutes": "15"
        ])
        XCTAssertEqual(
            failure(parsed),
            .scheduleUnavailable(.beforeNextEventRequiresForegroundCalendarIntegration)
        )

        let container = try triggerContainer()
        let savedContainer = SharedContainer.shared
        SharedContainer.shared = container
        defer { SharedContainer.shared = savedContainer }

        let response = await TriggerTools.create(args: [
            "title": "Calendar reminder",
            "prompt": "Synthetic prompt",
            "schedule": "before_next_event",
            "beforeMinutes": "15"
        ])

        XCTAssertEqual(
            response,
            TriggerTools.invalidCreateArgumentsMessage(.scheduleUnavailable(
                .beforeNextEventRequiresForegroundCalendarIntegration
            ))
        )
        XCTAssertTrue(try ModelContext(container).fetch(FetchDescriptor<Trigger>()).isEmpty)
    }

    private func makeScheduler(
        headlessRun: @escaping @MainActor (String, SettingsSnapshot, ModelContext, Int) async -> HeadlessAgentRunResult
    ) -> TriggerScheduler {
        TriggerScheduler(
            registrar: Registrar(),
            submitter: Submitter(),
            executionSafetyStore: SafetyStore(),
            headlessRun: headlessRun
        )
    }

    private func dueOneTimeTrigger() throws -> (ModelContext, Trigger) {
        let container = try triggerContainer()
        let context = ModelContext(container)
        let trigger = dueOneTimeTrigger(in: context, title: "One time", prompt: "Synthetic prompt")
        try context.save()
        return (context, trigger)
    }

    private func dueOneTimeTrigger(
        in context: ModelContext,
        title: String,
        prompt: String
    ) -> Trigger {
        let trigger = Trigger(
            title: title,
            prompt: prompt,
            scheduleType: .once,
            fireDate: Date().addingTimeInterval(-60)
        )
        trigger.nextFireAt = Date().addingTimeInterval(-30)
        context.insert(trigger)
        return trigger
    }

    private func toolResult(status: ToolResultStatus) -> ToolResult {
        ToolResult(
            invocationID: UUID(),
            status: status,
            displayText: "Synthetic tool result",
            modelText: "Synthetic tool result",
            structuredPayload: nil,
            privacyLevel: .low,
            metricsSummary: "synthetic",
            errorCode: nil
        )
    }

    private func assessment(
        status: BackgroundToolExecutionAssessment.Status
    ) -> BackgroundToolExecutionAssessment {
        BackgroundToolExecutionAssessment(
            status: status,
            intent: .webSearch,
            routedToolIDs: ["web.search"],
            backgroundCapableToolIDs: [],
            policyAllowedToolIDs: [],
            availableTools: []
        )
    }

    private func triggerContainer() throws -> ModelContainer {
        try ModelContainer(
            for: Trigger.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
    }

    private func failure(
        _ result: Result<TriggerCreateArguments, TriggerCreateArgumentError>
    ) -> TriggerCreateArgumentError? {
        guard case .failure(let error) = result else { return nil }
        return error
    }
}
