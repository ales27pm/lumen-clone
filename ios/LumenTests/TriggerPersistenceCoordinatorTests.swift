import BackgroundTasks
import SwiftData
import XCTest
@testable import Lumen

@MainActor
final class TriggerPersistenceCoordinatorTests: XCTestCase {
    private final class Registrar: BackgroundTaskRegistering {
        func register(identifier: String, handler: @escaping (BGTask) -> Void) -> Bool { true }
    }

    private final class Submitter: BackgroundTaskSubmitting {
        private(set) var submittedIdentifiers: [String] = []
        private(set) var cancelAllCount = 0

        func submit(_ request: BGTaskRequest) throws {
            submittedIdentifiers.append(request.identifier)
        }

        func cancelAllTaskRequests() {
            cancelAllCount += 1
        }
    }

    private final class SafetyStore: TriggerExecutionSafetyStoring {
        var autonomousExecutionSuspensionTokens: Set<String> = []
    }

    private struct SyntheticSaveError: Error {
        let privateDetail: String
    }

    func testFailedMutationRestoresStateSuspendsExecutionAndRedactsFailure() {
        var value = "persisted"
        var didSchedule = false
        var didSuspend = false

        value = "pending"
        let outcome = TriggerPersistenceCoordinator.attempt(
            operation: .pause,
            save: {
                throw SyntheticSaveError(privateDetail: "/private/user/trigger.sqlite")
            },
            restore: { value = "persisted" },
            onSaved: { didSchedule = true },
            onFailure: {
                didSuspend = true
                return true
            }
        )

        guard case .failed(let failure) = outcome else {
            return XCTFail("Expected a typed persistence failure")
        }
        XCTAssertEqual(value, "persisted")
        XCTAssertFalse(didSchedule)
        XCTAssertTrue(didSuspend)
        XCTAssertEqual(failure.operation, .pause)
        XCTAssertTrue(failure.autonomousExecutionSuspended)
        XCTAssertTrue(failure.userMessage.contains("Retry"))
        XCTAssertTrue(failure.userMessage.contains("Automatic trigger execution is suspended"))
        XCTAssertFalse(failure.userMessage.contains("/private/user/trigger.sqlite"))
        XCTAssertFalse(failure.errorCode.contains("/"))
    }

    func testSuccessfulMutationSchedulesOnlyAfterSave() {
        var events: [String] = []

        let outcome = TriggerPersistenceCoordinator.attempt(
            operation: .create,
            save: { events.append("save") },
            restore: { events.append("restore") },
            onSaved: { events.append("schedule") },
            onFailure: {
                events.append("suspend")
                return true
            }
        )

        XCTAssertEqual(outcome, .saved)
        XCTAssertEqual(events, ["save", "schedule"])
    }

    func testFailedCreateRemovesOnlyPendingTriggerAndKeepsDraftDataAvailable() throws {
        let container = try ModelContainer(
            for: Trigger.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
        let context = ModelContext(container)
        let draftTitle = "Synthetic draft title"
        let pending = Trigger(title: draftTitle, prompt: "Synthetic prompt", scheduleType: .once)
        context.insert(pending)

        let outcome = TriggerPersistenceCoordinator.attempt(
            operation: .create,
            save: { throw SyntheticSaveError(privateDetail: "private create failure") },
            restore: { context.delete(pending) },
            onSaved: { XCTFail("A failed save must not report success") },
            onFailure: { true }
        )

        guard case .failed(let failure) = outcome else {
            return XCTFail("Expected create to fail")
        }
        XCTAssertTrue(try context.fetch(FetchDescriptor<Trigger>()).isEmpty)
        XCTAssertEqual(draftTitle, "Synthetic draft title")
        XCTAssertEqual(failure.operation, .create)
    }

    func testFailedDeleteInIsolatedContextPreservesOriginalTrigger() throws {
        let container = try ModelContainer(
            for: Trigger.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
        let context = ModelContext(container)
        let trigger = Trigger(title: "Keep me", prompt: "Synthetic prompt", scheduleType: .daily)
        context.insert(trigger)
        try context.save()

        let outcome = TriggerPersistenceCoordinator.delete(
            triggerID: trigger.id,
            container: container,
            save: { _ in throw SyntheticSaveError(privateDetail: "private delete failure") },
            onSaved: { XCTFail("A failed save must not report success") },
            onFailure: { true }
        )

        guard case .failed(let failure) = outcome else {
            return XCTFail("Expected delete to fail")
        }
        let verificationContext = ModelContext(container)
        let restored = try verificationContext.fetch(FetchDescriptor<Trigger>())
        XCTAssertEqual(restored.map(\.id), [trigger.id])
        XCTAssertEqual(restored.first?.title, "Keep me")
        XCTAssertEqual(failure.operation, .delete)
    }

    func testSuccessfulDeleteUsesIsolatedContextAndPreservesUnrelatedViewContextWork() throws {
        let container = try ModelContainer(
            for: Trigger.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
        let viewContext = ModelContext(container)
        let deleted = Trigger(title: "Delete me", prompt: "Synthetic prompt", scheduleType: .once)
        let unrelated = Trigger(title: "Persisted title", prompt: "Unrelated prompt", scheduleType: .daily)
        viewContext.insert(deleted)
        viewContext.insert(unrelated)
        try viewContext.save()
        unrelated.title = "Unsaved view edit"

        let outcome = TriggerPersistenceCoordinator.delete(
            triggerID: deleted.id,
            container: container,
            onSaved: {},
            onFailure: { true }
        )

        XCTAssertEqual(outcome, .saved)
        XCTAssertEqual(unrelated.title, "Unsaved view edit")
        XCTAssertTrue(viewContext.hasChanges)

        let verificationContext = ModelContext(container)
        let persisted = try verificationContext.fetch(FetchDescriptor<Trigger>())
        XCTAssertFalse(persisted.contains { $0.id == deleted.id })
        XCTAssertEqual(persisted.first(where: { $0.id == unrelated.id })?.title, "Persisted title")
    }

    func testFailedPostRunSavePersistsInterlockUntilMatchingRetrySucceeds() {
        let submitter = Submitter()
        let safetyStore = SafetyStore()
        let scheduler = TriggerScheduler(
            registrar: Registrar(),
            submitter: submitter,
            executionSafetyStore: safetyStore
        )
        let triggerID = UUID()
        let expectedToken = TriggerScheduler.persistenceSafetyToken(operation: .run, triggerID: triggerID)

        let failure = scheduler.persistRunState(triggerID: triggerID) {
            throw SyntheticSaveError(privateDetail: "/private/user/post-run.sqlite")
        }

        XCTAssertEqual(failure?.operation, .run)
        XCTAssertTrue(failure?.autonomousExecutionSuspended == true)
        XCTAssertFalse(failure?.userMessage.contains("/private/user/post-run.sqlite") == true)
        XCTAssertEqual(safetyStore.autonomousExecutionSuspensionTokens, [expectedToken])
        XCTAssertEqual(submitter.cancelAllCount, 1)
        XCTAssertFalse(scheduler.scheduleBackgroundRefresh())

        let retryFailure = scheduler.persistRunState(triggerID: triggerID) {}

        XCTAssertNil(retryFailure)
        XCTAssertTrue(safetyStore.autonomousExecutionSuspensionTokens.isEmpty)
    }

    func testSchedulerSafetyInterlockPersistsPolicyAndBlocksSubmission() {
        let submitter = Submitter()
        let safetyStore = SafetyStore()
        let scheduler = TriggerScheduler(
            registrar: Registrar(),
            submitter: submitter,
            executionSafetyStore: safetyStore
        )

        let triggerID = UUID()
        let otherTriggerID = UUID()
        let safetyToken = TriggerScheduler.persistenceSafetyToken(operation: .pause, triggerID: triggerID)
        let otherSafetyToken = TriggerScheduler.persistenceSafetyToken(operation: .update, triggerID: otherTriggerID)
        scheduler.suspendAutonomousExecutionAfterPersistenceFailure(token: safetyToken)
        scheduler.suspendAutonomousExecutionAfterPersistenceFailure(token: otherSafetyToken)

        XCTAssertEqual(safetyStore.autonomousExecutionSuspensionTokens, [safetyToken, otherSafetyToken])
        XCTAssertTrue(scheduler.isAutonomousExecutionSuspended)
        XCTAssertEqual(submitter.cancelAllCount, 2)
        XCTAssertFalse(scheduler.scheduleBackgroundRefresh())
        XCTAssertTrue(submitter.submittedIdentifiers.isEmpty)

        scheduler.resumeAutonomousExecutionAfterSuccessfulPersistence(triggerID: UUID())

        XCTAssertTrue(scheduler.isAutonomousExecutionSuspended, "An unrelated successful save must not clear a failed pause")
        XCTAssertFalse(scheduler.scheduleBackgroundRefresh())

        scheduler.resumeAutonomousExecutionAfterSuccessfulPersistence(triggerID: triggerID)

        XCTAssertEqual(safetyStore.autonomousExecutionSuspensionTokens, [otherSafetyToken])
        XCTAssertTrue(scheduler.isAutonomousExecutionSuspended)
        scheduler.resumeAutonomousExecutionAfterSuccessfulPersistence(triggerID: otherTriggerID)

        XCTAssertTrue(safetyStore.autonomousExecutionSuspensionTokens.isEmpty)
        XCTAssertTrue(scheduler.scheduleBackgroundRefresh())
        XCTAssertEqual(submitter.submittedIdentifiers, [
            TriggerScheduler.refreshIdentifier,
            TriggerScheduler.processIdentifier,
        ])
    }

    func testSupersedingSuccessClearsEveryTokenForSameTriggerAndNeverAnotherTrigger() {
        let submitter = Submitter()
        let safetyStore = SafetyStore()
        let scheduler = TriggerScheduler(
            registrar: Registrar(),
            submitter: submitter,
            executionSafetyStore: safetyStore
        )
        let resolvedTriggerID = UUID()
        let otherTriggerID = UUID()
        let sameTriggerTokens: Set<String> = [
            TriggerScheduler.persistenceSafetyToken(operation: .pause, triggerID: resolvedTriggerID),
            TriggerScheduler.persistenceSafetyToken(operation: .update, triggerID: resolvedTriggerID),
            TriggerScheduler.persistenceSafetyToken(operation: .run, triggerID: resolvedTriggerID),
        ]
        let otherTriggerToken = TriggerScheduler.persistenceSafetyToken(operation: .delete, triggerID: otherTriggerID)
        safetyStore.autonomousExecutionSuspensionTokens = sameTriggerTokens.union([otherTriggerToken])

        scheduler.resumeAutonomousExecutionAfterSuccessfulPersistence(triggerID: resolvedTriggerID)

        XCTAssertEqual(safetyStore.autonomousExecutionSuspensionTokens, [otherTriggerToken])
        XCTAssertEqual(scheduler.autonomouslySuspendedTriggerIDs, [otherTriggerID])
    }

    func testRunTokenRecoveryPausesOnceClearsPresentAndAbsentTokensAndDoesNotRerun() throws {
        let container = try ModelContainer(
            for: Trigger.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
        let context = ModelContext(container)
        let trigger = Trigger(title: "Resolve without rerun", prompt: "Synthetic prompt", scheduleType: .daily)
        let originalLastRunAt = Date(timeIntervalSince1970: 1234)
        trigger.lastRunAt = originalLastRunAt
        trigger.lastResult = "Prior run result"
        trigger.isPaused = false
        trigger.nextFireAt = Date(timeIntervalSince1970: 9999)
        context.insert(trigger)
        try context.save()

        let absentTriggerID = UUID()
        let runToken = TriggerScheduler.persistenceSafetyToken(operation: .run, triggerID: trigger.id)
        let absentToken = TriggerScheduler.persistenceSafetyToken(operation: .delete, triggerID: absentTriggerID)
        let submitter = Submitter()
        let safetyStore = SafetyStore()
        safetyStore.autonomousExecutionSuspensionTokens = [runToken, absentToken]
        let scheduler = TriggerScheduler(
            registrar: Registrar(),
            submitter: submitter,
            executionSafetyStore: safetyStore
        )
        var saveCount = 0

        let outcome = scheduler.resolveAutonomousExecutionSuspension(container: container) { recoveryContext in
            saveCount += 1
            try recoveryContext.save()
        }

        XCTAssertEqual(outcome, .saved)
        XCTAssertEqual(saveCount, 1)
        let verificationContext = ModelContext(container)
        let recovered = try XCTUnwrap(try verificationContext.fetch(FetchDescriptor<Trigger>()).first)
        XCTAssertTrue(recovered.isPaused)
        XCTAssertNil(recovered.nextFireAt)
        XCTAssertEqual(recovered.lastRunAt, originalLastRunAt)
        XCTAssertEqual(recovered.lastResult, "Prior run result")
        XCTAssertTrue(safetyStore.autonomousExecutionSuspensionTokens.isEmpty)
        XCTAssertEqual(submitter.submittedIdentifiers, [
            TriggerScheduler.refreshIdentifier,
            TriggerScheduler.processIdentifier,
        ])
    }

    func testRecoverySaveFailureRestoresTriggersKeepsAllTokensAndDoesNotSchedule() throws {
        let container = try ModelContainer(
            for: Trigger.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
        let context = ModelContext(container)
        let trigger = Trigger(title: "Keep active", prompt: "Synthetic prompt", scheduleType: .daily)
        let originalNextFireAt = Date(timeIntervalSince1970: 9999)
        trigger.isPaused = false
        trigger.nextFireAt = originalNextFireAt
        context.insert(trigger)
        try context.save()

        let absentTriggerID = UUID()
        let runToken = TriggerScheduler.persistenceSafetyToken(operation: .run, triggerID: trigger.id)
        let absentToken = TriggerScheduler.persistenceSafetyToken(operation: .delete, triggerID: absentTriggerID)
        let expectedTokens: Set<String> = [runToken, absentToken]
        let submitter = Submitter()
        let safetyStore = SafetyStore()
        safetyStore.autonomousExecutionSuspensionTokens = expectedTokens
        let scheduler = TriggerScheduler(
            registrar: Registrar(),
            submitter: submitter,
            executionSafetyStore: safetyStore
        )
        var saveCount = 0

        let outcome = scheduler.resolveAutonomousExecutionSuspension(container: container) { _ in
            saveCount += 1
            throw SyntheticSaveError(privateDetail: "private recovery failure")
        }

        guard case .failed(let failure) = outcome else {
            return XCTFail("Expected recovery persistence failure")
        }
        XCTAssertEqual(failure.operation, .recovery)
        XCTAssertEqual(saveCount, 1)
        XCTAssertFalse(trigger.isPaused)
        XCTAssertEqual(trigger.nextFireAt, originalNextFireAt)
        let verificationContext = ModelContext(container)
        let persisted = try XCTUnwrap(try verificationContext.fetch(FetchDescriptor<Trigger>()).first)
        XCTAssertFalse(persisted.isPaused)
        XCTAssertEqual(persisted.nextFireAt, originalNextFireAt)
        XCTAssertEqual(safetyStore.autonomousExecutionSuspensionTokens, expectedTokens)
        XCTAssertTrue(submitter.submittedIdentifiers.isEmpty)
    }

    func testSuccessfulRecoveryUsesIsolatedContextAndPreservesUnrelatedViewContextWork() throws {
        let container = try ModelContainer(
            for: Trigger.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
        let viewContext = ModelContext(container)
        let affected = Trigger(title: "Pause me", prompt: "Affected prompt", scheduleType: .daily)
        affected.isPaused = false
        affected.nextFireAt = Date(timeIntervalSince1970: 9999)
        let unrelated = Trigger(title: "Persisted title", prompt: "Unrelated prompt", scheduleType: .daily)
        viewContext.insert(affected)
        viewContext.insert(unrelated)
        try viewContext.save()
        unrelated.title = "Unsaved view edit"

        let token = TriggerScheduler.persistenceSafetyToken(operation: .pause, triggerID: affected.id)
        let safetyStore = SafetyStore()
        safetyStore.autonomousExecutionSuspensionTokens = [token]
        let scheduler = TriggerScheduler(
            registrar: Registrar(),
            submitter: Submitter(),
            executionSafetyStore: safetyStore
        )
        var saveCount = 0

        let outcome = scheduler.resolveAutonomousExecutionSuspension(container: container) { recoveryContext in
            saveCount += 1
            try recoveryContext.save()
        }

        XCTAssertEqual(outcome, .saved)
        XCTAssertEqual(saveCount, 1)
        XCTAssertEqual(unrelated.title, "Unsaved view edit")
        XCTAssertTrue(viewContext.hasChanges)
        XCTAssertTrue(safetyStore.autonomousExecutionSuspensionTokens.isEmpty)

        let verificationContext = ModelContext(container)
        let persisted = try verificationContext.fetch(FetchDescriptor<Trigger>())
        let recovered = try XCTUnwrap(persisted.first(where: { $0.id == affected.id }))
        XCTAssertTrue(recovered.isPaused)
        XCTAssertNil(recovered.nextFireAt)
        XCTAssertEqual(persisted.first(where: { $0.id == unrelated.id })?.title, "Persisted title")
    }

    func testUserDefaultsSafetyInterlockSurvivesStoreRecreation() throws {
        let suiteName = "TriggerPersistenceCoordinatorTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let token = TriggerScheduler.persistenceSafetyToken(operation: .pause, triggerID: UUID())

        let firstStore = UserDefaultsTriggerExecutionSafetyStore(defaults: defaults)
        firstStore.autonomousExecutionSuspensionTokens = [token]

        let recreatedStore = UserDefaultsTriggerExecutionSafetyStore(defaults: defaults)
        XCTAssertEqual(recreatedStore.autonomousExecutionSuspensionTokens, [token])
    }

    func testTriggersViewHasNoSilentSaveOrUnconditionalDismissPath() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let triggersViewURL = repoRoot.appendingPathComponent("ios/Lumen/Views/TriggersView.swift")
        guard FileManager.default.fileExists(atPath: triggersViewURL.path) else {
            throw XCTSkip("Source-layout guard requires a test host that can read the repository checkout.")
        }
        let source = try String(
            contentsOf: triggersViewURL,
            encoding: .utf8
        )

        XCTAssertFalse(source.contains("try? modelContext.save()"))
        XCTAssertTrue(source.contains("TriggerPersistenceCoordinator.attempt"))
        XCTAssertTrue(source.contains("case .saved:"))
        XCTAssertTrue(source.contains("case .failed(let failure):"))
        XCTAssertTrue(source.contains("suspendAutonomousExecutionAfterPersistenceFailure"))
        XCTAssertTrue(source.contains("runTriggerWithPersistenceOutcome"))
        XCTAssertTrue(source.contains("case .persistRun"))
        XCTAssertTrue(source.contains("successfullyDeletedTriggerIDs"))
        XCTAssertTrue(source.contains("visibleTriggers"))
        XCTAssertTrue(source.contains("Button(\"Resolve safely\")"))
        XCTAssertTrue(source.contains("triggers.resolvePersistenceSafetyInterlock"))
        XCTAssertTrue(source.contains("resolveAutonomousExecutionSuspension"))

        let schedulerSource = try String(
            contentsOf: repoRoot.appendingPathComponent("ios/Lumen/Services/TriggerScheduler.swift"),
            encoding: .utf8
        )
        XCTAssertTrue(schedulerSource.contains("case .deferred(let issue), .cancelled(let issue):"))
        XCTAssertTrue(schedulerSource.contains("case .persistenceFailed(let failure):"))
        XCTAssertTrue(schedulerSource.contains("message: failure.userMessage"))

        let orchestratorSource = try String(
            contentsOf: repoRoot.appendingPathComponent("ios/Lumen/Background/BackgroundOrchestrator.swift"),
            encoding: .utf8
        )
        XCTAssertTrue(orchestratorSource.contains("guard scanOutcome.backgroundTaskSucceeded else { return false }"))
    }
}
