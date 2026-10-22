import Foundation
import Observation
import Testing
@testable import Lumen

@MainActor
@Suite(.serialized)
struct ModelSettingsOperationOwnerTests {
    @Test func detachedPresentationCannotStartWork() {
        let bus = AppCancellationBus()
        let owner = ModelSettingsOperationOwner(cancellationBus: bus)
        var ran = false
        let work = owner.start(operation: { ran = true }, completion: { _ in Issue.record("Detached UI received a result") })
        #expect(work == nil)
        #expect(!ran)
        #expect(!owner.isBusy)
        #expect(bus.activeRegistrationCount(category: .modelMaintenance) == 0)
    }

    @Test func successfulCompletionClearsOwnershipBeforeDelivery() async throws {
        let bus = AppCancellationBus()
        let configuration = ModelSettingsConfigurationState()
        let owner = ModelSettingsOperationOwner(cancellationBus: bus, configurationState: configuration)
        let previousRevision = configuration.revision
        owner.activate()
        var received: Int?
        let work = try #require(owner.start(operation: { 42 }, completion: { result in
            #expect(!owner.isBusy)
            #expect(bus.activeRegistrationCount(category: .modelMaintenance) == 0)
            #expect(configuration.revision != previousRevision)
            received = try? result.get()
        }))
        #expect(owner.isBusy)
        #expect(bus.activeRegistrationCount(category: .modelMaintenance) == 1)
        await work.value
        #expect(received == 42)
    }

    @Test func userCancellationKeepsAdmissionClosedUntilWorkerDrain() async throws {
        let bus = AppCancellationBus()
        let owner = ModelSettingsOperationOwner(cancellationBus: bus)
        let worker = SuspendedWorker()
        owner.activate()
        var cancelled = false
        let work = try #require(owner.start(operation: {
            await worker.run()
            return "late success from an uncooperative worker"
        }, completion: { result in
            switch result {
            case .success: Issue.record("Cancellation became a successful UI result")
            case .failure(let error): cancelled = error is CancellationError
            }
        }))
        await worker.waitUntilStarted()
        owner.cancel()
        #expect(owner.isBusy)
        #expect(bus.activeRegistrationCount(category: .modelMaintenance) == 1)
        #expect(owner.start(operation: {}, completion: { _ in }) == nil)
        await worker.release()
        await work.value
        #expect(cancelled)
        #expect(!owner.isBusy)
        #expect(bus.activeRegistrationCount(category: .modelMaintenance) == 0)
    }

    @Test func recreatedPresentationSeesARevisionAfterCancelledOldWorkerPublishes() async throws {
        let bus = AppCancellationBus()
        let configuration = ModelSettingsConfigurationState()
        let oldOwner = ModelSettingsOperationOwner(cancellationBus: bus, configurationState: configuration)
        let worker = SuspendedWorker()
        var authoritative = "old package"
        var obsoleteCompletions = 0
        oldOwner.activate()
        let old = try #require(oldOwner.start(operation: {
            await worker.run()
            // The store may have crossed its last cancellation check before
            // finishing its synchronous atomic publication.
            authoritative = "new package"
        }, completion: { _ in obsoleteCompletions += 1 }))
        await worker.waitUntilStarted()
        oldOwner.deactivate()

        // Navigation replaces the section, rather than reactivating oldOwner.
        let currentOwner = ModelSettingsOperationOwner(cancellationBus: bus, configurationState: configuration)
        currentOwner.activate()
        let snapshotRevision = currentOwner.configurationRevision
        var presented = authoritative
        let revisionChanges = AsyncStream<Bool>.makeStream()
        withObservationTracking {
            _ = currentOwner.configurationRevision
        } onChange: {
            revisionChanges.continuation.yield(true)
        }
        #expect(!currentOwner.isBusy)
        #expect(oldOwner.isBusy)
        #expect(configuration.revision == snapshotRevision)

        await worker.release()
        await old.value
        revisionChanges.continuation.finish()
        var changes = revisionChanges.stream.makeAsyncIterator()
        #expect(await changes.next() == true)
        #expect(obsoleteCompletions == 0)
        #expect(old.isCancelled)
        #expect(currentOwner.configurationRevision != snapshotRevision)
        #expect(currentOwner.configurationRevision == oldOwner.configurationRevision)
        if currentOwner.configurationRevision != snapshotRevision { presented = authoritative }
        #expect(presented == "new package")
        #expect(!oldOwner.isBusy)
        #expect(bus.activeRegistrationCount(category: .modelMaintenance) == 0)
    }

    @Test func failureInvalidatesConfigurationBeforeItsCurrentCompletion() async throws {
        let bus = AppCancellationBus()
        let configuration = ModelSettingsConfigurationState()
        let owner = ModelSettingsOperationOwner(cancellationBus: bus, configurationState: configuration)
        let previousRevision = configuration.revision
        owner.activate()
        var receivedFailure = false
        let work = try #require(owner.start(operation: { throw SyntheticFailure.expected }, completion: { result in
            #expect(configuration.revision != previousRevision)
            if case .failure(let error) = result { receivedFailure = error is SyntheticFailure }
        }))
        #expect(configuration.revision == previousRevision)
        await work.value
        #expect(receivedFailure)
    }

    @Test func cancellationBeforeExecutionStillInvalidatesConfigurationAfterDrain() async throws {
        let bus = AppCancellationBus()
        let configuration = ModelSettingsConfigurationState()
        let owner = ModelSettingsOperationOwner(cancellationBus: bus, configurationState: configuration)
        let previousRevision = configuration.revision
        owner.activate()
        var operationRan = false
        var receivedCancellation = false
        let work = try #require(owner.start(operation: { operationRan = true }, completion: { result in
            #expect(configuration.revision != previousRevision)
            if case .failure(let error) = result { receivedCancellation = error is CancellationError }
        }))
        owner.cancel()
        #expect(configuration.revision == previousRevision)
        #expect(owner.isBusy)
        await work.value
        #expect(!operationRan)
        #expect(receivedCancellation)
        #expect(!owner.isBusy)
        #expect(bus.activeRegistrationCount(category: .modelMaintenance) == 0)
    }

    @Test func targetedMemoryPressureCancellationDrainsTheOwnedOperation() async throws {
        let bus = AppCancellationBus()
        let owner = ModelSettingsOperationOwner(cancellationBus: bus)
        let worker = SuspendedWorker()
        owner.activate()
        var cancelled = false
        let work = try #require(owner.start(operation: { await worker.run() }, completion: { result in
            if case .failure(let error) = result { cancelled = error is CancellationError }
        }))
        await worker.waitUntilStarted()
        bus.cancel(.modelMaintenance)
        #expect(owner.isBusy)
        #expect(work.isCancelled)
        #expect(owner.start(operation: {}, completion: { _ in }) == nil)
        await worker.release()
        await work.value
        #expect(cancelled)
        #expect(!owner.isBusy)
        #expect(bus.activeRegistrationCount(category: .modelMaintenance) == 0)
    }

    @Test func sceneCancellationReachesModelMaintenance() async throws {
        let bus = AppCancellationBus()
        let owner = ModelSettingsOperationOwner(cancellationBus: bus)
        let worker = SuspendedWorker()
        owner.activate()
        let work = try #require(owner.start(operation: { await worker.run() }, completion: { _ in }))
        await worker.waitUntilStarted()
        #expect(AppCancellationCategory.modelMaintenance.isSceneSensitive)
        bus.cancelAllSceneSensitive()
        #expect(work.isCancelled)
        #expect(owner.isBusy)
        await worker.release()
        await work.value
        #expect(!owner.isBusy)
        #expect(bus.activeRegistrationCount(category: .modelMaintenance) == 0)
    }

    @Test func disappearanceAndReappearanceSuppressOldTerminalUpdates() async throws {
        let bus = AppCancellationBus()
        let owner = ModelSettingsOperationOwner(cancellationBus: bus)
        let worker = SuspendedWorker()
        owner.activate()
        var updates: [String] = []
        let old = try #require(owner.start(operation: {
            await worker.run()
            throw SyntheticFailure.expected
        }, completion: { _ in updates.append("stale") }))
        await worker.waitUntilStarted()
        owner.deactivate()
        owner.activate()
        #expect(owner.isBusy)
        #expect(owner.start(operation: {}, completion: { _ in }) == nil)
        await worker.release()
        await old.value
        #expect(updates.isEmpty)
        #expect(!owner.isBusy)
        let current = try #require(owner.start(operation: { "current" }, completion: { result in
            if case .success(let value) = result { updates.append(value) }
        }))
        await current.value
        #expect(updates == ["current"])
        #expect(bus.activeRegistrationCount(category: .modelMaintenance) == 0)
    }

    @Test func failureIsDeliveredOnceAndReleasesRegistration() async throws {
        let bus = AppCancellationBus()
        let owner = ModelSettingsOperationOwner(cancellationBus: bus)
        owner.activate()
        var failures = 0
        let work = try #require(owner.start(operation: { throw SyntheticFailure.expected }, completion: { result in
            if case .failure(let error) = result {
                #expect(error is SyntheticFailure)
                failures += 1
            }
        }))
        await work.value
        #expect(failures == 1)
        #expect(!owner.isBusy)
        #expect(bus.activeRegistrationCount(category: .modelMaintenance) == 0)
    }

    @Test func completedTaskDoesNotRetainItsOwner() async throws {
        let bus = AppCancellationBus()
        var owner: ModelSettingsOperationOwner? = .init(cancellationBus: bus)
        weak var released = owner
        owner?.activate()
        let work = try #require(owner?.start(operation: {}, completion: { _ in }))
        await work.value
        owner = nil
        #expect(released == nil)
    }

    private enum SyntheticFailure: Error { case expected }

    /// Intentionally ignores cancellation until released, so a task handle
    /// being cancelled cannot be confused with its worker having drained.
    private actor SuspendedWorker {
        private var started = false
        private var startedWaiters: [CheckedContinuation<Void, Never>] = []
        private var continuation: CheckedContinuation<Void, Never>?

        func run() async {
            started = true
            for waiter in startedWaiters { waiter.resume() }
            startedWaiters.removeAll()
            await withCheckedContinuation { continuation = $0 }
        }

        func waitUntilStarted() async {
            guard !started else { return }
            await withCheckedContinuation { startedWaiters.append($0) }
        }

        func release() {
            continuation?.resume()
            continuation = nil
        }
    }
}
