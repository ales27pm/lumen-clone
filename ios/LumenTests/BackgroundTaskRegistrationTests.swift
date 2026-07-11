import BackgroundTasks
import XCTest
@testable import Lumen

@MainActor
final class BackgroundTaskRegistrationTests: XCTestCase {
    private final class Registrar: BackgroundTaskRegistering {
        var results: [String: [Bool]]
        private(set) var registeredIdentifiers: [String] = []

        init(results: [String: [Bool]]) {
            self.results = results
        }

        func register(identifier: String, handler: @escaping (BGTask) -> Void) -> Bool {
            registeredIdentifiers.append(identifier)
            guard var queued = results[identifier], !queued.isEmpty else { return true }
            let result = queued.removeFirst()
            results[identifier] = queued
            return result
        }
    }

    func testClassicRegistrationsAreLaunchTimedIdempotentAndRetryFailures() {
        let registrar = Registrar(results: [
            TriggerScheduler.refreshIdentifier: [false, true],
            TriggerScheduler.processIdentifier: [true]
        ])
        let scheduler = TriggerScheduler(registrar: registrar)

        let first = scheduler.registerTasks(beforeApplicationLaunchCompletion: true)
        XCTAssertEqual(first.count, 2)
        XCTAssertTrue(first.allSatisfy(\.beforeApplicationLaunchCompletion))
        XCTAssertFalse(first.first(where: { $0.identifier == TriggerScheduler.refreshIdentifier })?.succeeded ?? true)
        XCTAssertTrue(first.first(where: { $0.identifier == TriggerScheduler.processIdentifier })?.succeeded ?? false)

        let retry = scheduler.registerTasks(beforeApplicationLaunchCompletion: true)
        XCTAssertEqual(retry.map(\.identifier), [TriggerScheduler.refreshIdentifier])
        XCTAssertTrue(retry[0].succeeded)
        XCTAssertTrue(scheduler.registerTasks(beforeApplicationLaunchCompletion: true).isEmpty)
    }

    func testContinuedRegistrationRecordsLaunchTimingAndRetriesFailure() throws {
        guard #available(iOS 26.0, *) else { throw XCTSkip("BGContinuedProcessingTask requires iOS 26") }
        let identifier = TriggerScheduler.continuedProcessingRegistrationIdentifier
        let registrar = Registrar(results: [identifier: [false, true]])
        let coordinator = BackgroundContinuedProcessingCoordinator(registrar: registrar)

        let first = coordinator.registerHandlerBeforeApplicationLaunchCompletion()
        XCTAssertFalse(first.succeeded)
        XCTAssertTrue(first.beforeApplicationLaunchCompletion)

        let retry = coordinator.registerHandlerBeforeApplicationLaunchCompletion()
        XCTAssertTrue(retry.succeeded)
        XCTAssertTrue(retry.beforeApplicationLaunchCompletion)
        XCTAssertEqual(registrar.registeredIdentifiers, [identifier, identifier])
    }
}
