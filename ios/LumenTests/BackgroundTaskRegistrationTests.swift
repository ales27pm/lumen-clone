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

    func testContinuedRegistrationUsesConcreteIdentifierAndRetriesFailure() throws {
        guard #available(iOS 26.0, *) else { throw XCTSkip("BGContinuedProcessingTask requires iOS 26") }
        let identifier = TriggerScheduler.continuedProcessingIdentifier(for: "test-token")
        let registrar = Registrar(results: [identifier: [false, true]])
        let coordinator = BackgroundContinuedProcessingCoordinator(registrar: registrar)
        coordinator.markApplicationLaunchCompleted()

        let first = coordinator.registerHandler(identifier: identifier)
        XCTAssertFalse(first.succeeded)
        XCTAssertFalse(first.beforeApplicationLaunchCompletion)

        let retry = coordinator.registerHandler(identifier: identifier)
        XCTAssertTrue(retry.succeeded)
        XCTAssertFalse(retry.beforeApplicationLaunchCompletion)
        XCTAssertFalse(identifier.contains("*"))
        XCTAssertEqual(registrar.registeredIdentifiers, [identifier, identifier])
    }

    func testContinuedRegistrationRejectsWildcardPatternAsAHandlerIdentifier() throws {
        guard #available(iOS 26.0, *) else { throw XCTSkip("BGContinuedProcessingTask requires iOS 26") }
        let registrar = Registrar(results: [:])
        let coordinator = BackgroundContinuedProcessingCoordinator(registrar: registrar)

        let outcome = coordinator.registerHandler(identifier: TriggerScheduler.continuedProcessingRegistrationIdentifier)

        XCTAssertFalse(outcome.succeeded)
        XCTAssertEqual(outcome.errorDomain, "BGTaskScheduler.invalidContinuedProcessingIdentifier")
        XCTAssertTrue(registrar.registeredIdentifiers.isEmpty)
    }

    func testSystemContinuedRegistrationMatchesAdvertisedWildcard() throws {
        guard #available(iOS 26.0, *) else { throw XCTSkip("BGContinuedProcessingTask requires iOS 26") }
        let coordinator = BackgroundContinuedProcessingCoordinator(registrar: SystemBackgroundTaskRegistrar.shared)
        coordinator.markApplicationLaunchCompleted()
        let identifier = TriggerScheduler.continuedProcessingIdentifier(for: UUID().uuidString)

        let outcome = coordinator.registerHandler(identifier: identifier)

        XCTAssertTrue(outcome.succeeded)
        XCTAssertFalse(outcome.beforeApplicationLaunchCompletion)
        XCTAssertNil(outcome.errorDomain)
        XCTAssertNil(outcome.errorCode)
    }
}
