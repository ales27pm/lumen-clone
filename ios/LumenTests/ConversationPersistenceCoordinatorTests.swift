import XCTest
import SwiftUI
@testable import Lumen

@MainActor
final class ConversationPersistenceCoordinatorTests: XCTestCase {
    override func setUp() async throws {
        try await super.setUp()
        #if DEBUG
        DeferredMaintenanceQueue.shared.resetForTesting()
        #endif
    }

    override func tearDown() async throws {
        #if DEBUG
        DeferredMaintenanceQueue.shared.resetForTesting()
        #endif
        try await super.tearDown()
    }

    func testSuccessfulSaveRecordsConversationBudgetAfterSave() {
        let budget = DiskWriteBudget(
            oneMinuteLimit: 100_000,
            fifteenMinuteLimit: 100_000,
            dayLimit: 100_000
        )
        var didSave = false

        let outcome = ConversationPersistenceCoordinator.attemptSave(
            estimatedBytes: 8_192,
            operation: "chat.conversation.save",
            budget: budget
        ) {
            didSave = true
        }

        XCTAssertTrue(didSave)
        XCTAssertEqual(outcome, .saved)
        XCTAssertEqual(budget.snapshot().bytesByCategory24Hours[.conversation], 8_192)
    }

    func testFailedSaveDoesNotRecordBudgetAndEmitsOnlyRedactedDiagnostics() throws {
        let privateDetail = "/private/var/mobile/secret-chat.sqlite user@example.com"
        let budget = DiskWriteBudget(
            oneMinuteLimit: 100_000,
            fifteenMinuteLimit: 100_000,
            dayLimit: 100_000
        )
        let signals = ConversationPersistenceSignalCapture()
        defer { signals.stop() }

        let outcome = ConversationPersistenceCoordinator.attemptSave(
            estimatedBytes: 8_192,
            operation: "chat.conversation.save",
            budget: budget
        ) {
            throw SyntheticConversationSaveError(privateDetail: privateDetail)
        }

        guard case .failed(let failure) = outcome else {
            return XCTFail("Expected a typed persistence failure")
        }
        XCTAssertEqual(failure.errorCode, "syntheticconversationsaveerror")
        XCTAssertEqual(failure.estimatedBytes, 8_192)
        XCTAssertFalse(failure.userMessage.contains(privateDetail))
        XCTAssertEqual(budget.snapshot().bytesByCategory24Hours[.conversation], 0)

        let signal = try XCTUnwrap(signals.signals.last)
        XCTAssertEqual(signal.kind, .fallbackUsed)
        XCTAssertEqual(signal.values["source"], "conversation-persistence")
        XCTAssertEqual(signal.values["errorcode"], "syntheticconversationsaveerror")
        XCTAssertEqual(signal.values["fallbackbehavior"], "retain-dirty-context")
        XCTAssertFalse(signal.values.values.contains { $0.contains(privateDetail) })
        XCTAssertFalse(signal.values.values.contains { $0.contains("user@example.com") })
    }

    func testDeferredSaveFailureDoesNotRecordBudgetAndNotifiesCaller() async {
        #if DEBUG
        let budget = DiskWriteBudget(
            oneMinuteLimit: 100_000,
            fifteenMinuteLimit: 100_000,
            dayLimit: 100_000
        )
        let queue = DeferredMaintenanceQueue.shared
        queue.updateScenePhase(.active)
        queue.forceForegroundGraceElapsedForTesting()
        let failureReported = expectation(description: "deferred failure reported")
        var capturedFailure: ConversationPersistenceFailure?

        ConversationPersistenceCoordinator.enqueueDeferredSave(
            estimatedBytes: 4_096,
            operation: "voice.conversation.save",
            deferredKey: "test-voice-conversation-save",
            deferredCategory: .conversation,
            budget: budget,
            queue: queue,
            save: {
                throw SyntheticConversationSaveError(privateDetail: "raw private deferred error")
            }
        ) { failure in
            capturedFailure = failure
            failureReported.fulfill()
        }

        await fulfillment(of: [failureReported], timeout: 1)
        XCTAssertEqual(capturedFailure?.errorCode, "syntheticconversationsaveerror")
        XCTAssertEqual(budget.snapshot().bytesByCategory24Hours[.conversation], 0)
        #endif
    }

    func testChatAndVoiceViewsDoNotSilentlyDiscardModelContextSaveErrors() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let paths = [
            "ios/Lumen/Views/ChatView.swift",
            "ios/Lumen/Views/VoiceModeView.swift"
        ]

        for path in paths {
            let source = try String(
                contentsOf: repoRoot.appendingPathComponent(path),
                encoding: .utf8
            )
            XCTAssertFalse(source.contains("try? modelContext.save()"), "\(path) must not discard SwiftData save errors")
            XCTAssertTrue(source.contains("ConversationPersistenceCoordinator"), "\(path) must use the shared persistence path")
        }
    }
}

private struct SyntheticConversationSaveError: Error {
    let privateDetail: String
}

private final class ConversationPersistenceSignalCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var observerID: UUID?
    private var storedSignals: [PersistentRuntimeDiagnosticSignal] = []

    var signals: [PersistentRuntimeDiagnosticSignal] {
        lock.lock()
        defer { lock.unlock() }
        return storedSignals
    }

    init() {
        observerID = PersistentRuntimeDiagnosticsObserver.shared.addObserver { [weak self] signal in
            guard signal.kind == .fallbackUsed,
                  signal.values["source"] == "conversation-persistence" else { return }
            self?.lock.lock()
            self?.storedSignals.append(signal)
            self?.lock.unlock()
        }
    }

    func stop() {
        guard let observerID else { return }
        PersistentRuntimeDiagnosticsObserver.shared.removeObserver(observerID)
        self.observerID = nil
    }
}
