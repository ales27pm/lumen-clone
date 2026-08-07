import XCTest
@testable import Lumen

@MainActor
final class ToolApprovalPersistenceCoordinatorTests: XCTestCase {
    func testRunningClaimIsPersistedBeforeExecutionAndConsumesApprovalOnce() async throws {
        let fixture = makeFixture()
        var savedStatuses: [ToolStatus?] = []

        let claim = ToolApprovalPersistenceCoordinator.claimForExecution(
            message: fixture.message,
            toolID: fixture.pending.toolID,
            payloadArguments: fixture.payload,
            policyAllowed: true,
            policyDeniedResult: "blocked",
            save: {
                savedStatuses.append(fixture.message.status)
            }
        )

        guard case .claimed(let claimed) = claim else {
            return XCTFail("Expected a durable execution claim")
        }
        XCTAssertEqual(savedStatuses, [.running])
        XCTAssertEqual(claimed.pendingActionID, fixture.pending.pendingActionID)
        XCTAssertNil(ToolApprovalQueue.shared.resolve(fixture.pending.pendingActionID))

        var didExecute = false
        let executed: Bool? = await ToolApprovalPersistenceCoordinator.executeIfClaimed(claim) { pending in
            didExecute = true
            return pending.arguments == fixture.pending.arguments
        }
        XCTAssertTrue(didExecute)
        XCTAssertEqual(executed, true)
    }

    func testRunningClaimSaveFailureFailsClosedWithoutExecuting() async throws {
        let fixture = makeFixture()
        let privateDetail = "/private/var/mobile/approval.sqlite user@example.com"
        let signals = ToolApprovalPersistenceSignalCapture()
        defer { signals.stop() }
        var saveAttempts = 0

        let claim = ToolApprovalPersistenceCoordinator.claimForExecution(
            message: fixture.message,
            toolID: fixture.pending.toolID,
            payloadArguments: fixture.payload,
            policyAllowed: true,
            policyDeniedResult: "blocked",
            save: {
                saveAttempts += 1
                throw SyntheticToolApprovalSaveError(privateDetail: privateDetail)
            }
        )

        guard case .blocked(let failure) = claim else {
            return XCTFail("Expected the action to be blocked by persistence")
        }
        XCTAssertEqual(saveAttempts, 2, "The second attempt persists the fail-closed marker")
        XCTAssertEqual(failure.operation, .runningClaim)
        XCTAssertEqual(failure.errorCode, "synthetictoolapprovalsaveerror")
        XCTAssertEqual(fixture.message.status, .failed)
        XCTAssertTrue(fixture.message.toolResult?.contains("was not run") == true)
        XCTAssertFalse(fixture.message.toolResult?.contains(privateDetail) == true)
        XCTAssertNil(ToolApprovalQueue.shared.resolve(fixture.pending.pendingActionID))

        var didExecute = false
        let executed: Bool? = await ToolApprovalPersistenceCoordinator.executeIfClaimed(claim) { _ in
            didExecute = true
            return true
        }
        XCTAssertFalse(didExecute)
        XCTAssertNil(executed)

        XCTAssertTrue(signals.signals.contains { signal in
            signal.values["operation"] == ToolApprovalPersistenceOperation.runningClaim.rawValue
                && signal.values["errorcode"] == "synthetictoolapprovalsaveerror"
                && !signal.values.values.contains(where: { $0.contains(privateDetail) })
                && !signal.values.values.contains(where: { $0.contains("user@example.com") })
        })
    }

    func testInvalidAndPolicyBlockedApprovalsPersistDeniedAndCannotReplay() {
        let invalidMessage = ChatMessage(
            role: .tool,
            content: "subject: hello",
            toolName: "mail.draft",
            toolStatus: .pendingApproval
        )
        var invalidSavedStatus: ToolStatus?

        let invalid = ToolApprovalPersistenceCoordinator.claimForExecution(
            message: invalidMessage,
            toolID: "mail.draft",
            payloadArguments: ["subject": "hello"],
            policyAllowed: true,
            policyDeniedResult: "blocked",
            save: { invalidSavedStatus = invalidMessage.status }
        )

        XCTAssertEqual(invalid, .rejected)
        XCTAssertEqual(invalidSavedStatus, .denied)
        XCTAssertEqual(invalidMessage.status, .denied)
        XCTAssertTrue(invalidMessage.toolResult?.contains("cannot be verified") == true)

        let blockedFixture = makeFixture()
        var blockedSavedStatus: ToolStatus?
        let blocked = ToolApprovalPersistenceCoordinator.claimForExecution(
            message: blockedFixture.message,
            toolID: blockedFixture.pending.toolID,
            payloadArguments: blockedFixture.payload,
            policyAllowed: false,
            policyDeniedResult: "This action is blocked by policy.",
            save: { blockedSavedStatus = blockedFixture.message.status }
        )

        XCTAssertEqual(blocked, .rejected)
        XCTAssertEqual(blockedSavedStatus, .denied)
        XCTAssertEqual(blockedFixture.message.toolResult, "This action is blocked by policy.")
        XCTAssertNil(ToolApprovalQueue.shared.resolve(blockedFixture.pending.pendingActionID))
    }

    func testUserDenialClearsApprovalEvenWhenBothPersistenceAttemptsFail() {
        let fixture = makeFixture()
        let privateDetail = "private denial detail alex@example.com"
        var saveAttempts = 0

        let outcome = ToolApprovalPersistenceCoordinator.persistDenied(
            message: fixture.message,
            result: "Denied by user.",
            pendingActionID: fixture.pending.pendingActionID,
            save: {
                saveAttempts += 1
                throw SyntheticToolApprovalSaveError(privateDetail: privateDetail)
            }
        )

        guard case .failed(let failure) = outcome else {
            return XCTFail("Expected typed denial persistence failure")
        }
        XCTAssertEqual(saveAttempts, 2)
        XCTAssertEqual(failure.operation, .userDenied)
        XCTAssertEqual(fixture.message.status, .failed)
        XCTAssertTrue(fixture.message.toolResult?.contains("remains blocked") == true)
        XCTAssertFalse(fixture.message.toolResult?.contains(privateDetail) == true)
        XCTAssertNil(ToolApprovalQueue.shared.resolve(fixture.pending.pendingActionID))
    }

    func testTerminalSaveFailurePersistsFailureMarkerAndNeverReenablesReplay() async {
        let fixture = makeFixture()
        let claim = ToolApprovalPersistenceCoordinator.claimForExecution(
            message: fixture.message,
            toolID: fixture.pending.toolID,
            payloadArguments: fixture.payload,
            policyAllowed: true,
            policyDeniedResult: "blocked",
            save: {}
        )
        guard case .claimed = claim else {
            return XCTFail("Expected a durable claim")
        }

        var terminalSaveStatuses: [ToolStatus?] = []
        let terminal = ToolApprovalPersistenceCoordinator.persistTerminal(
            message: fixture.message,
            status: .completed,
            result: "Action completed.",
            save: {
                terminalSaveStatuses.append(fixture.message.status)
                if terminalSaveStatuses.count == 1 {
                    throw SyntheticToolApprovalSaveError(privateDetail: "private terminal result")
                }
            }
        )

        guard case .failed(let failure) = terminal else {
            return XCTFail("Expected typed terminal persistence failure")
        }
        XCTAssertEqual(failure.operation, .terminalOutcome)
        XCTAssertEqual(terminalSaveStatuses, [.completed, .failed])
        XCTAssertEqual(fixture.message.status, .failed)
        XCTAssertTrue(fixture.message.toolResult?.contains("may have completed") == true)
        XCTAssertTrue(fixture.message.toolResult?.contains("will not run again") == true)
        XCTAssertFalse(fixture.message.toolResult?.contains("private terminal result") == true)
        XCTAssertNil(ToolApprovalQueue.shared.resolve(fixture.pending.pendingActionID))

        let replay = ToolApprovalPersistenceCoordinator.claimForExecution(
            message: fixture.message,
            toolID: fixture.pending.toolID,
            payloadArguments: fixture.payload,
            policyAllowed: true,
            policyDeniedResult: "blocked",
            save: { XCTFail("A terminal message must not be saved as a new claim") }
        )
        var didReplay = false
        let replayResult: Bool? = await ToolApprovalPersistenceCoordinator.executeIfClaimed(replay) { _ in
            didReplay = true
            return true
        }
        XCTAssertEqual(replay, .rejected)
        XCTAssertFalse(didReplay)
        XCTAssertNil(replayResult)
    }

    func testMessageBubbleUsesTypedApprovalPersistenceBoundary() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repoRoot.appendingPathComponent("ios/Lumen/Views/MessageBubble.swift"),
            encoding: .utf8
        )

        XCTAssertFalse(source.contains("try? modelContext.save()"))
        XCTAssertTrue(source.contains("ToolApprovalPersistenceCoordinator.claimForExecution"))
        XCTAssertTrue(source.contains("ToolApprovalPersistenceCoordinator.executeIfClaimed"))
        XCTAssertTrue(source.contains("ToolApprovalPersistenceCoordinator.persistTerminal"))
        XCTAssertTrue(source.contains("ToolApprovalPersistenceCoordinator.persistDenied"))
    }

    private func makeFixture() -> (
        message: ChatMessage,
        pending: ExecutorPendingApproval,
        payload: [String: String]
    ) {
        let pending = ToolApprovalQueue.shared.enqueue(
            toolID: "mail.draft",
            toolName: "Draft Email",
            arguments: [
                "to": "sam@example.com",
                "subject": "Timeline",
                "body": "Status green"
            ]
        )
        let payload = ToolApprovalPayloadCodec.displayArguments(
            for: pending,
            visibleArguments: pending.arguments.stringCoerced
        )
        let message = ChatMessage(
            role: .tool,
            content: ToolApprovalPayloadCodec.serialize(payload),
            toolName: pending.toolID,
            toolStatus: .pendingApproval
        )
        return (message, pending, payload)
    }
}

private struct SyntheticToolApprovalSaveError: Error {
    let privateDetail: String
}

private final class ToolApprovalPersistenceSignalCapture: @unchecked Sendable {
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
                  signal.values["source"] == "tool-approval-persistence" else { return }
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
