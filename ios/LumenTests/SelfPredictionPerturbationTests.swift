import Foundation
import XCTest
@testable import Lumen

/// Real registry/ledger components with synthetic tools and forecast fixtures.
/// These tests establish observation/feedback contracts, not model adaptation,
/// confidence calibration, native permission behavior, or live tool correctness.
@MainActor
final class SelfPredictionPerturbationTests: XCTestCase {
    private static let toolID = "perturbation.test"
    private static let otherToolID = "perturbation.other"

    private var context: ToolExecutionContext {
        .init(isForeground: true, appState: nil, modelContext: nil,
              permissionRegistry: .shared, metricsStore: .shared)
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func invocation(toolID: String = "perturbation.test", source: ToolInvocationSource = .modelProposed) -> ToolInvocation {
        .init(id: UUID(), toolID: toolID, arguments: [:], source: source,
              conversationID: nil, turnID: nil, createdAt: Date())
    }

    private func forecast(
        toolID: String = "perturbation.test", runtime: String = "coreAI",
        origin: SelfPredictionLedger.ProbabilityOrigin = .model
    ) -> SelfPredictionLedger.Context {
        .init(runID: UUID(), toolID: toolID, runtime: runtime,
              probabilityOrigin: origin, probability: origin == .model ? 0.8 : nil)
    }

    private func snapshot(
        _ registry: SecureToolRegistry, runtime: String = "coreAI",
        toolIDs: Set<String> = ["perturbation.test"], now: Date = Date()
    ) async throws -> SelfPredictionFeedback.Snapshot {
        switch await registry.predictionFeedback(runtime: runtime, toolIDs: toolIDs, now: now) {
        case .available(let snapshot): return snapshot
        case .unavailable:
            XCTFail("Expected available synthetic registry observations")
            throw NSError(domain: "SelfPredictionPerturbationTests", code: 1)
        }
    }

    private func entry(
        _ registry: SecureToolRegistry, runtime: String = "coreAI",
        toolID: String = "perturbation.test"
    ) async throws -> SelfPredictionFeedback.ToolSummary {
        let feedback = try await snapshot(registry, runtime: runtime, toolIDs: [toolID])
        return try XCTUnwrap(feedback.entries.first { $0.toolID == toolID })
    }

    private func events(_ ledger: SelfPredictionLedger) async throws -> [[String: Any]] {
        let data = try await ledger.exportJSON()
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        return try XCTUnwrap(object["events"] as? [[String: Any]])
    }

    func testRemovalAndRecoveryUpdateObservationsWithoutRewritingOriginalForecast() async throws {
        let url = try directory()
        let ledger = SelfPredictionLedger(directoryURL: url)
        var executions = 0
        let tool = PerturbationTool { invocation in
            executions += 1
            return Self.result(invocation.id)
        }
        let available = SecureToolRegistry(tools: [tool], predictionLedger: ledger)
        let missing = SecureToolRegistry(tools: [], predictionLedger: ledger)

        let first = await available.execute(invocation(), context: context, predictionContext: forecast())
        XCTAssertEqual(first.status, .success)
        let originalBytes = try Data(contentsOf: url.appendingPathComponent(SelfPredictionLedger.fileName))

        let removed = await missing.execute(invocation(), context: context, predictionContext: forecast())
        XCTAssertEqual(removed.status, .unavailable)
        let afterRemoval = try await entry(missing)
        XCTAssertEqual(afterRemoval.completedCount, 2)
        XCTAssertEqual(afterRemoval.successCount, 1)
        XCTAssertEqual(afterRemoval.lastStatus, .unavailable)
        XCTAssertEqual(afterRemoval.transition?.from, .success)
        XCTAssertEqual(afterRemoval.transition?.to, .unavailable)

        let restored = await available.execute(invocation(), context: context, predictionContext: forecast())
        XCTAssertEqual(restored.status, .success)
        let recovered = try await entry(available)
        XCTAssertEqual(recovered.completedCount, 3)
        XCTAssertEqual(recovered.successCount, 2)
        XCTAssertEqual(recovered.lastStatus, .success)
        XCTAssertEqual(recovered.transition?.from, .unavailable)
        XCTAssertEqual(recovered.transition?.to, .success)
        XCTAssertGreaterThan(recovered.lastSeq, afterRemoval.lastSeq)
        XCTAssertEqual(executions, 2)

        let recorded = try await events(ledger)
        XCTAssertEqual(recorded[0]["probability"] as? Double, 0.8)
        XCTAssertEqual(recorded.filter { $0["kind"] as? String == "outcome" }.compactMap { $0["toolBodyStarted"] as? Bool }, [true, false, true])
        XCTAssertTrue(try Data(contentsOf: url.appendingPathComponent(SelfPredictionLedger.fileName)).starts(with: originalBytes))
    }

    func testPolicyDenialAndExplicitApprovalRecoveryDoNotMutateGlobalPermissions() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        var executions = 0
        let tool = PerturbationTool(requiresApproval: true) { invocation in
            executions += 1
            return Self.result(invocation.id)
        }
        let registry = SecureToolRegistry(tools: [tool], predictionLedger: ledger)

        let denied = await registry.execute(invocation(source: .backgroundTrigger), context: context, predictionContext: forecast())
        let approval = await registry.execute(invocation(), context: context, predictionContext: forecast())
        XCTAssertEqual(denied.status, .denied)
        XCTAssertEqual(approval.status, .requiresApproval)
        XCTAssertEqual(executions, 0)

        let approved = await registry.execute(invocation(source: .userApproved), context: context, predictionContext: forecast())
        XCTAssertEqual(approved.status, .success)
        XCTAssertEqual(executions, 1)
        let feedback = try await entry(registry)
        XCTAssertEqual(feedback.completedCount, 3)
        XCTAssertEqual(feedback.successCount, 1)
        XCTAssertEqual(feedback.transition?.from, .requiresApproval)
        XCTAssertEqual(feedback.transition?.to, .success)
        let recorded = try await events(ledger)
        XCTAssertEqual(recorded.filter { $0["kind"] as? String == "outcome" }.compactMap { $0["toolBodyStarted"] as? Bool }, [false, false, true])
    }

    func testBodyFailureAndRecoveryKeepTheToolDefinitionUnchanged() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        var bodyStatus: ToolResultStatus = .success
        let registry = SecureToolRegistry(tools: [PerturbationTool { invocation in
            Self.result(invocation.id, status: bodyStatus)
        }], predictionLedger: ledger)
        let definitions = registry.definitions()

        _ = await registry.execute(invocation(), context: context, predictionContext: forecast())
        bodyStatus = .failed
        let failed = await registry.execute(invocation(), context: context, predictionContext: forecast())
        XCTAssertEqual(failed.status, .failed)
        let afterFailure = try await entry(registry)
        XCTAssertEqual(afterFailure.transition?.from, .success)
        XCTAssertEqual(afterFailure.transition?.to, .failed)
        bodyStatus = .success
        _ = await registry.execute(invocation(), context: context, predictionContext: forecast())

        let recovered = try await entry(registry)
        XCTAssertEqual(recovered.successCount, 2)
        XCTAssertEqual(recovered.lastStatus, .success)
        XCTAssertEqual(recovered.transition?.from, .failed)
        XCTAssertEqual(recovered.transition?.to, .success)
        XCTAssertEqual(registry.definitions().map(\.id), definitions.map(\.id))
        XCTAssertEqual(registry.definitions().map(\.requiresUserApproval), definitions.map(\.requiresUserApproval))
    }

    func testDifferentToolAndRuntimeObservationsCannotContaminateTheRequestedScope() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        let registry = SecureToolRegistry(tools: [
            PerturbationTool { Self.result($0.id) },
            PerturbationTool(id: Self.otherToolID) { Self.result($0.id, status: .failed) }
        ], predictionLedger: ledger)
        _ = await registry.execute(invocation(), context: context, predictionContext: forecast())
        _ = await registry.execute(invocation(), context: context, predictionContext: forecast(runtime: "llama"))
        _ = await registry.execute(invocation(toolID: Self.otherToolID), context: context, predictionContext: forecast(toolID: Self.otherToolID))

        let feedback = try await snapshot(registry)
        XCTAssertEqual(feedback.entries.map(\.toolID), [Self.toolID])
        XCTAssertEqual(feedback.entries.first?.completedCount, 1)
        XCTAssertEqual(feedback.entries.first?.successCount, 1)
    }

    func testCancelledAndUnmatchedPredictionsDoNotBecomeCompletedObservations() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        var executions = 0
        let registry = SecureToolRegistry(tools: [PerturbationTool { invocation in
            executions += 1
            return Self.result(invocation.id)
        }], predictionLedger: ledger)
        _ = await registry.execute(invocation(), context: context, predictionContext: forecast())
        try await ledger.begin(context: forecast(), id: UUID())
        let call = invocation()
        let executionContext = context
        let forecastContext = forecast()
        let cancelled = await Task { @MainActor in
            withUnsafeCurrentTask { $0?.cancel() }
            return await registry.execute(call, context: executionContext, predictionContext: forecastContext)
        }.value
        XCTAssertEqual(cancelled.status, .failed)
        XCTAssertEqual(executions, 1)

        let feedback = try await snapshot(registry)
        XCTAssertEqual(feedback.entries.first?.completedCount, 1)
        XCTAssertEqual(feedback.entries.first?.successCount, 1)
        XCTAssertEqual(feedback.pendingCount, 1)
        XCTAssertEqual(feedback.cancelledCount, 1)
    }

    func testObservedStatusesDoNotRequireOrInventModelProbabilities() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        let registry = SecureToolRegistry(tools: [PerturbationTool { Self.result($0.id) }], predictionLedger: ledger)
        for origin in [SelfPredictionLedger.ProbabilityOrigin.missing, .invalid, .hostRepair] {
            _ = await registry.execute(invocation(), context: context, predictionContext: forecast(origin: origin))
        }
        let feedback = try await entry(registry)
        XCTAssertEqual(feedback.completedCount, 3)
        XCTAssertEqual(feedback.successCount, 3)
        let recorded = try await events(ledger)
        let predictions = recorded.filter { $0["kind"] as? String == "prediction" }
        XCTAssertEqual(predictions.count, 3)
        XCTAssertTrue(predictions.allSatisfy { $0["probability"] == nil })
    }

    func testRecordingAblationDoesNotChangeToolExecutionOrInventFeedback() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        var executions = 0
        let registry = SecureToolRegistry(tools: [PerturbationTool { invocation in
            executions += 1
            return Self.result(invocation.id)
        }], predictionLedger: ledger)

        let unrecorded = await registry.execute(invocation(), context: context)
        let withoutRecording = try await snapshot(registry)
        XCTAssertTrue(withoutRecording.entries.isEmpty)
        let unrecordedEvents = try await events(ledger)
        XCTAssertTrue(unrecordedEvents.isEmpty)
        let recorded = await registry.execute(invocation(), context: context, predictionContext: forecast())
        let withRecording = try await entry(registry)
        XCTAssertEqual(unrecorded.status, recorded.status)
        XCTAssertEqual(executions, 2)
        XCTAssertEqual(withRecording.completedCount, 1)
        XCTAssertEqual(withRecording.successCount, 1)
    }

    func testOmittingFeedbackRenderingLeavesRecordedReceiptsUnchanged() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        let registry = SecureToolRegistry(tools: [PerturbationTool { Self.result($0.id) }], predictionLedger: ledger)
        _ = await registry.execute(invocation(), context: context, predictionContext: forecast())
        let before = try await ledger.exportJSON()
        let feedback = await registry.predictionFeedback(runtime: "coreAI", toolIDs: [Self.toolID])

        let included = SelfPredictionFeedback.render(feedback, maxCharacters: 640)
        let omitted = SelfPredictionFeedback.render(feedback, maxCharacters: 0)
        XCTAssertFalse(included.isEmpty)
        XCTAssertTrue(omitted.isEmpty)
        let after = try await ledger.exportJSON()
        XCTAssertEqual(after, before)
    }

    func testCorruptFeedbackIsUnavailableInsteadOfAHealthyEmptyHistory() async throws {
        let url = try directory()
        try Data("synthetic corrupt ledger\n".utf8).write(to: url.appendingPathComponent(SelfPredictionLedger.fileName))
        let registry = SecureToolRegistry(tools: [], predictionLedger: SelfPredictionLedger(directoryURL: url))
        if case .available = await registry.predictionFeedback(runtime: "coreAI", toolIDs: [Self.toolID]) {
            XCTFail("Corrupt storage must not become an available empty history")
        }
        let empty = SecureToolRegistry(tools: [], predictionLedger: SelfPredictionLedger(directoryURL: try directory()))
        let emptySnapshot = try await snapshot(empty)
        XCTAssertTrue(emptySnapshot.entries.isEmpty)
    }

    func testRecordingCapacityFailureCannotExposeOldHealthyFeedbackCounts() async throws {
        let url = try directory()
        let ledger = SelfPredictionLedger(directoryURL: url, maxEvents: 2)
        var executions = 0
        let registry = SecureToolRegistry(tools: [PerturbationTool { invocation in
            executions += 1
            return Self.result(invocation.id)
        }], predictionLedger: ledger)
        let first = await registry.execute(invocation(), context: context, predictionContext: forecast())
        XCTAssertEqual(first.status, .success)
        let priorFeedback = try await entry(registry)
        XCTAssertEqual(priorFeedback.completedCount, 1)
        let file = url.appendingPathComponent(SelfPredictionLedger.fileName)
        let originalBytes = try Data(contentsOf: file)

        let secondInvocation = invocation()
        let second = await registry.execute(secondInvocation, context: context, predictionContext: forecast())
        XCTAssertEqual(second.status, .success)
        XCTAssertEqual(executions, 2)
        XCTAssertEqual(registry.lastPredictionRecordingFailure,
                       .init(invocationID: secondInvocation.id, stage: .prediction))
        if case .available = await registry.predictionFeedback(runtime: "coreAI", toolIDs: [Self.toolID]) {
            XCTFail("A recording gap must not expose the prior healthy count as complete feedback")
        }
        XCTAssertEqual(try Data(contentsOf: file), originalBytes)
    }

    func testRecentWindowReflectsRecoveryWithoutErasingTheFullLedger() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        var bodyStatus: ToolResultStatus = .failed
        let registry = SecureToolRegistry(tools: [PerturbationTool { Self.result($0.id, status: bodyStatus) }], predictionLedger: ledger)
        for _ in 0..<9 {
            _ = await registry.execute(invocation(), context: context, predictionContext: forecast())
        }
        bodyStatus = .success
        _ = await registry.execute(invocation(), context: context, predictionContext: forecast())
        let feedback = try await entry(registry)
        XCTAssertEqual(feedback.completedCount, 8)
        XCTAssertEqual(feedback.successCount, 1)
        XCTAssertEqual(feedback.transition?.from, .failed)
        XCTAssertEqual(feedback.transition?.to, .success)
        let recorded = try await events(ledger)
        XCTAssertEqual(recorded.filter { $0["kind"] as? String == "outcome" }.count, 10)
    }

    func testStaleAndFutureHostDatesAreExcludedWithoutSleepingOrRewritingEvents() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        let registry = SecureToolRegistry(tools: [PerturbationTool { Self.result($0.id) }], predictionLedger: ledger)
        _ = await registry.execute(invocation(), context: context, predictionContext: forecast())
        let stored = try await ledger.exportJSON()
        let stale = try await snapshot(registry, now: Date().addingTimeInterval(25 * 60 * 60))
        XCTAssertTrue(stale.entries.isEmpty)
        XCTAssertEqual(stale.staleCount, 1)
        let future = try await snapshot(registry, now: Date().addingTimeInterval(-60))
        XCTAssertTrue(future.entries.isEmpty)
        XCTAssertEqual(future.futureCount, 1)
        let after = try await ledger.exportJSON()
        XCTAssertEqual(after, stored)
    }

    func testFeedbackNeverCopiesToolResultText() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        let marker = "SYNTHETIC_PRIVATE_RESULT_CONTENT"
        let registry = SecureToolRegistry(tools: [PerturbationTool { Self.result($0.id, text: marker) }], predictionLedger: ledger)
        _ = await registry.execute(invocation(), context: context, predictionContext: forecast())
        let feedback = await registry.predictionFeedback(runtime: "coreAI", toolIDs: [Self.toolID])
        XCTAssertFalse(SelfPredictionFeedback.render(feedback).contains(marker))
        let exported = try await ledger.exportJSON()
        XCTAssertFalse(String(decoding: exported, as: UTF8.self).contains(marker))
    }

    private static func result(_ id: UUID, status: ToolResultStatus = .success, text: String = "synthetic result") -> ToolResult {
        .init(invocationID: id, status: status, displayText: text, modelText: text,
              structuredPayload: nil, privacyLevel: .low, metricsSummary: "synthetic", errorCode: nil)
    }
}

private struct PerturbationTool: LocalTool {
    let id: ToolID
    let requiresApproval: Bool
    let body: @MainActor (ToolInvocation) async -> ToolResult

    init(id: ToolID = "perturbation.test", requiresApproval: Bool = false,
         body: @escaping @MainActor (ToolInvocation) async -> ToolResult) {
        self.id = id
        self.requiresApproval = requiresApproval
        self.body = body
    }

    var definition: SecureToolDefinition {
        .init(id: id, displayName: "Synthetic perturbation", description: "Component test tool",
              category: .readOnly, requiredPermissions: [], supportsBackgroundExecution: true,
              requiresUserApproval: requiresApproval, argumentSchemaDescription: "{}",
              resultPrivacyLevel: .low, maxOutputCharacters: 1000)
    }

    func validateArguments(_ arguments: [String: String]) throws {}
    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        await body(invocation)
    }
}
