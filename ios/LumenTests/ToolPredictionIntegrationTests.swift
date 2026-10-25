import Foundation
import XCTest
@testable import Lumen

@MainActor
final class ToolPredictionIntegrationTests: XCTestCase {
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func invocation(source: ToolInvocationSource = .modelProposed) -> ToolInvocation {
        ToolInvocation(id: UUID(), toolID: "prediction.test", arguments: [:], source: source,
                       conversationID: nil, turnID: nil, createdAt: Date())
    }

    private func prediction() -> SelfPredictionLedger.Context {
        .init(runID: UUID(), toolID: "caller.cannot.bind.this", runtime: "coreAI",
              probabilityOrigin: .model, probability: 0.8)
    }

    private var context: ToolExecutionContext {
        .init(isForeground: true, appState: nil, modelContext: nil,
              permissionRegistry: .shared, metricsStore: .shared)
    }

    private func events(_ ledger: SelfPredictionLedger) async throws -> [[String: Any]] {
        let data = try await ledger.exportJSON()
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        return try XCTUnwrap(object["events"] as? [[String: Any]])
    }

    func testDurablePredictionExistsBeforeBodyAndActualStatusWinsOverText() async throws {
        let url = try directory()
        let ledger = SelfPredictionLedger(directoryURL: url)
        let file = url.appendingPathComponent(SelfPredictionLedger.fileName)
        var before = Data()
        var executions = 0
        let tool = PredictionTestTool { invocation in
            executions += 1
            before = (try? Data(contentsOf: file)) ?? Data()
            let lines = before.split(separator: 0x0A)
            XCTAssertEqual(lines.count, 1, "The acknowledged prediction must already be on disk")
            if let line = lines.first {
                let first = try? JSONSerialization.jsonObject(with: Data(line)) as? [String: Any]
                XCTAssertEqual(first?["kind"] as? String, "prediction")
                XCTAssertEqual(first?["id"] as? String, invocation.id.uuidString)
                XCTAssertEqual(first?["toolID"] as? String, "prediction.test")
            }
            return Self.result(invocation.id, status: .failed, text: "success")
        }
        let registry = SecureToolRegistry(tools: [tool], predictionLedger: ledger)
        let call = invocation()
        let result = await registry.execute(call, context: context, predictionContext: prediction())
        XCTAssertEqual(executions, 1)
        XCTAssertEqual(result.status, .failed)
        let recorded = try await events(ledger)
        XCTAssertEqual(recorded.count, 2)
        XCTAssertEqual(recorded[1]["status"] as? String, "failed")
        XCTAssertEqual(recorded[1]["toolBodyStarted"] as? Bool, true)
        XCTAssertEqual(recorded[1]["cancelled"] as? Bool, false)
        XCTAssertTrue(try Data(contentsOf: file).starts(with: before), "The before record is append-only")
        XCTAssertNil(registry.lastPredictionRecordingFailure)
    }

    func testDenialAndApprovalRecordActualPolicyWithoutStartingBody() async throws {
        for source in [ToolInvocationSource.modelProposed, .backgroundTrigger] {
            let ledger = SelfPredictionLedger(directoryURL: try directory())
            var executions = 0
            let tool = PredictionTestTool(requiresApproval: true) { invocation in
                executions += 1
                return Self.result(invocation.id)
            }
            let registry = SecureToolRegistry(tools: [tool], predictionLedger: ledger)
            let result = await registry.execute(invocation(source: source), context: context, predictionContext: prediction())
            let expected: ToolResultStatus = source == .backgroundTrigger ? .denied : .requiresApproval
            XCTAssertEqual(result.status, expected)
            XCTAssertEqual(executions, 0)
            let recorded = try await events(ledger)
            XCTAssertEqual(recorded[1]["status"] as? String, expected.rawValue)
            XCTAssertEqual(recorded[1]["toolBodyStarted"] as? Bool, false)
        }
    }

    func testUnavailableToolIsRecordedWithoutBody() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        let registry = SecureToolRegistry(tools: [], predictionLedger: ledger)
        let result = await registry.execute(invocation(), context: context, predictionContext: prediction())
        XCTAssertEqual(result.status, .unavailable)
        let recorded = try await events(ledger)
        XCTAssertEqual(recorded[1]["status"] as? String, "unavailable")
        XCTAssertEqual(recorded[1]["toolBodyStarted"] as? Bool, false)
    }

    func testLegacyCallerWithoutPredictionContextDoesNotInventLedgerCoverage() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        let tool = PredictionTestTool { Self.result($0.id) }
        let registry = SecureToolRegistry(tools: [tool], predictionLedger: ledger)
        let result = await registry.execute(invocation(), context: context)
        XCTAssertEqual(result.status, .success)
        let recorded = try await events(ledger)
        XCTAssertTrue(recorded.isEmpty)
    }

    func testPredictionStorageFailureDoesNotRetryOrReplaceToolResult() async throws {
        let file = try directory().appendingPathComponent("not-a-directory")
        try Data("synthetic".utf8).write(to: file)
        let ledger = SelfPredictionLedger(directoryURL: file)
        var executions = 0
        let tool = PredictionTestTool { invocation in
            executions += 1
            return Self.result(invocation.id, text: "actual")
        }
        let registry = SecureToolRegistry(tools: [tool], predictionLedger: ledger)
        let call = invocation()
        let result = await registry.execute(call, context: context, predictionContext: prediction())
        XCTAssertEqual(executions, 1)
        XCTAssertEqual(result.status, .success)
        XCTAssertEqual(result.modelText, "actual")
        XCTAssertEqual(registry.lastPredictionRecordingFailure, .init(invocationID: call.id, stage: .prediction))
        XCTAssertEqual(try String(contentsOf: file, encoding: .utf8), "synthetic")
    }

    func testOutcomeStorageFailureDoesNotRerunToolOrInventOutcome() async throws {
        let url = try directory()
        let ledger = SelfPredictionLedger(directoryURL: url)
        let file = url.appendingPathComponent(SelfPredictionLedger.fileName)
        let preserved = url.appendingPathComponent("preserved-prediction.jsonl")
        var executions = 0
        let tool = PredictionTestTool { invocation in
            executions += 1
            do {
                try FileManager.default.moveItem(at: file, to: preserved)
                try FileManager.default.createDirectory(at: file, withIntermediateDirectories: false)
            } catch { XCTFail("Synthetic storage fault setup failed: \(error)") }
            return Self.result(invocation.id)
        }
        let registry = SecureToolRegistry(tools: [tool], predictionLedger: ledger)
        let call = invocation()
        let result = await registry.execute(call, context: context, predictionContext: prediction())
        XCTAssertEqual(executions, 1)
        XCTAssertEqual(result.status, .success)
        XCTAssertEqual(registry.lastPredictionRecordingFailure, .init(invocationID: call.id, stage: .outcome))
        let stored = try Data(contentsOf: preserved).split(separator: 0x0A)
        XCTAssertEqual(stored.count, 1)
        let event = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(stored[0])) as? [String: Any])
        XCTAssertEqual(event["kind"] as? String, "prediction")
    }

    func testCancellationAfterDurableBeginPreventsBodyButStillRecordsOutcome() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        var executions = 0
        let tool = PredictionTestTool { invocation in
            executions += 1
            return Self.result(invocation.id)
        }
        let registry = SecureToolRegistry(tools: [tool], predictionLedger: ledger)
        let call = invocation()
        let executionContext = context
        let forecast = prediction()
        let result = await Task { @MainActor in
            withUnsafeCurrentTask { $0?.cancel() }
            return await registry.execute(call, context: executionContext, predictionContext: forecast)
        }.value
        XCTAssertEqual(executions, 0)
        XCTAssertEqual(result.status, .failed)
        XCTAssertEqual(result.errorCode, "cancelled")
        let recorded = try await events(ledger)
        XCTAssertEqual(recorded.count, 2)
        XCTAssertEqual(recorded[1]["toolBodyStarted"] as? Bool, false)
        XCTAssertEqual(recorded[1]["cancelled"] as? Bool, true)
    }

    func testBodyCannotBindOutcomeToAnotherInvocation() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        let tool = PredictionTestTool { _ in Self.result(UUID(), text: "untrusted success") }
        let registry = SecureToolRegistry(tools: [tool], predictionLedger: ledger)
        let call = invocation()
        let result = await registry.execute(call, context: context, predictionContext: prediction())
        XCTAssertEqual(result.invocationID, call.id)
        XCTAssertEqual(result.status, .failed)
        XCTAssertEqual(result.errorCode, "result_invocation_mismatch")
        XCTAssertFalse(result.modelText.contains("untrusted"))
        let recorded = try await events(ledger)
        XCTAssertEqual(recorded[1]["id"] as? String, call.id.uuidString)
        XCTAssertEqual(recorded[1]["status"] as? String, "failed")
        XCTAssertEqual(recorded[1]["toolBodyStarted"] as? Bool, true)
    }

    private static func result(_ id: UUID, status: ToolResultStatus = .success, text: String = "actual") -> ToolResult {
        .init(invocationID: id, status: status, displayText: text, modelText: text, structuredPayload: nil,
              privacyLevel: .low, metricsSummary: "synthetic", errorCode: nil)
    }
}

private struct PredictionTestTool: LocalTool {
    let requiresApproval: Bool
    let body: @MainActor (ToolInvocation) async -> ToolResult

    init(requiresApproval: Bool = false, body: @escaping @MainActor (ToolInvocation) async -> ToolResult) {
        self.requiresApproval = requiresApproval
        self.body = body
    }

    var definition: SecureToolDefinition {
        .init(id: "prediction.test", displayName: "Synthetic", description: "Test fixture", category: .readOnly,
              requiredPermissions: [], supportsBackgroundExecution: false, requiresUserApproval: requiresApproval,
              argumentSchemaDescription: "{}", resultPrivacyLevel: .low, maxOutputCharacters: 1000)
    }
    func validateArguments(_ arguments: [String: String]) throws {}
    func execute(invocation: ToolInvocation, context: ToolExecutionContext) async -> ToolResult {
        await body(invocation)
    }
}
