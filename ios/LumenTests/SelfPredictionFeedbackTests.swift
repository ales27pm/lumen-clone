import Foundation
import XCTest
@testable import Lumen

@MainActor
final class SelfPredictionFeedbackTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_800_000_000)

    private func record(
        tool: String = "weather", runtime: String = "coreAI", seq: Int = 2,
        status: SelfPredictionLedger.OutcomeStatus? = .success,
        age: TimeInterval = 1, cancelled: Bool = false
    ) -> SelfPredictionFeedback.Record {
        let date = now.addingTimeInterval(-age)
        return .init(runtime: runtime, toolID: tool, predictionDate: date,
                     outcome: status.map { .init(seq: seq, timestamp: date, status: $0, cancelled: cancelled) })
    }

    private func snapshot(
        _ records: [SelfPredictionFeedback.Record], tools: Set<String> = ["weather"], runtime: String = "coreAI"
    ) -> SelfPredictionFeedback.Snapshot {
        SelfPredictionFeedback.project(records: records, runtime: runtime, toolIDs: tools, now: now)
    }

    private func object(_ text: String) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }

    func testOnlyExactRuntimeAndRequestedToolsContribute() {
        let value = snapshot([
            record(status: .success), record(runtime: "llama", seq: 4, status: .failed),
            record(tool: "contacts.search", seq: 6, status: .denied),
            record(tool: "contacts.search", seq: 7, status: nil),
            record(runtime: "CoreAI", seq: 8, status: .unavailable),
        ])
        XCTAssertEqual(value.entries.count, 1)
        XCTAssertEqual(value.entries.first?.completedCount, 1)
        XCTAssertEqual(value.entries.first?.lastStatus, .success)
        XCTAssertEqual(value.pendingCount, 0)
    }

    func testLatestEightEligibleOutcomesDefineCountsAndTransition() {
        let records = (1...10).map { seq in
            record(seq: seq, status: seq <= 2 ? .success : .failed)
        }
        let entry = snapshot(records).entries.first
        XCTAssertEqual(entry?.completedCount, 8)
        XCTAssertEqual(entry?.successCount, 0)
        XCTAssertEqual(entry?.lastStatus, .failed)
        XCTAssertEqual(entry?.lastSeq, 10)
        XCTAssertNil(entry?.transition)
    }

    func testImmediatePreviousEligibleStatusDeterminesFactualTransition() {
        let value = snapshot([
            record(seq: 2, status: .success), record(seq: 4, status: .failed),
            record(seq: 6, status: .success, cancelled: true), record(seq: 7, status: nil),
            record(seq: 8, status: .unavailable),
        ])
        let entry = value.entries.first
        XCTAssertEqual(entry?.completedCount, 3)
        XCTAssertEqual(entry?.successCount, 1)
        XCTAssertEqual(entry?.transition, SelfPredictionFeedback.StatusTransition(from: .failed, to: .unavailable))
        XCTAssertEqual(value.pendingCount, 1)
        XCTAssertEqual(value.cancelledCount, 1)
    }

    func testOneObservationDoesNotInventTransition() {
        XCTAssertNil(snapshot([record(status: .denied)]).entries.first?.transition)
    }

    func testSequenceOrdersHistoryEvenWhenClockMovesBack() {
        let value = snapshot([
            record(seq: 2, status: .success, age: 10),
            record(seq: 4, status: .failed, age: 20),
        ])
        XCTAssertEqual(value.entries.first?.lastSeq, 4)
        XCTAssertEqual(value.entries.first?.lastObservedAt, now.addingTimeInterval(-20))
        XCTAssertEqual(value.entries.first?.transition, SelfPredictionFeedback.StatusTransition(from: .success, to: .failed))
    }

    func testOnlyThreeMostRecentToolsAreRetainedBySequence() {
        let records = (1...5).map { record(tool: "tool.\($0)", seq: $0, age: TimeInterval($0)) }
        let value = snapshot(records, tools: Set((1...5).map { "tool.\($0)" }))
        XCTAssertEqual(value.entries.map(\.toolID), ["tool.5", "tool.4", "tool.3"])
        XCTAssertEqual(value.omittedToolCount, 2)
    }

    func testFreshnessBoundaryAndExclusionCountersAreDisjoint() {
        let value = snapshot([
            record(seq: 2, age: SelfPredictionFeedback.maximumAge),
            record(seq: 4, age: SelfPredictionFeedback.maximumAge + 0.001),
            record(seq: 6, age: -0.001), record(seq: 7, status: nil),
            record(seq: 8, cancelled: true),
            record(seq: 9, status: nil, age: SelfPredictionFeedback.maximumAge + 1),
            record(seq: 10, age: -1, cancelled: true),
        ])
        XCTAssertEqual(value.entries.first?.completedCount, 1)
        XCTAssertEqual(value.entries.first?.lastSeq, 2)
        XCTAssertEqual(value.pendingCount, 1)
        XCTAssertEqual(value.cancelledCount, 1)
        XCTAssertEqual(value.staleCount, 2)
        XCTAssertEqual(value.futureCount, 2)
    }

    func testOldHistoryDoesNotCreateARecentTransition() {
        let value = snapshot([
            record(seq: 2, status: .success, age: SelfPredictionFeedback.maximumAge + 1),
            record(seq: 4, status: .failed),
        ])
        XCTAssertEqual(value.entries.first?.completedCount, 1)
        XCTAssertNil(value.entries.first?.transition)
    }

    func testEveryActualStatusRemainsDistinctWithoutProbabilityOrCause() throws {
        let statuses: [SelfPredictionLedger.OutcomeStatus] = [.success, .failed, .denied, .requiresApproval, .unavailable]
        for status in statuses {
            let value = snapshot([record(status: status)])
            XCTAssertEqual(value.entries.first?.lastStatus, status)
            XCTAssertEqual(value.entries.first?.successCount, status == .success ? 1 : 0)
            let rendered = try object(SelfPredictionFeedback.render(.available(value)))
            XCTAssertEqual(rendered["scope"] as? String, "historical_tool_results_not_current_capability")
            XCTAssertNil(rendered["probability"])
            XCTAssertNil(rendered["permission"])
            XCTAssertNil(rendered["readiness"])
        }
    }

    func testRendererDropsWholeEntriesAndCountsAllOmissions() throws {
        let tools = (1...4).map { String(repeating: String($0), count: 120) }
        let value = snapshot(tools.enumerated().map { record(tool: $0.element, seq: $0.offset + 1) }, tools: Set(tools))
        let text = SelfPredictionFeedback.render(.available(value), maxCharacters: 640)
        XCTAssertLessThanOrEqual(text.count, 640)
        let rendered = try object(text)
        let retained = try XCTUnwrap(rendered["tools"] as? [[String: Any]])
        XCTAssertFalse(retained.isEmpty)
        XCTAssertLessThan(retained.count, 3)
        XCTAssertEqual(retained.compactMap { $0["tool"] as? String }, Array(value.entries.prefix(retained.count)).map(\.toolID))
        XCTAssertEqual(rendered["omittedTools"] as? Int, 4 - retained.count)
        XCTAssertEqual(rendered["windowHours"] as? Int, 24)
        XCTAssertEqual(rendered["outcomesPerTool"] as? Int, 8)
    }

    func testUnavailableAndAvailableWithoutObservationsRemainDistinct() throws {
        let empty = try object(SelfPredictionFeedback.render(.available(snapshot([]))))
        let unavailable = try object(SelfPredictionFeedback.render(.unavailable))
        XCTAssertEqual(empty["state"] as? String, "available")
        XCTAssertEqual((empty["tools"] as? [[String: Any]])?.count, 0)
        XCTAssertEqual(unavailable["state"] as? String, "unavailable")
        XCTAssertNil(unavailable["tools"])
        XCTAssertNil(unavailable["excluded"])
    }

    func testInsufficientBudgetReturnsNoPartialData() {
        for budget in [-1, 0, 1] {
            XCTAssertEqual(SelfPredictionFeedback.render(.available(snapshot([record()])), maxCharacters: budget), "")
            XCTAssertEqual(SelfPredictionFeedback.render(.unavailable, maxCharacters: budget), "")
        }
    }

    func testLedgerFeedbackReadsExistingEvidenceWithoutChangingWireBytes() async throws {
        let directory = try temporaryDirectory()
        let ledger = SelfPredictionLedger(directoryURL: directory)
        let id = UUID()
        try await ledger.begin(context: .init(runID: UUID(), toolID: "weather", runtime: "coreAI", probabilityOrigin: .missing), id: id)
        try await ledger.complete(id: id, status: .failed, toolBodyStarted: true, cancelled: false)
        let before = try await ledger.exportJSON()
        let value = try await ledger.feedbackSnapshot(runtime: "coreAI", toolIDs: ["weather"])
        let after = try await ledger.exportJSON()
        XCTAssertEqual(before, after)
        XCTAssertEqual(value.entries.first?.lastStatus, .failed)
        XCTAssertEqual(value.entries.first?.completedCount, 1)
        let restored = SelfPredictionLedger(directoryURL: directory)
        let restoredValue = try await restored.feedbackSnapshot(runtime: "coreAI", toolIDs: ["weather"])
        XCTAssertEqual(restoredValue.entries.first?.lastStatus, .failed)
        XCTAssertEqual(restoredValue.entries.first?.lastSeq, value.entries.first?.lastSeq)
    }

    func testNeverCreatedLedgerIsEmptyWithoutCreatingAFile() async throws {
        let directory = try temporaryDirectory()
        let ledger = SelfPredictionLedger(directoryURL: directory)
        let value = try await ledger.feedbackSnapshot(runtime: "coreAI", toolIDs: ["weather"])
        XCTAssertTrue(value.entries.isEmpty)
        XCTAssertEqual(value.pendingCount, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent(SelfPredictionLedger.fileName).path))
    }

    func testCorruptLedgerErrorDoesNotBecomeEmptyFeedback() async throws {
        let directory = try temporaryDirectory()
        try Data("synthetic malformed JSON\n".utf8).write(to: directory.appendingPathComponent(SelfPredictionLedger.fileName))
        let ledger = SelfPredictionLedger(directoryURL: directory)
        do {
            _ = try await ledger.feedbackSnapshot(runtime: "coreAI", toolIDs: ["weather"])
            XCTFail("Corruption must remain an explicit failure")
        } catch {
            XCTAssertEqual(error as? SelfPredictionLedger.Failure, .corruptFile)
        }
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
}
