import Foundation
import CoreFoundation
import XCTest
@testable import Lumen

final class SelfPredictionLedgerTests: XCTestCase {
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func context(origin: SelfPredictionLedger.ProbabilityOrigin = .model,
                         probability: Double? = 0.8, toolID: String = "device.status",
                         runtime: String = "coreAI") -> SelfPredictionLedger.Context {
        .init(runID: UUID(), toolID: toolID, runtime: runtime,
              probabilityOrigin: origin, probability: probability)
    }

    private func exported(_ ledger: SelfPredictionLedger) async throws -> [String: Any] {
        let data = try await ledger.exportJSON()
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func canonicalLines(_ events: [[String: Any]]) throws -> Data {
        // Match the writer's JSONEncoder number formatting. JSONSerialization on
        // macOS expands 0.8 to 0.80000000000000004, which is intentionally noncanonical.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var bytes = Data()
        for event in events {
            let values = try event.mapValues { value -> AgentJSONValue in
                if let number = value as? NSNumber {
                    return CFGetTypeID(number) == CFBooleanGetTypeID()
                        ? .bool(number.boolValue) : .number(number.doubleValue)
                }
                if let string = value as? String { return .string(string) }
                throw CocoaError(.coderInvalidValue)
            }
            bytes.append(try encoder.encode(values))
            bytes.append(0x0A)
        }
        return bytes
    }

    private func assertFailure(_ expected: SelfPredictionLedger.Failure,
                               file: StaticString = #filePath, line: UInt = #line,
                               operation: () async throws -> Void) async {
        do {
            try await operation()
            XCTFail("Expected a typed ledger failure", file: file, line: line)
        } catch {
            XCTAssertEqual(error as? SelfPredictionLedger.Failure, expected, file: file, line: line)
        }
    }

    func testBeginDurablyWritesPredictionBeforeOutcomeAndExportHasOnlyWireFields() async throws {
        let url = try directory()
        let ledger = SelfPredictionLedger(directoryURL: url)
        let id = UUID()
        let metadata = context()
        try await ledger.begin(context: metadata, id: id)

        // Read the real file before completing: begin's acknowledgement is not RAM-only.
        let bytes = try Data(contentsOf: url.appendingPathComponent(SelfPredictionLedger.fileName))
        XCTAssertEqual(bytes.last, 0x0A)
        let persisted = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        XCTAssertEqual(persisted["kind"] as? String, "prediction")
        XCTAssertEqual(persisted["id"] as? String, id.uuidString)
        XCTAssertEqual(persisted["runID"] as? String, metadata.runID.uuidString)
        XCTAssertEqual(persisted["probability"] as? Double, 0.8)

        try await ledger.complete(id: id, status: .failed, toolBodyStarted: true, cancelled: false)
        let root = try await exported(ledger)
        XCTAssertEqual(Set(root.keys), ["schemaVersion", "coverageScope", "events"])
        XCTAssertEqual(root["schemaVersion"] as? String, "self_prediction_ledger.v1")
        XCTAssertEqual(root["coverageScope"] as? String, "structured_tool_invocations_stored_window")
        let events = try XCTUnwrap(root["events"] as? [[String: Any]])
        XCTAssertEqual(events.compactMap { $0["seq"] as? Int }, [1, 2])
        XCTAssertEqual(Set(events[0].keys), ["kind", "seq", "id", "runID", "toolID", "runtime", "probabilityOrigin", "probability", "timestamp"])
        XCTAssertEqual(Set(events[1].keys), ["kind", "seq", "id", "status", "toolBodyStarted", "cancelled", "timestamp"])
        XCTAssertEqual(events[1]["id"] as? String, id.uuidString)
        XCTAssertEqual(events[1]["status"] as? String, "failed")
        XCTAssertEqual(events[1]["toolBodyStarted"] as? Bool, true)
        let summary = try await ledger.summary()
        XCTAssertEqual(summary.eventCount, 2)
        XCTAssertEqual(summary.pendingCount, 0)
    }

    func testReopenPreservesPendingAndContinuesPhysicalAppendOrder() async throws {
        let url = try directory()
        let idA = UUID(), idB = UUID()
        let first = SelfPredictionLedger(directoryURL: url)
        try await first.begin(context: context(), id: idA)
        try await first.begin(context: context(origin: .missing, probability: nil), id: idB)
        let reopened = SelfPredictionLedger(directoryURL: url)
        let pending = try await reopened.summary()
        XCTAssertEqual(pending.pendingCount, 2)
        try await reopened.complete(id: idB, status: .denied, toolBodyStarted: false, cancelled: false)
        try await reopened.complete(id: idA, status: .success, toolBodyStarted: true, cancelled: false)
        let root = try await exported(reopened)
        let events = try XCTUnwrap(root["events"] as? [[String: Any]])
        XCTAssertEqual(events.compactMap { $0["seq"] as? Int }, [1, 2, 3, 4])
        XCTAssertEqual(events[2]["id"] as? String, idB.uuidString)
        XCTAssertNil(events[1]["probability"])
        XCTAssertEqual(events[1]["probabilityOrigin"] as? String, "missing")
    }

    func testExportSnapshotCountsDescribeTheExactExportedEvents() async throws {
        let ledger = SelfPredictionLedger(directoryURL: try directory())
        let completedID = UUID(), pendingID = UUID()
        try await ledger.begin(context: context(), id: completedID)
        try await ledger.complete(id: completedID, status: .success, toolBodyStarted: true, cancelled: false)
        try await ledger.begin(context: context(origin: .missing, probability: nil), id: pendingID)
        let snapshot = try await ledger.exportSnapshot()
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: snapshot.data) as? [String: Any])
        let events = try XCTUnwrap(root["events"] as? [[String: Any]])
        let predictions = events.filter { $0["kind"] as? String == "prediction" }
        let outcomes = events.filter { $0["kind"] as? String == "outcome" }
        XCTAssertEqual(snapshot.summary.eventCount, events.count)
        XCTAssertEqual(snapshot.summary.predictionCount, predictions.count)
        XCTAssertEqual(snapshot.summary.outcomeCount, outcomes.count)
        XCTAssertEqual(snapshot.summary.pendingCount, predictions.count - outcomes.count)
        // Subsequent appends cannot mutate the already returned export or its counts.
        try await ledger.complete(id: pendingID, status: .unavailable, toolBodyStarted: false, cancelled: false)
        XCTAssertEqual(snapshot.summary.pendingCount, 1)
        XCTAssertEqual(events.count, 3)
    }

    func testDuplicateAndOrphanCallsNeverAppend() async throws {
        let url = try directory()
        let ledger = SelfPredictionLedger(directoryURL: url)
        let id = UUID()
        try await ledger.begin(context: context(), id: id)
        await assertFailure(.duplicatePrediction) { try await ledger.begin(context: self.context(), id: id) }
        await assertFailure(.unknownPrediction) {
            try await ledger.complete(id: UUID(), status: .success, toolBodyStarted: true, cancelled: false)
        }
        try await ledger.complete(id: id, status: .requiresApproval, toolBodyStarted: false, cancelled: false)
        let before = try Data(contentsOf: url.appendingPathComponent(SelfPredictionLedger.fileName))
        await assertFailure(.duplicateOutcome) {
            try await ledger.complete(id: id, status: .success, toolBodyStarted: true, cancelled: false)
        }
        let after = try Data(contentsOf: url.appendingPathComponent(SelfPredictionLedger.fileName))
        XCTAssertEqual(before, after)
        let summary = try await ledger.summary()
        XCTAssertEqual(summary.eventCount, 2)
    }

    func testProbabilityOriginsAndBoundedTokensRejectInvalidMetadataWithoutCreatingFile() async throws {
        let url = try directory()
        let ledger = SelfPredictionLedger(directoryURL: url)
        let invalid = [
            context(probability: nil), context(probability: .nan), context(probability: .infinity),
            context(probability: -0.01), context(probability: 1.01),
            context(origin: .missing, probability: 0.8), context(origin: .invalid, probability: 0.8),
            context(origin: .hostRepair, probability: 0.8), context(toolID: "DEVICE.status"),
            context(toolID: "document with private content"), context(toolID: String(repeating: "a", count: 129)),
            context(runtime: ""), context(runtime: "raw\ncontent"), context(runtime: String(repeating: "a", count: 49))
        ]
        for metadata in invalid {
            await assertFailure(.invalidPrediction) { try await ledger.begin(context: metadata, id: UUID()) }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.appendingPathComponent(SelfPredictionLedger.fileName).path))
        for origin in [SelfPredictionLedger.ProbabilityOrigin.missing, .invalid, .hostRepair] {
            try await ledger.begin(context: context(origin: origin, probability: nil), id: UUID())
        }
        let root = try await exported(ledger)
        let events = try XCTUnwrap(root["events"] as? [[String: Any]])
        XCTAssertEqual(events.compactMap { $0["probabilityOrigin"] as? String }, ["missing", "invalid", "host_repair"])
        XCTAssertTrue(events.allSatisfy { $0["probability"] == nil })
    }

    func testEventLimitReservesPendingOutcomesAndNeverRotates() async throws {
        let url = try directory()
        let ledger = SelfPredictionLedger(directoryURL: url, maxEvents: 4)
        let idA = UUID(), idB = UUID()
        try await ledger.begin(context: context(), id: idA)
        try await ledger.begin(context: context(), id: idB)
        await assertFailure(.capacityExceeded) { try await ledger.begin(context: self.context(), id: UUID()) }
        try await ledger.complete(id: idB, status: .unavailable, toolBodyStarted: false, cancelled: false)
        try await ledger.complete(id: idA, status: .success, toolBodyStarted: true, cancelled: false)
        let before = try Data(contentsOf: url.appendingPathComponent(SelfPredictionLedger.fileName))
        await assertFailure(.capacityExceeded) { try await ledger.begin(context: self.context(), id: UUID()) }
        XCTAssertEqual(try Data(contentsOf: url.appendingPathComponent(SelfPredictionLedger.fileName)), before)
        let summary = try await ledger.summary()
        XCTAssertTrue(summary.isAtCapacity)
        XCTAssertEqual(summary.eventCount, 4)
    }

    func testByteLimitReservesOutcomeAndAllowsCompletionAtCapacity() async throws {
        let url = try directory()
        // One bounded prediction plus its 512-byte outcome reservation fits; two cannot.
        let ledger = SelfPredictionLedger(directoryURL: url, maxBytes: 1000)
        let emptySummary = try await ledger.summary()
        XCTAssertFalse(emptySummary.isAtCapacity)
        let id = UUID()
        try await ledger.begin(context: context(), id: id)
        await assertFailure(.capacityExceeded) { try await ledger.begin(context: self.context(), id: UUID()) }
        try await ledger.complete(id: id, status: .requiresApproval, toolBodyStarted: false, cancelled: true)
        let bytes = try Data(contentsOf: url.appendingPathComponent(SelfPredictionLedger.fileName))
        XCTAssertLessThanOrEqual(bytes.count, 1000)
        let reopened = SelfPredictionLedger(directoryURL: url, maxBytes: 1000)
        let summary = try await reopened.summary()
        XCTAssertEqual(summary.eventCount, 2)
        XCTAssertEqual(summary.pendingCount, 0)
    }

    func testCorruptOrAmbiguousExistingFilesArePreservedAndRefused() async throws {
        let source = try directory()
        let clean = SelfPredictionLedger(directoryURL: source)
        try await clean.begin(context: context(), id: UUID())
        let valid = try Data(contentsOf: source.appendingPathComponent(SelfPredictionLedger.fileName))
        let line = String(decoding: valid.dropLast(), as: UTF8.self)
        let corruptions: [Data] = [
            Data(valid.dropLast()), // A partial final line must not be silently ignored.
            valid + Data([0x0A]), // Empty interior line.
            valid + valid, // Duplicate prediction/sequence.
            Data((line.replacingOccurrences(of: "\"seq\":1", with: "\"seq\":2") + "\n").utf8),
            Data((line.replacingOccurrences(of: "{", with: "{\"unknown\":true,", options: [], range: line.startIndex..<line.index(after: line.startIndex)) + "\n").utf8),
            Data((line.replacingOccurrences(of: "{", with: "{\"seq\":1,", options: [], range: line.startIndex..<line.index(after: line.startIndex)) + "\n").utf8),
            Data("{\"kind\":\"outcome\"}\n".utf8), Data([0xFF, 0x0A])
        ]
        for bytes in corruptions {
            let url = try directory()
            let path = url.appendingPathComponent(SelfPredictionLedger.fileName)
            try bytes.write(to: path)
            let ledger = SelfPredictionLedger(directoryURL: url)
            await assertFailure(.corruptFile) { _ = try await ledger.exportJSON() }
            await assertFailure(.corruptFile) { try await ledger.begin(context: self.context(), id: UUID()) }
            XCTAssertEqual(try Data(contentsOf: path), bytes)
        }
    }

    func testOversizedExistingFileAndInvalidLimitsAreTypedFailures() async throws {
        let url = try directory()
        let path = url.appendingPathComponent(SelfPredictionLedger.fileName)
        let bytes = Data(repeating: 0x41, count: 33)
        try bytes.write(to: path)
        let oversized = SelfPredictionLedger(directoryURL: url, maxBytes: 32)
        await assertFailure(.capacityExceeded) { _ = try await oversized.exportJSON() }
        XCTAssertEqual(try Data(contentsOf: path), bytes)
        let invalid = SelfPredictionLedger(directoryURL: url, maxEvents: 0)
        await assertFailure(.invalidConfiguration) { _ = try await invalid.exportJSON() }
    }

    func testCanonicalFilesStillRequireUniquePredictionsAndMatchingSingleOutcomes() async throws {
        let url = try directory()
        let ledger = SelfPredictionLedger(directoryURL: url)
        let id = UUID()
        try await ledger.begin(context: context(), id: id)
        try await ledger.complete(id: id, status: .success, toolBodyStarted: true, cancelled: false)
        let root = try await exported(ledger)
        let events = try XCTUnwrap(root["events"] as? [[String: Any]])
        let controlURL = try directory()
        try canonicalLines(events).write(to: controlURL.appendingPathComponent(SelfPredictionLedger.fileName))
        let control = try await SelfPredictionLedger(directoryURL: controlURL).summary()
        XCTAssertEqual(control.eventCount, 2, "The unmodified fixture must be canonical before corruption tests")
        var duplicatePrediction = events[0]
        duplicatePrediction["seq"] = 2
        var orphanOutcome = events[1]
        orphanOutcome["seq"] = 1
        var duplicateOutcome = events[1]
        duplicateOutcome["seq"] = 3
        var invalidProbability = events[0]
        invalidProbability["probability"] = 1.1
        var mixedShape = events[1]
        mixedShape["runID"] = UUID().uuidString
        let invalidSequences = [
            [events[0], duplicatePrediction], [orphanOutcome],
            events + [duplicateOutcome], [invalidProbability], [events[0], mixedShape]
        ]
        for invalidEvents in invalidSequences {
            let directoryURL = try directory()
            let bytes = try canonicalLines(invalidEvents)
            let path = directoryURL.appendingPathComponent(SelfPredictionLedger.fileName)
            try bytes.write(to: path)
            let reopened = SelfPredictionLedger(directoryURL: directoryURL)
            await assertFailure(.corruptFile) { _ = try await reopened.exportJSON() }
            XCTAssertEqual(try Data(contentsOf: path), bytes)
        }
    }

    func testUncertainAppendPoisonsActorInsteadOfClaimingOrRetryingSuccess() async throws {
        let parent = try directory()
        let blockedDirectory = parent.appendingPathComponent("blocked", isDirectory: true)
        let ledger = SelfPredictionLedger(directoryURL: blockedDirectory)
        _ = try await ledger.exportJSON() // Lazy load observes no file, creates nothing.
        let obstruction = Data("synthetic obstruction".utf8)
        try obstruction.write(to: blockedDirectory)
        await assertFailure(.storageFailure) { try await ledger.begin(context: self.context(), id: UUID()) }
        await assertFailure(.storageFailure) { _ = try await ledger.exportJSON() }
        XCTAssertEqual(try Data(contentsOf: blockedDirectory), obstruction)
    }

    func testCancelledTaskCanPersistTerminalOutcome() async throws {
        let url = try directory()
        let ledger = SelfPredictionLedger(directoryURL: url)
        let id = UUID()
        try await ledger.begin(context: context(), id: id)
        let completion = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            XCTAssertTrue(Task.isCancelled)
            try await ledger.complete(id: id, status: .failed, toolBodyStarted: false, cancelled: true)
        }
        try await completion.value
        let root = try await exported(ledger)
        let events = try XCTUnwrap(root["events"] as? [[String: Any]])
        XCTAssertEqual(events.last?["cancelled"] as? Bool, true)
        XCTAssertEqual(events.last?["toolBodyStarted"] as? Bool, false)
        let summary = try await ledger.summary()
        XCTAssertEqual(summary.pendingCount, 0)
    }

    func testConcurrentPredictionsAreSerializedWithUniqueContiguousSequences() async throws {
        let url = try directory()
        let ledger = SelfPredictionLedger(directoryURL: url)
        let metadata = context()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<20 {
                group.addTask { try await ledger.begin(context: metadata, id: UUID()) }
            }
            try await group.waitForAll()
        }
        let reopened = SelfPredictionLedger(directoryURL: url)
        let root = try await exported(reopened)
        let events = try XCTUnwrap(root["events"] as? [[String: Any]])
        XCTAssertEqual(events.compactMap { $0["seq"] as? Int }, Array(1...20))
        XCTAssertEqual(Set(events.compactMap { $0["id"] as? String }).count, 20)
    }

    func testSymlinkIsNotFollowedOrOverwritten() async throws {
        let url = try directory()
        let target = url.appendingPathComponent("unrelated")
        let bytes = Data("synthetic unrelated bytes".utf8)
        try bytes.write(to: target)
        let path = url.appendingPathComponent(SelfPredictionLedger.fileName)
        try FileManager.default.createSymbolicLink(at: path, withDestinationURL: target)
        let ledger = SelfPredictionLedger(directoryURL: url)
        await assertFailure(.corruptFile) { _ = try await ledger.exportJSON() }
        XCTAssertEqual(try Data(contentsOf: target), bytes)
    }

    func testHostClockRollbackDoesNotInvalidateSequenceOrderedEvidence() async throws {
        let url = try directory()
        let ledger = SelfPredictionLedger(directoryURL: url)
        let id = UUID()
        try await ledger.begin(context: context(), id: id)
        try await ledger.complete(id: id, status: .success, toolBodyStarted: true, cancelled: false)
        let root = try await exported(ledger)
        var events = try XCTUnwrap(root["events"] as? [[String: Any]])
        events[0]["timestamp"] = "2026-10-25T02:00:00Z"
        events[1]["timestamp"] = "2026-10-25T01:59:00Z"
        let bytes = try canonicalLines(events)
        try bytes.write(to: url.appendingPathComponent(SelfPredictionLedger.fileName))
        let reopened = SelfPredictionLedger(directoryURL: url)
        let restored = try await exported(reopened)
        let restoredEvents = try XCTUnwrap(restored["events"] as? [[String: Any]])
        XCTAssertEqual(restoredEvents.compactMap { $0["seq"] as? Int }, [1, 2])
        XCTAssertEqual(restoredEvents[1]["timestamp"] as? String, "2026-10-25T01:59:00Z")
    }
}
