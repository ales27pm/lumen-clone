import XCTest
@testable import Lumen

final class MemoryCaptureQueueTests: XCTestCase {
    override func setUp() {
        super.setUp()
        DiskWriteBudget.shared.setGenerationActive(false)
    }

    override func tearDown() {
        DiskWriteBudget.shared.setGenerationActive(false)
        super.tearDown()
    }

    @MainActor
    func testEnqueueStoresTrimmedCaptureAndDedupesContent() throws {
        let fileURL = temporaryQueueURL()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let first = try MemoryCaptureQueue.enqueue(
            content: "  Remember that I prefer terse status updates.  ",
            kind: .preference,
            source: "app-intent-pending",
            topic: "preferences",
            createdAt: Date(timeIntervalSince1970: 1_700_000_000),
            fileURL: fileURL
        )
        let duplicate = try MemoryCaptureQueue.enqueue(
            content: "remember that i prefer terse status updates.",
            kind: .preference,
            source: "app-intent-pending",
            topic: "preferences",
            fileURL: fileURL
        )

        let pending = try MemoryCaptureQueue.loadPending(fileURL: fileURL)
        XCTAssertEqual(pending.count, 1)
        XCTAssertEqual(first.id, duplicate.id)
        XCTAssertEqual(pending.first?.content, "Remember that I prefer terse status updates.")
        XCTAssertEqual(pending.first?.kind, .preference)
        XCTAssertEqual(pending.first?.source, "app-intent-pending")
        XCTAssertEqual(pending.first?.topic, "preferences")
    }

    @MainActor
    func testDrainPromotesAndRemovesBoundedCaptures() async throws {
        let fileURL = temporaryQueueURL()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        _ = try MemoryCaptureQueue.enqueue(content: "first", fileURL: fileURL)
        _ = try MemoryCaptureQueue.enqueue(content: "second", fileURL: fileURL)

        var promoted: [String] = []
        let result = await MemoryCaptureQueue.drain(maxItems: 1, fileURL: fileURL) { capture in
            promoted.append(capture.content)
        }

        XCTAssertEqual(result.attempted, 1)
        XCTAssertEqual(result.promoted, 1)
        XCTAssertEqual(result.remaining, 1)
        XCTAssertNil(result.skippedReason)
        XCTAssertEqual(promoted, ["first"])
        XCTAssertEqual(try MemoryCaptureQueue.loadPending(fileURL: fileURL).map(\.content), ["second"])
    }

    @MainActor
    func testDrainRetainsFailedAndUnattemptedCaptures() async throws {
        let fileURL = temporaryQueueURL()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        _ = try MemoryCaptureQueue.enqueue(content: "first", fileURL: fileURL)
        _ = try MemoryCaptureQueue.enqueue(content: "second", fileURL: fileURL)

        let result = await MemoryCaptureQueue.drain(maxItems: 2, fileURL: fileURL) { capture in
            if capture.content == "first" {
                throw TestPromotionError.embeddingUnavailable
            }
        }

        let pending = try MemoryCaptureQueue.loadPending(fileURL: fileURL)
        XCTAssertEqual(result.attempted, 1)
        XCTAssertEqual(result.promoted, 0)
        XCTAssertEqual(result.remaining, 2)
        XCTAssertNotNil(result.lastError)
        XCTAssertEqual(pending.map(\.content), ["first", "second"])
        XCTAssertEqual(pending.first?.retryCount, 1)
        XCTAssertNotNil(pending.first?.lastError)
    }

    @MainActor
    func testDrainSkipsWhenLimitIsZero() async throws {
        let fileURL = temporaryQueueURL()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        _ = try MemoryCaptureQueue.enqueue(content: "first", fileURL: fileURL)

        let result = await MemoryCaptureQueue.drain(maxItems: 0, fileURL: fileURL) { _ in
            XCTFail("zero-limit drain must not promote")
        }

        XCTAssertEqual(result.skippedReason, "empty_drain_limit")
        XCTAssertEqual(result.remaining, 1)
    }

    @MainActor
    func testDiagnosticsSnapshotExposesQueueMetadataOnly() async throws {
        let fileURL = temporaryQueueURL()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let oldest = Date(timeIntervalSince1970: 1_700_000_000)
        _ = try MemoryCaptureQueue.enqueue(
            content: "private preference text",
            createdAt: oldest,
            fileURL: fileURL
        )
        _ = try MemoryCaptureQueue.enqueue(
            content: "another local memory",
            createdAt: oldest.addingTimeInterval(120),
            fileURL: fileURL
        )

        _ = await MemoryCaptureQueue.drain(maxItems: 1, fileURL: fileURL) { _ in
            throw TestPromotionError.embeddingUnavailable
        }

        let snapshot = try MemoryCaptureQueue.diagnosticsSnapshot(fileURL: fileURL)

        XCTAssertEqual(snapshot.pendingCount, 2)
        XCTAssertEqual(snapshot.oldestCreatedAt, oldest)
        XCTAssertEqual(snapshot.maxRetryCount, 1)
        XCTAssertEqual(snapshot.lastError, "embeddingunavailable")
    }

    private func temporaryQueueURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("lumen-memory-capture-\(UUID().uuidString)", isDirectory: false)
            .appendingPathExtension("json")
    }
}

private enum TestPromotionError: Error {
    case embeddingUnavailable
}
