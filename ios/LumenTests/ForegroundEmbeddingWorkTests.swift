import Foundation
import XCTest
@testable import Lumen

@MainActor
final class ForegroundEmbeddingWorkTests: XCTestCase {
    func testColdForegroundOperationPreparesBeforeItsFirstVectorRequest() async throws {
        var loaded = false
        var calls: [String] = []
        let result = try await ForegroundEmbeddingWork.run(isForeground: true) {
            calls.append("prepare")
            loaded = true
            return true
        } operation: {
            XCTAssertTrue(loaded)
            calls.append("embed")
            return 1
        }
        XCTAssertEqual(result, 1)
        XCTAssertEqual(calls, ["prepare", "embed"])
    }

    func testUnavailableSelectedEncoderDoesNotStartPersistenceOrIndexReplacement() async {
        var submitted = false
        do {
            _ = try await ForegroundEmbeddingWork.run(isForeground: true, prepare: { false }) {
                submitted = true
            }
            XCTFail("An unavailable selected encoder must stop the explicit operation")
        } catch ForegroundEmbeddingWork.Failure.embeddingUnavailable {
        } catch {
            XCTFail("Unexpected error category")
        }
        XCTAssertFalse(submitted)
    }

    func testBackgroundEntryNeverPreparesOrEmbeds() async {
        var prepared = false
        var submitted = false
        do {
            _ = try await ForegroundEmbeddingWork.run(isForeground: false, prepare: {
                prepared = true
                return true
            }) {
                submitted = true
            }
            XCTFail("Background entry must be rejected")
        } catch ForegroundEmbeddingWork.Failure.notForeground {
        } catch {
            XCTFail("Unexpected error category")
        }
        XCTAssertFalse(prepared)
        XCTAssertFalse(submitted)
    }

    func testCancellationDuringPreparationDoesNotSubmitEmbeddingWork() async {
        var submitted = false
        let task = Task { @MainActor in
            try await ForegroundEmbeddingWork.run(isForeground: true, prepare: {
                withUnsafeCurrentTask { $0?.cancel() }
                return true
            }) {
                submitted = true
            }
        }
        do {
            _ = try await task.value
            XCTFail("Cancellation must be terminal")
        } catch is CancellationError {
        } catch {
            XCTFail("Unexpected error category")
        }
        XCTAssertFalse(submitted)
    }

    func testCancellationAfterEmbeddingDoesNotPublishLateSuccess() async {
        let task = Task { @MainActor in
            try await ForegroundEmbeddingWork.run(isForeground: true, prepare: { true }) {
                withUnsafeCurrentTask { $0?.cancel() }
                return 1
            }
        }
        do {
            _ = try await task.value
            XCTFail("A cancelled operation must not report late success")
        } catch is CancellationError {
        } catch {
            XCTFail("Unexpected error category")
        }
    }

    func testPreparationFailureKeepsItsErrorAndNeverStartsTheOperation() async {
        enum PreparationError: Error { case storageUnavailable }
        var submitted = false
        do {
            _ = try await ForegroundEmbeddingWork.run(isForeground: true, prepare: {
                throw PreparationError.storageUnavailable
            }) {
                submitted = true
            }
            XCTFail("Preparation failure must propagate")
        } catch PreparationError.storageUnavailable {
        } catch {
            XCTFail("Unexpected error category")
        }
        XCTAssertFalse(submitted)
    }

    func testUserSourceAndMemoryActionsUseTheForegroundPreparationBoundary() throws {
        let iosRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let sources = try String(contentsOf: iosRoot.appendingPathComponent("Lumen/Views/SourcesView.swift"), encoding: .utf8)
        let memory = try String(contentsOf: iosRoot.appendingPathComponent("Lumen/Views/MemoryView.swift"), encoding: .utf8)
        let bookmark = try String(contentsOf: iosRoot.appendingPathComponent("Lumen/Views/MessageBubble.swift"), encoding: .utf8)
        XCTAssertEqual(sources.components(separatedBy: "ForegroundEmbeddingWork.run(").count - 1, 2)
        XCTAssertTrue(sources.contains("private func startIndexing("))
        XCTAssertEqual(sources.components(separatedBy: "startIndexing {").count - 1, 3)
        XCTAssertTrue(memory.contains("ForegroundEmbeddingWork.run("))
        XCTAssertTrue(bookmark.contains("ForegroundEmbeddingWork.run("))
        let boundary = try String(contentsOf: iosRoot.appendingPathComponent("Lumen/Services/ForegroundEmbeddingWork.swift"), encoding: .utf8)
        XCTAssertTrue(boundary.contains("ModelLoader.ensureEmbedLoaded("))
        XCTAssertTrue(boundary.contains("intent: .userChat"))
    }
}
