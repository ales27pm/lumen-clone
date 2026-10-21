import Foundation
import XCTest
@testable import Lumen

@MainActor
final class CoreAIModelReplacementLifecycleTests: XCTestCase {
    private enum Failure: Error, Equatable { case admission, operation }

    func testAlreadyCancelledMutationDoesNotAdmitOrCommit() async {
        var calls: [String] = []
        let task = Task { @MainActor in
            withUnsafeCurrentTask { $0?.cancel() }
            try await CoreAITextEmbeddingModelStore.withModelReplacement(
                begin: { calls.append("begin") }, end: { calls.append("end") }
            ) { calls.append("commit") }
        }
        do { try await task.value; XCTFail("Cancellation must propagate") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertTrue(calls.isEmpty)
    }

    func testCancellationDuringDrainReleasesGateWithoutCommitting() async {
        var calls: [String] = []
        let task = Task { @MainActor in
            try await CoreAITextEmbeddingModelStore.withModelReplacement(
                begin: { calls.append("begin"); withUnsafeCurrentTask { $0?.cancel() } },
                end: { calls.append("end") }
            ) { calls.append("commit") }
        }
        do { try await task.value; XCTFail("Cancellation must propagate") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(calls, ["begin", "end"])
    }

    func testAdmissionFailureDoesNotReleaseSomeoneElsesGate() async {
        var calls: [String] = []
        do {
            try await CoreAITextEmbeddingModelStore.withModelReplacement(
                begin: { throw Failure.admission }, end: { calls.append("end") }
            ) { calls.append("commit") }
            XCTFail("Admission must fail")
        } catch { XCTAssertEqual(error as? Failure, .admission) }
        XCTAssertTrue(calls.isEmpty)
    }

    func testOperationFailureReleasesGateAndKeepsErrorCategory() async {
        var calls: [String] = []
        do {
            try await CoreAITextEmbeddingModelStore.withModelReplacement(
                begin: { calls.append("begin") }, end: { calls.append("end") }
            ) { calls.append("commit"); throw Failure.operation }
            XCTFail("Operation must fail")
        } catch { XCTAssertEqual(error as? Failure, .operation) }
        XCTAssertEqual(calls, ["begin", "commit", "end"])
    }

    func testCommittedOperationCompletesCleanupDespiteLateCancellation() async throws {
        var ended = false
        let task = Task { @MainActor in
            try await CoreAITextEmbeddingModelStore.withModelReplacement(
                begin: {}, end: { ended = true }
            ) {
                withUnsafeCurrentTask { $0?.cancel() }
                return "committed"
            }
        }
        let result = try await task.value
        XCTAssertEqual(result, "committed")
        XCTAssertTrue(ended)
    }

    func testSelectionRevisionDetectsABAAndKeepsExactCommittedSnapshot() throws {
        let name = "CoreAISelectionStateTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let state = CoreAITextEmbeddingModelStore.SelectionState(defaults: defaults)
        let firstLegacy = state.commit(false)
        let coreAI = state.commit(true)
        let secondLegacy = state.commit(false)
        XCTAssertFalse(firstLegacy.selected)
        XCTAssertTrue(coreAI.selected)
        XCTAssertFalse(secondLegacy.selected)
        XCTAssertNotEqual(firstLegacy.revision, secondLegacy.revision)
        XCTAssertEqual(state.snapshot, secondLegacy)
        XCTAssertNotEqual(state.snapshot, firstLegacy)
    }

    func testStaleRestorationDoesNotChangeTheProviderOrUnloadIt() async throws {
        let before = CoreAITextEmbeddingModelStore.selectionSnapshot
        let loaded = await CoreAITextEmbeddingService.shared.isLoaded()
        let stale = CoreAITextEmbeddingModelStore.SelectionSnapshot(selected: before.selected, revision: UUID())
        let restored = try await CoreAITextEmbeddingModelStore.restoreSelection(!before.selected, ifUnchanged: stale)
        XCTAssertFalse(restored)
        XCTAssertEqual(CoreAITextEmbeddingModelStore.selectionSnapshot, before)
        let afterLoaded = await CoreAITextEmbeddingService.shared.isLoaded()
        XCTAssertEqual(afterLoaded, loaded)
    }
}
