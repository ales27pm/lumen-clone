import Foundation
import XCTest
@testable import Lumen

@MainActor
final class EmbeddingSelectionTransactionTests: XCTestCase {
    private enum Failure: Error, Equatable { case load, commit, restore }
    private typealias Snapshot = CoreAITextEmbeddingModelStore.SelectionSnapshot

    /// Every explicit choice gets a revision, even when its Boolean is unchanged.
    private final class Selection {
        var snapshot = Snapshot(selected: true, revision: UUID())
        var ids = "old"
        var restored: [Bool] = []

        @discardableResult func choose(_ selected: Bool) -> Snapshot {
            snapshot = Snapshot(selected: selected, revision: UUID())
            return snapshot
        }

        func restore(_ selected: Bool, ifUnchanged expected: Snapshot) -> Bool {
            guard snapshot == expected else { return false }
            restored.append(selected)
            choose(selected)
            return true
        }
    }

    func testFailedLegacyLoadRestoresPreviouslySelectedCoreAIAndIDs() async {
        let selection = Selection()
        var attemptedIDs = selection.ids
        do {
            try await ModelRuntimeController().preservingEmbeddingSelection(
                for: .embedding,
                readSelection: { selection.snapshot },
                chooseLegacy: { selection.choose(false) },
                restoreSelection: { selection.restore($0, ifUnchanged: $1) },
                isCurrent: { selection.ids == attemptedIDs },
                rollback: { selection.ids = "old" }
            ) {
                selection.ids = "attempted"
                attemptedIDs = selection.ids
                throw Failure.load
            }
            XCTFail("Loading must fail")
        } catch { XCTAssertEqual(error as? Failure, .load) }
        XCTAssertTrue(selection.snapshot.selected)
        XCTAssertEqual(selection.ids, "old")
        XCTAssertEqual(selection.restored, [true])
    }

    func testFailedSelectionCommitRetainsCoreAIAndPreviouslySavedIDs() async {
        let selection = Selection()
        do {
            try await ModelRuntimeController().preservingEmbeddingSelection(
                for: .embedding,
                readSelection: { selection.snapshot },
                chooseLegacy: { selection.choose(false) },
                restoreSelection: { selection.restore($0, ifUnchanged: $1) },
                isCurrent: { selection.ids == "old" }
            ) { throw Failure.commit }
            XCTFail("A rejected commit must fail")
        } catch { XCTAssertEqual(error as? Failure, .commit) }
        XCTAssertTrue(selection.snapshot.selected)
        XCTAssertEqual(selection.ids, "old")
    }

    func testSuccessfulLegacyLoadKeepsTheNewProviderSelection() async throws {
        let selection = Selection()
        try await ModelRuntimeController().preservingEmbeddingSelection(
            for: .embedding,
            readSelection: { selection.snapshot },
            chooseLegacy: { selection.choose(false) },
            restoreSelection: { _, _ in XCTFail("Successful work must not roll back"); return false }
        ) { selection.ids = "new" }
        XCTAssertFalse(selection.snapshot.selected)
        XCTAssertEqual(selection.ids, "new")
    }

    func testCancellationAlsoRestoresThePreviousProvider() async {
        let selection = Selection()
        let task = Task { @MainActor in
            try await ModelRuntimeController().preservingEmbeddingSelection(
                for: .embedding,
                readSelection: { selection.snapshot },
                chooseLegacy: { selection.choose(false) },
                restoreSelection: { selection.restore($0, ifUnchanged: $1) }
            ) {
                withUnsafeCurrentTask { $0?.cancel() }
                try Task.checkCancellation()
            }
        }
        do {
            try await task.value
            XCTFail("Cancellation must propagate")
        } catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertTrue(selection.snapshot.selected)
    }

    func testChatLoadFailureDoesNotOverrideAnIndependentEmbeddingSelection() async {
        let selection = Selection()
        do {
            try await ModelRuntimeController().preservingEmbeddingSelection(
                for: .chat,
                readSelection: { selection.snapshot },
                chooseLegacy: { XCTFail("Chat work must not choose an embedding provider"); return selection.snapshot },
                restoreSelection: { _, _ in XCTFail("Chat work must not change embeddings"); return false }
            ) { selection.choose(false); throw Failure.load }
            XCTFail("Loading must fail")
        } catch { XCTAssertEqual(error as? Failure, .load) }
        XCTAssertFalse(selection.snapshot.selected)
    }

    func testRestorationFailureIsReportedWithoutRestoringIDs() async {
        let selection = Selection()
        var restoredIDs = false
        do {
            try await ModelRuntimeController().preservingEmbeddingSelection(
                for: .embedding,
                readSelection: { selection.snapshot },
                chooseLegacy: { selection.choose(false) },
                restoreSelection: { _, _ in throw Failure.restore },
                rollback: { restoredIDs = true }
            ) { throw Failure.load }
            XCTFail("Restoration must fail")
        } catch { XCTAssertEqual(error as? Failure, .restore) }
        XCTAssertFalse(restoredIDs)
    }

    func testFailedOldLoadDoesNotRollBackNewerPersistedSelection() async {
        let selection = Selection()
        var attemptedIDs = selection.ids
        do {
            try await ModelRuntimeController().preservingEmbeddingSelection(
                for: .embedding,
                readSelection: { selection.snapshot },
                chooseLegacy: { selection.choose(false) },
                restoreSelection: { selection.restore($0, ifUnchanged: $1) },
                isCurrent: { selection.ids == attemptedIDs },
                rollback: { XCTFail("A failed old load cannot overwrite newer IDs") }
            ) {
                selection.ids = "attempted"
                attemptedIDs = selection.ids
                await Task.yield()
                selection.choose(false)
                selection.ids = "newer"
                throw Failure.load
            }
            XCTFail("Loading must fail")
        } catch { XCTAssertEqual(error as? Failure, .load) }
        XCTAssertFalse(selection.snapshot.selected)
        XCTAssertEqual(selection.ids, "newer")
        XCTAssertTrue(selection.restored.isEmpty)
    }

    func testProviderABAChangeWithSameIDsDoesNotRestoreAnOldChoice() async {
        let selection = Selection()
        do {
            try await ModelRuntimeController().preservingEmbeddingSelection(
                for: .embedding,
                readSelection: { selection.snapshot },
                chooseLegacy: { selection.choose(false) },
                restoreSelection: { selection.restore($0, ifUnchanged: $1) },
                rollback: { XCTFail("The newer provider revision must retain its IDs") }
            ) {
                selection.choose(true)
                selection.choose(false)
                throw Failure.load
            }
            XCTFail("Loading must fail")
        } catch { XCTAssertEqual(error as? Failure, .load) }
        XCTAssertFalse(selection.snapshot.selected)
        XCTAssertTrue(selection.restored.isEmpty)
    }

    func testIndependentChatIDChangeStillRestoresTheOwnedProviderWithoutOverwritingIDs() async {
        let selection = Selection()
        do {
            try await ModelRuntimeController().preservingEmbeddingSelection(
                for: .embedding,
                readSelection: { selection.snapshot },
                chooseLegacy: { selection.choose(false) },
                restoreSelection: { selection.restore($0, ifUnchanged: $1) },
                isCurrent: { selection.ids == "old" },
                rollback: { XCTFail("The independent newer chat selection must survive") }
            ) {
                selection.ids = "new-chat"
                throw Failure.load
            }
            XCTFail("Loading must fail")
        } catch { XCTAssertEqual(error as? Failure, .load) }
        XCTAssertTrue(selection.snapshot.selected)
        XCTAssertEqual(selection.ids, "new-chat")
        XCTAssertEqual(selection.restored, [true])
    }

    func testNewerProviderBeforeCommitPreventsTheOldCommit() async {
        let selection = Selection()
        var committed = false
        do {
            try await ModelRuntimeController().preservingEmbeddingSelection(
                for: .embedding,
                readSelection: { selection.snapshot },
                chooseLegacy: {
                    let owned = selection.choose(false)
                    selection.choose(true)
                    return owned
                },
                restoreSelection: { selection.restore($0, ifUnchanged: $1) }
            ) { committed = true }
            XCTFail("A superseded request must fail")
        } catch ModelRuntimeController.Failure.selectionSuperseded {
        } catch { XCTFail("Unexpected failure category") }
        XCTAssertFalse(committed)
        XCTAssertTrue(selection.snapshot.selected)
        XCTAssertTrue(selection.restored.isEmpty)
    }

    func testIDsChangedWhileProviderRestoreWasSuspendedArePreserved() async {
        let selection = Selection()
        do {
            try await ModelRuntimeController().preservingEmbeddingSelection(
                for: .embedding,
                readSelection: { selection.snapshot },
                chooseLegacy: { selection.choose(false) },
                restoreSelection: {
                    let restored = selection.restore($0, ifUnchanged: $1)
                    await Task.yield()
                    selection.ids = "newer"
                    return restored
                },
                isCurrent: { selection.ids == "old" },
                rollback: { XCTFail("Reentrant newer IDs must survive provider restoration") }
            ) { throw Failure.load }
            XCTFail("Loading must fail")
        } catch { XCTAssertEqual(error as? Failure, .load) }
        XCTAssertEqual(selection.ids, "newer")
    }

    func testSuccessfulButSupersededLoadDoesNotPublishSuccess() async {
        let selection = Selection()
        do {
            try await ModelRuntimeController().preservingEmbeddingSelection(
                for: .embedding,
                readSelection: { selection.snapshot },
                chooseLegacy: { selection.choose(false) },
                restoreSelection: { selection.restore($0, ifUnchanged: $1) },
                rollback: { XCTFail("A newer provider revision cannot be rolled back") }
            ) { selection.choose(true) }
            XCTFail("A stale completed load must not report success")
        } catch ModelRuntimeController.Failure.selectionSuperseded {
        } catch { XCTFail("Unexpected failure category") }
        XCTAssertTrue(selection.snapshot.selected)
        XCTAssertTrue(selection.restored.isEmpty)
    }

    func testFailedProviderSelectionDoesNotRunThePersistedCommit() async {
        let selection = Selection()
        var committed = false
        do {
            try await ModelRuntimeController().preservingEmbeddingSelection(
                for: .embedding,
                readSelection: { selection.snapshot },
                chooseLegacy: { throw CancellationError() },
                restoreSelection: { _, _ in XCTFail("No revision was changed"); return false }
            ) { committed = true }
            XCTFail("Cancellation must propagate")
        } catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertFalse(committed)
        XCTAssertTrue(selection.snapshot.selected)
        XCTAssertEqual(selection.ids, "old")
    }
}
