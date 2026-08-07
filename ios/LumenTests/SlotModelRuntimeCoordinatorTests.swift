import Foundation
import Testing
@testable import Lumen

@MainActor
struct SlotModelRuntimeCoordinatorTests {
    @Test func selectionEventUsesSelectedForFirstPreferredCandidate() {
        let coordinator = SlotModelRuntimeCoordinator.shared
        let preferredID = UUID().uuidString

        #expect(coordinator.selectionEvent(index: 0, candidateID: preferredID, preferredID: preferredID) == "selected")
    }

    @Test func selectionEventUsesFallbackForNonPreferredFirstCandidate() {
        let coordinator = SlotModelRuntimeCoordinator.shared
        let preferredID = UUID().uuidString

        #expect(coordinator.selectionEvent(index: 0, candidateID: UUID().uuidString, preferredID: preferredID) == "fallback_selected")
    }

    @Test func selectionEventUsesFallbackForLaterCandidate() {
        let coordinator = SlotModelRuntimeCoordinator.shared

        #expect(coordinator.selectionEvent(index: 1, candidateID: UUID().uuidString, preferredID: nil) == "fallback_selected")
    }

    @Test func unassignedLoadedChatFallbackIsNotAnAcceptedRuntimePath() {
        #expect(LumenModelSlotContract.runtimePathKind(for: "loadedChatFallback") == .unknown)
        #expect(!LumenModelSlotContract.executor.acceptsRuntimePath("loadedChatFallback"))
    }

    @Test func noAssignmentCannotContinueAnAlreadyLoadedChatRuntime() async {
        let coordinator = SlotModelRuntimeCoordinator.shared
        await coordinator.configure(assignments: [:], contextSize: 2_048, preferExclusiveChatRuntime: true)

        #expect(await coordinator.hasLoadedRuntimeReadyForContinuation(slot: .executor) == false)
    }

    @Test func deletingTheLoadedSharedBasePlansAFullChatRuntimeUnload() {
        let sharedPath = "/models/qwen3-shared.gguf"
        let plan = ModelRuntimeController.chatUnloadPlan(
            resolvedPath: sharedPath,
            loadedSharedPath: sharedPath,
            loadedSlotPaths: [.cortex: sharedPath, .executor: sharedPath]
        )

        #expect(plan == .allChat)
    }
}
