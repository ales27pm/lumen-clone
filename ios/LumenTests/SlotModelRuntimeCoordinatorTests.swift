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
}
