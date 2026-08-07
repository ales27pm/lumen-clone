import Foundation
import Testing
@testable import Lumen

@MainActor
struct ModelProvisioningReceiptTests {
    @Test func deferredSetupDoesNotCountAsConsentOrCompletion() {
        let defaults = isolatedDefaults("deferred")

        #expect(ModelProvisioningReceipt.markDeferred(family: .qwen3, defaults: defaults))
        #expect(ModelProvisioningReceipt.status(family: .qwen3, defaults: defaults) == .deferred)
        #expect(!ModelProvisioningReceipt.isConsented(family: .qwen3, defaults: defaults))
        #expect(!ModelProvisioningReceipt.isCurrent(family: .qwen3, defaults: defaults))
    }

    @Test func completedReceiptRequiresTheAtomicPairAndMatchingPlan() throws {
        let defaults = isolatedDefaults("completed")
        let planID = ModelProvisioningReceipt.catalogIdentity(for: .qwen3)
        _ = try PersistedModelSelectionStore.commit(
            chatModelID: "chat-a",
            embeddingModelID: "embed-a",
            family: .qwen3,
            provisioningPlanID: planID,
            defaults: defaults
        )

        #expect(ModelProvisioningReceipt.markCurrent(
            family: .qwen3,
            chatModelID: "chat-a",
            embeddingModelID: "embed-a",
            defaults: defaults
        ))
        #expect(ModelProvisioningReceipt.isCurrent(family: .qwen3, defaults: defaults))
        #expect(!ModelProvisioningReceipt.isCurrent(
            family: .qwen3,
            chatModelID: "different-chat",
            embeddingModelID: "embed-a",
            defaults: defaults
        ))

        _ = try PersistedModelSelectionStore.commit(
            chatModelID: "chat-a",
            embeddingModelID: "embed-a",
            family: .qwen3,
            provisioningPlanID: nil,
            defaults: defaults
        )
        #expect(!ModelProvisioningReceipt.isCurrent(family: .qwen3, defaults: defaults))
    }

    @Test func staleFamilyFingerprintDoesNotAuthorizeAnotherFamily() {
        let defaults = isolatedDefaults("family")
        #expect(ModelProvisioningReceipt.markConsented(family: .qwen3, defaults: defaults))
        #expect(ModelProvisioningReceipt.isConsented(family: .qwen3, defaults: defaults))
        #expect(!ModelProvisioningReceipt.isConsented(family: .qwen25, defaults: defaults))
    }

    @Test func relaunchAfterTargetSelectionWriteRestoresPreviousCompletedState() throws {
        let defaults = isolatedDefaults("journal-selection-crash")
        try installCompletedState(
            family: .qwen3,
            chatModelID: "previous-chat",
            embeddingModelID: "previous-embed",
            defaults: defaults
        )
        #expect(ModelProvisioningSwitchJournalStore.prepare(targetFamily: .qwen25, defaults: defaults))
        #expect(ModelProvisioningReceipt.status(family: .qwen3, defaults: defaults) == .completed)
        #expect(ModelProvisioningReceipt.status(family: .qwen25, defaults: defaults) == nil)

        _ = try PersistedModelSelectionStore.commit(
            chatModelID: "target-chat",
            embeddingModelID: "target-embed",
            family: .qwen25,
            provisioningPlanID: ModelProvisioningReceipt.catalogIdentity(for: .qwen25),
            defaults: defaults
        )

        let relaunched = UserSettings(defaults: defaults)
        let recovered = try #require(PersistedModelSelectionStore.load(defaults: defaults))
        #expect(relaunched.activeChatModelID == "previous-chat")
        #expect(relaunched.activeEmbeddingModelID == "previous-embed")
        #expect(recovered.familyID == LumenModelFamily.qwen3.rawValue)
        #expect(ModelProvisioningReceipt.isCurrent(family: .qwen3, defaults: defaults))
        #expect(defaults.data(forKey: ModelProvisioningSwitchJournalStore.defaultsKey) == nil)
    }

    @Test func relaunchAfterTargetReceiptWriteStillRollsBackPreparedTransaction() throws {
        let defaults = isolatedDefaults("journal-receipt-crash")
        try installCompletedState(
            family: .qwen3,
            chatModelID: "previous-chat",
            embeddingModelID: "previous-embed",
            defaults: defaults
        )
        #expect(ModelProvisioningSwitchJournalStore.prepare(targetFamily: .qwen25, defaults: defaults))
        let targetPlanID = ModelProvisioningReceipt.catalogIdentity(for: .qwen25)
        _ = try PersistedModelSelectionStore.commit(
            chatModelID: "target-chat",
            embeddingModelID: "target-embed",
            family: .qwen25,
            provisioningPlanID: targetPlanID,
            defaults: defaults
        )
        #expect(ModelProvisioningReceipt.markCurrent(
            family: .qwen25,
            chatModelID: "target-chat",
            embeddingModelID: "target-embed",
            defaults: defaults
        ))

        let relaunched = UserSettings(defaults: defaults)
        let recovered = try #require(PersistedModelSelectionStore.load(defaults: defaults))
        #expect(relaunched.activeChatModelID == "previous-chat")
        #expect(relaunched.activeEmbeddingModelID == "previous-embed")
        #expect(recovered.familyID == LumenModelFamily.qwen3.rawValue)
        #expect(ModelProvisioningReceipt.isCurrent(family: .qwen3, defaults: defaults))
        #expect(ModelProvisioningReceipt.status(family: .qwen25, defaults: defaults) == nil)
        #expect(defaults.data(forKey: ModelProvisioningSwitchJournalStore.defaultsKey) == nil)
    }

    @Test func relaunchRollbackRemovesTargetReceiptWhenNoPreviousReceiptExisted() throws {
        let defaults = isolatedDefaults("journal-no-previous-receipt")
        _ = try PersistedModelSelectionStore.commit(
            chatModelID: "previous-chat",
            embeddingModelID: "previous-embed",
            family: .qwen3,
            provisioningPlanID: nil,
            defaults: defaults
        )
        #expect(ModelProvisioningSwitchJournalStore.prepare(targetFamily: .qwen25, defaults: defaults))
        _ = try PersistedModelSelectionStore.commit(
            chatModelID: "target-chat",
            embeddingModelID: "target-embed",
            family: .qwen25,
            provisioningPlanID: ModelProvisioningReceipt.catalogIdentity(for: .qwen25),
            defaults: defaults
        )
        #expect(ModelProvisioningReceipt.markCurrent(
            family: .qwen25,
            chatModelID: "target-chat",
            embeddingModelID: "target-embed",
            defaults: defaults
        ))

        let relaunched = UserSettings(defaults: defaults)
        let recovered = try #require(PersistedModelSelectionStore.load(defaults: defaults))
        #expect(relaunched.activeChatModelID == "previous-chat")
        #expect(relaunched.activeEmbeddingModelID == "previous-embed")
        #expect(recovered.familyID == LumenModelFamily.qwen3.rawValue)
        #expect(recovered.provisioningPlanID == nil)
        #expect(defaults.data(forKey: ModelProvisioningReceipt.defaultsKey) == nil)
        #expect(defaults.data(forKey: ModelProvisioningSwitchJournalStore.defaultsKey) == nil)
    }

    @Test func relaunchAfterCommittedMarkerRetainsExactTargetState() throws {
        let defaults = isolatedDefaults("journal-committed-crash")
        try installCompletedState(
            family: .qwen3,
            chatModelID: "previous-chat",
            embeddingModelID: "previous-embed",
            defaults: defaults
        )
        #expect(ModelProvisioningSwitchJournalStore.prepare(targetFamily: .qwen25, defaults: defaults))
        let targetPlanID = ModelProvisioningReceipt.catalogIdentity(for: .qwen25)
        _ = try PersistedModelSelectionStore.commit(
            chatModelID: "target-chat",
            embeddingModelID: "target-embed",
            family: .qwen25,
            provisioningPlanID: targetPlanID,
            defaults: defaults
        )
        #expect(ModelProvisioningReceipt.markCurrent(
            family: .qwen25,
            chatModelID: "target-chat",
            embeddingModelID: "target-embed",
            defaults: defaults
        ))
        #expect(ModelProvisioningSwitchJournalStore.markCommitted(
            targetFamily: .qwen25,
            chatModelID: "target-chat",
            embeddingModelID: "target-embed",
            provisioningPlanID: targetPlanID,
            defaults: defaults
        ))

        let relaunched = UserSettings(defaults: defaults)
        let retained = try #require(PersistedModelSelectionStore.load(defaults: defaults))
        #expect(relaunched.activeChatModelID == "target-chat")
        #expect(relaunched.activeEmbeddingModelID == "target-embed")
        #expect(retained.familyID == LumenModelFamily.qwen25.rawValue)
        #expect(retained.provisioningPlanID == targetPlanID)
        #expect(ModelProvisioningReceipt.isCurrent(family: .qwen25, defaults: defaults))
        #expect(defaults.data(forKey: ModelProvisioningSwitchJournalStore.defaultsKey) == nil)
    }

    @Test func relaunchRejectsCommittedMarkerWhenTargetStateIsTorn() throws {
        let defaults = isolatedDefaults("journal-committed-torn")
        try installCompletedState(
            family: .qwen3,
            chatModelID: "previous-chat",
            embeddingModelID: "previous-embed",
            defaults: defaults
        )
        #expect(ModelProvisioningSwitchJournalStore.prepare(targetFamily: .qwen25, defaults: defaults))
        let targetPlanID = ModelProvisioningReceipt.catalogIdentity(for: .qwen25)
        _ = try PersistedModelSelectionStore.commit(
            chatModelID: "target-chat",
            embeddingModelID: "target-embed",
            family: .qwen25,
            provisioningPlanID: targetPlanID,
            defaults: defaults
        )
        #expect(ModelProvisioningReceipt.markCurrent(
            family: .qwen25,
            chatModelID: "target-chat",
            embeddingModelID: "target-embed",
            defaults: defaults
        ))
        #expect(ModelProvisioningSwitchJournalStore.markCommitted(
            targetFamily: .qwen25,
            chatModelID: "target-chat",
            embeddingModelID: "target-embed",
            provisioningPlanID: targetPlanID,
            defaults: defaults
        ))

        _ = try PersistedModelSelectionStore.commit(
            chatModelID: "torn-chat",
            embeddingModelID: "target-embed",
            family: .qwen25,
            provisioningPlanID: targetPlanID,
            defaults: defaults
        )

        let relaunched = UserSettings(defaults: defaults)
        let recovered = try #require(PersistedModelSelectionStore.load(defaults: defaults))
        #expect(relaunched.activeChatModelID == "previous-chat")
        #expect(relaunched.activeEmbeddingModelID == "previous-embed")
        #expect(recovered.familyID == LumenModelFamily.qwen3.rawValue)
        #expect(ModelProvisioningReceipt.isCurrent(family: .qwen3, defaults: defaults))
        #expect(defaults.data(forKey: ModelProvisioningSwitchJournalStore.defaultsKey) == nil)
    }

    private func installCompletedState(
        family: LumenModelFamily,
        chatModelID: String,
        embeddingModelID: String,
        defaults: UserDefaults
    ) throws {
        _ = try PersistedModelSelectionStore.commit(
            chatModelID: chatModelID,
            embeddingModelID: embeddingModelID,
            family: family,
            provisioningPlanID: ModelProvisioningReceipt.catalogIdentity(for: family),
            defaults: defaults
        )
        #expect(ModelProvisioningReceipt.markCurrent(
            family: family,
            chatModelID: chatModelID,
            embeddingModelID: embeddingModelID,
            defaults: defaults
        ))
    }

    private func isolatedDefaults(_ suffix: String) -> UserDefaults {
        let name = "ModelProvisioningReceiptTests.\(suffix).\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }
}
