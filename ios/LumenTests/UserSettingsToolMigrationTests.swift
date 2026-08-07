import Testing
import Foundation
@testable import Lumen

struct UserSettingsToolMigrationTests {
    @Test func freshInstallEnablesAllRegistryTools() async throws {
        let defaults = isolatedDefaults("fresh")
        let settings = UserSettings(defaults: defaults)
        let registry = Set(ToolRegistry.all.map { ToolRouteGuard.canonicalToolID($0.id) })
        #expect(settings.enabledToolIDs == registry)
    }

    @Test func legacyEnabledToolIDsMigrateToDisabledOptOut() async throws {
        let defaults = isolatedDefaults("legacy")
        defaults.set(["weather"], forKey: "enabledToolIDs")
        let settings = UserSettings(defaults: defaults)
        #expect(settings.enabledToolIDs.contains("weather"))
        #expect(!settings.enabledToolIDs.contains("memory.save"))
        #expect(defaults.array(forKey: "disabledToolIDs") != nil)
    }

    @Test func toggleToolMutatesDerivedEnabledSet() async throws {
        let defaults = isolatedDefaults("toggle")
        let settings = UserSettings(defaults: defaults)
        settings.toggleTool("weather")
        #expect(!settings.enabledToolIDs.contains("weather"))
        settings.toggleTool("weather")
        #expect(settings.enabledToolIDs.contains("weather"))
    }

    @Test func enabledToolSetterCanonicalizesIncomingIDs() async throws {
        let defaults = isolatedDefaults("setter")
        let settings = UserSettings(defaults: defaults)
        settings.enabledToolIDs = ["WEATHER", "memory.search"]
        #expect(settings.enabledToolIDs.contains("weather"))
        #expect(settings.enabledToolIDs.contains("memory.recall"))
    }

    @Test func persistedDisabledIDsSurviveRegistryChurn() async throws {
        let defaults = isolatedDefaults("disabled-churn")
        defaults.set(["legacy.future.tool", "WEATHER"], forKey: "disabledToolIDs")
        let settings = UserSettings(defaults: defaults)
        #expect(!settings.enabledToolIDs.contains("weather"))
        let persisted = Set((defaults.array(forKey: "disabledToolIDs") as? [String]) ?? [])
        #expect(persisted.contains("legacy.future.tool"))
    }

    @Test func networkToolsSettingPersistsIntoSnapshots() async throws {
        let defaults = isolatedDefaults("network-tools")
        let settings = UserSettings(defaults: defaults)
        #expect(settings.networkToolsEnabled == false)

        settings.networkToolsEnabled = true

        #expect(UserSettings(defaults: defaults).networkToolsEnabled == true)
        #expect(SettingsSnapshot.loadFromDisk(defaults: defaults).networkToolsEnabled == true)
    }

    @Test func legacyModelSelectionMigratesIntoOneCanonicalRecord() async throws {
        let defaults = isolatedDefaults("model-selection-legacy")
        defaults.set("legacy-chat", forKey: "activeChatModelID")
        defaults.set("legacy-embed", forKey: "activeEmbeddingModelID")
        defaults.set(LumenModelFamily.qwen25.rawValue, forKey: "selectedModelFamilyID")

        let settings = UserSettings(defaults: defaults)
        let record = try #require(PersistedModelSelectionStore.load(defaults: defaults))

        #expect(settings.activeChatModelID == "legacy-chat")
        #expect(settings.activeEmbeddingModelID == "legacy-embed")
        #expect(record.chatModelID == "legacy-chat")
        #expect(record.embeddingModelID == "legacy-embed")
        #expect(record.familyID == LumenModelFamily.qwen25.rawValue)
    }

    @Test func atomicModelSelectionPersistsPairFamilyAndPlanTogether() async throws {
        let defaults = isolatedDefaults("model-selection-atomic")
        let settings = UserSettings(defaults: defaults)

        try settings.commitActiveModelSelection(
            chatModelID: "chat-v2",
            embeddingModelID: "embed-v2",
            family: .qwen3,
            provisioningPlanID: "plan-v2"
        )

        let record = try #require(PersistedModelSelectionStore.load(defaults: defaults))
        let reloaded = UserSettings(defaults: defaults)
        let snapshot = SettingsSnapshot.loadFromDisk(defaults: defaults)
        #expect(record.chatModelID == "chat-v2")
        #expect(record.embeddingModelID == "embed-v2")
        #expect(record.familyID == LumenModelFamily.qwen3.rawValue)
        #expect(record.provisioningPlanID == "plan-v2")
        #expect(reloaded.activeChatModelID == "chat-v2")
        #expect(reloaded.activeEmbeddingModelID == "embed-v2")
        #expect(snapshot.activeChatModelID == "chat-v2")
        #expect(snapshot.activeEmbeddingModelID == "embed-v2")
    }

    @Test func canonicalSelectionWinsOverTornLegacyMirrors() async throws {
        let defaults = isolatedDefaults("model-selection-torn-legacy")
        _ = try PersistedModelSelectionStore.commit(
            chatModelID: "canonical-chat",
            embeddingModelID: "canonical-embed",
            family: .qwen3,
            provisioningPlanID: "canonical-plan",
            defaults: defaults
        )
        defaults.set("stale-chat", forKey: "activeChatModelID")
        defaults.removeObject(forKey: "activeEmbeddingModelID")

        let settings = UserSettings(defaults: defaults)
        #expect(settings.activeChatModelID == "canonical-chat")
        #expect(settings.activeEmbeddingModelID == "canonical-embed")
    }

    @Test func unrelatedPreferenceChangesPreserveTheCompletedProvisioningPlan() async throws {
        let defaults = isolatedDefaults("model-selection-plan-preserved")
        let settings = UserSettings(defaults: defaults)
        try settings.commitActiveModelSelection(
            chatModelID: "verified-chat",
            embeddingModelID: "verified-embed",
            family: .qwen3,
            provisioningPlanID: "verified-plan"
        )

        settings.temperature = 0.25
        settings.networkToolsEnabled = true

        let record = try #require(PersistedModelSelectionStore.load(defaults: defaults))
        #expect(record.chatModelID == "verified-chat")
        #expect(record.embeddingModelID == "verified-embed")
        #expect(record.familyID == LumenModelFamily.qwen3.rawValue)
        #expect(record.provisioningPlanID == "verified-plan")
    }

    @Test func assigningTheSameModelIDsPreservesTheCompletedProvisioningPlan() async throws {
        let defaults = isolatedDefaults("model-selection-same-ids")
        let settings = UserSettings(defaults: defaults)
        try settings.commitActiveModelSelection(
            chatModelID: "verified-chat",
            embeddingModelID: "verified-embed",
            family: .qwen3,
            provisioningPlanID: "verified-plan"
        )

        settings.activeChatModelID = "verified-chat"
        settings.activeEmbeddingModelID = "verified-embed"

        let record = try #require(PersistedModelSelectionStore.load(defaults: defaults))
        #expect(record.chatModelID == "verified-chat")
        #expect(record.embeddingModelID == "verified-embed")
        #expect(record.provisioningPlanID == "verified-plan")
    }

    @Test func changingTheChatModelIDClearsTheCompletedProvisioningPlan() async throws {
        let defaults = isolatedDefaults("model-selection-changed-chat")
        let settings = UserSettings(defaults: defaults)
        try settings.commitActiveModelSelection(
            chatModelID: "verified-chat",
            embeddingModelID: "verified-embed",
            family: .qwen3,
            provisioningPlanID: "verified-plan"
        )

        settings.activeChatModelID = "replacement-chat"

        let record = try #require(PersistedModelSelectionStore.load(defaults: defaults))
        #expect(record.chatModelID == "replacement-chat")
        #expect(record.embeddingModelID == "verified-embed")
        #expect(record.provisioningPlanID == nil)
    }

    @Test func changingTheEmbeddingModelIDClearsTheCompletedProvisioningPlan() async throws {
        let defaults = isolatedDefaults("model-selection-changed-embedding")
        let settings = UserSettings(defaults: defaults)
        try settings.commitActiveModelSelection(
            chatModelID: "verified-chat",
            embeddingModelID: "verified-embed",
            family: .qwen3,
            provisioningPlanID: "verified-plan"
        )

        settings.activeEmbeddingModelID = "replacement-embed"

        let record = try #require(PersistedModelSelectionStore.load(defaults: defaults))
        #expect(record.chatModelID == "verified-chat")
        #expect(record.embeddingModelID == "replacement-embed")
        #expect(record.provisioningPlanID == nil)
    }

    private func isolatedDefaults(_ suffix: String) -> UserDefaults {
        let name = "UserSettingsToolMigrationTests.\(suffix).\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }
}
