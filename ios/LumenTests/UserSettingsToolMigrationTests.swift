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

    private func isolatedDefaults(_ suffix: String) -> UserDefaults {
        let name = "UserSettingsToolMigrationTests.\(suffix).\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }
}
