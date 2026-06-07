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

    private func isolatedDefaults(_ suffix: String) -> UserDefaults {
        let name = "UserSettingsToolMigrationTests.\(suffix).\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }
}
