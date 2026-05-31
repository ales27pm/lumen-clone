import Foundation
#if canImport(AppIntents)
import AppIntents

@available(iOS 16.0, *)
struct LumenAppShortcuts: AppShortcutsProvider {
    @AppShortcutsBuilder
    static var appShortcuts: [AppShortcut] {
        AppShortcut(intent: LumenAskIntent(), phrases: ["Ask Lumen \(.applicationName)"], shortTitle: "Ask Lumen", systemImageName: "bubble.left")
        AppShortcut(intent: LumenMemorySearchIntent(), phrases: ["Search Lumen memory in \(.applicationName)"], shortTitle: "Search Memory", systemImageName: "magnifyingglass")
        AppShortcut(intent: LumenAddMemoryIntent(), phrases: ["Add memory in \(.applicationName)"], shortTitle: "Add Memory", systemImageName: "brain")
        AppShortcut(intent: LumenRunTriggerIntent(), phrases: ["Run Lumen trigger in \(.applicationName)"], shortTitle: "Run Trigger", systemImageName: "bolt")
    }
}
#endif
