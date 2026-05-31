import Foundation
import SwiftData
#if canImport(AppIntents)
import AppIntents

@available(iOS 16.0, *)
struct LumenRunTriggerIntent: AppIntent {
    static var title: LocalizedStringResource = "Run Lumen Trigger"
    static var openAppWhenRun = false

    @Parameter(title: "Trigger Name") var triggerName: String

    @MainActor
    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let name = triggerName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, name.count <= 120 else { return .result(value: "Trigger name must be 1...120 characters.") }
        guard let container = SharedContainer.shared else {
            return .result(value: LumenIntentResultRenderer.degraded("trigger store unavailable"))
        }
        let ctx = ModelContext(container)
        let all: [Trigger]
        do {
            all = try ctx.fetch(FetchDescriptor<Trigger>())
        } catch {
            return .result(value: LumenIntentResultRenderer.degraded("trigger fetch failed"))
        }
        let foldedName = name.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
        let exact = all.filter { $0.title.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current) == foldedName }
        let trigger: Trigger
        if exact.count == 1 {
            trigger = exact[0]
        } else if exact.count > 1 {
            return .result(value: "Multiple triggers matched exactly. Open Lumen to choose one.")
        } else {
            let partial = all.filter { $0.title.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current).contains(foldedName) }
            guard partial.count == 1 else {
                return .result(value: partial.isEmpty ? "No matching trigger found." : "Multiple triggers matched. Open Lumen to choose one.")
            }
            trigger = partial[0]
        }
        if LumenIntentPolicy.requiresOpenAppForSensitiveAction(trigger.prompt) {
            return .result(value: LumenIntentResultRenderer.openAppRequired("trigger may require sensitive tools"))
        }
        let result = await TriggerScheduler.shared.runTrigger(trigger, context: ctx, settings: SettingsSnapshot.loadFromDisk(), notify: false) ?? "No result."
        return .result(value: String(result.prefix(500)))
    }
}
#endif
