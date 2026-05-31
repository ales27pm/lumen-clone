import Foundation
@testable import Lumen

struct ToolScenario: Sendable {
    let toolID: String
    let positivePrompts: [String]
    let negativePrompts: [String]
    let expectedIntentNames: [String]
    let requiresApproval: Bool
    let expectedArgumentKeys: Set<String>
    let forbiddenToolIDs: Set<String>
}

enum ToolScenarioCatalog {
    static let all: [ToolScenario] = ToolRegistry.all.map { tool in
        ToolScenario(
            toolID: tool.id,
            positivePrompts: [defaultPositivePrompt(for: tool.id)],
            negativePrompts: ["Tell me a joke"],
            expectedIntentNames: [expectedIntentName(for: tool.id)],
            requiresApproval: tool.requiresApproval,
            expectedArgumentKeys: expectedArgs(for: tool.id),
            forbiddenToolIDs: defaultForbidden(for: tool.id)
        )
    }

    private static func defaultPositivePrompt(for toolID: String) -> String {
        switch toolID {
        case "calendar.create": return "Create an event tomorrow at 5 called dentist"
        case "calendar.list": return "What's on my calendar today?"
        case "reminders.create": return "Remind me to buy screws tomorrow"
        case "reminders.list": return "Show my pending reminders"
        case "contacts.search": return "Find Jordan in my contacts"
        default: return "Use \(toolID)"
        }
    }

    private static func expectedIntentName(for toolID: String) -> String { toolID.split(separator: ".").first.map(String.init) ?? toolID }
    private static func expectedArgs(for toolID: String) -> Set<String> {
        switch toolID {
        case "web.fetch": return ["url"]
        case "web.search", "maps.search", "photos.search", "rag.search": return ["query"]
        default: return []
        }
    }
    private static func defaultForbidden(for toolID: String) -> Set<String> {
        switch toolID {
        case "photos.search": return ["web.search"]
        case "web.search": return ["maps.search"]
        case "alarm.schedule": return ["reminders.create"]
        default: return []
        }
    }
}
