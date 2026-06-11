import Testing
@testable import Lumen

struct ToolRegistryCoverageTests {
    private let expectedToolCount = ToolRegistry.all.count
    @Test func registryIntegrity() {
        let tools = ToolRegistry.all
        #expect(tools.count == expectedToolCount)
        let ids = tools.map(\.id)
        #expect(Set(ids).count == tools.count)

        for tool in tools {
            #expect(!tool.name.isEmpty)
            #expect(!tool.description.isEmpty)
            #expect(!tool.icon.isEmpty)
            #expect(!tool.tint.isEmpty)
            #expect(ToolRouteGuard.canonicalToolID(tool.id) == tool.id)
            if let key = tool.permissionKey {
                #expect(PermissionKind(usageDescriptionKey: key) != nil)
            }
        }
    }

    @Test func everyRegisteredToolHasScenarioCoverage() {
        let scenarioIDs = ToolScenarioCatalog.all.map(\.toolID)
        #expect(scenarioIDs.count == expectedToolCount)
        #expect(Set(scenarioIDs).count == expectedToolCount)
        let registered = Set(ToolRegistry.all.map(\.id))
        #expect(Set(scenarioIDs) == registered)
    }

    @Test func commonToolCommandAliasesCanonicalizeToRegisteredTools() {
        let aliases = [
            "schedule.alarm": "alarm.schedule",
            "countdown.alarm": "alarm.countdown",
            "request.alarm.authorization": "alarm.request_authorization",
            "alarm.authorization.status": "alarm.authorization_status",
            "cancel.alarm": "alarm.cancel",
            "contacts.lookup": "contacts.search",
            "calendar.read": "calendar.list",
            "memory.search": "memory.recall",
            "rag.search.secure": "rag.search"
        ]

        for (alias, expected) in aliases {
            let canonical = ToolRouteGuard.canonicalToolID(alias)
            #expect(canonical == expected)
            #expect(ToolRegistry.find(id: alias)?.id == expected)
        }
    }
}
