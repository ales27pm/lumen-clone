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
}
