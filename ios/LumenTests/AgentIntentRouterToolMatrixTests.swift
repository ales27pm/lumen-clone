import Testing
@testable import Lumen

struct AgentIntentRouterToolMatrixTests {
    @Test func allRegisteredToolsReachableFromIntentRouters() {
        let registered = Set(ToolRegistry.all.map(\.id))
        let intentRouterCovered = Set(UserIntent.allCases.flatMap { IntentRouter.allowedToolIDs(for: $0) })
        let agentRouterCovered = Set(AgentIntentRouter.Intent.allCases.flatMap { AgentIntentRouter.allowedToolIDs(for: $0) })
        let combined = intentRouterCovered.union(agentRouterCovered)

        let uncovered = registered.subtracting(combined).sorted()
        #expect(uncovered.isEmpty, "Registered tool \(uncovered.first ?? "") has no intent-routing coverage.")
    }

    @Test func scenariosHavePositiveAndNegativeAndConflictChecks() {
        for scenario in ToolScenarioCatalog.all {
            #expect(!scenario.positivePrompts.isEmpty)
            #expect(!scenario.negativePrompts.isEmpty)
        }

        let photoDecision = IntentRouter.classify("Search photos for selfies")
        #expect(IntentRouter.isToolAllowed("photos.search", for: photoDecision))
        #expect(!IntentRouter.isToolAllowed("web.search", for: photoDecision))

        let webDecision = IntentRouter.classify("Search web for DIY bunker plans")
        #expect(IntentRouter.isToolAllowed("web.search", for: webDecision))
        #expect(!IntentRouter.isToolAllowed("maps.search", for: webDecision))

        let fetchDecision = AgentIntentRouter.decide(userMessage: "Read this URL: https://example.com")
        #expect(fetchDecision.allowedToolIDs.contains("web.fetch"))
    }
}
