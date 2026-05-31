import Testing
@testable import Lumen

struct AgentIntentRouterTests {

    @Test func explainTradeoffsDefaultsToChat() async throws {
        let decision = AgentIntentRouter.decide(userMessage: "Explain tradeoffs between precision and recall…")
        #expect(decision.intent == .conversation)
    }

    @Test func rememberStatementRoutesToMemory() async throws {
        let decision = AgentIntentRouter.decide(userMessage: "Remember that I prefer concise bullet points…")
        #expect(decision.intent == .memorySave)
    }

    @Test func callAlexRoutesToPhoneCall() async throws {
        let decision = AgentIntentRouter.decide(userMessage: "Call Alex…")
        #expect(decision.intent == .phoneCall)
    }
}
