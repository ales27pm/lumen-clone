import Testing
import Foundation
@testable import Lumen

struct AgentIntentRouterTests {

    @Test func explainTradeoffsDefaultsToChat() async throws {
        let decision = AgentIntentRouter.decide(userMessage: "Explain tradeoffs between precision and recall…")
        #expect(decision.intent == .chat)
        #expect(decision.confidenceSource == "compatibility:intentrouter-direct-answer")
    }

    @Test func rememberStatementRoutesToMemory() async throws {
        let decision = AgentIntentRouter.decide(userMessage: "Remember that I prefer concise bullet points…")
        #expect(decision.intent == .memory)
        #expect(decision.allowedToolIDs.contains("memory.save"))
        #expect(decision.confidenceSource == "compatibility:intentrouter-tool-scope")
    }

    @Test func callAlexRoutesToPhoneCall() async throws {
        let decision = AgentIntentRouter.decide(userMessage: "Call Alex…")
        #expect(decision.intent == .phoneCall)
    }

    @Test func attachmentsAreExplicitlyPromptContextOnly() async throws {
        let attachment = ChatAttachment(
            id: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!,
            name: "notes.txt",
            kind: .text,
            path: "/tmp/notes.txt",
            byteSize: 12
        )
        let decision = AgentIntentRouter.decide(userMessage: "Summarize this", attachments: [attachment])
        #expect(decision.attachmentsWerePresent)
        #expect(!decision.attachmentsAffectRouting)
        #expect(decision.reason.contains("prompt context only"))
    }

    @Test func clarificationKeepsProductionIntentAndCompatibilityName() async throws {
        let decision = AgentIntentRouter.decide(userMessage: "Call")
        #expect(decision.shouldAskClarification)
        #expect(decision.intent == .phoneCall)
        #expect(decision.compatibilityIntentName == "clarify")
        #expect(decision.confidenceSource == "compatibility:clarification-required")
    }
}
