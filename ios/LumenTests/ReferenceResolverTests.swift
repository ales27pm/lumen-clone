import Foundation
import Testing
@testable import Lumen

struct ReferenceResolverTests {
    @Test func resolvesPronounFromRecentConversation() async throws {
        let history: [(role: MessageRole, content: String)] = [
            (.user, "I need to call Julie Charlebois after lunch"),
            (.assistant, "Got it, Julie Charlebois.")
        ]

        let result = ReferenceResolver.resolve(
            prompt: "call her",
            history: history,
            relevantMemories: []
        )

        #expect(result.rewrittenPrompt.lowercased().contains("call julie charlebois"))
        #expect(result.confidence >= 0.7)
        #expect(!result.resolvedReferences.isEmpty)
    }

    @Test func resolvesPronounFromPersonMemory() async throws {
        let memories = [
            MemoryContextItem(content: "Preferred contact: Alex Morgan", scope: .person, authority: .referenceOnly, createdAt: nil, expiresAt: nil, source: "test", topic: "contact")
        ]

        let result = ReferenceResolver.resolve(prompt: "text him about dinner", history: [], relevantMemories: memories)
        #expect(result.rewrittenPrompt.lowercased().contains("text alex morgan"))
    }

    @Test func resolvesPreviousOneFromCurrentTurnToolLedger() async throws {
        let entry = ToolLedgerEntry(conversationID: UUID(), turnID: UUID(), intent: .webSearch, toolID: "web.search", query: "q=best pizza", result: "ok")
        let result = ReferenceResolver.resolve(prompt: "use previous one", history: [], relevantMemories: [], currentTurnLedger: [entry])
        #expect(result.rewrittenPrompt.lowercased().contains("previous web.search result"))
        #expect(result.diagnostics.contains("resolved_deictic_from_current_turn_toolledger"))
    }

    @Test func leavesPromptUnchangedWhenNoSafeReferentExists() async throws {
        let result = ReferenceResolver.resolve(prompt: "call her", history: [], relevantMemories: [])
        #expect(result.rewrittenPrompt == "call her")
        #expect(result.confidence == 0)
        #expect(result.diagnostics.contains("pronoun_detected_no_safe_referent"))
    }

    @Test func leavesPreviousOneUnchangedWithoutLedger() async throws {
        let result = ReferenceResolver.resolve(prompt: "use previous one", history: [], relevantMemories: [])
        #expect(result.rewrittenPrompt == "use previous one")
        #expect(result.diagnostics.contains("deictic_detected_without_current_turn_toolledger"))
    }
}
