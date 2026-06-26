import XCTest
import SwiftData
@testable import Lumen

final class LegacyGroundingBridgeTests: XCTestCase {
    @MainActor func testBuildDoesNotCrash() async throws {
        let schema = Schema([MemoryItem.self, RAGChunk.self]); let c = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(c)
        ctx.insert(MemoryItem(content: "lumen adapter contract requires executor preflight", kind: .project, topic: "adapter"))
        try? ctx.save()
        let turn = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        let out = try await LegacyGroundingBridge().build(userMessage: "adapter", conversationID: nil, turnID: nil, history: [], modelContext: ctx, turn: turn)
        XCTAssertGreaterThanOrEqual(out.grounding.toolCount, 0)
        XCTAssertEqual(out.grounding.contextProfile, ContextPolicyProfile.chat.rawValue)
        XCTAssertGreaterThan(out.grounding.maxInputTokens ?? 0, 0)
        XCTAssertGreaterThanOrEqual(out.grounding.estimatedTokens, 0)
        XCTAssertEqual(out.grounding.memoryTierCounts?["semantic"], 1)
        XCTAssertEqual(out.grounding.contextQueryExpanded, false)
    }

    @MainActor func testBuildExpandsRetrievalQueryFromRecentHistory() async throws {
        let schema = Schema([MemoryItem.self, RAGChunk.self])
        let c = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(c)
        ctx.insert(MemoryItem(content: "adapter preflight must validate role adapter paths before executor streaming", kind: .project, topic: "adapter"))
        try? ctx.save()

        let history: [(role: MessageRole, content: String)] = [
            (.assistant, "Earlier we were discussing adapter preflight failures.")
        ]
        let turn = AssistantTurnContext(task: .chat, input: "what about that?", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        let out = try await LegacyGroundingBridge().build(userMessage: "what about that?", conversationID: nil, turnID: nil, history: history, modelContext: ctx, turn: turn)

        XCTAssertEqual(out.grounding.contextQueryExpanded, true)
        XCTAssertEqual(out.grounding.memoryTierCounts?["semantic"], 1)
    }
}
