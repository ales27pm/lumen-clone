import XCTest
import SwiftData
@testable import Lumen

final class LegacyGroundingBridgeTests: XCTestCase {
    @MainActor func testBuildDoesNotCrash() async {
        let schema = Schema([MemoryItem.self, RAGChunk.self]); let c = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(c)
        let turn = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        let out = await LegacyGroundingBridge().build(userMessage: "hello", conversationID: nil, turnID: nil, history: [], modelContext: ctx, turn: turn)
        XCTAssertGreaterThanOrEqual(out.grounding.toolCount, 0)
    }
}
