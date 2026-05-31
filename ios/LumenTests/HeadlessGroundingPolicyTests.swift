import XCTest
import SwiftData
@testable import Lumen

final class HeadlessGroundingPolicyTests: XCTestCase {
    @MainActor func testBackgroundToolsFiltered() async {
        let schema = Schema([MemoryItem.self, RAGChunk.self]); let c = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(c)
        let turn = AssistantTurnContext(task: .backgroundTrigger, input: "status", isForeground: false, lowPowerMode: true, thermalState: .nominal)
        let out = await LegacyGroundingBridge().build(userMessage: "status", conversationID: nil, turnID: nil, history: [], modelContext: ctx, turn: turn)
        XCTAssertFalse(out.secureTools.contains { $0.id == "open.url" })
    }
}
