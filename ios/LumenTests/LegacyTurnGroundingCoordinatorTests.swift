import XCTest
import SwiftData
@testable import Lumen

final class LegacyTurnGroundingCoordinatorTests: XCTestCase {
    @MainActor func testBoundedInjection() async {
        let schema = Schema([MemoryItem.self, RAGChunk.self]); let c = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(c)
        let out = await LegacyTurnGroundingCoordinator.shared.build(userMessage: "hello", conversationID: nil, turnID: UUID(), history: [], modelContext: ctx, isBackground: true, task: .backgroundTrigger)
        XCTAssertLessThanOrEqual(out.promptInjection.count, 2000)
    }

    func testRoleAffectsCacheKeyDigest() {
        let base = LegacyGroundingCache.digest("hello\nrole=planner")
        let other = LegacyGroundingCache.digest("hello\nrole=executor")
        XCTAssertNotEqual(base, other)
    }

    func testRoleMetadataAppearsInAssembledPrompt() {
        let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: "sys", baseUserMessage: "hi", sections: [], policy: .slotAgent, roleMetadata: "slotAgent")
        XCTAssertTrue(assembled.userMessage.contains("[ROLE STAGE]"))
        XCTAssertTrue(assembled.userMessage.contains("slotAgent"))
    }

}
