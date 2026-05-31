import XCTest
import SwiftData
@testable import Lumen

final class MemorySearchToolTests: XCTestCase {
    @MainActor func testValidationAndBoundedOutput() async {
        let schema = Schema([MemoryItem.self]); let container = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(container)
        ctx.insert(MemoryItem(content: String(repeating: "abc ", count: 200), kind: .fact, source: "manual"))
        try? ctx.save()
        let tool = MemorySearchTool()
        let inv = ToolInvocation(id: UUID(), toolID: "memory.search", arguments: ["query":"abc","limit":"1"], source: .system, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await tool.execute(invocation: inv, context: .init(isForeground: true, appState: nil, modelContext: ctx, permissionRegistry: .shared, metricsStore: .shared))
        XCTAssertEqual(res.status, .success)
        XCTAssertLessThanOrEqual(res.displayText.count, tool.definition.maxOutputCharacters + 20)
    }
}
