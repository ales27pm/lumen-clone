import XCTest
import SwiftData
import SwiftUI
@testable import Lumen

final class MemorySearchToolTests: XCTestCase {
    @MainActor func testValidationAndBoundedOutput() async {
        ResourceBudgetGate.testSnapshotOverride = .init(
            scenePhase: .background,
            lowPowerModeEnabled: true,
            thermalState: .nominal,
            recentMemoryWarningCount: 0,
            lastMemoryWarningAt: nil
        )
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let schema = Schema([MemoryItem.self]); let container = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(container)
        ctx.insert(MemoryItem(content: String(repeating: "abc ", count: 200), kind: .fact, source: "manual"))
        try? ctx.save()
        let tool = MemorySearchTool()
        let inv = ToolInvocation(id: UUID(), toolID: "memory.search", arguments: ["query":"abc","limit":"1"], source: .system, conversationID: nil, turnID: nil, createdAt: Date())
        let res = await tool.execute(invocation: inv, context: .init(isForeground: true, appState: nil, modelContext: ctx, permissionRegistry: .shared, metricsStore: .shared))
        XCTAssertEqual(res.status, .success)
        XCTAssertTrue(res.modelText.contains("abc"))
        XCTAssertEqual(res.structuredPayload?["mode"], "lexical_fallback")
        XCTAssertEqual(res.structuredPayload?["diagnostic"], "memory.recall: lowPowerMode=true")
        XCTAssertLessThanOrEqual(res.displayText.count, tool.definition.maxOutputCharacters + 20)
    }
}
