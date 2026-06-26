import XCTest
import SwiftData
@testable import Lumen

final class MemoryContextBuilderTests: XCTestCase {
    @MainActor func testBudgetBounded() {
        let schema = Schema([MemoryItem.self]); let container = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(container)
        for i in 0..<10 { ctx.insert(MemoryItem(content: "memory \(i) lorem ipsum", kind: .fact)) }
        try? ctx.save()
        let r = MemoryContextBuilder.build(query: "memory", budgetChars: 80, context: ctx)
        XCTAssertLessThanOrEqual(r.totalChars, 80)
        XCTAssertGreaterThanOrEqual(r.totalTokens, 0)
        XCTAssertEqual(r.candidateCount, 10)
    }

    @MainActor func testHierarchyKeepsDurableSemanticMemoryVisible() {
        let schema = Schema([MemoryItem.self])
        let container = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(container)
        for index in 0..<5 {
            ctx.insert(MemoryItem(content: "lumen session note \(index) about adapters", kind: .conversation, source: "chat"))
        }
        ctx.insert(MemoryItem(content: "lumen adapter contract must require preflight before executor streaming", kind: .project, source: "manual", topic: "adapters"))
        try? ctx.save()

        let r = MemoryContextBuilder.build(query: "lumen adapter", budgetChars: 220, context: ctx)

        XCTAssertTrue(r.selected.contains { $0.memoryKind == .project })
        XCTAssertGreaterThan(r.tierCounts["semantic"] ?? 0, 0)
        XCTAssertGreaterThan(r.tierCounts["working"] ?? 0, 0)
        XCTAssertTrue(r.hierarchyPassApplied)
    }

    @MainActor func testPinnedMemoryUsesPinnedTierReason() {
        let schema = Schema([MemoryItem.self])
        let container = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(container)
        let pinned = MemoryItem(content: "always call me Alexis", kind: .preference)
        pinned.isPinned = true
        ctx.insert(pinned)
        ctx.insert(MemoryItem(content: "recent unrelated chat", kind: .conversation))
        try? ctx.save()

        let r = MemoryContextBuilder.build(query: "unrelated", budgetChars: 120, context: ctx)

        XCTAssertEqual(r.selected.first?.id, pinned.id)
        XCTAssertEqual(r.tierCounts["pinned"], 1)
        XCTAssertEqual(r.reasons[pinned.id], "pinned:pinned")
    }
}
