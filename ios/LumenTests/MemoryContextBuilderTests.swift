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
    }
}
