import XCTest
import SwiftData
@testable import Lumen

final class MemoryConsolidatorTests: XCTestCase {
    @MainActor func testDedupe() async {
        let schema = Schema([MemoryItem.self]); let c = try! ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let ctx = ModelContext(c)
        ctx.insert(MemoryItem(content: "same", kind: .fact)); ctx.insert(MemoryItem(content: "same", kind: .fact)); try? ctx.save()
        await MemoryConsolidator.consolidate(context: ctx)
        let all = (try? ctx.fetch(FetchDescriptor<MemoryItem>())) ?? []
        XCTAssertEqual(all.count, 1)
    }
}
