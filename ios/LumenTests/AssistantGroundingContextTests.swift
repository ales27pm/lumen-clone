import XCTest
@testable import Lumen

final class AssistantGroundingContextTests: XCTestCase {
    func testCodable() throws {
        let g = AssistantGroundingContext(memoryCount: 1, ragCount: 2, toolCount: 3, estimatedChars: 1200)
        let d = try JSONEncoder().encode(g)
        _ = try JSONDecoder().decode(AssistantGroundingContext.self, from: d)
    }
}
