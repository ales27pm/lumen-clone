import XCTest
@testable import Lumen

final class PromptGroundingRendererTests: XCTestCase {
    func testBudgetedPrompt() {
        let sections = [PromptGroundingSection(title: "A", content: String(repeating: "x", count: 300), estimatedChars: 300, sourceIDs: [], privacyLevel: .low)]
        let rendered = PromptGroundingRenderer.renderForPrompt(sections, maxChars: 100)
        XCTAssertLessThanOrEqual(rendered.count, 100)
    }
}
