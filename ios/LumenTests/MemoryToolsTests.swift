import Testing
@testable import Lumen

@MainActor
struct MemoryToolsTests {
    @Test func saveRejectsEmptyContent() async {
        let result = await MemoryTools.save(content: "   \n\t  ", kind: "fact")
        #expect(result == "Need content.")
    }
}
