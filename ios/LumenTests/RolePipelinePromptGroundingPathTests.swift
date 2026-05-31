import XCTest
@testable import Lumen

final class RolePipelinePromptGroundingPathTests: XCTestCase {
    func testRoleMetadataTruncationIsReported() {
        let role = String(repeating: "r", count: 240)
        let assembled = LegacyPromptAssembler.assemble(baseSystemPrompt: "sys", baseUserMessage: "hi", sections: [], policy: .rolePipeline, roleMetadata: role)
        XCTAssertTrue(assembled.truncationOccurred)
        XCTAssertFalse(assembled.userMessage.contains(role))
    }
}
