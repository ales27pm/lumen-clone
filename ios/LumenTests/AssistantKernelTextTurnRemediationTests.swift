import XCTest
@testable import Lumen

@MainActor
final class AssistantKernelTextTurnRemediationTests: XCTestCase {
    func testRunTextTurnRejectsEmbedding() async {
        let kernel = AssistantKernel()
        let context = AssistantTurnContext(task: .embedding, input: "x", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        do {
            _ = try await kernel.runTextTurn(context)
            XCTFail("Embedding should not run through text turn")
        } catch AssistantKernel.KernelError.unsupportedTaskForTextTurn(let task) {
            XCTAssertEqual(task, .embedding)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testRunTextTurnRejectsSafetyClassification() async {
        let kernel = AssistantKernel()
        let context = AssistantTurnContext(task: .safetyClassification, input: "x", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        do {
            _ = try await kernel.runTextTurn(context)
            XCTFail("Safety classification should not run through text turn")
        } catch AssistantKernel.KernelError.unsupportedTaskForTextTurn(let task) {
            XCTAssertEqual(task, .safetyClassification)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

}
