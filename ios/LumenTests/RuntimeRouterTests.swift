import XCTest
@testable import Lumen

final class RuntimeRouterTests: XCTestCase {
    func testEmbeddingPrefersCoreMLWhenAvailable() {
        let tempURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("fake.mlmodelc")
        FileManager.default.createFile(atPath: tempURL.path, contents: Data(), attributes: nil)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let router = AssistantRuntimeRouter(coreML: CoreMLRuntimeAdapter(modelURL: tempURL))
        let context = AssistantTurnContext(task: .embedding, input: "x", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        XCTAssertEqual(router.runtime(for: context), .coreML)
    }

    func testBackgroundTriggerUsesFallbackWhenConstrained() {
        let router = AssistantRuntimeRouter()
        let context = AssistantTurnContext(task: .backgroundTrigger, input: "x", isForeground: false, lowPowerMode: true, thermalState: .serious)
        XCTAssertEqual(router.runtime(for: context), .deterministicFallback)
    }

    func testChatFallsBackWhenHeavyRuntimeDisallowed() {
        let foundation = FoundationModelsRuntimeAdapter()
        let router = AssistantRuntimeRouter(foundation: foundation, llama: .init(isAvailable: true, unavailableReason: nil))
        let context = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: true, thermalState: .nominal)
        XCTAssertEqual(router.runtime(for: context), .deterministicFallback)
    }

    func testChatUsesLlamaOnlyWhenHeavyRuntimeAllowed() {
        let foundation = FoundationModelsRuntimeAdapter()
        let router = AssistantRuntimeRouter(foundation: foundation, llama: .init(isAvailable: true, unavailableReason: nil))
        let context = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        XCTAssertEqual(router.runtime(for: context), .llama)
    }

    func testChatDoesNotUseCoreMLTextFallback() {
        let tempURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("fake-chat.mlmodelc")
        FileManager.default.createFile(atPath: tempURL.path, contents: Data(), attributes: nil)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let router = AssistantRuntimeRouter(llama: .init(isAvailable: false), coreML: CoreMLRuntimeAdapter(modelURL: tempURL))
        let context = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        XCTAssertEqual(router.runtime(for: context), .deterministicFallback)
    }

}
