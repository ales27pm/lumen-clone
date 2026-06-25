import XCTest
@testable import Lumen

final class RuntimeRouterTests: XCTestCase {
    func testEmbeddingDoesNotUseStagedCoreMLEvenWhenModelFileExists() {
        let tempURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("fake.mlmodelc")
        FileManager.default.createFile(atPath: tempURL.path, contents: Data(), attributes: nil)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let router = AssistantRuntimeRouter(coreML: CoreMLRuntimeAdapter(modelURL: tempURL))
        let context = AssistantTurnContext(task: .embedding, input: "x", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        let selection = router.selection(for: context)
        XCTAssertEqual(selection.runtime, .deterministicFallback)
        XCTAssertTrue(selection.reason.contains("implementation missing"))
    }

    func testBackgroundTriggerUsesFallbackWhenConstrained() {
        let router = AssistantRuntimeRouter()
        let context = AssistantTurnContext(task: .backgroundTrigger, input: "x", isForeground: false, lowPowerMode: true, thermalState: .serious)
        XCTAssertEqual(router.runtime(for: context), .deterministicFallback)
    }

    func testChatFallsBackWhenHeavyRuntimeDisallowed() {
        let foundation = FoundationModelsRuntimeAdapter()
        let router = AssistantRuntimeRouter(foundation: foundation, llama: .init(generateHandler: { _ in "ok" }))
        let context = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal, allowHeavyRuntime: false)
        let selection = router.selection(for: context)
        XCTAssertEqual(selection.runtime, .deterministicFallback)
        XCTAssertEqual(selection.reason, "foregroundInteractive: heavyRuntime=false")
    }

    func testChatLowPowerStillUsesLlamaWhenForegroundAndAvailable() {
        let foundation = FoundationModelsRuntimeAdapter()
        let router = AssistantRuntimeRouter(foundation: foundation, llama: .init(generateHandler: { _ in "ok" }))
        let context = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: true, thermalState: .nominal)
        let selection = router.selection(for: context)
        XCTAssertEqual(selection.runtime, .llama)
        XCTAssertEqual(selection.reason, "llama available")
    }

    func testChatDoesNotUseUnwiredFoundationModelsEvenWhenPreferred() {
        let foundation = FoundationModelsRuntimeAdapter()
        let router = AssistantRuntimeRouter(foundation: foundation, llama: .init(isAvailable: false))
        let context = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal, prefersFoundationModels: true)
        let selection = router.selection(for: context)
        XCTAssertEqual(selection.runtime, .deterministicFallback)
        XCTAssertNotEqual(selection.runtime, .foundationModels)
    }

    func testChatUsesLlamaWhenGenerationAdapterIsWiredAndHeavyRuntimeAllowed() {
        let foundation = FoundationModelsRuntimeAdapter()
        let router = AssistantRuntimeRouter(foundation: foundation, llama: .init(generateHandler: { request in
            "llama: \(request.prompt)"
        }))
        let context = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        XCTAssertEqual(router.runtime(for: context), .llama)
    }

    func testSwiftLlamaRemainsPreferredOverStagedFoundationModelsWhenAvailableAndAllowed() {
        let foundation = FoundationModelsRuntimeAdapter()
        let llama = LlamaRuntimeAdapter(generateHandler: { request in
            "llama: \(request.prompt)"
        })
        let router = AssistantRuntimeRouter(foundation: foundation, llama: llama)
        let context = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal, prefersFoundationModels: true)
        let selection = router.selection(for: context)
        XCTAssertEqual(selection.runtime, .llama)
        XCTAssertEqual(selection.reason, "llama available")
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
