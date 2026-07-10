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

    func testRunTextTurnDoesNotSelectUnwiredLlamaMarkedAvailable() async throws {
        let router = AssistantRuntimeRouter(llama: .init(isAvailable: true, unavailableReason: nil))
        let kernel = AssistantKernel(router: router)
        let context = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal)

        let output = try await kernel.runTextTurn(context)

        XCTAssertEqual(output, "Diagnostic deterministic runtime response.")
        XCTAssertEqual(kernel.selectRuntime(for: context), .deterministicFallback)
    }

    func testRunTextTurnUsesInjectedLlamaGenerationAdapter() async throws {
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { request in
                "wired: \(request.prompt)"
            })
        )
        let kernel = AssistantKernel(router: router)
        let context = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal)

        let output = try await kernel.runTextTurn(context)

        XCTAssertEqual(output, "wired: hello")
        XCTAssertEqual(kernel.selectRuntime(for: context), .llama)
    }

    func testRunTextTurnRecordsCorrelatedModelTurnForRealRuntime() async throws {
        AgentBehaviorTraceRecorder.clear()
        let e2eRunID = UUID()
        let agentRunID = UUID()
        let conversationID = UUID()
        let turnID = UUID()
        let correlation = AgentTraceCorrelation(
            scenarioID: "training-general-chat",
            e2eRunID: e2eRunID,
            agentRunID: agentRunID,
            conversationID: conversationID,
            turnID: turnID
        )
        let router = AssistantRuntimeRouter(
            llama: .init(generateHandler: { request in
                "model output for \(request.prompt)"
            })
        )
        let kernel = AssistantKernel(router: router)
        let context = AssistantTurnContext(
            task: .chat,
            input: "Explain precision and recall.",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal,
            prefersFoundationModels: false,
            traceCorrelation: correlation,
            allowedToolIDs: ["weather"]
        )

        _ = try await kernel.runTextTurn(context)

        let trace = AgentBehaviorTraceRecorder.recent(limit: 1).last
        XCTAssertEqual(trace?.event, .modelTurn)
        XCTAssertEqual(trace?.stage, "chat-text-turn")
        XCTAssertEqual(trace?.scenarioID, "training-general-chat")
        XCTAssertEqual(trace?.e2eRunID, e2eRunID)
        XCTAssertEqual(trace?.agentRunID, agentRunID)
        XCTAssertEqual(trace?.conversationID, conversationID)
        XCTAssertEqual(trace?.turnID, turnID)
        XCTAssertEqual(trace?.runtimePath, "agent-model")
        XCTAssertEqual(trace?.modelLoaded, true)
        XCTAssertEqual(trace?.allowedToolIDs, ["weather"])
        XCTAssertFalse(trace?.rawOutputPrefix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
        AgentBehaviorTraceRecorder.clear()
    }

}
