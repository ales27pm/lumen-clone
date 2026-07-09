import XCTest
@testable import Lumen

final class SelfModelSnapshotTests: XCTestCase {
    func testSnapshotUsesRuntimeSourcesWithoutRawPromptPayload() {
        let turn = AssistantTurnContext(
            task: .toolDecision,
            input: "secret raw user prompt should not enter snapshot",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal
        )
        let budget = ContextBudgetAllocator.allocate(for: turn, maxInputTokens: 800)
        let snapshot = SelfModelSnapshotBuilder.build(
            turn: turn,
            budget: budget,
            selectedRuntime: .init(runtime: .llama, reason: "llama available"),
            tools: [
                DeviceStatusTool().definition,
                OpenURLTool().definition
            ],
            availableBackendKinds: [LLMBackendKind.gguf.rawValue],
            now: Date(timeIntervalSince1970: 0)
        )
        let rendered = SelfModelContextProvider.render(snapshot, maxChars: 4_000)

        XCTAssertEqual(snapshot.schemaVersion, "0.1.0")
        XCTAssertEqual(snapshot.agent.activeSlot, LumenModelSlot.executor.rawValue)
        XCTAssertEqual(snapshot.runtime.selectedRuntime, AssistantRuntimeKind.llama.rawValue)
        XCTAssertEqual(snapshot.runtime.selectedRuntimePathKind, LumenRuntimePathKind.llamaGGUF.rawValue)
        XCTAssertTrue(snapshot.tools.available.contains("device.status"))
        XCTAssertTrue(snapshot.tools.requiresApproval.contains("open.url"))
        XCTAssertTrue(snapshot.tools.backgroundSafe.contains("device.status"))
        XCTAssertFalse(rendered.contains("secret raw user prompt"))
    }

    @MainActor
    func testBackgroundSnapshotUsesPolicyFilteredReadOnlyTools() async {
        let turn = AssistantTurnContext(
            task: .backgroundTrigger,
            input: "background check",
            isForeground: false,
            lowPowerMode: true,
            thermalState: .nominal
        )
        let context = ToolExecutionContext(
            isForeground: false,
            appState: nil,
            modelContext: nil,
            permissionRegistry: .shared,
            metricsStore: .shared
        )
        let tools = await SecureToolRegistry.shared.availableDefinitions(context: context, source: .backgroundTrigger)
        let budget = ContextBudgetAllocator.allocate(for: turn, maxInputTokens: 800)
        let snapshot = SelfModelSnapshotBuilder.build(
            turn: turn,
            budget: budget,
            selectedRuntime: .init(runtime: .deterministicFallback, reason: "background heavy runtime disallowed"),
            tools: tools,
            now: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(snapshot.app.mode, "background")
        XCTAssertEqual(snapshot.agent.activeSlot, LumenModelSlot.mouth.rawValue)
        XCTAssertTrue(snapshot.tools.available.contains("device.status"))
        XCTAssertFalse(snapshot.tools.available.contains("open.url"))
        XCTAssertFalse(snapshot.tools.requiresApproval.contains("open.url"))
        XCTAssertTrue(Set(snapshot.tools.backgroundSafe).isSubset(of: Set(snapshot.tools.available)))
    }

    func testSelfModelContextProviderBoundsOutput() {
        let turn = AssistantTurnContext(
            task: .chat,
            input: "hello",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal
        )
        let budget = ContextBudgetAllocator.allocate(for: turn, maxInputTokens: 800)
        let manyTools = (0..<100).map { index in
            SecureToolDefinition(
                id: "test.tool.\(index)",
                displayName: "Tool \(index)",
                description: "Synthetic test tool",
                category: .readOnly,
                requiredPermissions: [],
                supportsBackgroundExecution: true,
                requiresUserApproval: false,
                argumentSchemaDescription: "{}",
                resultPrivacyLevel: .low,
                maxOutputCharacters: 100
            )
        }
        let snapshot = SelfModelSnapshotBuilder.build(
            turn: turn,
            budget: budget,
            selectedRuntime: .init(runtime: .foundationModels, reason: "preferred on-device foundation runtime"),
            tools: manyTools,
            now: Date(timeIntervalSince1970: 0)
        )
        let section = SelfModelContextProvider.section(for: snapshot, budget: budget)

        XCTAssertEqual(section.title, SelfModelContextProvider.sectionTitle)
        XCTAssertLessThanOrEqual(section.content.count, min(budget.charSections.runtime, 1_600))
        XCTAssertEqual(section.privacyLevel, .low)
        XCTAssertTrue(section.sourceIDs.contains("selfModelSnapshot/0.1.0"))
    }

    func testSelfImprovementMaintenanceSnapshotDoesNotExposeRawPromptOrLoadedRuntime() {
        let turn = AssistantTurnContext(
            task: .backgroundTrigger,
            input: "secret raw prompt should not enter self-improvement snapshot",
            isForeground: true,
            lowPowerMode: false,
            thermalState: .nominal,
            prefersFoundationModels: false,
            allowHeavyRuntime: false,
            maxTokens: 128
        )
        let budget = ContextBudgetAllocator.allocate(for: turn, maxInputTokens: 512)
        let snapshot = SelfModelSnapshotBuilder.build(
            turn: turn,
            budget: budget,
            selectedRuntime: .init(runtime: .unavailable, reason: "self-improvement runtime maintenance does not load models"),
            tools: [
                DeviceStatusTool().definition,
                OpenURLTool().definition
            ],
            availableBackendKinds: [],
            activeSlot: .rem,
            now: Date(timeIntervalSince1970: 0)
        )
        let rendered = SelfModelContextProvider.render(snapshot, maxChars: 2_000)

        XCTAssertEqual(snapshot.agent.activeSlot, LumenModelSlot.rem.rawValue)
        XCTAssertEqual(snapshot.runtime.selectedRuntime, AssistantRuntimeKind.unavailable.rawValue)
        XCTAssertEqual(snapshot.runtime.availableBackendKinds, [])
        XCTAssertEqual(snapshot.runtime.embeddingAvailable, false)
        XCTAssertFalse(rendered.contains("secret raw prompt"))
    }
}
