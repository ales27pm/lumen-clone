import XCTest
@testable import Lumen

final class SelfModelSnapshotTests: XCTestCase {
    private func snapshot(embeddingAvailable: Bool? = nil) -> SelfModelSnapshot {
        let turn = AssistantTurnContext(task: .chat, input: "hello", isForeground: true, lowPowerMode: false, thermalState: .nominal)
        return SelfModelSnapshotBuilder.build(
            turn: turn,
            budget: ContextBudgetAllocator.allocate(for: turn, maxInputTokens: 800),
            selectedRuntime: .init(runtime: .coreML, reason: "registered generative backend"),
            tools: [DeviceStatusTool().definition, OpenURLTool().definition],
            availableBackendKinds: [LLMBackendKind.coreML.rawValue],
            embeddingAvailable: embeddingAvailable,
            now: Date(timeIntervalSince1970: 0)
        )
    }

    func testGenerativeBackendDoesNotEstablishEmbeddingOrSlotReadiness() {
        let unknown = snapshot()
        XCTAssertNil(unknown.runtime.embeddingAvailable)
        XCTAssertEqual(unknown.runtime.embeddingReadiness, .unknown)
        XCTAssertEqual(unknown.runtime.readiness, .unknown)
        XCTAssertFalse(unknown.agent.declaredSlots.isEmpty)
        XCTAssertTrue(unknown.agent.availableSlots.isEmpty)
        XCTAssertEqual(unknown.agent.slotAvailability, .unknown)
        XCTAssertEqual(unknown.evidence.manifestFreshness, "unknown")
        XCTAssertEqual(snapshot(embeddingAvailable: false).runtime.embeddingReadiness, .unavailable)
        XCTAssertEqual(snapshot(embeddingAvailable: true).runtime.embeddingReadiness, .available)
    }

    func testSourceRevisionRequiresImmutableBuildMetadata() {
        let revision = String(repeating: "A", count: 40)
        XCTAssertEqual(SelfModelSnapshotBuilder.validatedSourceRevision(revision), revision.lowercased())
        for invalid in ["unknown", "$(LUMEN_GIT_SHA)", "main", "abc123", "line\nrevision"] {
            XCTAssertNil(SelfModelSnapshotBuilder.validatedSourceRevision(invalid))
        }
    }

    func testCompactRenderingKeepsWholeFactsAndReportsOmissions() {
        let value = snapshot()
        XCTAssertEqual(SelfModelContextProvider.render(value, maxChars: 100), "")
        let rendered = SelfModelContextProvider.render(value, maxChars: 600)
        XCTAssertFalse(rendered.isEmpty)
        XCTAssertLessThanOrEqual(rendered.count, 600)
        XCTAssertTrue(rendered.contains("selectedRuntime=coreML\n"))
        XCTAssertTrue(rendered.contains("mustNotBypassApproval=true\n"))
        XCTAssertTrue(rendered.contains("embeddingReadiness=unknown\n"))
        XCTAssertTrue(rendered.contains("mode=foreground\n"))
        XCTAssertTrue(rendered.contains("slotAvailability=unknown\n"))
        XCTAssertTrue(rendered.split(separator: "\n").last?.hasPrefix("omittedFields=") == true)
        XCTAssertTrue(rendered.split(separator: "\n").allSatisfy { $0.contains("=") && !$0.hasSuffix("=") || $0.hasPrefix("availableSlots=") })
    }

    func testSnapshotRoundTripAndLegacyReadinessRemainConservative() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let value = snapshot(embeddingAvailable: true)
        let data = try encoder.encode(value)
        XCTAssertEqual(try decoder.decode(SelfModelSnapshot.self, from: data), value)

        var legacy = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        legacy["schemaVersion"] = "0.1.0"
        var app = try XCTUnwrap(legacy["app"] as? [String: Any])
        app.removeValue(forKey: "sourceRevision")
        legacy["app"] = app
        var agent = try XCTUnwrap(legacy["agent"] as? [String: Any])
        agent["availableSlots"] = ["cortex", "executor"]
        agent.removeValue(forKey: "declaredSlots")
        agent.removeValue(forKey: "slotAvailability")
        legacy["agent"] = agent
        var runtime = try XCTUnwrap(legacy["runtime"] as? [String: Any])
        runtime.removeValue(forKey: "readiness")
        runtime.removeValue(forKey: "embeddingReadiness")
        legacy["runtime"] = runtime
        var tools = try XCTUnwrap(legacy["tools"] as? [String: Any])
        tools.removeValue(forKey: "availabilityScope")
        legacy["tools"] = tools
        var evidence = try XCTUnwrap(legacy["evidence"] as? [String: Any])
        evidence["manifestFreshness"] = "bundled"
        legacy["evidence"] = evidence

        let decoded = try decoder.decode(SelfModelSnapshot.self, from: JSONSerialization.data(withJSONObject: legacy))
        XCTAssertEqual(decoded.schemaVersion, "0.1.0")
        XCTAssertEqual(decoded.agent.declaredSlots, ["cortex", "executor"])
        XCTAssertTrue(decoded.agent.availableSlots.isEmpty)
        XCTAssertNil(decoded.runtime.embeddingAvailable)
        XCTAssertEqual(decoded.runtime.embeddingReadiness, .unknown)
        XCTAssertEqual(decoded.evidence.manifestFreshness, "unknown")
        XCTAssertEqual(try decoder.decode(SelfModelSnapshot.CapabilityAvailability.self, from: Data("\"future_status\"".utf8)), .unknown)
    }

    func testGemmaSnapshotDescribesOneAssistantForEveryGenerativeRole() {
        let tasks: [AssistantTaskKind] = [.chat, .agentPlan, .toolDecision, .summarization, .memoryExtraction]
        for task in tasks {
            let turn = AssistantTurnContext(
                task: task,
                input: "hello",
                isForeground: true,
                lowPowerMode: false,
                thermalState: .nominal
            )
            let snapshot = SelfModelSnapshotBuilder.build(
                turn: turn,
                budget: ContextBudgetAllocator.allocate(for: turn, maxInputTokens: 800),
                selectedRuntime: .init(runtime: .coreAI, reason: "Gemma selected"),
                tools: [DeviceStatusTool().definition],
                activeSlot: .executor
            )

            XCTAssertEqual(snapshot.agent.logicalIdentity, "lumen")
            XCTAssertEqual(snapshot.agent.activeSlot, "assistant")
            XCTAssertEqual(snapshot.agent.declaredSlots, ["assistant"])
            XCTAssertEqual(snapshot.agent.availableSlots, [])
            XCTAssertEqual(snapshot.agent.slotAvailability, .unknown)
            XCTAssertEqual(snapshot.runtime.selectedRuntimePathKind, LumenRuntimePathKind.coreAI.rawValue)
            XCTAssertTrue(snapshot.tools.available.contains("device.status"))
        }
    }

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

        XCTAssertEqual(snapshot.schemaVersion, "0.2.0")
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
        XCTAssertTrue(section.sourceIDs.contains("selfModelSnapshot/0.2.0"))
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
        XCTAssertNil(snapshot.runtime.embeddingAvailable)
        XCTAssertEqual(snapshot.runtime.embeddingReadiness, .unknown)
        XCTAssertFalse(rendered.contains("secret raw prompt"))
    }
}
