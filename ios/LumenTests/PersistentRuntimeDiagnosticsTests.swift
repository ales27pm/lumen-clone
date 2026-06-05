import XCTest
import SwiftUI
@testable import Lumen

final class PersistentRuntimeDiagnosticsTests: XCTestCase {
    func testCampaignStorePersistsAndRestoresEnabledCampaign() async throws {
        let store = try makeStore()
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: true, maxRunsPerScenario: 3, delayBetweenRunsSeconds: 2, scenarios: [.plainFastPrompt, .agentFastPrompt])
        try await store.saveCampaign(campaign)
        let restored = await store.loadCampaign()
        XCTAssertEqual(restored, campaign)
    }

    func testRunnerSkipsModelScenarioWhenNoModelLoadedInsteadOfFailing() async throws {
        let store = try makeStore()
        let runner = PersistentRuntimeDiagnosticsRunner(store: store)
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: false, scenarios: [.plainFastPrompt])
        let record = await runner.runOnce(campaign)
        XCTAssertNotEqual(record?.status, .failed)
        XCTAssertEqual(record?.scenario, .plainFastPrompt)
    }

    func testPlainFastPromptExpectationFailsIfFinalCharsExceedFastCap() {
        let result = PersistentRuntimeDiagnosticsRunner.evaluatePlainFastPrompt(finalChars: PromptBudgetConstants.fastInteractiveTotalChars + 1, estimatedTokens: 10, latencyClass: .fastInteractive)
        XCTAssertEqual(result.status, .failed)
        XCTAssertEqual(result.code, "fast_prompt_too_large")
    }

    func testDeveloperTraceBypassIsMarkedExpected() {
        let selection = PromptLatencyClassifier.classify(userMessage: "Yo", attachments: [], developerTraceModeEnabled: true, reasoningCaptureEnabled: true, modelName: "chat")
        XCTAssertEqual(selection.latencyClass, .developerTrace)
    }

    func testDiskWriteGateBuffersAndDefersDiagnosticsDuringGeneration() async throws {
        let store = try makeStore()
        let lease = DiskWriteBudget.shared.beginGeneration()
        await store.appendEvent(PersistentDiagnosticEvent(code: "diagnostic_write", message: "safe synthetic event"))
        XCTAssertEqual(await store.readLogDataForExport(), Data())
        lease.end()
        await store.flushBufferedIfPossible()
        let data = await store.readLogDataForExport()
        XCTAssertFalse(data.isEmpty)
    }

    func testCrashResumeDetectionMarksUnfinishedActiveRunInterrupted() async throws {
        let store = try makeStore()
        var state = PersistentDiagnosticState()
        let runID = UUID()
        let campaignID = UUID()
        state.activeRunID = runID
        state.activeCampaignID = campaignID
        state.activeScenario = .lifecycleCancellation
        state.activeStartedAt = Date(timeIntervalSince1970: 100)
        state.activeLaunchUUID = UUID()
        try await store.saveState(state)

        let record = try await store.markUnfinishedRunInterrupted(launchUUID: UUID(), startupAt: Date(timeIntervalSince1970: 200))
        XCTAssertEqual(record?.id, runID)
        XCTAssertEqual(record?.campaignID, campaignID)
        XCTAssertEqual(record?.status, .interrupted)
        XCTAssertEqual(record?.failureSummary, "interrupted_or_terminated")
    }

    func testRedactionRemovesPromptMemoryAndFileContentsFromLogEvents() {
        let event = PersistentDiagnosticEvent(
            code: "unsafe event",
            message: "prompt=My private question memory=secret file=/private/var/mobile/doc.txt email user@example.com",
            values: ["prompt": "My private question", "path": "/private/var/mobile/doc.txt"]
        )
        XCTAssertFalse(event.message.contains("My private question"))
        XCTAssertFalse(event.message.contains("secret"))
        XCTAssertFalse(event.message.contains("/private"))
        XCTAssertFalse(event.message.contains("user@example.com"))
        XCTAssertFalse(event.values.description.contains("My private question"))
    }

    @MainActor func testScenarioSelectionPausesWhenResourceBudgetDeniesHeavyWork() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        let store = try makeStore()
        let runner = PersistentRuntimeDiagnosticsRunner(store: store)
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: false, scenarios: [.plainFastPrompt])
        let record = await runner.runOnce(campaign)
        XCTAssertEqual(record?.status, .skipped)
        #endif
    }


    @MainActor func testThermalResourceGateUsesSimulatedDeniedSnapshotWhenRealStateAllowsWork() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .fair, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        let store = try makeStore()
        let runner = PersistentRuntimeDiagnosticsRunner(store: store)
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: false, scenarios: [.thermalResourceGate])

        let record = await runner.runOnce(campaign)

        XCTAssertNil(ResourceBudgetGate.testSnapshotOverride)
        XCTAssertEqual(record?.status, .passed)
        XCTAssertEqual(record?.events.last?.code, "resource_gate_policy_passed")
        XCTAssertEqual(record?.metrics.didFallback, true)
        XCTAssertEqual(record?.metrics.fallbackReason, "resource_gate_probe")
        XCTAssertEqual(record?.metrics.realScenePhase, "active")
        XCTAssertEqual(record?.metrics.realThermalState, DeviceThermalState.fair.rawValue)
        XCTAssertEqual(record?.metrics.realDenied, false)
        XCTAssertEqual(record?.metrics.simulatedScenePhase, "background")
        XCTAssertEqual(record?.metrics.simulatedThermalState, DeviceThermalState.serious.rawValue)
        XCTAssertEqual(record?.metrics.simulatedDenied, true)
        #endif
    }

    func testPersistentDiagnosticsStateCapsCompletedRunIDs() async throws {
        let store = try makeStore()
        var state = PersistentDiagnosticState()
        let ids = (0..<250).map { _ in UUID() }
        state.completedRunIDs = ids

        try await store.saveState(state)

        let restored = try XCTUnwrap(await store.loadState())
        XCTAssertEqual(restored.completedRunIDs.count, PersistentDiagnosticState.maxCompletedRunIDs)
        XCTAssertEqual(restored.completedRunIDs.first, ids[50])
        XCTAssertEqual(restored.completedRunIDs.last, ids[249])
    }

    func testDefaultExporterBoundsNormalExportSizeAndRecentLogLines() async throws {
        let store = try makeStore()
        for index in 0..<700 {
            await store.appendEvent(PersistentDiagnosticEvent(code: "bounded_export", message: "safe synthetic event \(index)"))
        }

        let exporter = PersistentRuntimeDiagnosticsExporter(store: store)
        let url = try await exporter.export()
        let data = try Data(contentsOf: url)
        let text = try String(contentsOf: url)

        XCTAssertLessThanOrEqual(data.count, 1_100_000)
        XCTAssertFalse(text.contains("safe synthetic event 0"))
        XCTAssertTrue(text.contains("safe synthetic event 699"))
    }

    func testAgentFastPromptScenarioUsesFastPathAndBoundedGroundingMetrics() {
        let request = AgentRequest(systemPrompt: "diagnostic", history: [], userMessage: "Yo", temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 64, maxSteps: 1, availableTools: ToolRegistry.all, relevantMemories: [])
        let result = SlotAgentService.fastGroundingResult(for: request, options: .default)
        XCTAssertTrue(SlotAgentService.shouldUseFastAgentPath(request))
        XCTAssertTrue(result.bridgedTools.isEmpty)
        XCTAssertLessThanOrEqual(result.userMessage.count + result.systemPrompt.count, PromptBudgetConstants.fastInteractiveTotalChars)
    }

    func testExporterExcludesRawSensitiveContent() async throws {
        let store = try makeStore()
        await store.appendEvent(PersistentDiagnosticEvent(code: "redaction", message: "prompt=Sensitive prompt memory=Private memory file=/tmp/secret.txt"))
        let exporter = PersistentRuntimeDiagnosticsExporter(store: store)
        let url = try await exporter.export()
        let text = try String(contentsOf: url)
        XCTAssertFalse(text.contains("Sensitive prompt"))
        XCTAssertFalse(text.contains("Private memory"))
        XCTAssertFalse(text.contains("/tmp/secret"))
    }

    @MainActor func testAgentToolPromptDiagnosticUsesDryRunWithoutLiveSlotStream() async throws {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        #endif
        AppCancellationBus.shared.cancel(.chatGeneration)
        let before = AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration)
        let store = try makeStore()
        let runner = PersistentRuntimeDiagnosticsRunner(store: store)
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: false, scenarios: [.dryRunPromptBudgetOnly])

        let record = await runner.runOnce(campaign)

        XCTAssertEqual(record?.status, .passed)
        XCTAssertEqual(record?.scenario, .dryRunPromptBudgetOnly)
        XCTAssertEqual(record?.metrics.didUseFastPath, false)
        XCTAssertLessThanOrEqual(record?.metrics.groundingChars ?? Int.max, 4_000)
        XCTAssertLessThanOrEqual(record?.metrics.groundingSectionCount ?? Int.max, 6)
        XCTAssertEqual(record?.metrics.inputToolCount, 2)
        XCTAssertEqual(record?.metrics.bridgedToolCount, 2)
        XCTAssertFalse(record?.events.contains { $0.code == PersistentRuntimeDiagnosticSignalKind.slotAgentStart.rawValue } ?? true)
        XCTAssertEqual(AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration), before)
    }

    @MainActor func testAgentToolPromptDryRunProducesBoundedGroundingWithoutCancellationRegistration() async {
        AppCancellationBus.shared.cancel(.chatGeneration)
        let before = AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration)
        let request = AgentRequest(systemPrompt: "diagnostic", history: [], userMessage: "Search the web for SwiftData cancellation patterns", temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 64, maxSteps: 2, availableTools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") }, relevantMemories: [], conversationID: UUID(), turnID: UUID())

        let result = await SlotAgentService.shared.prepareGroundedRequestForDiagnostics(request, options: .init(modelContext: nil, conversationID: request.conversationID, turnID: request.turnID, groundingMode: .slotAgent, allowDegradedGrounding: true, preventDoubleGrounding: true, diagnosticsEnabled: true))

        XCTAssertFalse(SlotAgentService.shouldUseFastAgentPath(request))
        XCTAssertLessThanOrEqual(result.userMessage.count + result.systemPrompt.count, 4_000)
        XCTAssertLessThanOrEqual(result.sections.count, 6)
        XCTAssertEqual(result.bridgedTools.count, request.availableTools.count)
        XCTAssertEqual(AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration), before)
    }

    @MainActor func testLiveSlotAgentStreamUnregistersCancellationBusOnNormalCompletion() async {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        #endif
        AppCancellationBus.shared.cancel(.chatGeneration)
        let before = AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration)
        let request = AgentRequest(systemPrompt: "diagnostic", history: [], userMessage: "Search the web for SwiftData cancellation patterns", temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 64, maxSteps: 2, availableTools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") }, relevantMemories: [], conversationID: UUID(), turnID: UUID())

        for await _ in SlotAgentService.shared.run(request, options: .init(modelContext: nil, conversationID: request.conversationID, turnID: request.turnID, groundingMode: .slotAgent, allowDegradedGrounding: true, preventDoubleGrounding: true, diagnosticsEnabled: true)) {}
        try? await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertEqual(AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration), before)
    }

    @MainActor func testLiveSlotAgentStreamUnregistersCancellationBusOnCancellation() async {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        #endif
        AppCancellationBus.shared.cancel(.chatGeneration)
        let before = AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration)
        let request = AgentRequest(systemPrompt: "diagnostic", history: [], userMessage: "Search the web for SwiftData cancellation patterns", temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 64, maxSteps: 2, availableTools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") }, relevantMemories: [], conversationID: UUID(), turnID: UUID())
        let stream = SlotAgentService.shared.run(request, options: .init(modelContext: nil, conversationID: request.conversationID, turnID: request.turnID, groundingMode: .slotAgent, allowDegradedGrounding: true, preventDoubleGrounding: true, diagnosticsEnabled: true))
        XCTAssertGreaterThanOrEqual(AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration), before)
        let task = Task {
            for await _ in stream {}
        }

        task.cancel()
        AppCancellationBus.shared.cancel(.chatGeneration)
        _ = await task.result
        try? await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertEqual(AppCancellationBus.shared.activeRegistrationCount(category: .chatGeneration), before)
    }

    @MainActor func testLiveSlotAgentDiagnosticSequenceIncludesPostGroundingMilestones() async {
        #if DEBUG
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        defer { ResourceBudgetGate.testSnapshotOverride = nil }
        #endif
        let capturedKinds = DiagnosticSignalKindCapture()
        let observerID = PersistentRuntimeDiagnosticsObserver.shared.addObserver { signal in
            capturedKinds.append(signal.kind)
        }
        defer { PersistentRuntimeDiagnosticsObserver.shared.removeObserver(observerID) }
        let request = AgentRequest(systemPrompt: "diagnostic", history: [], userMessage: "Search the web for SwiftData cancellation patterns", temperature: 0, topP: 1, repetitionPenalty: 1, maxTokens: 64, maxSteps: 2, availableTools: ToolRegistry.all.filter { $0.id.hasPrefix("web.") }, relevantMemories: [], conversationID: UUID(), turnID: UUID())

        for await _ in SlotAgentService.shared.run(request, options: .init(modelContext: nil, conversationID: request.conversationID, turnID: request.turnID, groundingMode: .slotAgent, allowDegradedGrounding: true, preventDoubleGrounding: true, diagnosticsEnabled: true)) {}

        let kinds = capturedKinds.snapshot()
        XCTAssertTrue(kinds.contains(.slotAgentGroundingComplete))
        XCTAssertTrue(kinds.contains(.slotAgentEffectiveRequestBuilt))
        XCTAssertTrue(kinds.contains(.slotAgentDeterministicAnswerBuilt))
        XCTAssertTrue(kinds.contains(.slotAgentDoneYielded))
        XCTAssertTrue(kinds.contains(.slotAgentEndEmitted))
        XCTAssertTrue(kinds.contains(.slotAgentContinuationFinished))
    }



    func testAutomaticCampaignNeverSchedulesManualOnlyScenarios() {
        let campaign = PersistentDiagnosticCampaign(enabled: true, runContinuously: true, scenarios: [.plainFastPrompt, .lifecycleCancellation, .liveAgentStream, .agentToolPrompt, .sandboxedToolPlanOnly])
        let automatic = campaign.automaticOnly()
        XCTAssertEqual(automatic.scenarios, [.plainFastPrompt, .sandboxedToolPlanOnly])
        XCTAssertFalse(automatic.scenarios.contains { $0.automationPolicy != .automatic })
    }

    func testLiveAgentStreamCannotRunWithoutExplicitUserRequest() async throws {
        let store = try makeStore()
        let runner = PersistentRuntimeDiagnosticsRunner(store: store)
        let record = await runner.runLiveAgentStream(explicitUserRequested: false)
        XCTAssertNil(record)
    }

    @MainActor func testResourceGateAllowsNominalAndFairActiveState() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertTrue(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .fair, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertTrue(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = nil
    }

    @MainActor func testResourceGateDeniesSeriousCriticalAndBackground() {
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .serious, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .active, lowPowerModeEnabled: false, thermalState: .critical, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = .init(scenePhase: .background, lowPowerModeEnabled: false, thermalState: .nominal, recentMemoryWarningCount: 0, lastMemoryWarningAt: nil)
        XCTAssertFalse(ResourceBudgetGate.allowsHeavyModelWork(reason: ModelLoadIntent.userChat.rawValue))
        ResourceBudgetGate.testSnapshotOverride = nil
    }

    func testLifecycleProbePassesAfterInactiveBackgroundActiveCycle() async {
        let controller = LifecycleProbeController()
        let record = PersistentDiagnosticRunRecord(campaignID: UUID(), scenario: .lifecycleCancellation, status: .running)
        _ = await controller.arm(record: record)
        _ = await controller.record(phase: .inactive)
        _ = await controller.record(phase: .background)
        let result = await controller.record(phase: .active)
        XCTAssertEqual(result?.record.status, .passed)
        XCTAssertEqual(result?.record.metrics.appBecameInactiveOrBackgroundDuringRun, true)
    }

    func testLifecycleProbeSkipsWhenNoTransitionOccurs() async {
        let controller = LifecycleProbeController()
        let record = PersistentDiagnosticRunRecord(campaignID: UUID(), scenario: .lifecycleCancellation, status: .running)
        _ = await controller.arm(record: record)
        let finalized = await controller.finalizeWithoutTransition()
        XCTAssertEqual(finalized?.status, .skipped)
        XCTAssertEqual(finalized?.metrics.appBecameInactiveOrBackgroundDuringRun, false)
    }

    func testDiagnosticsWriteBufferCapsRecordCountAndBatchesWrites() async throws {
        let store = try makeStore()
        for index in 0..<60 {
            await store.appendEvent(PersistentDiagnosticEvent(code: "batched", message: "safe synthetic event \(index)"))
        }
        let data = await store.readLogDataForExport(full: true)
        let text = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(text.contains("safe synthetic event 59"))

        var state = PersistentDiagnosticState()
        state.records = (0..<520).map { _ in PersistentDiagnosticRunRecord(campaignID: UUID(), scenario: .plainFastPrompt, status: .passed) }
        try await store.saveState(state)
        let restored = try XCTUnwrap(await store.loadState())
        XCTAssertEqual(restored.records.count, 500)
    }

    func testAgentCancellationPersistsCancelledStateNotInterrupted() async throws {
        let store = try makeStore()
        let coordinator = AgentRunCoordinator(store: store)
        let record = PersistentDiagnosticRunRecord(campaignID: UUID(), scenario: .agentCancellation, status: .running)
        let task = Task {
            await coordinator.run(record: record, cancellationReason: "unit-test-cancel") { starting in
                try await Task.sleep(nanoseconds: 1_000_000_000)
                return starting
            }
        }
        try? await Task.sleep(nanoseconds: 20_000_000)
        await coordinator.cancelActive(reason: "unit-test-cancel")
        let result = await task.value
        XCTAssertEqual(result.status, .cancelled)
        XCTAssertEqual(result.metrics.didCancel, true)
        XCTAssertEqual(result.metrics.cancellationReason, "unit-test-cancel")
        XCTAssertNotEqual(result.status, .interrupted)
    }


    func testFullDiagnosticsExportDoesNotDuplicateFlushedRingEntries() async throws {
        let store = try makeStore()
        for index in 0..<50 {
            await store.appendEvent(PersistentDiagnosticEvent(code: "no_duplicate", message: "no duplicate event \(index)"))
        }
        let data = await store.readLogDataForExport(full: true)
        let text = String(data: data, encoding: .utf8) ?? ""
        XCTAssertEqual(text.components(separatedBy: "no_duplicate").count - 1, 50)
    }


    func testDiagnosticsExportReadsLegacySingleEntryJSONLLines() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let store = PersistentRuntimeDiagnosticsStore(directoryURL: directory)
        let legacyEntry = PersistentDiagnosticLogEntry(kind: "event", recordID: nil, campaignID: nil, event: PersistentDiagnosticEvent(code: "legacy_line", message: "legacy event"), record: nil)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let line = String(data: try encoder.encode(legacyEntry), encoding: .utf8) ?? ""
        try (line + "\n").write(to: directory.appendingPathComponent("persistent-runtime-diagnostics.jsonl"), atomically: true, encoding: .utf8)

        let data = await store.readLogDataForExport(full: true)
        let text = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(text.contains("legacy_line"))
        XCTAssertTrue(text.contains("legacy event"))
    }

    func testMetricKitExportUsesSummariesAndRetentionCountsOnlyRawPayloads() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let store = MetricKitDiagnosticsStore(directoryURL: directory)
        let payload = Data("{}".utf8)
        for _ in 0..<51 {
            await store.persistMetricPayload(payload)
            try? await Task.sleep(nanoseconds: 1_000_000)
        }

        let urls = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let rawPayloads = urls.filter { $0.lastPathComponent.hasPrefix("mxmetric-") && !$0.lastPathComponent.hasSuffix(".summary.json") }
        let summaries = await store.exportSummaryPayloadURLs()

        XCTAssertEqual(rawPayloads.count, 50)
        XCTAssertEqual(summaries.count, 50)
        XCTAssertTrue(summaries.allSatisfy { $0.lastPathComponent.hasSuffix(".summary.json") })
    }


    private func makeStore() throws -> PersistentRuntimeDiagnosticsStore {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        return PersistentRuntimeDiagnosticsStore(directoryURL: url)
    }
}

private final class DiagnosticSignalKindCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var kinds: [PersistentRuntimeDiagnosticSignalKind] = []

    func append(_ kind: PersistentRuntimeDiagnosticSignalKind) {
        lock.lock()
        kinds.append(kind)
        lock.unlock()
    }

    func snapshot() -> [PersistentRuntimeDiagnosticSignalKind] {
        lock.lock()
        defer { lock.unlock() }
        return kinds
    }
}
